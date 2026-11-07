"""Learn a joint four-reader state distribution in the SAM decoder's native grid."""
import argparse
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/joint_prompt_protocol.md"


class StateDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        bits = torch.tensor(list(product((0.,1.),repeat=4)))
        self.register_buffer("bits",bits,persistent=False)
        multi = bits.sum(1)>=2
        index = torch.zeros(16,dtype=torch.long)
        index[multi]=torch.arange(1,12)
        self.register_buffer("index",index,persistent=False)

    def forward(self, unary, interactions):
        with torch.autocast(unary.device.type,enabled=False):
            values = torch.einsum("bkhw,sk->bshw",unary.float(),self.bits.float())
            corrections = torch.cat((torch.zeros_like(interactions[:,:1]),interactions),dim=1).float()
            values = values + corrections[:,self.index]
            conditions = [self.bits[:,k].bool() for k in range(4)]
            count = self.bits.sum(1)
            conditions.extend(count>=k for k in range(1,5))
            odds = torch.stack([values[:,condition].logsumexp(1)-values[:,~condition].logsumexp(1)
                                for condition in conditions],dim=1)
            return odds[:,:4], odds[:,4:], values.log_softmax(1)


class JointPrompts(prompt.PromptRanks):
    def __init__(self,root,pretrained=True,image_size=512,backbone="vit_b"):
        super().__init__(root,pretrained,image_size,backbone)
        with torch.random.fork_rng(devices=[]):
            self.interaction=nn.Conv2d(37,11,1)
        nn.init.zeros_(self.interaction.weight)
        nn.init.zeros_(self.interaction.bias)
        self.states=StateDistribution()

    def distribution(self, embedding, guide, image):
        unary, raw, features = super().forward(embedding,guide,image,return_features=True)
        residual = self.interaction(torch.cat((features,guide.clamp(-12,12)/8,2*image-1),1))
        ranks, supports, law = self.states(unary,residual)
        return ranks,raw,supports,law

    def forward(self,embedding,guide,image):
        ranks,raw,_,_=self.distribution(embedding,guide,image)
        return ranks,raw


def supervised(model,embedding,guide,image,truth):
    ranks,raw,supports,law=model.distribution(embedding,guide,image)
    state=(truth.long()*truth.new_tensor([8,4,2,1],dtype=torch.long)[None,:,None,None]).sum(1)
    extra=0.65*vote.mask_loss(supports,vote.votes(truth))+0.20*F.nll_loss(law,state)
    return ranks,raw,extra


def check(args):
    images,truth,_,uids=native.inputs(args.data)
    _,inner,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(group,valid,args.fold)
    ids=library+checked
    prior=args.texture/f"features-{args.fold}-inner.npz"
    meta=incumbent.read(prior.with_suffix(".json"))
    if meta["indices"]!=ids or meta["payload"]!=rc.hash_file(prior):
        raise ValueError("Incumbent guide identity differs")
    with np.load(prior,allow_pickle=False) as z:
        masks=np.unpackbits(z["old"],axis=-1).astype(bool)
        guide=incumbent.anchor_logits(z["base"].astype(np.float32),masks)
    directory=args.run/f"fold-{args.fold}-{args.variant}"
    fit=incumbent.read(directory/"fit.json")
    if (fit["scope"]!="inner" or fit["variant"]!=args.variant or fit["steps"]!=prompt.STEPS
            or fit["train_indices"]!=sorted(inner) or fit["source"]!=rc.hash_file(prompt.__file__)
            or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["checkpoint"]!=rc.hash_file(directory/"final.pt")):
        raise ValueError("Paired SAM fit identity differs")
    cache=incumbent.read(args.public/"image_features.json")
    if (cache["array"]!=rc.hash_file(args.public/"image_features.npy") or cache["public_weights"]!=prompt.WEIGHT
            or cache["data"]!=rc.DATA_HASH):
        raise ValueError("Lakers public image features differ")
    features=np.load(args.public/"image_features.npy",mmap_mode="r",allow_pickle=False)
    cls=JointPrompts if args.variant=="joint" else prompt.PromptRanks
    model=cls(args.public,pretrained=False,image_size=512).cuda().eval()
    model.load_state_dict(torch.load(directory/"final.pt",map_location="cpu",weights_only=True))
    logits=prompt.predict(model,features,ids,images,guide)
    n=len(library)
    grid=[float(scoring.dice(truth[library],(1-w)*guide[:n]+w*logits[:n]>0).mean()) for w in incumbent.WEIGHTS]
    w=incumbent.WEIGHTS[int(np.argmax(grid))]
    lock={"fold":args.fold,"variant":args.variant,"library":library,"grid":grid,"weight":w,
          "model":fit["checkpoint"],"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)}
    rc.write_json(args.run/f"selection-{args.fold}-{args.variant}.json",lock)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite SAM prediction")
    pred=(1-w)*guide+w*logits>0
    scores={"reference":float(scoring.dice(truth[checked],masks[n:]).mean()),
            "standalone":float(scoring.dice(truth[checked],logits[n:]>0).mean()),
            "candidate":float(scoring.dice(truth[checked],pred[n:]).mean())}
    path=args.run/f"pred-{args.fold}-{args.variant}.npz"
    np.savez_compressed(path,ids=ids,masks=np.packbits(pred,axis=-1),logits=logits.astype(np.float16))
    rc.write_json(args.run/f"inner-{args.fold}-{args.variant}.json",{**lock,"check":checked,"scores":scores,
                  "prediction":rc.hash_file(path)})
    print("JOINT_PROMPT_CHECK",args.fold,args.variant,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("encode","fit","check"))
    for key in ("run","data","public","align","crossfit"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--texture",type=Path)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--variant",choices=("control","joint"),default="joint")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    args.image_size=512;args.scope="inner";args.feature_cache=None
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    if args.phase=="encode":
        prompt.encode(args,verified_folds=(args.fold,))
    elif args.phase=="fit":
        use_joint=args.variant=="joint"
        prompt.train(args,JointPrompts if use_joint else prompt.PromptRanks,
                     supervised if use_joint else None,args.variant,PROTOCOL)
    else:
        check(args)
