"""Object-scale single-slice prompting with supervision in the original pixel grid."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/roi_prompt_protocol.md"
FORMAL = "799311c1c176f2d8239c619e4c6f35a67fa0ec9300c60c3194cb87d97ade58a3"


def geometry(logits, local):
    centers = np.full((len(logits),2),63.5,np.float32)
    sides = np.full(len(logits),128.,np.float32)
    if local:
        for j,case in enumerate(logits):
            yy,xx = np.where((case>0).sum(0)>=2)
            if len(xx):
                low,high = np.quantile(np.stack((xx,yy),axis=1),[0.02,0.98],axis=0)
                centers[j] = (low+high)/2
                sides[j] = np.clip(2.5*(high-low+1).max()+8,48,128)
    return centers,sides


class View:
    def __init__(self, centers, sides, local, device, features=None):
        self.local = local
        self.features = features
        if local:
            xy = torch.as_tensor(centers,device=device,dtype=torch.float32)
            scale = torch.as_tensor(sides,device=device,dtype=torch.float32)/128
            matrix = native.matrix_for(scale,torch.zeros_like(scale),torch.ones_like(scale))
            self.source,self.back = native.grids(xy,matrix,128)
            self.window = ((1-self.back.abs().amax(-1))*8).clamp(0,1)[:,None]

    def crop(self, image, index):
        if not self.local:
            return image
        return F.grid_sample(image,self.source[index],align_corners=True,padding_mode="border")

    def decode(self, model, embedding, guide_rot, image_rot, index, rotation):
        if not self.local:
            return model(embedding,guide_rot,image_rot)
        guide = torch.rot90(guide_rot,-rotation,(-2,-1))
        image = torch.rot90(image_rot,-rotation,(-2,-1))
        local_guide = torch.rot90(self.crop(guide,index),rotation,(-2,-1))
        local_image = torch.rot90(self.crop(image,index),rotation,(-2,-1))
        fields = model(embedding,local_guide,local_image)
        native_fields = []
        for rotated in fields:
            restored = F.grid_sample(torch.rot90(rotated,-rotation,(-2,-1)).float(),self.back[index],
                                     align_corners=True,padding_mode="border")
            value = guide+self.window[index]*(restored-guide)
            native_fields.append(torch.rot90(value,rotation,(-2,-1)))
        return tuple(native_fields)


def training_bank(args, uids):
    _,inner,_ = native.splits(uids,args.fold)
    path = args.crossfit/f"oof-{args.fold}.npz"
    info = incumbent.read(path.with_suffix(".json"))
    if rc.hash_file(path)!=info["payload"]:
        raise ValueError("OOF payload changed")
    with np.load(path,allow_pickle=False) as z:
        ids=z["ids"].tolist()
        if ids!=sorted(inner):
            raise ValueError("Ineligible prompt training patients")
        guide=z["logits"].astype(np.float32)
    return ids,guide,info


@torch.inference_mode()
def encode_cases(args, images, ids, view, target):
    model=prompt.image_encoder(args.public,512)
    start=time.monotonic()
    for begin in range(0,len(ids),4):
        subset=ids[begin:begin+4]
        ix=torch.arange(begin,begin+len(subset),device="cuda")
        image=align.registered_ct(torch.from_numpy(images[subset,None]).cuda(),-1,-1).float()
        crop=view.crop(image,ix)
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                embedding=model(prompt.image_input(crop,k,512)).float()
            if not torch.isfinite(embedding).all():
                raise ValueError("Nonfinite single-slice embedding")
            target[subset,k]=embedding.cpu().numpy()
        if begin%100==0:
            print("VIEW_FEATURE",args.variant,begin,len(ids),time.monotonic()-start,flush=True)


def prepare(args):
    images,_,_,uids=native.inputs(args.data)
    ids,guide,info=training_bank(args,uids)
    shift=incumbent.read(args.align/f"fold-{args.fold}-inner-registered/fit.json")["applied_shift"]
    if shift!=[-1,-1]:
        raise ValueError("Unverified training registration")
    centers,sides=geometry(guide,args.variant=="local")
    view=View(centers,sides,args.variant=="local","cuda")
    path=args.run/f"features-{args.fold}-{args.variant}.npy"
    if path.exists():
        raise FileExistsError(path)
    cache=np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(len(images),4,256,32,32))
    encode_cases(args,images,ids,view,cache)
    cache.flush();del cache
    rc.write_json(path.with_suffix(".json"),{"ids":ids,"fold":args.fold,"variant":args.variant,
                  "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"oof":info["payload"],
                  "weights":prompt.WEIGHT,"data":rc.DATA_HASH,"payload":rc.hash_file(path),
                  "centers":centers.tolist(),"sides":sides.tolist()})


def fit(args):
    _,_,_,uids=native.inputs(args.data)
    ids,guide,info=training_bank(args,uids)
    path=args.run/f"features-{args.fold}-{args.variant}.npy"
    owner=incumbent.read(path.with_suffix(".json"))
    centers,sides=geometry(guide,args.variant=="local")
    if (owner["ids"]!=ids or owner["oof"]!=info["payload"] or owner["payload"]!=rc.hash_file(path)
            or owner["source"]!=rc.hash_file(__file__) or owner["protocol"]!=rc.hash_file(PROTOCOL)
            or not np.array_equal(centers,owner["centers"]) or not np.array_equal(sides,owner["sides"])):
        raise ValueError("Training view identity differs")
    cache=np.load(path,mmap_mode="r",allow_pickle=False)
    view=View(centers,sides,args.variant=="local","cuda",cache)
    view.receipt=rc.hash_file(path.with_suffix(".json"))
    prompt.train(args,variant=args.variant,study_protocol=PROTOCOL,view=view)


@torch.inference_mode()
def predict(args, model, images, ids, guide):
    centers,sides=geometry(guide,args.variant=="local")
    view=View(centers,sides,args.variant=="local","cuda")
    vision=prompt.image_encoder(args.public,512)
    result=[]
    for begin in range(0,len(ids),4):
        subset=ids[begin:begin+4]
        ix=torch.arange(begin,begin+len(subset),device="cuda")
        image=align.registered_ct(torch.from_numpy(images[subset,None]).cuda(),-1,-1).float()
        crop=view.crop(image,ix)
        raw=torch.from_numpy(guide[begin:begin+len(subset)]).cuda()
        total=0
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                emb=vision(prompt.image_input(crop,k,512)).float()
                logits,_=view.decode(model,emb,torch.rot90(raw,k,(-2,-1)),
                                     torch.rot90(image/255,k,(-2,-1)),ix,k)
            total=total+torch.rot90(logits,-k,(-2,-1))
        result.append((total/4).cpu().numpy())
    return np.concatenate(result),sides


def check(args):
    images,truth,_,uids=native.inputs(args.data)
    _,fit_ids,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(group,valid,args.fold)
    ids=library+checked
    if rc.hash_file(args.formal/"assessment/outer_report.json")!=FORMAL:
        raise ValueError("Accepted reference differs")
    choice=incumbent.read(args.formal/"selection.json")["folds"][args.fold]
    owner=args.encoder if choice["variant"]=="adapted" else args.ct
    prior=args.texture/f"features-{args.fold}-inner.npz"
    bank=owner/f"pred-{args.fold}-{choice['variant']}.npz"
    record=incumbent.read(owner/f"inner-{args.fold}-{choice['variant']}.json")
    if (record["check"]!=checked or record["library"]!=library or record["prediction"]!=rc.hash_file(bank)
            or incumbent.read(prior.with_suffix(".json"))["payload"]!=rc.hash_file(prior)):
        raise ValueError("Guide identity differs")
    with np.load(prior,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Original case order differs")
        base=incumbent.anchor_logits(z["base"].astype(np.float32),np.unpackbits(z["old"],axis=-1).astype(bool))
    with np.load(bank,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Accepted case order differs")
        original=np.unpackbits(z["masks"],axis=-1).astype(bool)
        w=choice["weight"]
        guide=incumbent.anchor_logits((1-w)*base+w*z["logits"].astype(np.float32),original)
    folder=args.run/f"fold-{args.fold}-{args.variant}"
    fitted=incumbent.read(folder/"fit.json")
    if (fitted["train_indices"]!=sorted(fit_ids) or fitted["scope"]!="inner" or fitted["steps"]!=prompt.STEPS
            or fitted["variant"]!=args.variant or fitted["source"]!=rc.hash_file(prompt.__file__)
            or fitted["protocol"]!=rc.hash_file(PROTOCOL) or fitted["checkpoint"]!=rc.hash_file(folder/"final.pt")):
        raise ValueError("View model training identity differs")
    model=prompt.PromptRanks(args.public,pretrained=False,image_size=512).cuda().eval()
    model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
    logits,sides=predict(args,model,images,ids,guide)
    if not np.isfinite(logits).all():raise ValueError("Nonfinite restored output")
    count=len(library)
    grid=[float(scoring.dice(truth[library],(1-v)*guide[:count]+v*logits[:count]>0).mean()) for v in incumbent.WEIGHTS]
    v=incumbent.WEIGHTS[int(np.argmax(grid))]
    lock={"fold":args.fold,"variant":args.variant,"library":library,"grid":grid,"weight":v,"model":fitted["checkpoint"],
          "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"reference":record["prediction"]}
    rc.write_json(args.run/f"selection-{args.fold}-{args.variant}.json",lock)
    masks=(1-v)*guide[count:]+v*logits[count:]>0
    scores={"reference":float(scoring.dice(truth[checked],original[count:]).mean()),
            "standalone":float(scoring.dice(truth[checked],logits[count:]>0).mean()),
            "candidate":float(scoring.dice(truth[checked],masks).mean())}
    path=args.run/f"pred-{args.fold}-{args.variant}.npz"
    np.savez_compressed(path,ids=checked,masks=np.packbits(masks,axis=-1),logits=logits[count:].astype(np.float16))
    rc.write_json(args.run/f"inner-{args.fold}-{args.variant}.json",{**lock,"check":checked,"scores":scores,
                  "prediction":rc.hash_file(path),"check_side_percentiles":np.percentile(sides[count:],[0,25,50,75,100]).tolist()})
    print("ROI_CHECK",args.fold,args.variant,scores,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("prepare","fit","check"))
    for key in ("run","data","public","align","crossfit"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("formal","encoder","ct","texture"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--variant",choices=("global","local"),required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    args.image_size=512;args.scope="inner";args.feature_cache=None;args.backbone="vit_b"
    cv2.setNumThreads(1);torch.set_num_threads(3)
    {"prepare":prepare,"fit":fit,"check":check}[args.phase](args)
