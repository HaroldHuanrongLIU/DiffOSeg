"""Single-slice masked CT pretraining for the native Hiera decoder skips."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import OFFSETS, mean_saved_logits
from reproduction.semantic_fusion import ct_and_split

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/masked_ct_protocol.md"
MODES = ("visible", "hidden")
PRE_STEPS = 2500


class NativeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2,16,3,padding=1), nn.GroupNorm(8,16), nn.SiLU())
        self.native = rc.Residual(16,16)
        self.middle = nn.Sequential(rc.Residual(16,32,2), rc.Residual(32,32))
        self.fuse = rc.Residual(48,16)

    def forward(self, image, hidden):
        observed = torch.where(hidden, 0., 2*image-1)
        fine = self.native(self.stem(torch.cat((observed, hidden.to(image.dtype)),1)))
        mid = self.middle(fine)
        up = F.interpolate(mid, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((fine,up),1)), mid


def block_mask(permutations, device):
    blocks = np.zeros((len(permutations),256), dtype=bool)
    np.put_along_axis(blocks, permutations[:,:128], True, axis=1)
    small = torch.from_numpy(blocks).to(device).reshape(-1,1,16,16)
    return small.repeat_interleave(8,-1).repeat_interleave(8,-2)


def reconstruction_target(image):
    dx = F.pad(image[:,:,:,1:]-image[:,:,:,:-1], (0,1,0,0))
    dy = F.pad(image[:,:,1:]-image[:,:,:-1], (0,0,0,1))
    return torch.cat((2*image-1,4*dx,4*dy),1)


def region_loss(prediction, target, selected, spatial_weight):
    weight = selected*spatial_weight
    error = F.smooth_l1_loss(prediction.float(), target, beta=.05, reduction="none")
    return (error.mean(1,keepdim=True)*weight).sum()/weight.sum()


def read(path):
    return json.loads(path.read_text())


def fitting_ids(uids, fold):
    outer, train, valid = native.splits(uids,fold)
    patients = lambda ids: {str(uids[i]).split("_")[0] for i in ids}
    if patients(train) & patients(valid+outer["test_indices"]):
        raise ValueError("SSL and supervised patients must exclude validation and outer test")
    return sorted(train)


def pretrain(args):
    images, uids = ct_and_split(args)
    ids = fitting_ids(uids,args.fold)
    shift = read(args.align/f"fold-{args.fold}-inner-registered/fit.json")["applied_shift"]
    if shift != [-1,-1]:
        raise ValueError("The CT registration changed")
    folder = args.run/(f"smoke-pre-{args.mode}-{args.fold}" if args.smoke else f"pre-{args.mode}-{args.fold}")
    folder.mkdir(exist_ok=False)
    steps = 5 if args.smoke else PRE_STEPS
    ids = ids[:32] if args.smoke else ids
    seed = 245017+args.fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    view = align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*shift).float()/255
    _,inverse,counts = np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    weights = .45/len(ids)+.55/(len(counts)*counts[inverse])
    encoder = NativeEncoder().cuda()
    head = nn.Conv2d(16,3,1).cuda()
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=.0006, fused=True, weight_decay=.0002)
    coord = torch.arange(128,device="cuda")-63.5
    importance = 1+2*torch.exp(-(coord[:,None].square()+coord[None].square())/(2*22**2))
    schedule = hashlib.sha256()
    with (folder/"train.jsonl").open("x",buffering=1) as out:
        for step in range(1,steps+1):
            batch = rng.choice(len(ids),16,p=weights)
            rotation = int(rng.integers(4))
            perms = rng.random((16,256)).argsort(axis=1)
            schedule.update(batch.astype("<i4").tobytes()+bytes([rotation])+perms[:,:128].astype("uint8").tobytes())
            image = torch.rot90(view[batch],rotation,(-2,-1))
            hidden = block_mask(perms,"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                fine,_ = encoder(image,hidden)
                prediction = head(fine)
            target = reconstruction_target(image)
            loss = region_loss(prediction,target,hidden if args.mode=="hidden" else ~hidden,importance)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,4,error_if_nonfinite=True)
            opt.param_groups[0]["lr"] = .0006*min(step/60,1)*(.05+.95*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            if step==1 or step%250==0 or step==steps:
                results = {"step":step, "hidden":region_loss(prediction,target,hidden,importance).item(),
                           "visible":region_loss(prediction,target,~hidden,importance).item()}
                out.write(json.dumps(results,allow_nan=False)+"\n")
    checkpoint = folder/"encoder.pt"
    torch.save(encoder.state_dict(),checkpoint)
    record = {"fold":args.fold,"mode":args.mode,"train":ids,"steps":steps,"seed":seed,
              "schedule":schedule.hexdigest(),"checkpoint":rc.hash_file(checkpoint),"source":rc.hash_file(__file__),
              "protocol":rc.hash_file(PROTOCOL),"data":rc.DATA_HASH,"registration":shift,"smoke":args.smoke}
    rc.write_json(folder/"receipt.json",record)
    print("CT_PRETRAIN",args.fold,args.mode,steps,record["schedule"],flush=True)


class MaskedRanks(hierarchy.HierarchicalRanks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.ct = NativeEncoder()
            self.projections = nn.ModuleList([nn.Conv2d(16,32,1),nn.Conv2d(32,64,1)])
        for proj in self.projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def spatial_features(self, embedding, guide, image):
        coarse, high = hierarchy.unpack(embedding,True)
        ct = self.ct(image,torch.zeros_like(image,dtype=torch.bool))
        return coarse,[skip+proj(feature) for skip,proj,feature in zip(high,self.projections,ct,strict=True)]


def pre_identity(args, ids, mode):
    folder = args.run/f"pre-{mode}-{args.fold}"
    r = read(folder/"receipt.json")
    if (r["fold"]!=args.fold or r["mode"]!=mode or r["train"]!=ids or r["steps"]!=PRE_STEPS or r["smoke"]
            or r["source"]!=rc.hash_file(__file__) or r["protocol"]!=rc.hash_file(PROTOCOL)
            or r["data"]!=rc.DATA_HASH or r["checkpoint"]!=rc.hash_file(folder/"encoder.pt")):
        raise ValueError("Wrong fold, schedule or source for CT pretraining")
    return folder, r


def train(args):
    _,uids = ct_and_split(args)
    ids = fitting_ids(uids,args.fold)
    folder,r = pre_identity(args,ids,args.mode)
    _,control = pre_identity(args,ids,"visible" if args.mode=="hidden" else "hidden")
    if control["schedule"]!=r["schedule"]:
        raise ValueError("Matched pretraining schedules differ")
    state = torch.load(folder/"encoder.pt",map_location="cpu",weights_only=True)

    class InitializedRanks(MaskedRanks):
        def __init__(self,*params,**kwargs):
            super().__init__(*params,**kwargs)
            self.ct.load_state_dict(state)

    label=f"{args.mode}-{args.seed_index}"
    prompt.train(args,decoder_type=InitializedRanks,variant=label,study_protocol=PROTOCOL,
                 view=cache_owner.verified_parent(args),public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index])
    out = args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{label}")
    if not args.smoke:
        ordinary = (args.parent/f"fold-{args.fold}-pyramid" if args.seed_index==0 and args.fold<2 else
                    args.ensemble/f"fold-{args.fold}-seed-{args.seed_index}")
        def first(d):
            with (d/"train.jsonl").open() as stream:
                return json.loads(next(stream))["loss"]
        if not np.isclose(first(out),first(ordinary),rtol=2e-4,atol=4e-5):
            raise ValueError("Initial zero-residual supervision differs from E")
    rc.write_json(out/"pretraining.json",{"pre":r["checkpoint"],"pre_receipt":rc.hash_file(folder/"receipt.json"),
                  "fit":rc.hash_file(out/"fit.json"),"source":rc.hash_file(__file__)})


def lock(args):
    if (rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT or
            rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER):
        raise ValueError("Formal E anchor changed")
    images,uids = ct_and_split(args)
    fit = fitting_ids(uids,args.fold)
    _,_,valid = native.splits(uids,args.fold)
    group = type("Group",(),{"series_uid":uids})()
    library, check = polar.calibration_split(group,valid,args.fold)
    ids = library+check
    path,field = verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:
            raise ValueError("Wrong guide cases")
        guide=z["adapted"]
    reference=args.stable/f"pred-{args.fold}.npz"
    old=read(args.stable/f"selection-{args.fold}.json")
    if (old["library"]!=library or old["check"]!=check or old["prediction"]!=rc.hash_file(reference)
            or old["field"]!=field["payload"] or old["choices"]["adapted"]["weight"]!=.25):
        raise ValueError("Frozen E comparison differs")
    with np.load(reference,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:
            raise ValueError("Archived patient order differs")
        masks={"reference":z["adapted"]}
    model=MaskedRanks(args.public).cuda().eval()
    view=cache_owner.verified_parent(args)
    members={}
    for name in MODES:
        directory,pre=pre_identity(args,fit,name)
        packed=[]
        hashes=[]
        for i,offset in enumerate(OFFSETS):
            label=f"{name}-{i}"
            fdir=args.run/f"fold-{args.fold}-{label}"
            record=read(fdir/"fit.json")
            binding=read(fdir/"pretraining.json")
            if (record["train_indices"]!=fit or record["steps"]!=5000 or record["scope"]!="inner"
                    or record["seed"]!=50171+args.fold+offset or record["variant"]!=label
                    or record["architecture"]!="InitializedRanks" or record["public_weights"]!=hierarchy.WEIGHT
                    or record["view_receipt"]!=view.receipt or record["protocol"]!=rc.hash_file(PROTOCOL)
                    or record["source"]!=rc.hash_file(prompt.__file__) or record["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                    or record["checkpoint"]!=rc.hash_file(fdir/"final.pt") or binding["fit"]!=rc.hash_file(fdir/"fit.json")
                    or binding["pre"]!=pre["checkpoint"] or binding["pre_receipt"]!=rc.hash_file(directory/"receipt.json")
                    or binding["source"]!=rc.hash_file(__file__)):
                raise ValueError("Invalid supervised CT model identity")
            model.load_state_dict(torch.load(fdir/"final.pt",weights_only=True,map_location="cpu"))
            z=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
            if not np.isfinite(z).all():
                raise ValueError("Nonfinite CT segmentation output")
            packed.append(z)
            hashes.append(record["checkpoint"])
        masks[name]=np.packbits(.75*guide+.25*mean_saved_logits(packed)>0,axis=-1)
        members[name]=hashes
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":check,
        "reference":old["prediction"],"field":field["payload"],"members":members,"prediction":rc.hash_file(dst),
        "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("CT_INNER_LOCK",args.fold,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("pre","fit","lock"))
    for key in ("run","parent","ensemble","formal","stable","data","crossfit","public","align"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--mode",choices=MODES,default="hidden")
    parser.add_argument("--seed-index",type=int,choices=range(3),default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    {"pre":pretrain,"fit":train,"lock":lock}[args.phase](args)
