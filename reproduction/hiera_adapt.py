"""Adapt the image decoder neck and last Hiera blocks on eligible LIDC patients."""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import canonical_guide as canonical
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits
from reproduction.trajectory_sam import select_weight

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/hiera_adapt_protocol.md"
STABLE = PROTOCOL.with_name("stable_refinement_protocol.md")
REFERENCE = "5f30f6c9500bd196c4f4f9cf8dbe8b0153f27bd8135d88254d1b079d6c9f189f"
STEPS = 2500


def parameter_groups(model, tuned):
    late = list(model.encoder.trunk.blocks[-4:].parameters())
    projections = list(model.encoder.neck.convs.parameters())
    for layer in (model.head.decoder.conv_s0,model.head.decoder.conv_s1):
        projections.extend(layer.parameters())
    for param in late+projections:
        param.requires_grad_(tuned)
    projection_ids = {id(p) for p in projections}
    head = [p for p in model.head.parameters() if p.requires_grad and id(p) not in projection_ids]
    groups = [{"params":head,"base_lr":.00005,"lr":.00005,"weight_decay":.0002}]
    if tuned:
        groups.extend([{"params":late,"base_lr":.00001,"lr":.00001,"weight_decay":.003},
                       {"params":projections,"base_lr":.00004,"lr":.00004,"weight_decay":.003}])
    return groups


class AdaptiveImageModel(nn.Module):
    def __init__(self,public,warm):
        super().__init__()
        state = hierarchy.open_public(public)
        self.encoder = hierarchy.image_encoder(state)
        self.head = hierarchy.HierarchicalRanks(public)
        self.head.load_state_dict(warm)
        self.register_buffer("no_memory",state["no_mem_embed"].reshape(1,256,1,1))

    def forward(self, guide, image, rotation):
        original = torch.rot90(image,-rotation,(-2,-1))*255
        normalized = prompt.image_input(original,rotation,512)
        high,middle,deep = self.encoder(normalized)["backbone_fpn"]
        embedding = hierarchy.pack(self.head.decoder.conv_s0(high),self.head.decoder.conv_s1(middle),deep+self.no_memory)
        # Match the half-precision public feature representation used for warm training and validation.
        return self.head(embedding.half().float(),guide,image)


def warm_identity(args,training):
    folder = args.parent/f"fold-{args.fold}-pyramid"
    fit = json.loads((folder/"fit.json").read_text())
    if (fit["steps"]!=5000 or fit["scope"]!="inner" or fit["variant"]!="pyramid"
            or fit["train_indices"]!=sorted(training) or fit["public_weights"]!=hierarchy.WEIGHT
            or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
            or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
        raise ValueError("Warm single-slice decoder identity differs")
    return torch.load(folder/"final.pt",weights_only=True,map_location="cpu"),fit["checkpoint"]


def train(args):
    images,truth,_,uids = native.inputs(args.data)
    _,training,_ = native.splits(uids,args.fold)
    warm,initial = warm_identity(args,training)
    path = args.crossfit/f"oof-{args.fold}.npz"
    if rc.hash_file(path)!=json.loads(path.with_suffix(".json").read_text())["payload"]:
        raise ValueError("Out-of-fit training guide changed")
    with np.load(path,allow_pickle=False) as bank:
        ids=bank["ids"].tolist();prior=bank["logits"].astype(np.float32)
    if ids!=sorted(training):raise ValueError("Guide includes ineligible cases")
    shift=json.loads((args.align/f"fold-{args.fold}-inner-registered/fit.json").read_text())["applied_shift"]
    if shift!=[-1,-1]:raise ValueError("The registered single-slice convention changed")
    tuned=args.variant=="tuned"
    seed=195171+args.fold;steps=5 if args.smoke else STEPS
    folder=args.run/(f"smoke-{args.fold}-{args.variant}" if args.smoke else f"fold-{args.fold}-{args.variant}")
    folder.mkdir(exist_ok=False)
    if args.smoke:ids,prior=ids[:32],prior[:32]
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=AdaptiveImageModel(args.public,warm).cuda()
    groups=parameter_groups(model,tuned)
    model.train();model.encoder.eval()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    shadow=[q for p,q in zip(model.parameters(),ema.parameters()) if p.requires_grad]
    opt=torch.optim.AdamW(groups,fused=True)
    x=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*shift).float()/255
    y=torch.from_numpy(truth[ids]).cuda().float();g=torch.from_numpy(prior).cuda()
    _,inv,n=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    weights=torch.from_numpy(.45/len(ids)+.55/(len(n)*n[inv])).float().cuda()
    rc.write_json(folder/"plan.json",{"fold":args.fold,"scope":"inner","seed":seed,"steps":steps,"variant":args.variant,
        "train_indices":ids,"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"oof":rc.hash_file(path),
        "initial":initial,"public_weights":hierarchy.WEIGHT,"group_parameters":[sum(p.numel() for p in group["params"]) for group in groups]})
    start=time.monotonic()
    with (folder/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(weights,12,replacement=True)
            k=int(torch.randint(4,()).item())
            image=torch.rot90(x[ix],k,(-2,-1));target=torch.rot90(y[ix],k,(-2,-1))
            guide=torch.rot90(g[ix],k,(-2,-1))
            guide=guide*(.75+.5*torch.rand(len(ix),1,1,1,device="cuda"))+(torch.rand(len(ix),1,1,1,device="cuda")-.5)*2
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                logits,raw=model(guide,image,k)
            radius=(logits.sigmoid().sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
            actual=(target.sum((-2,-1)).clamp_min(1)/math.pi).sqrt()
            loss=vote.mask_loss(logits,target)+.4*vote.mask_loss(raw,target)+.08*F.smooth_l1_loss(radius,actual,beta=.5)
            if step==1:
                first={"ids":[ids[j] for j in ix.tolist()],"rotation":k,"loss":float(loss.item())}
                if tuned and not args.smoke:
                    paired=json.loads((args.run/f"fold-{args.fold}-frozen/first.json").read_text())
                    if (first["ids"]!=paired["ids"] or k!=paired["rotation"]
                            or not np.isclose(first["loss"],paired["loss"],rtol=2e-4,atol=4e-5)):
                        raise ValueError(f"Paired first batch or loss differs: {first} vs {paired}")
                rc.write_json(folder/"first.json",first)
            opt.zero_grad(set_to_none=True);loss.backward()
            if tuned and args.smoke:
                for group in groups[1:]:
                    if not any(p.grad is not None and p.grad.abs().max().item()>0 for p in group["params"]):
                        raise ValueError("A registered image adaptation group has no gradient")
            torch.nn.utils.clip_grad_norm_(parameters,4,error_if_nonfinite=True)
            schedule=min(1,step/100)*(.04+.96*(1+math.cos(math.pi*step/steps))/2)
            for group in opt.param_groups:group["lr"]=group["base_lr"]*schedule
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(shadow,parameters,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":float(loss.item()),"seconds":time.monotonic()-start}
                log.write(json.dumps(row,allow_nan=False)+"\n");print("IMAGE_ADAPT",args.fold,args.variant,row,flush=True)
    torch.save(ema.state_dict(),folder/"final.pt")
    rc.write_json(folder/"fit.json",{"checkpoint":rc.hash_file(folder/"final.pt"),"plan":rc.hash_file(folder/"plan.json"),
                                    "seconds":time.monotonic()-start})


@torch.inference_mode()
def predict(model,images,ids,guide):
    result=[]
    for start in range(0,len(ids),12):
        cases=ids[start:start+12]
        x=align.registered_ct(torch.from_numpy(images[cases,None]).cuda(),-1,-1).float()/255
        g=torch.from_numpy(guide[start:start+len(cases)]).cuda()
        total=0
        for k in range(4):
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                z,_=model(torch.rot90(g,k,(-2,-1)),torch.rot90(x,k,(-2,-1)),k)
            total=total+torch.rot90(z,-k,(-2,-1))
        result.append((total/4).cpu().numpy())
    return np.concatenate(result)


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,training,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(group,valid,args.fold)
    ids=library+checked;n=len(library)
    if rc.hash_file(args.stable/"summary.json")!=REFERENCE:raise ValueError("Stable inner study differs")
    field,receipt=verified_field(args.stable,"inner",args.fold,STABLE)
    with np.load(field,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong registered image order")
        guide=z["adapted"]
    sealed=json.loads((args.stable/f"inner-{args.fold}.json").read_text())
    bank=args.stable/f"pred-{args.fold}.npz"
    if sealed["prediction"]!=rc.hash_file(bank):raise ValueError("Stable predictions differ")
    with np.load(bank,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Stable reference order differs")
        outputs={"reference":z["adapted"]}
    view=cache_owner.verified_parent(args)
    other=[]
    seeds=canonical.model_sources(args,training,args.fold)
    decoder=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    for directory,_ in seeds[1:]:
        decoder.load_state_dict(torch.load(directory/"final.pt",weights_only=True,map_location="cpu"))
        other.append(prompt.predict(decoder,view.features,ids,images,guide).astype(np.float16))
    del decoder
    warm,initial=warm_identity(args,training)
    choices={}
    for name in ("frozen","tuned"):
        folder=args.run/f"fold-{args.fold}-{name}"
        fit=json.loads((folder/"fit.json").read_text());plan=json.loads((folder/"plan.json").read_text())
        if (fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or fit["plan"]!=rc.hash_file(folder/"plan.json")
                or plan["steps"]!=STEPS or plan["scope"]!="inner" or plan["initial"]!=initial
                or plan["train_indices"]!=sorted(training) or plan["variant"]!=name
                or plan["source"]!=rc.hash_file(__file__) or plan["protocol"]!=rc.hash_file(PROTOCOL)):
            raise ValueError("Unverified adapted image model")
        model=AdaptiveImageModel(args.public,warm).cuda().eval()
        model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
        member=predict(model,images,ids,guide)
        del model
        mean=mean_saved_logits([member.astype(np.float16),*other])
        if not np.isfinite(mean).all():raise ValueError("Nonfinite adapted image prediction")
        weight,grid=select_weight(truth[library],guide[:n],mean[:n])
        choices[name]={"weight":weight,"grid":grid,"checkpoint":fit["checkpoint"]}
        outputs[name]=np.packbits((1-weight)*guide+weight*mean>0,axis=-1)
        outputs["single_"+name]=np.packbits(member>0,axis=-1)
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**outputs)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,"choices":choices,
        "field":receipt["payload"],"reference":sealed["prediction"],"prediction":rc.hash_file(dst),
        "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("IMAGE_ADAPT_LOCK",args.fold,choices,flush=True)


def score(args):
    locks=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    if any(r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz") for f,r in enumerate(locks)):
        raise ValueError("Incomplete image adaptation pair")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(locks):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Wrong image adaptation patient order")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference","frozen","tuned","single_frozen","single_tuned")}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores})
        print("IMAGE_ADAPT_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("fit","lock","score"))
    for name in ("run","parent","ensemble","data","public","stable","crossfit","align"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0)
    parser.add_argument("--variant",choices=("frozen","tuned"),default="frozen")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    {"fit":train,"lock":lock,"score":score}[args.phase](args)
