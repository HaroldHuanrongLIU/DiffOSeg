"""Train the E decoder using a fixed patient-excluded pixel-risk budget."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import OFFSETS,mean_saved_logits
from reproduction.semantic_fusion import ct_and_split

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/risk_supervision_protocol.md"
ROUTING="1400b0319bf2f2f5d330df4e941e43fa76520b1c2ce478a183815d3106819de3"
MODES=("margin","event")


def region_loss(logits,truth,region):
    ce=F.binary_cross_entropy_with_logits(logits.float(),truth.float(),reduction="none").mean(1)
    weight=region.float()
    return ((ce*weight).sum((-2,-1))/weight.sum((-2,-1)).clamp_min(1)).mean()


def read(path):
    return json.loads(path.read_text())


def train(args):
    if rc.hash_file(args.routing/"summary.json")!=ROUTING:raise ValueError("Risk ranking study identity changed")
    path=args.routing/f"pred-{args.fold}.npz";route=read(args.routing/f"seal-{args.fold}.json")
    if rc.hash_file(path)!=route["prediction"]:raise ValueError("Unverified risk masks")
    with np.load(path,allow_pickle=False) as z,np.load(args.crossfit/f"oof-{args.fold}.npz",allow_pickle=False) as oof:
        ids=z["ids"].tolist()
        if ids!=oof["ids"].tolist():raise ValueError("Training region order differs")
        use=np.unpackbits(z["selected_"+args.mode],axis=-1)
        other=np.unpackbits(z["selected_"+("event" if args.mode=="margin" else "margin")],axis=-1)
        if not np.array_equal(use.sum((1,2)),other.sum((1,2))):raise ValueError("Unmatched case budgets")
        masks=torch.from_numpy(use[:32] if args.smoke else use).cuda()
    label=f"{args.mode}-{args.seed_index}"
    folder=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{label}")
    def objective(model,embedding,guide,image,labels,indices,k,step):
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            logits,raw=model(embedding,guide,image)
        ordinary=prompt.segmentation_loss(logits,raw,labels)
        region=torch.rot90(masks[indices],k,(-2,-1))
        aux=region_loss(logits,labels,region)
        if step==1:
            info={"batch":indices.tolist(),"rotation":k,"ordinary":ordinary.item(),"aux":aux.item(),
                  "counts":region.sum((-2,-1)).tolist()}
            if args.smoke:
                grad=torch.autograd.grad(aux,logits,retain_graph=True)[0]
                if not torch.isfinite(grad).all() or grad.abs().sum().item()==0 or grad.masked_select(~region[:,None].bool()).abs().sum().item()!=0:
                    raise ValueError("Auxiliary gradient escaped the selected region")
            else:
                owner=args.parent/f"fold-{args.fold}-pyramid" if args.seed_index==0 else args.ensemble/f"fold-{args.fold}-seed-{args.seed_index}"
                with (owner/"train.jsonl").open() as st:expect=json.loads(next(st))["loss"]
                if not np.isclose(info["ordinary"],expect,rtol=2e-4,atol=4e-5):raise ValueError("Initial E supervision differs")
                if args.mode=="event":
                    previous=read(args.run/f"fold-{args.fold}-margin-{args.seed_index}/first.json")
                    if any(info[k]!=previous[k] for k in ("batch","rotation","counts")):
                        raise ValueError("Paired examples, rotation or pixel count differ")
            rc.write_json(folder/"first.json",info)
        return ordinary+.15*min((step-1)/299,1.)*aux
    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=label,study_protocol=PROTOCOL,
        view=cache_owner.verified_parent(args),public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index],
        paired_loss=objective)
    rc.write_json(folder/"risk.json",{"source":rc.hash_file(__file__),"fit":rc.hash_file(folder/"fit.json"),
                 "first":rc.hash_file(folder/"first.json"),"regions":route["prediction"]})


def lock(args):
    if rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT:
        raise ValueError("E reference identity changed")
    images,uids=ct_and_split(args)
    _,train_ids,valid=native.splits(uids,args.fold)
    lib,check=polar.calibration_split(type("Groups",(),{"series_uid":uids})(),valid,args.fold)
    ids=lib+check
    path,fr=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("E field order differs")
        guide=z["adapted"]
    old=read(args.stable/f"selection-{args.fold}.json");fp=args.stable/f"pred-{args.fold}.npz"
    if old["prediction"]!=rc.hash_file(fp) or old["field"]!=fr["payload"] or old["choices"]["adapted"]["weight"]!=.25 or old["library"]!=lib or old["check"]!=check:
        raise ValueError("E prediction binding differs")
    with np.load(fp,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("E prediction order differs")
        masks={"reference":z["adapted"]}
    view=cache_owner.verified_parent(args);model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    members={};region=rc.hash_file(args.routing/f"pred-{args.fold}.npz")
    for mode in MODES:
        copies=[];members[mode]=[]
        for j,offset in enumerate(OFFSETS):
            label=f"{mode}-{j}";folder=args.run/f"fold-{args.fold}-{label}"
            fit=read(folder/"fit.json");extra=read(folder/"risk.json")
            if (fit["train_indices"]!=sorted(train_ids) or fit["scope"]!="inner" or fit["steps"]!=5000 or fit["variant"]!=label
                    or fit["seed"]!=50171+args.fold+offset or fit["architecture"]!="HierarchicalRanks"
                    or fit["view_receipt"]!=view.receipt or fit["public_weights"]!=hierarchy.WEIGHT or fit["protocol"]!=rc.hash_file(PROTOCOL)
                    or fit["source"]!=rc.hash_file(prompt.__file__) or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                    or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or extra["fit"]!=rc.hash_file(folder/"fit.json")
                    or extra["source"]!=rc.hash_file(__file__) or extra["first"]!=rc.hash_file(folder/"first.json") or extra["regions"]!=region):
                raise ValueError("Invalid risk supervision identity")
            model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
            z=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
            if not np.isfinite(z).all():raise ValueError("Nonfinite rank logits")
            copies.append(z);members[mode].append(fit["checkpoint"])
        mean=mean_saved_logits(copies)
        masks[mode]=np.packbits(.75*guide+.25*mean>0,axis=-1)
        masks["standalone_"+mode]=np.packbits(mean>0,axis=-1)
    out=args.run/f"pred-{args.fold}.npz";np.savez_compressed(out,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"members":members,
        "reference":old["prediction"],"field":fr["payload"],"regions":region,"prediction":rc.hash_file(out),
        "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("RISK_SUPERVISION_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("fit","lock"));p.add_argument("--fold",type=int,choices=(0,1),required=True)
    for key in ("run","data","parent","crossfit","ensemble","formal","stable","public","routing"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--mode",choices=MODES,default="event");p.add_argument("--seed-index",type=int,choices=range(3),default=0)
    p.add_argument("--smoke",action="store_true");p.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2)
    {"fit":train,"lock":lock}[args.phase](args)
