"""Matched auxiliary objectives for spatially fine, reader-specific contour errors."""
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

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/rank_detail_protocol.md"
DIAGNOSTIC="519090d2ca4cfa39207252a14543e0f494d849681a26d3e2af55b90b095df56d"
MODES=("all","fine")


def detail_loss(probability,truth,mode):
    residual=probability.float()-truth.float()
    contrast=residual-residual.mean(1,keepdim=True)
    if mode=="fine":
        average=F.avg_pool2d(contrast,2,2).repeat_interleave(2,-2).repeat_interleave(2,-1)
        contrast=contrast-average
    energy=contrast.square().sum((1,2,3))
    return (2.4 if mode=="fine" else 1.)*(energy/(truth.sum((1,2,3))+16)).mean()


def train(args):
    if rc.hash_file(args.diagnostic/"summary.json")!=DIAGNOSTIC:
        raise ValueError("Fitting-only frequency decomposition changed")
    label=f"{args.mode}-{args.seed_index}"
    folder=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{label}")
    def objective(model,embedding,guide,image,labels,indices,rotation,step):
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            logits,raw=model(embedding,guide,image)
        standard=prompt.segmentation_loss(logits,raw,labels)
        extra=detail_loss(logits.sigmoid(),labels,args.mode)
        if step==1:
            info={"batch":indices.tolist(),"rotation":rotation,"supervised":standard.item(),"extra":extra.item()}
            if args.smoke:
                grad=torch.autograd.grad(extra,logits,retain_graph=True)[0]
                if not torch.isfinite(grad).all() or grad.abs().sum().item()==0:
                    raise ValueError("Missing rank-detail gradient")
            else:
                original=(args.parent/f"fold-{args.fold}-pyramid" if args.seed_index==0 and args.fold<2 else
                          args.ensemble/f"fold-{args.fold}-seed-{args.seed_index}")
                with (original/"train.jsonl").open() as stream:
                    expect=json.loads(next(stream))["loss"]
                if not np.isclose(info["supervised"],expect,rtol=2e-4,atol=4e-5):
                    raise ValueError("First ordinary loss differs from E")
                if args.mode=="fine":
                    control=json.loads((args.run/f"fold-{args.fold}-all-{args.seed_index}/first.json").read_text())
                    if info["batch"]!=control["batch"] or info["rotation"]!=control["rotation"]:
                        raise ValueError("Unmatched sampling stream")
            rc.write_json(folder/"first.json",info)
        return standard+(.35*min((step-1)/299,1.))*extra
    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=label,study_protocol=PROTOCOL,
        view=cache_owner.verified_parent(args),public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index],
        paired_loss=objective)
    rc.write_json(folder/"detail.json",{"source":rc.hash_file(__file__),"fit":rc.hash_file(folder/"fit.json"),
                 "first":rc.hash_file(folder/"first.json"),"diagnostic":DIAGNOSTIC})


def lock(args):
    if (rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or
            rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT):
        raise ValueError("Formal E changed")
    images,uids=ct_and_split(args)
    _,fit,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    lib,check=polar.calibration_split(group,valid,args.fold)
    ids=lib+check
    path,fr=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids: raise ValueError("Invalid deterministic guide order")
        guide=z["adapted"]
    old=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    archive=args.stable/f"pred-{args.fold}.npz"
    if (old["prediction"]!=rc.hash_file(archive) or old["field"]!=fr["payload"] or
            old["library"]!=lib or old["check"]!=check or old["choices"]["adapted"]["weight"]!=.25):
        raise ValueError("Frozen E reference differs")
    with np.load(archive,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids: raise ValueError("Wrong E prediction order")
        masks={"reference":z["adapted"]}
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    view=cache_owner.verified_parent(args)
    members={}
    for mode in MODES:
        packed=[];members[mode]=[]
        for i,offset in enumerate(OFFSETS):
            label=f"{mode}-{i}";folder=args.run/f"fold-{args.fold}-{label}"
            record=json.loads((folder/"fit.json").read_text())
            auxiliary=json.loads((folder/"detail.json").read_text())
            if (record["train_indices"]!=sorted(fit) or record["scope"]!="inner" or record["steps"]!=5000
                    or record["seed"]!=50171+args.fold+offset or record["variant"]!=label
                    or record["architecture"]!="HierarchicalRanks" or record["view_receipt"]!=view.receipt
                    or record["public_weights"]!=hierarchy.WEIGHT or record["source"]!=rc.hash_file(prompt.__file__)
                    or record["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                    or record["checkpoint"]!=rc.hash_file(folder/"final.pt") or record["protocol"]!=rc.hash_file(PROTOCOL)
                    or auxiliary["fit"]!=rc.hash_file(folder/"fit.json") or auxiliary["first"]!=rc.hash_file(folder/"first.json")
                    or auxiliary["source"]!=rc.hash_file(__file__) or auxiliary["diagnostic"]!=DIAGNOSTIC):
                raise ValueError("Frozen detail supervision identity differs")
            model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
            logits=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
            if not np.isfinite(logits).all():raise ValueError("Nonfinite rank-detail output")
            packed.append(logits);members[mode].append(record["checkpoint"])
        masks[mode]=np.packbits(.75*guide+.25*mean_saved_logits(packed)>0,axis=-1)
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"members":members,
          "reference":old["prediction"],"field":fr["payload"],"prediction":rc.hash_file(dst),
          "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("DETAIL_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","lock"))
    p.add_argument("--fold",type=int,choices=(2,3),required=True)
    for key in ("run","data","parent","crossfit","ensemble","formal","stable","public","diagnostic"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--mode",choices=MODES,default="fine")
    p.add_argument("--seed-index",type=int,choices=range(3),default=0)
    p.add_argument("--smoke",action="store_true")
    p.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    {"fit":train,"lock":lock}[args.phase](args)
