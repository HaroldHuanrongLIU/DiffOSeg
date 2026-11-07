"""Recover continuous single-slice prompts from archived binary predictions."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from reproduction import gated_hierarchy as public_cache
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import rater_area
from reproduction.prefix_utility import accepted_inner_guide
from reproduction.seed_ensemble import OFFSETS,mean_saved_logits
from reproduction.trajectory_sam import select_weight

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/canonical_guide_protocol.md"
INNER_PARENT="1eb13a03183e78b509bc51d537564300ed5b601f1c60201d06f0f954e771422b"
DISTANCES=9
CONTRAST_BINS=6


def distance_code(mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.all() or not mask.any():
        return np.full(mask.shape,8+DISTANCES*int(mask.all()),dtype=np.uint8)
    distance=np.where(mask,ndimage.distance_transform_edt(mask),ndimage.distance_transform_edt(~mask))
    return (np.clip(np.floor(distance-.5),0,8)+DISTANCES*mask).astype(np.uint8)


def codes(image,masks):
    middle=masks.sum(0)>=2
    exterior=ndimage.distance_transform_edt(~middle) if middle.any() else np.full(middle.shape,np.inf)
    ring=(exterior>1)&(exterior<=8)
    if middle.any() and ring.any():
        low=float(np.median(image[ring]));high=float(np.median(image[middle]))
    else:
        low,high=np.quantile(image,[.1,.9])
    relative=(image.astype(np.float64)-low)/max(high-low,12.)
    contrast=np.searchsorted([0,.25,.5,.75,1.],relative,side="right").astype(np.uint8)
    return np.stack([distance_code(m) for m in masks]),contrast


def patient_cell_mean(total,count):
    patient_values=np.divide(total,count,out=np.zeros_like(total),where=count>0)
    available=(count>0).sum(0)
    mean=np.divide(patient_values.sum(0),available,out=np.zeros_like(total[0]),where=available>0)
    return mean,available


def calibrate(images,masks,logits,patient_names):
    _,assigned=np.unique(patient_names,return_inverse=True)
    npat=int(assigned.max()+1)
    coarse_sum=np.zeros((npat,4,2*DISTANCES));coarse_count=np.zeros_like(coarse_sum,dtype=int)
    fine_sum=np.zeros((npat,4,2*DISTANCES*CONTRAST_BINS));fine_count=np.zeros_like(fine_sum,dtype=int)
    for image,mask,values,patient in zip(images,masks,logits,assigned,strict=True):
        distance,contrast=codes(image,mask)
        for rank in range(4):
            weights=np.log1p(np.abs(values[rank])).ravel()
            for key,total,count in ((distance[rank],coarse_sum,coarse_count),
                                    (distance[rank]*CONTRAST_BINS+contrast,fine_sum,fine_count)):
                n=total.shape[-1]
                freq=np.bincount(key.ravel(),minlength=n)
                sums=np.bincount(key.ravel(),weights=weights,minlength=n)
                total[patient,rank]+=np.divide(sums,freq,out=np.zeros_like(sums),where=freq>0)
                count[patient,rank]+=(freq>0)
    base,support=patient_cell_mean(coarse_sum,coarse_count)
    for rank in range(4):
        for start in (0,DISTANCES):
            available=np.flatnonzero(support[rank,start:start+DISTANCES]>0)
            for j in range(DISTANCES):
                i=start+j
                if support[rank,i]==0:
                    base[rank,i]=base[rank,start+available[np.argmin(abs(available-j))]] if len(available) else np.log1p(.5+j)
    fine,n=patient_cell_mean(fine_sum,fine_count)
    fine=fine.reshape(4,2*DISTANCES,CONTRAST_BINS);n=n.reshape(fine.shape)
    fine=(fine*n+5*base[:,:,None])/(n+5)
    return (np.expm1(base).clip(.05,12).astype(np.float32),
            np.expm1(fine).clip(.05,12).astype(np.float32))


def reconstruct(image,mask,table):
    distance,contrast=codes(image,mask)
    if table.ndim==3:
        magnitude=table[np.arange(4)[:,None,None],distance,contrast[None]]
    else:
        magnitude=table[np.arange(4)[:,None,None],distance]
    return np.where(mask,magnitude,-magnitude).astype(np.float32)


def model_sources(args,training,fold):
    if rc.hash_file(args.ensemble/"summary.json")!=INNER_PARENT:
        raise ValueError("Registered seed ensemble changed")
    rows=[]
    for i,offset in enumerate(OFFSETS):
        history=i==0 and fold<2
        directory=(args.parent/f"fold-{fold}-pyramid") if history else (args.ensemble/f"fold-{fold}-seed-{i}")
        fit=json.loads((directory/"fit.json").read_text())
        if (fit["train_indices"]!=sorted(training) or fit["steps"]!=prompt.STEPS or fit["scope"]!="inner"
                or fit.get("seed",50171+fold)!=50171+fold+offset or fit["public_weights"]!=hierarchy.WEIGHT
                or fit["view_receipt"]!=public_cache.FEATURE_RECEIPT or fit["checkpoint"]!=rc.hash_file(directory/"final.pt")):
            raise ValueError("Inner seed identity changed")
        rows.append((directory,fit["checkpoint"]))
    return rows


def lock(args):
    if rc.hash_file(args.data)!=rc.DATA_HASH:raise ValueError("Wrong LIDC data")
    with args.data.open("rb") as stream:cases=list(pickle.load(stream).values())
    images=np.stack([case["image"] for case in cases]);uids=[case["series_uid"] for case in cases]
    _,fit,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(group,valid,args.fold)
    ids=library+checked;n=len(library)
    original,known,previous,_=accepted_inner_guide(args,library,checked)
    shift=json.loads((args.align/f"fold-{args.fold}-inner-registered/fit.json").read_text())["applied_shift"]
    registered=[rater_area.register_image(images[i],shift) for i in ids]
    base,conditional=calibrate(registered[:n],original[:n],known[:n],[str(uids[i]).split("_")[0] for i in library])
    parameters=args.run/f"table-{args.fold}.npz"
    np.savez(parameters,shape=base,conditional=conditional)
    seeds=model_sources(args,fit,args.fold)
    view=public_cache.verified_parent(args)
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    # Only library labels are used for fusion. Check labels are read by score() after all four locks.
    labels=np.stack([sorted(cases[i]["masks"],key=np.count_nonzero) for i in library]).astype(bool)
    choices={};packed={"reference":np.packbits(original,axis=-1)}
    for name,table in (("conditional",conditional),("shape",base)):
        guide=np.stack([reconstruct(im,mask,table) for im,mask in zip(registered,original)])
        if not np.array_equal(guide>0,original):raise ValueError("Canonical signs changed")
        logits=[]
        for directory,_ in seeds:
            model.load_state_dict(torch.load(directory/"final.pt",weights_only=True,map_location="cpu"))
            logits.append(prompt.predict(model,view.features,ids,images,guide).astype(np.float16))
        average=mean_saved_logits(logits)
        if not np.isfinite(average).all():raise ValueError("Nonfinite canonical inference")
        weight,grid=select_weight(labels,guide[:n],average[:n])
        packed[name]=np.packbits((1-weight)*guide+weight*average>0,axis=-1)
        choices[name]={"weight":weight,"library_scores":grid}
    path=args.run/f"pred-{args.fold}.npz";np.savez_compressed(path,ids=ids,**packed)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,"shift":shift,
                  "previous":previous["prediction"],"weights":[s for _,s in seeds],"choices":choices,"table":rc.hash_file(parameters),
                  "prediction":rc.hash_file(path),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("CANONICAL_LOCK",args.fold,{k:v["weight"] for k,v in choices.items()},flush=True)


def score(args):
    locks=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in range(4)]
    for f,r in enumerate(locks):
        if r["fold"]!=f or rc.hash_file(args.run/f"pred-{f}.npz")!=r["prediction"]:
            raise ValueError("Incomplete canonical prediction seals")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(locks):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Wrong inner patient order")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference","conditional","shape")}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores})
        print("CANONICAL_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("lock","score"))
    for name in ("run","parent","ensemble","data","public","formal","encoder","ct","texture","align"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    args=parser.parse_args();{"lock":lock,"score":score}[args.phase](args)
