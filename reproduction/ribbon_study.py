"""Inner validation of an appearance-conditioned contour field and its implied area."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import native_local as native
from reproduction import native_study as ns
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import rater_area as area
from reproduction import ribbon_preference as ribbon
from reproduction import texture_study as texture

read=texture.read
WEIGHTS=(0.,.25,.5,.75,1.)


def dice(y,p):
    return ns.dice(y,p).mean((0,1)).item()


def setup(args,mode):
    d=args.run/f"fold-{args.fold}-{mode}"
    fit,plan=read(d/"fit.json"),read(d/"plan.json")
    model=ribbon.Ribbon().cuda().eval()
    if (fit["source"]!=rc.hash_file(ribbon.__file__) or fit["checkpoint"]!=rc.hash_file(d/"final.pt")
            or fit["protocol"]!=rc.hash_file(ribbon.PROTOCOL) or fit["steps"]!=ribbon.STEPS):
        raise ValueError("Ribbon checkpoint identity")
    model.load_state_dict(torch.load(d/"final.pt",map_location="cpu",weights_only=True))
    return model,fit,plan


def inner(args):
    cached=args.texture/f"features-{args.fold}-inner.npz"
    receipt=read(cached.with_suffix(".json"))
    if rc.hash_file(cached)!=receipt["payload"] or receipt["comparator"]!=texture.REFERENCE:
        raise ValueError("Unexpected image-only guide cache")
    with np.load(cached,allow_pickle=False) as z:
        ids=z["ids"].tolist()
        reference=np.unpackbits(z["old"],axis=-1).astype(bool)
        base=texture.anchor_logits(z["base"].astype(np.float32),reference)
    ims,labels,_,uids=native.inputs(args.data)
    _,train,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    lib,check=polar.calibration_split(group,valid,args.fold)
    if ids!=lib+check or receipt["indices"]!=ids:
        raise ValueError("Wrong patient partition")
    model,fit,plan=setup(args,"inner")
    if plan["train_indices"]!=train:raise ValueError("Ribbon trained on wrong patients")
    out=ribbon.predict(model,ims[ids],reference,fit["shift"])
    if not np.isfinite(out).all():raise ValueError("Nonfinite shape proposal")
    np.savez_compressed(args.run/f"ribbon-inner-{args.fold}.npz",ids=ids,sdf=out.astype("float16"))
    truth=labels[ids].astype(bool)
    n=len(lib); radius=np.sqrt((out>0).sum((2,3))/np.pi)
    offset=np.zeros(4)
    grid=[]
    def prediction(indices,m,w):
        if m=="logit":
            return (1-w)*base[indices]+w*out[indices]/.65>0
        return np.stack([area.apply_radii(base[i],reference[i],radius[i],w,offset) for i in indices])
    for m in ("logit","area"):
        for w in WEIGHTS:
            grid.append({"method":m,"weight":w,"dice":dice(truth[:n],prediction(np.arange(n),m,w))})
    best=max(grid,key=lambda row:row["dice"])
    result=prediction(np.arange(n,len(ids)),best["method"],best["weight"])
    target=np.sqrt(truth.sum((2,3))/np.pi)
    oldr=np.sqrt(reference.sum((2,3))/np.pi)
    scores={"reference":dice(truth[n:],reference[n:]),"ribbon_only":dice(truth[n:],out[n:]>0),"candidate":dice(truth[n:],result)}
    record={"source":rc.hash_file(__file__),"ribbon":fit["checkpoint"],"protocol":fit["protocol"],
            "reference_hash":texture.REFERENCE,"fold":args.fold,"library":lib,"check":check,"grid":grid,"best":best,"scores":scores,
            "radius_rmse":{"baseline":float(np.mean((target[n:]-oldr[n:])**2)**.5),"ribbon":float(np.mean((target[n:]-radius[n:])**2)**.5)}}
    rc.write_json(args.run/f"inner-{args.fold}.json",record)
    print("RIBBON_INNER",args.fold,best,scores,record["radius_rmse"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("run","data","texture"):p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--fold",required=True,type=int,choices=range(4))
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(3);inner(args)
