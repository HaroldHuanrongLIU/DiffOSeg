"""Check an image-semantic adapter and its independent radius output on inner patients."""
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
from reproduction import semantic_preference as method
from reproduction import texture_study as prior


def correct(z,mask,rad,weight,offset):
    return np.stack([area.apply_radii(a,b,c,weight,offset) for a,b,c in zip(z,mask,rad)])


def run(args, backend=method, load=method.load_model):
    f=args.fold
    ims,labels,_,uids=native.inputs(args.data);_,fit,valid=native.splits(uids,f)
    group=type("Groups",(),{"series_uid":uids})()
    lib,check=polar.calibration_split(group,valid,f)
    ids=lib+check;n=len(lib)
    folder=args.run/f"fold-{f}-inner"
    record=prior.read(folder/"fit.json");plan=prior.read(folder/"plan.json")
    if plan["train_indices"]!=fit or record["checkpoint"]!=rc.hash_file(folder/"final.pt") or record["source"]!=rc.hash_file(backend.__file__):
        raise ValueError("Semantic fit identity")
    cached=args.texture/f"features-{f}-inner.npz"
    owner=prior.read(cached.with_suffix(".json"))
    if owner["payload"]!=rc.hash_file(cached) or owner["indices"]!=ids or owner["comparator"]!=prior.REFERENCE:
        raise ValueError("Cached feature patient order")
    with np.load(cached,allow_pickle=False) as d:
        masks=np.unpackbits(d["old"],axis=-1).astype(bool);ctr=d["centers"]
        base=prior.anchor_logits(d["base"].astype("float32"),masks)
    net=load(folder/"final.pt")
    z,r=backend.predict(net,ims[ids,None],ctr,record["shift"])
    if not np.isfinite(z).all() or not np.isfinite(r).all():raise ValueError("Nonfinite transformer outputs")
    truth=labels[ids].astype(bool)
    gr=[]
    for w in prior.WEIGHTS:
        mix=(1-w)*base[:n]+w*z[:n]
        gr.append(float(ns.dice(truth[:n],mix>0).mean()))
    w=prior.WEIGHTS[int(np.argmax(gr))]
    merged=(1-w)*base+w*z
    y=merged>0
    target=np.sqrt(truth.sum((2,3))/np.pi)
    offset=0.5*np.median(target[:n]-r[:n],0)
    radi=[]
    for t in prior.WEIGHTS:
        m=correct(merged[:n],y[:n],r[:n],t,offset)
        radi.append(float(ns.dice(truth[:n],m).mean()))
    t=prior.WEIGHTS[int(np.argmax(radi))]
    out=correct(merged[n:],y[n:],r[n:],t,offset)
    score=lambda x:float(ns.dice(truth[n:],x).mean())
    s={"reference":score(masks[n:]),"semantic_only":score(z[n:]>0),"blend":score(y[n:]),"candidate":score(out)}
    np.savez_compressed(args.run/f"outputs-inner-{f}.npz",ids=ids,logits=z.astype("float16"),radii=r)
    rc.write_json(args.run/f"inner-{f}.json",{"fold":f,"library":lib,"check":check,"source":rc.hash_file(__file__),
                  "protocol":record["protocol"],"weights":record["checkpoint"],"blend":w,"radius_fraction":t,
                  "bias":offset.tolist(),"blend_grid":gr,"radius_grid":radi,"scores":s})
    print("SEMANTIC_CHECK",f,w,t,s,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("data","run","texture"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    a=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(3);run(a)
