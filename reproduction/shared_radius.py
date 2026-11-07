"""Estimate the value of a common radius correction across the four reader ranks."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from reproduction import component_utility as components
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import rater_area
from reproduction import nested_guides as nested
from reproduction.prefix_utility import prefix_dice
from reproduction.shell_occupancy import unpack_level

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/shared_radius_protocol.md"
OFFSETS=np.arange(-3.,3.001,.25)
ZERO=len(OFFSETS)//2
SHRINK=(0.,.25,.5,.75,1.)
MODES=("common","independent","point","constant")


def area(radius,offset,pixels):
    return np.rint(np.pi*np.maximum(radius+offset,0.)**2).astype(int).clip(0,pixels)


def best(scores):
    order=np.argsort(np.abs(OFFSETS),kind="stable")
    return OFFSETS[order[np.argmax(scores[...,order],axis=-1)]]


@dataclass
class Example:
    common: np.ndarray
    rank: np.ndarray
    radii: np.ndarray
    order: np.ndarray
    curves: np.ndarray


def example(image,u,s,reference,packed,truth=None):
    q=np.where(reference,1.,-1.).astype(np.float32)*np.maximum(abs((u+s)*.5),.05)
    ru,rs,rq=(np.sqrt(x.sum((1,2))/np.pi) for x in ((u>0),(s>0),reference))
    global_data=np.r_[ru/16,rs/16,rq/16,rs-ru,(rs-ru).mean(),np.std(rs-ru),np.ptp(rq)/8,
                      np.count_nonzero((u>0)!=(s>0),axis=(1,2))/np.maximum(reference.sum((1,2)),1)]
    ct=components.normalized_ct(image,reference).reshape(4,-1)
    high=unpack_level(packed[512:],4).reshape(32,-1)
    rank_features=[];orders=[];curves=[]
    for r in range(4):
        z=q[r].ravel();order=np.argsort(-z,kind="stable");orders.append(order.astype(np.uint16))
        size=area(rq[r],OFFSETS,len(z))
        lower=area(rq[r],OFFSETS-.25,len(z));upper=area(rq[r],OFFSETS+.25,len(z))
        pixels=order[:max(1,int(upper[-1]))]
        h=high[:,pixels].astype(np.float32);h/=np.sqrt((h*h).mean(0)+1e-4)
        left,right=u[r].ravel()[pixels],s[r].ravel()[pixels]
        points=np.vstack((ct[:,pixels],left.clip(-12,12)/8,right.clip(-12,12)/8,
                          (left>0)!=(right>0),z[pixels].clip(-12,12)/8,h))
        cumulative=np.pad(np.cumsum(points,axis=1,dtype=np.float32),((0,0),(1,0)))
        lengths=upper-lower
        band=(cumulative[:,upper]-cumulative[:,lower])/np.maximum(lengths,1)
        empty=lengths==0
        band[:,empty]=points[:,np.clip(lower[empty]-1,0,len(pixels)-1)]
        rank_features.append(np.column_stack((OFFSETS/3,size/1024.,band.T)))
        if truth is not None:curves.append(prefix_dice(truth[r],order,size))
    rank_features=np.asarray(rank_features,np.float32)
    common=np.column_stack((np.broadcast_to(global_data,(len(OFFSETS),len(global_data))),OFFSETS/3,
                            rank_features.mean(0),rank_features[-1]-rank_features[0],rank_features.std(0))).astype(np.float32)
    rank=np.stack([np.column_stack((common,rank_features[r],np.full(len(OFFSETS),r/3))) for r in range(4)]).astype(np.float32)
    return Example(common,rank,rq,np.stack(orders),np.asarray(curves))


def masks_for(case,offset,strength):
    sizes=area(case.radii,strength*np.asarray(offset),case.order.shape[1])
    out=np.zeros(case.order.shape,dtype=bool)
    for r,count in enumerate(sizes):out[r,case.order[r,:count]]=True
    return out.reshape(4,128,128)


def fit(args,images,truth,uids,training,features,shift):
    if rc.hash_file(args.nested/"independent_quality.json")!=components.QUALITY:
        raise ValueError("Nested OOF bank has not passed the registered patient audit")
    ids,u,rec=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz";meta=json.loads(path.with_suffix(".json").read_text())
    if meta["payload"]!=rc.hash_file(path) or meta["base"]!=rec["payload"] or ids!=sorted(training):
        raise ValueError("Incorrect genuine OOF pairs")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong nested training order")
        s=z["logits"].astype(np.float32)
    xc=[];xr=[];curve=[];point=[]
    for i,a,b in zip(ids,u,s,strict=True):
        case=example(rater_area.register_image(images[i],shift),a,b,a+b>0,features[i,0],truth[i])
        xc.append(case.common);xr.append(case.rank);curve.append(case.curves)
        point.append(np.mean(np.sqrt(truth[i].sum((1,2))/np.pi)-case.radii))
    xc=np.stack(xc);xr=np.stack(xr);curve=np.stack(curve)
    gains=100*(curve-curve[:,:,ZERO:ZERO+1])
    _,inv,n=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    w=.5+.5*len(ids)/(len(n)*n[inv])
    settings=dict(max_iter=100,max_leaf_nodes=9,min_samples_leaf=100,learning_rate=.06,l2_regularization=8,
                  early_stopping=False,random_state=142139+args.fold)
    joint=HistGradientBoostingRegressor(**settings).fit(xc.reshape(-1,xc.shape[-1]),gains.mean(1).ravel(),sample_weight=np.repeat(w,len(OFFSETS)))
    individual=HistGradientBoostingRegressor(**settings).fit(xr.reshape(-1,xr.shape[-1]),gains.ravel(),sample_weight=np.repeat(w,4*len(OFFSETS)))
    direct=HistGradientBoostingRegressor(**settings).fit(xc[:,ZERO],point,sample_weight=w)
    models={"common":joint,"independent":individual,"point":direct}
    dst=args.run/f"models-{args.fold}.joblib";joblib.dump(models,dst,compress=3)
    rc.write_json(args.run/f"fit-{args.fold}.json",{"fold":args.fold,"training":ids,"original":rec["payload"],"nested":meta["payload"],
        "models":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    return models


def lock(args):
    if rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT or rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER:
        raise ValueError("Formal E comparator changed")
    images,truth,_,uids=native.inputs(args.data)
    _,training,valid=native.splits(uids,args.fold)
    grp=type("Groups",(),{"series_uid":uids})();lib,check=polar.calibration_split(grp,valid,args.fold)
    ids=lib+check;n=len(lib)
    u,s,reference,selection,view=components.e_teacher_pair(args,images,training,lib,check)
    shift=json.loads((args.align/f"fold-{args.fold}-inner-registered/fit.json").read_text())["applied_shift"]
    models=fit(args,images,truth,uids,training,view.features,shift)
    cases=[example(rater_area.register_image(images[i],shift),a,b,ref,view.features[i,0],truth[i] if j<n else None)
           for j,(i,a,b,ref) in enumerate(zip(ids,u,s,reference,strict=True))]
    cx=np.stack([c.common for c in cases]);rx=np.stack([c.rank for c in cases])
    common=best(models["common"].predict(cx.reshape(-1,cx.shape[-1])).reshape(len(ids),len(OFFSETS)))
    separate=best(models["independent"].predict(rx.reshape(-1,rx.shape[-1])).reshape(len(ids),4,len(OFFSETS)))
    point=models["point"].predict(cx[:,ZERO]).clip(-3,3)
    constant=float(best(np.mean([c.curves.mean(0) for c in cases[:n]],axis=0)))
    changes={"common":common,"independent":separate,"point":point,"constant":np.full(len(ids),constant)}
    original=np.stack([masks_for(c,0.,0.) for c in cases])
    if not np.array_equal(original,reference):raise ValueError("Zero radial correction does not reproduce E")
    arrays={"reference":np.packbits(reference,axis=-1)};choices={}
    for name,offsets in changes.items():
        strengths=(1.,) if name=="constant" else SHRINK
        grid=[]
        for strength in strengths:
            masks=np.stack([masks_for(c,off,strength) for c,off in zip(cases[:n],offsets[:n],strict=True)])
            grid.append(float(metrics.dice(truth[lib],masks).mean()))
        strength=strengths[int(np.argmax(grid))]
        masks=np.stack([masks_for(c,off,strength) for c,off in zip(cases,offsets,strict=True)])
        arrays[name]=np.packbits(masks,axis=-1);arrays["offset_"+name]=offsets
        choices[name]={"strength":strength,"grid":grid}
    path=args.run/f"pred-{args.fold}.npz";np.savez_compressed(path,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"choices":choices,"constant":constant,
        "reference":selection["prediction"],"fit":rc.hash_file(args.run/f"fit-{args.fold}.json"),"prediction":rc.hash_file(path),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("SHARED_RADIUS_LOCK",args.fold,{k:v["strength"] for k,v in choices.items()},flush=True)


def score(args):
    seals=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    if any(r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz") for f,r in enumerate(seals)):
        raise ValueError("Incomplete common radius seals")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Wrong radius score order")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference",*MODES)}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores});print("SHARED_RADIUS_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("lock","score"))
    for name in ("run","parent","ensemble","data","original","nested","public","stable","formal","align"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0)
    args=parser.parse_args();cv2.setNumThreads(1)
    {"lock":lock,"score":score}[args.phase](args)
