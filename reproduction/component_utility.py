"""Learn Dice flip utility on connected disagreements from strictly nested OOF predictions."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
import torch

from reproduction import canonical_guide
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import nested_guides as nested
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import rater_area
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits
from reproduction.shell_occupancy import unpack_level

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/component_utility_protocol.md"
QUALITY="3f536c2278d9bcaabe3a2db374d26c1d27b3964a1551fdb40d58ce536f5312d6"
METHODS=("scores","pixel","component")
THRESHOLDS=(None,.30,.15,.075,.03,0.)


def normalized_ct(image,mask):
    mid=mask.sum(0)>=2
    distance=ndimage.distance_transform_edt(~mid)
    ring=(~mid)&(distance>1)&(distance<=8)
    if mid.any() and ring.any():
        low=float(np.median(image[ring]));high=float(np.median(image[mid]))
    else:
        low,high=np.quantile(image,[.1,.9])
    normalized=((image.astype(np.float32)-low)/max(high-low,12.)).clip(-3,3)
    sm=cv2.GaussianBlur(normalized,(0,0),.9)
    return np.stack((normalized,sm,normalized-sm,np.hypot(*np.gradient(sm))))


def partition(u,s,current,pixel):
    eligible=(u>0)!=(s>0)
    points=np.flatnonzero(eligible)
    if pixel:return points,np.arange(len(points),dtype=np.intp)
    code=np.zeros(eligible.shape,np.int32);used=0
    for sign in (False,True):
        group,count=ndimage.label(eligible&(current==sign))
        active=group>0
        code[active]=group[active]+used
        used+=count
    return points,code.ravel()[points]-1


def aggregate(features,index,size):
    result=np.zeros((len(size),features.shape[1]),np.float64)
    np.add.at(result,index,features)
    return (result/size[:,None]).astype(np.float32)


@dataclass
class Components:
    points: np.ndarray
    group: np.ndarray
    count: np.ndarray
    initial: np.ndarray
    geometry: np.ndarray
    context: np.ndarray


def components(u,s,current,ct,high,rank,pixel):
    points,group=partition(u[rank],s[rank],current[rank],pixel)
    if not len(points):return None
    size=np.bincount(group)
    anchor=current[rank]
    radius=max(float(np.sqrt(anchor.sum()/np.pi)),1.)
    fg=np.argwhere(anchor)
    cy,cx=fg.mean(0) if len(fg) else (64.,64.)
    yy,xx=np.unravel_index(points,anchor.shape)
    direction=aggregate(anchor.ravel()[points,None],group,size)[:,0]
    if not np.all((direction==0)|(direction==1)):raise ValueError("A component spans both flip directions")
    def values(z):return z[rank].ravel()[points]
    us,ss=values(u),values(s)
    prior=np.column_stack((us.clip(-12,12),ss.clip(-12,12),(ss-us).clip(-12,12),
          (us>0).astype(float),u[:,yy,xx].clip(-12,12).mean(0),s[:,yy,xx].clip(-12,12).mean(0),
          (xx-cx)/radius,(yy-cy)/radius))
    mins=np.full((len(size),2),128,np.int32);maxs=np.full((len(size),2),-1,np.int32)
    pos=np.column_stack((yy,xx));np.minimum.at(mins,group,pos);np.maximum.at(maxs,group,pos)
    radii=np.sqrt((current.sum((1,2)))/np.pi)
    fixed=np.broadcast_to(np.r_[rank/3,radii/16],(len(size),5))
    geom=np.column_stack((fixed,direction,np.log1p(size),size/(anchor.sum()+1),(maxs-mins+1)/radius,
                          aggregate(prior,group,size))).astype(np.float32)
    local=high[:,yy,xx].astype(np.float32);local/=np.sqrt((local*local).mean(0)+1e-4)
    appearance=aggregate(np.vstack((ct[:,yy,xx],local)).T,group,size)
    return Components(points,group,size,direction,geom,np.column_stack((geom,appearance)))


def flip_utility(c,reference,truth):
    area=int(reference.sum());actual=int(truth.sum());hits=int(np.count_nonzero(reference&truth))
    current_dice=2*hits/(area+actual) if area+actual else 1.
    labelled=np.bincount(c.group,weights=truth.ravel()[c.points].astype(float),minlength=len(c.count))
    sign=1-2*c.initial
    denom=area+sign*c.count+actual
    flipped=np.divide(2*(hits+sign*labelled),denom,out=np.ones(len(c.count)),where=denom>0)
    return (flipped-current_dice)*(area+1)/c.count


def fit(args,features,images,truth,uids,training,shift):
    if rc.hash_file(args.nested/"independent_quality.json")!=QUALITY or not json.loads((args.nested/"independent_quality.json").read_text())["passed"]:
        raise ValueError("Independent nested patient lineage/quality gate differs")
    ids,base,old=nested.base_bank(args,uids,args.fold)
    file=args.nested/"nested-training"/f"oof-{args.fold}.npz";rec=json.loads(file.with_suffix(".json").read_text())
    if rec["payload"]!=rc.hash_file(file) or rec["base"]!=old["payload"] or ids!=sorted(training):
        raise ValueError("Wrong fitting-only nested score bank")
    with np.load(file,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Nested OOF ordering changed")
        stronger=z["logits"].astype(np.float32)
    names=[str(uids[i]).split("_")[0] for i in ids]
    _,inv,n=np.unique(names,return_inverse=True,return_counts=True)
    case_weight=.5+.5*len(ids)/(len(n)*n[inv])
    X={k:[] for k in METHODS};Y={k:[] for k in METHODS};W={k:[] for k in METHODS}
    for i,u,s,cw in zip(ids,base,stronger,case_weight,strict=True):
        ref=(u+s)>0
        ct=normalized_ct(rater_area.register_image(images[i],shift),ref)
        high=unpack_level(features[i,0,512:],4)
        for rank in range(4):
            for pixel in (False,True):
                c=components(u,s,ref,ct,high,rank,pixel)
                if c is None:continue
                gain=flip_utility(c,ref[rank],truth[i,rank].astype(bool))
                for method in (("pixel",) if pixel else ("scores","component")):
                    X[method].append(c.geometry if method=="scores" else c.context);Y[method].append(gain)
                    W[method].append(cw*c.count/(int(ref[rank].sum())+1))
    models={};counts={}
    for method in METHODS:
        y=np.concatenate(Y[method]);w=np.concatenate(W[method])
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=12,min_samples_leaf=50,
            learning_rate=.06,l2_regularization=8,early_stopping=False,random_state=710045+args.fold)
        model.fit(np.concatenate(X[method]),y,sample_weight=w/w.mean())
        models[method]=model;counts[method]=len(y)
    dst=args.run/f"fit-{args.fold}.joblib";joblib.dump(models,dst,compress=3)
    rc.write_json(dst.with_suffix(".json"),{"fold":args.fold,"training":ids,"models":rc.hash_file(dst),
        "counts":counts,"nested":rec["payload"],"quality":QUALITY,"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    return models


def predict_changes(image,u,s,reference,packed,models):
    ct=normalized_ct(image,reference);high=unpack_level(packed[512:],4)
    delta={k:np.full(reference.shape,-np.inf,np.float32) for k in METHODS}
    for rank in range(4):
        for pixel in (False,True):
            c=components(u,s,reference,ct,high,rank,pixel)
            if c is None:continue
            for k in (("pixel",) if pixel else ("scores","component")):
                score=models[k].predict(c.geometry if k=="scores" else c.context)
                delta[k][rank].ravel()[c.points]=score[c.group]
    return delta


def apply(reference,gain,threshold):
    return reference.copy() if threshold is None else reference^(gain>threshold)


def e_teacher_pair(args,images,training,lib,check):
    ids=lib+check
    field,fr=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(field,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong reference input order")
        guide=z["adapted"]
    ref=args.stable/f"pred-{args.fold}.npz";selection=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    if selection["prediction"]!=rc.hash_file(ref) or selection["field"]!=fr["payload"]:
        raise ValueError("Stable masks changed")
    with np.load(ref,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong ref order")
        reference=np.unpackbits(z["adapted"],axis=-1).astype(bool)
    view=cache_owner.verified_parent(args);model=hierarchy.HierarchicalRanks(args.public).cuda().eval();members=[]
    for folder,_ in canonical_guide.model_sources(args,training,args.fold):
        model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
        members.append(prompt.predict(model,view.features,ids,images,guide).astype(np.float16))
    candidate=mean_saved_logits(members)
    if not np.isfinite(candidate).all():raise ValueError("Nonfinite E component")
    w=selection["choices"]["adapted"]["weight"]
    replay=(1-w)*guide+w*candidate>0
    if not np.array_equal(replay,reference):raise ValueError("The same-run E masks did not reproduce")
    strong=(guide+candidate)*.5
    return guide,strong,reference,selection,view


def lock(args):
    if (rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT or
            rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER):
        raise ValueError("Formal E identity differs")
    images,truth,_,uids=native.inputs(args.data)
    _,training,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    lib,check=polar.calibration_split(group,valid,args.fold);ids=lib+check;n=len(lib)
    guide,strong,reference,selection,view=e_teacher_pair(args,images,training,lib,check)
    shift=json.loads((args.align/f"fold-{args.fold}-inner-registered/fit.json").read_text())["applied_shift"]
    models=fit(args,view.features,images,truth,uids,training,shift)
    deltas={k:[] for k in METHODS}
    for i,u,s,m in zip(ids,guide,strong,reference,strict=True):
        output=predict_changes(rater_area.register_image(images[i],shift),u,s,m,view.features[i,0],models)
        for k in METHODS:deltas[k].append(output[k])
    choices={};payload={"reference":np.packbits(reference,axis=-1),"eligible":np.packbits((guide>0)!=(strong>0),axis=-1)}
    for k in METHODS:
        values=np.stack(deltas[k]);grid=[]
        for threshold in THRESHOLDS:
            masks=apply(reference[:n],values[:n],threshold)
            grid.append(float(metrics.dice(truth[lib],masks).mean()))
        t=THRESHOLDS[int(np.argmax(grid))];result=apply(reference,values,t)
        if np.any((result!=reference)&((guide>0)==(strong>0))):raise ValueError("Changed an agreement pixel")
        choices[k]={"threshold":t,"grid":grid};payload[k]=np.packbits(result,axis=-1)
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**payload)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"reference":selection["prediction"],
        "fit":rc.hash_file(args.run/f"fit-{args.fold}.json"),"choices":choices,"prediction":rc.hash_file(dst),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("UTILITY_LOCK",args.fold,{k:v["threshold"] for k,v in choices.items()},flush=True)


def score(args):
    seals=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    if any(r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz") for f,r in enumerate(seals)):
        raise ValueError("Both component prediction seals are required")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Unverified component case order")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference",*METHODS)}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores});print("UTILITY_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("lock","score"))
    for key in ("run","parent","ensemble","data","public","stable","formal","original","nested","align"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0)
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    {"lock":lock,"score":score}[args.phase](args)
