"""Fold-local calibration and probability pooling of unmodified formal-E sources."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import canonical_guide as canonical
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/probability_pool_protocol.md"
TEMPERATURES=(1.,.5,.75,1.5,2.,3.,4.,.25)
WEIGHTS=(.25,0.,.125,.375,.5,.75,1.)
MODES=("logits","uncalibrated","probability")


def temperatures(scores,labels,weight):
    # logaddexp gives finite NLL even for a saturated foreground or background prediction.
    normalizer=max(weight.sum(),1e-12)
    losses=[float((np.logaddexp(0.,(1.-2*labels)*scores/t)*weight).sum()/normalizer) for t in TEMPERATURES]
    return TEMPERATURES[int(np.argmin(losses))],losses


def pooling(guide,members,tg,tm,weight,mode):
    g=guide.astype(np.float64)/tg
    z=members.astype(np.float64)/tm
    if mode!="logits":
        # Twice the centered sigmoid avoids loss of the sign near probability 0.5.
        g=np.tanh(g/2);z=np.tanh(z/2)
    return (1-weight)*g+weight*z.mean(0)>0


def lock(args):
    if (rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT or
        rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER):raise ValueError("E identity differs")
    images,truth,_,uids=native.inputs(args.data)
    _,train,valid=native.splits(uids,args.fold)
    lib,check=polar.calibration_split(type("Group",(),{"series_uid":uids})(),valid,args.fold)
    ids=lib+check;n=len(lib)
    path,fr=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Incorrect guide order")
        guide=z["adapted"]
    orig=args.stable/f"pred-{args.fold}.npz";old=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    if (old["prediction"]!=rc.hash_file(orig) or old["field"]!=fr["payload"] or old["library"]!=lib or old["check"]!=check
            or old["choices"]["adapted"]["weight"]!=.25):raise ValueError("Wrong E reference")
    with np.load(orig,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong E reference order")
        ref=np.unpackbits(z["adapted"],axis=-1).astype(bool)
    view=cache_owner.verified_parent(args);model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    copies=[];hashes=[]
    for directory,h in canonical.model_sources(args,train,args.fold):
        model.load_state_dict(torch.load(directory/"final.pt",weights_only=True,map_location="cpu"))
        z=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
        if not np.isfinite(z).all():raise ValueError("Nonfinite replay")
        copies.append(z);hashes.append(h)
    if hashes!=old["checkpoints"] or not np.array_equal(.75*guide+.25*mean_saved_logits(copies)>0,ref):
        raise ValueError("Could not reproduce every E pixel")
    disputed=np.logical_or.reduce([(v>0)!=(guide>0) for v in copies])
    loc=np.flatnonzero(disputed)
    g=guide.ravel()[loc];z=np.stack([v.ravel()[loc] for v in copies])
    prefix=loc<n*4*128**2
    idx=loc[prefix]
    y=truth[lib].astype(bool);labels=y.ravel()[idx].astype(np.float64)
    _,inv,counts=np.unique([str(uids[i]).split("_")[0] for i in lib],return_inverse=True,return_counts=True)
    case_weight=.5/n+.5/(len(counts)*counts[inv])
    mass=(ref[:n].sum((2,3))+y.sum((2,3))+1).ravel()
    w=case_weight[idx//(4*128**2)]/mass[idx//128**2]
    tg,gl=temperatures(g[prefix],labels,w)
    tm,ml=temperatures(z[:,prefix].astype(np.float64).ravel(),np.tile(labels,3),np.tile(w,3))
    inputs=args.run/f"scores-{args.fold}.npz"
    np.savez_compressed(inputs,ids=ids,index=loc,guide=g,members=z)
    masks={"reference":np.packbits(ref,axis=-1)};choices={}
    for mode in MODES:
        temps=(1.,1.) if mode=="uncalibrated" else (tg,tm)
        grid=[float(metrics.dice(y,ref[:n]).mean())]
        for weight in WEIGHTS:
            pred=ref[:n].copy()
            pred.ravel()[idx]=pooling(g[prefix],z[:,prefix],*temps,weight,mode)
            grid.append(float(metrics.dice(y,pred).mean()))
        selected=int(np.argmax(grid))
        pred=ref.copy()
        if selected:pred.ravel()[loc]=pooling(g,z,*temps,WEIGHTS[selected-1],mode)
        masks[mode]=np.packbits(pred,axis=-1)
        choices[mode]={"weight":None if not selected else WEIGHTS[selected-1],"grid":grid}
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"reference":old["prediction"],
        "field":fr["payload"],"members":hashes,"inputs":rc.hash_file(inputs),"choices":choices,
        "guide_temperature":tg,"member_temperature":tm,"guide_nll":gl,"member_nll":ml,
        "prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("PROB_POOL_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("run","parent","ensemble","stable","formal","data","public"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.set_defaults(backbone="sam2.1-tiny",image_size=512)
    args=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1);lock(args)
