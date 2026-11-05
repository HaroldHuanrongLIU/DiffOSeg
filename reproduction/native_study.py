"""Real single-image validation for the 90% preference-Dice target."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.special import expit,logit
import torch

from reproduction import native_local as loc
from reproduction import order_study as v9
from reproduction import joint_rater as joint
from reproduction import rank_contours as rc
from reproduction import polar_contours as polar
from reproduction import vote_eval as v3

V9="33226deadde97f57179f0c261b5631fd50c86b8f64e2956a06b56ab56a33c281"
MIX=(0.,0.25,0.5,0.75,1.)
SHRINK=(0.,0.5,1.)
read=lambda p:json.loads(p.read_text())


def ensure(ok,msg):
    if not ok:raise ValueError(msg)


def dice(truth,pred):
    d=truth.sum((-2,-1))+pred.sum((-2,-1))
    return np.divide(2*(truth&pred).sum((-2,-1)),d,out=np.ones(d.shape),where=d>0)


def setup(args,mode):
    ensure(rc.hash_file(args.v9/"assessment/fourfold_comparison.json")==V9,"v9 identity")
    oldargs=SimpleNamespace(fold=args.fold,data=args.data,run=args.v9,v1=args.v1,v2=args.v2,v3=args.v3,v4=args.v4,
                            fixed=args.v5,ordinal=args.v8,canonical=args.v7)
    data,outer,valid,old,weights,previous,parent,lock=v9.setup(oldargs,mode)
    expected=outer["train_indices"] if mode=="full" else rc.split_for(data,args.fold)[1]
    for stage in ("locator","local"):
        folder=args.run/f"fold-{args.fold}-{mode}-{stage}"
        plan,fit=read(folder/"plan.json"),read(folder/"fit.json")
        ensure(plan["train_indices"]==expected and plan["inner_valid_indices"]==(valid if mode=="inner" else []) and
               (fit["stage"],fit["mode"],fit["fold"],fit["steps"])==(stage,mode,args.fold,loc.STEPS[stage]) and
               fit["source"]==rc.hash_file(loc.__file__) and fit["protocol"]==rc.hash_file(loc.PROTOCOL) and
               fit["checkpoint"]==rc.hash_file(folder/"final.pt"),"local model patient/source identity")
    return oldargs,data,outer,valid,old,weights,previous,parent,lock


def moment_project(logits,locations,strength):
    if strength==0:return logits
    x=torch.as_tensor(logits)
    u=loc.coordinates(128,x.device).reshape(-1,2)
    z=x.reshape(len(x),4,-1)
    # Normal to the feasible centroid cell is a linear spatial logit tilt.
    init=torch.sigmoid(2*z)
    mass=init.sum(-1,keepdim=True)
    c=(init@u)/mass.clamp_min(1)
    mean=c.mean(1)
    delta=(mean.clamp(64.,65.)-mean)*strength
    loc_ok=(torch.as_tensor(locations)-64.5).abs()<4.
    delta=torch.where(loc_ok & ((mean-64.5).abs()<5),delta,0.)
    target=mean+delta
    direction=u[None]-mean[:,None]
    bias=torch.zeros_like(mean)
    for _ in range(3):
        tilt=(direction@bias[:,:,None]).squeeze(-1).clamp(-2.5,2.5)
        p=(2*(z+tilt[:,None])).sigmoid()
        den=p.sum(-1,keepdim=True).clamp_min(1)
        center=(p@u)/den
        centered=u[None,None]-center[:,:,None]
        grad=2*p*(1-p)/den
        jac=(centered.transpose(-1,-2)@(grad[:,:,:,None]*direction[:,None])).mean(1)
        step=torch.linalg.solve(jac+torch.eye(2)[None]*0.2,(target-center.mean(1))[:,:,None]).squeeze(-1)
        bias=(bias+step).clamp(-0.4,0.4)
    tilt=(direction@bias[:,:,None]).reshape(len(x),1,128,128).clamp(-2.5,2.5)
    return (x+tilt).numpy()


def evidence(args,mode):
    oldargs,data,outer,valid,old,weights,previous,parent,lock=setup(args,mode)
    lib,check=polar.calibration_split(data,valid,args.fold)
    ids=lib+check if mode=="inner" else outer["test_indices"]
    baseline,_,_,cart=v9.preference_features(oldargs,data,ids,mode,old,weights,previous,parent,lock)
    images=torch.from_numpy(np.stack([data.images[i] for i in ids])[:,None]).cuda()
    net1,net2=loc.get_models(args.run,args.fold,mode)
    new,centers=loc.infer(net1,net2,images)
    anchored,_=loc.infer(net1,net2,images,force_center=True)
    return data,ids,lib,check,logit(baseline.clip(1e-6,1-1e-6)),cart,new[:,:4],centers,anchored[:,:4]


def preference(base,new,centers,priors,weight,shrink):
    logits=(1-weight)*base+weight*new
    parts=[]
    for i in range(0,len(logits),12):
        z=moment_project(logits[i:i+12],centers[i:i+12],shrink)
        parts.extend(joint.preference(x,p) for x,p in zip(expit(z),priors[i:i+12]))
    return np.stack(parts)


def calibrate(args):
    data,ids,lib,check,base,cart,new,centers,fixed=evidence(args,"inner")
    n=len(lib); y=np.array([data.mask_labels[i] for i in ids],bool)
    old=preference(base,new,centers,cart,0.,0.)
    grid=[]
    for w in MIX:
        for s in SHRINK:
            seg=preference(base[:n],new[:n],centers[:n],cart[:n],w,s)
            grid.append({"weight":w,"shrink":s,"score":float(dice(y[:n],seg).mean())})
    best=max(grid,key=lambda x:x["score"])
    seg=preference(base[n:],new[n:],centers[n:],cart[n:],best["weight"],best["shrink"])
    # A fixed-origin intervention uses the same local model weights.
    ctrl=preference(base[n:],fixed[n:],np.full_like(centers[n:],64.5),cart[n:],best["weight"],best["shrink"])
    sc=dice(y[n:],seg)
    record={"fold":args.fold,"source":rc.hash_file(__file__),"protocol":rc.hash_file(loc.PROTOCOL),"v9":V9,
            "fit_indices":lib,"check_indices":check,"grid":grid,"selected":best,"check_candidate":float(sc.mean()),
            "check_v9":float(dice(y[n:],old[n:]).mean()),"check_centered":float(dice(y[n:],ctrl).mean()),
            "check_local":float(dice(y[n:],new[n:]>0).mean()),"check_each":sc.mean(0).tolist(),
            "locator_p95_error":float(np.percentile(np.linalg.norm(centers-loc.rater_center(torch.from_numpy(y)).numpy(),axis=1),95))}
    # Development features stay distinct from the sealed independent test artifacts.
    np.savez_compressed(args.run/f"inner-features-{args.fold}.npz",indices=np.array(ids),centers=centers,
                        base=base.astype("float16"),local=new.astype("float16"),fixed=fixed.astype("float16"))
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("INNER",args.fold,best,record["check_candidate"],record["check_v9"],record["check_local"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["calibrate"])
    for k in ("run","data","v1","v2","v3","v4","v5","v7","v8","v9"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--fold",type=int,required=True,choices=range(4))
    args=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(3)
    calibrate(args)
