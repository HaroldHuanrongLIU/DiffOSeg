"""Generate genuinely held-patient contour priors for learning residual errors."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/crossfit_protocol.md"
STEPS=4500


def partitions(inner,uids,fold):
    names=np.array(sorted({str(uids[i]).split("_")[0] for i in inner}))
    rng=np.random.RandomState(93011+fold)
    groups=np.array_split(rng.permutation(names),3)
    result=[]
    for group in groups:
        held=set(group);fit=[];predict=[]
        for i in inner:
            (predict if str(uids[i]).split("_")[0] in held else fit).append(i)
        result.append((fit,predict))
    return result


def training_partition(uids, fold, scope):
    outer, inner, validation = native.splits(uids, fold)
    if scope == "full":
        return outer["train_indices"], outer["test_indices"]
    if scope == "inner":
        return inner, validation
    raise ValueError("Unknown teacher training scope")


def train(args, *, singleton=False):
    cv2.setNumThreads(1);torch.set_num_threads(3)
    ims,masks,_,uids=native.inputs(args.data)
    eligible, excluded = training_partition(uids, args.fold, args.scope)
    fit,hold=partitions(eligible,uids,args.fold)[args.sub]
    if singleton:
        fit,hold=hold,fit
    steps=STEPS
    if args.smoke:fit,steps=fit[:48],5
    patients=[str(uids[i]).split("_")[0] for i in fit]
    shift,_=align.estimate_shift(ims[fit],masks[fit],patients)
    seed=(289331 if singleton else 88331)+args.fold+317*args.sub
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=True
    args.out.mkdir(parents=True,exist_ok=False)
    image=align.registered_ct(torch.from_numpy(ims[fit,None]).cuda(),*shift)
    truth=torch.from_numpy(masks[fit]).cuda().float()
    sd=torch.from_numpy(np.stack([[rc.signed(m) for m in y] for y in masks[fit]]).astype("float16")).cuda()
    all_y=torch.cat((truth,vote.votes(truth),sd.float()),1).half()
    centers=native.rater_center(truth)
    centers=torch.where((centers-64.5).abs().amax(1,keepdim=True)>4,centers,torch.full_like(centers,64.5))
    _,inv,n=np.unique(patients,return_inverse=True,return_counts=True)
    sampling=torch.from_numpy(.65/len(fit)+.35/(len(n)*n[inv])).float().cuda()
    net=vote.build().train()
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    params=list(net.parameters());avg=list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=.0009,weight_decay=.0002,fused=True)
    meta={"fold":args.fold,"sub":args.sub,"seed":seed,"steps":steps,"shift":shift,"data":rc.DATA_HASH,
          "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL if args.scope=="inner" else PROTOCOL.with_name("outer_protocol.md")),
          "scope":args.scope,"gpu":torch.cuda.get_device_name()}
    if singleton:
        meta["protocol"]=rc.hash_file(PROTOCOL.with_name("nested_guides_protocol.md"))
        meta["role"]="single_patient_group"
    rc.write_json(args.out/"plan.json",{**meta,"fit_indices":fit,"held_indices":hold,"excluded_indices":excluded,
                "excluded_inner_validation":excluded if args.scope=="inner" else []})
    start=time.monotonic()
    with (args.out/"log.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(sampling,20,replacement=True);u=torch.rand(20,9,device="cuda")
            scale=.65+.60*u[:,0]
            angle=torch.where(u[:,1]<.7,(u[:,2]*4).floor()*math.pi/2,(2*u[:,2]-1)*math.pi)
            flip=torch.where(u[:,3]<.5,-1.,1.)
            grid,_=align.transform_grids(centers[ix]+(u[:,4:6]-.5)*1.4,scale,angle,flip)
            x=F.grid_sample(image[ix].float()/255,grid,align_corners=True,padding_mode="border")
            x=(x.clamp_min(1e-6).pow(.85+.3*u[:,6,None,None,None])*(.85+.30*u[:,7,None,None,None])+
               .08*(u[:,8,None,None,None]-.5)).clamp(0,1)
            y=F.grid_sample(all_y[ix].float(),grid,align_corners=True,padding_mode="zeros")
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):z=net(x)
            p=(2*z[:,:4]).sigmoid()
            size=F.smooth_l1_loss((p.sum((-2,-1))+1).log(),(y[:,:4].sum((-2,-1))+1).log())
            loss=vote.mask_loss(z[:,:4],y[:,:4])+.7*vote.mask_loss(z[:,8:],y[:,4:8])+.15*size
            loss=loss+.1*F.smooth_l1_loss(z[:,4:8].tanh(),(y[:,8:]/(scale[:,None,None,None]*8)).clamp(-1,1))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.0009*min(1,step/100)*(.025+.975*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                row={"step":step,"loss":loss.item(),"seconds":time.monotonic()-start}
                log.write(json.dumps(row,allow_nan=False)+"\n");print(row,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    rc.write_json(args.out/"fit.json",{**meta,"checkpoint":rc.hash_file(args.out/"final.pt")})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--sub",type=int,choices=range(3),required=True)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scope",choices=("inner","full"),default="inner")
    p.add_argument("--smoke",action="store_true");p.add_argument("--single-group",action="store_true")
    args=p.parse_args();train(args,singleton=args.single_group)
