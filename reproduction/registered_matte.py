"""Calibrate a training-only continuous LIDC target for a shared opacity and rank decoder."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_views as views
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/registered_matte_protocol.md"
MODES=("vote","raw","registered")
STEPS=4000


def estimate_matte_shift(alpha,votes):
    h,w=alpha.shape[-2:]
    target=votes[:,3:h-3,3:w-3]
    mass=target.sum((1,2),dtype=np.float64)+1
    scores={}
    for dx in range(-2,3):
        for dy in range(-2,3):
            source=alpha[:,3-dy:h-3-dy,3-dx:w-3-dx]
            scores[dx,dy]=float(np.mean(np.square(source-target).sum((1,2),dtype=np.float64)/mass))
    shift=min(scores,key=lambda d:(scores[d],abs(d[0])+abs(d[1]),d))
    return shift,scores[0,0],scores[shift]


def spline(opacity):
    nodes=torch.arange(5,device=opacity.device,dtype=opacity.dtype)[None,:,None,None]
    return (1-(4*opacity-nodes).abs()).clamp_min(0)


class MatteHead(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.original=original
        with torch.random.fork_rng(devices=[]):
            self.continuous=nn.Conv2d(original.in_channels,1,1)
            self.coefficients=nn.Conv2d(original.in_channels,20,1)
        with torch.no_grad():
            self.continuous.weight.copy_(original.weight[:4].mean(0,keepdim=True))
            self.continuous.bias.copy_(original.bias[:4].mean().reshape(1))
            nn.init.zeros_(self.coefficients.weight)
            nn.init.zeros_(self.coefficients.bias)

    def forward(self,features):
        z=self.original(features)
        matte=self.continuous(features)
        basis=spline(matte.float().sigmoid())
        coefficients=self.coefficients(features).reshape(len(z),4,5,*z.shape[-2:])
        ranks=z[:,:4].float()+(coefficients*basis[:,None]).sum(2)
        return torch.cat((ranks,z[:,4:],matte),1)


def load_backbone(warm):
    model=vote.load_net(warm)
    model.head=MatteHead(model.head).to(device="cuda",memory_format=torch.channels_last)
    return model


def matte_loss(logit,soft,truth):
    roi=F.max_pool2d(truth.amax(1,keepdim=True),9,1,4)
    weight=.02+roi
    ce=F.binary_cross_entropy_with_logits(logit,soft,reduction="none")
    return ((ce*weight).sum((1,2,3))/weight.sum((1,2,3))).mean()


def train(args):
    images,truth,alpha,uids=native.inputs(args.data)
    _,fit_ids,valid=native.splits(uids,args.fold)
    warm,identity=views.warm_identity(args,fit_ids)
    folder=args.run/(f"smoke-{args.fold}-{args.mode}" if args.smoke else f"fold-{args.fold}-{args.mode}")
    folder.mkdir(exist_ok=False)
    ids=fit_ids[:32] if args.smoke else fit_ids
    votes=truth[ids].mean(1,dtype=np.float32)
    original=alpha[ids].astype(np.float32)/255
    shift,before,after=estimate_matte_shift(original,votes)
    y=torch.from_numpy(truth[ids]).cuda().float()
    if args.mode=="vote":
        soft=y.mean(1,keepdim=True)
    else:
        soft=torch.from_numpy(original[:,None]).cuda()
        if args.mode=="registered":
            soft=align.registered_ct(soft,*shift)
    signed=np.stack([[rc.signed(m) for m in panels] for panels in truth[ids]])
    targets=torch.cat((y,vote.votes(y),torch.from_numpy(signed).cuda(),soft),1).half()
    ct=align.registered_ct(torch.from_numpy(images[ids,None]).cuda(),*identity["applied_shift"]).float()/255
    centers=native.rater_center(y)
    centers=torch.where((centers-64.5).abs().amax(1,keepdim=True)>4,centers,64.5)
    _,inverse,count=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    sampling=.65/len(ids)+.35/(len(count)*count[inverse])
    seed=431157+args.fold
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    model=load_backbone(warm/"final.pt").train()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    params=list(model.parameters());avg=list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=.00012,weight_decay=.0001,fused=True)
    steps=5 if args.smoke else STEPS
    schedule=hashlib.sha256()
    with (folder/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=rng.choice(len(ids),20,p=sampling)
            random=rng.random((20,9),dtype=np.float32)
            schedule.update(ix.astype("<i4").tobytes()+random.tobytes())
            u=torch.from_numpy(random).cuda()
            angle=torch.where(u[:,0]<.65,(4*u[:,1]).floor()*math.pi/2,(2*u[:,1]-1)*math.pi)
            flip=torch.where(u[:,2]<.5,-1.,1.)
            scale=.65+.58*u[:,3]
            grid,_=align.transform_grids(centers[ix]+(u[:,4:6]-.5)*1.2,scale,angle,flip)
            x=F.grid_sample(ct[ix],grid,align_corners=True,padding_mode="border")
            photo=u[:,6:9,None,None]
            x=(x.clamp_min(1e-6).pow(.85+.3*photo[:,0:1])*(.87+.26*photo[:,1:2])+.07*(photo[:,2:3]-.5)).clamp(0,1)
            labels=F.grid_sample(targets[ix].float(),grid,align_corners=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                output=model(x)
            ranks=output[:,:4]
            area=F.smooth_l1_loss(((2*ranks).sigmoid().sum((-2,-1))+1).log(),(labels[:,:4].sum((-2,-1))+1).log())
            center=F.smooth_l1_loss(native.rater_center((2*ranks).sigmoid()),native.rater_center(labels[:,:4]))
            distance=F.smooth_l1_loss(output[:,4:8].tanh(),(labels[:,8:12]/(8*scale[:,None,None,None])).clamp(-1,1))
            ordinary=(vote.mask_loss(ranks,labels[:,:4])+.75*vote.mask_loss(output[:,8:12],labels[:,4:8])
                      +.15*distance+.20*area+.025*center)
            extra=matte_loss(output[:,12:13],labels[:,12:13],labels[:,:4])
            if args.smoke and step==1:
                grads=torch.autograd.grad(ordinary+.3*extra,(model.head.coefficients.weight,model.head.continuous.weight),retain_graph=True)
                if any(not torch.isfinite(g).all() or g.abs().sum().item()==0 for g in grads):
                    raise ValueError("The matte or rank readout did not receive supervision")
            loss=ordinary+.3*extra
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.00012*min(step/100,1)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(avg,params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                log.write(json.dumps({"step":step,"ordinary":ordinary.item(),"matte":extra.item()},allow_nan=False)+"\n")
    dst=folder/"final.pt"
    torch.save(ema.state_dict(),dst)
    rc.write_json(folder/"fit.json",{"fold":args.fold,"mode":args.mode,"steps":steps,"smoke":args.smoke,"seed":seed,
        "train":ids,"excluded_validation":valid,"warm":identity["checkpoint"],"image_shift":identity["applied_shift"],
        "estimated_alpha_shift":list(shift),"shift_error":[before,after],"schedule":schedule.hexdigest(),
        "checkpoint":rc.hash_file(dst),"data":rc.DATA_HASH,"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("MATTE_FIT",args.fold,args.mode,steps,shift,flush=True)


def lock(args):
    images,truth,train_ids,lib,check,base,mask,rec,centers=views.eval_context(args)
    warm,identity=views.warm_identity(args,train_ids)
    ids=lib+check
    old=align.predict(vote.load_net(warm/"final.pt"),images[ids,None],centers,identity["applied_shift"])[:,:4]
    if not np.isfinite(old).all():raise ValueError("Invalid warm predictions")
    arrays={"reference":np.packbits(mask,axis=-1),"initial":np.packbits(old>0,axis=-1)}
    choices={};members={};schedules=[];first=[];shifts=[]
    for name in MODES:
        folder=args.run/f"fold-{args.fold}-{name}"
        fit=views.read(folder/"fit.json")
        if (fit["fold"]!=args.fold or fit["mode"]!=name or fit["steps"]!=STEPS or fit["smoke"]
                or sorted(fit["train"])!=sorted(train_ids) or fit["warm"]!=identity["checkpoint"]
                or fit["data"]!=rc.DATA_HASH or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")
                or fit["source"]!=rc.hash_file(__file__) or fit["protocol"]!=rc.hash_file(PROTOCOL)):
            raise ValueError("Matte training identity differs")
        model=load_backbone(warm/"final.pt").eval()
        model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
        prediction=align.predict(model,images[ids,None],centers,identity["applied_shift"])[:,:4]
        if not np.isfinite(prediction).all():raise ValueError("Invalid matte predictions")
        final,choices[name]=views.insert(base,old,prediction,truth[lib].astype(bool))
        arrays[name]=np.packbits(final,axis=-1);arrays["standalone_"+name]=np.packbits(prediction>0,axis=-1)
        members[name]=fit["checkpoint"];schedules.append(fit["schedule"]);shifts.append(tuple(fit["estimated_alpha_shift"]))
        with (folder/"train.jsonl").open() as st:first.append(json.loads(next(st))["ordinary"])
    if len(set(schedules))!=1 or len(set(shifts))!=1 or not np.allclose(first,first[0],rtol=1e-5,atol=1e-5):
        raise ValueError("Matte ablation inputs differ")
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"reference":rec["prediction"],
        "members":members,"choices":choices,"warm":identity["checkpoint"],"schedule":schedules[0],
        "alpha_shift":list(shifts[0]),"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("MATTE_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","lock"))
    for key in ("run","data","align"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("stable","formal","parent","ensemble","public","local","texture"):
        p.add_argument("--"+key,type=Path)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--mode",choices=MODES,default="registered")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(3)
    {"fit":train,"lock":lock}[args.phase](args)
