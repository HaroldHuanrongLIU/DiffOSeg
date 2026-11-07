"""Prepare strictly held-out base representations and learn spatial/size corrections."""
import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction import crossfit_contours as cf
from reproduction import native_local as native
from reproduction import polar_contours as polar
from reproduction import native_study as ns
from reproduction import rank_contours as rc
from reproduction import rater_area as area
from reproduction import texture_study as tex
from reproduction import vote_contours as vote

STEPS=4200
PROTOCOL=cf.PROTOCOL


def teacher_logits(checkpoint, im, shift):
    model=vote.load_net(checkpoint)
    centers=np.full((len(im),2),64.5,np.float32)
    with torch.inference_mode():
        guesses=[]
        for x in torch.from_numpy(im).cuda().split(16):
            x=align.registered_ct(x,*shift).float()/255
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                z=model(x)
            c=native.rater_center(z[:,:4]>0)
            guesses.append(c.cpu().numpy())
        guess=np.concatenate(guesses)
        centers=np.where(((abs(guess-64.5).max(1)>4)&((abs(guess-64.5).max(1))<48))[:,None],guess,centers)
    return align.predict(model,im,centers,shift)[:,:4]


def prepare(args):
    images,truth,_,uids=native.inputs(args.data)
    eligible, excluded = cf.training_partition(uids, args.fold, args.scope)
    groups=cf.partitions(eligible,uids,args.fold)
    teacher_source=args.run/"source/reproduction/crossfit_contours.py"
    teacher_hash=rc.hash_file(teacher_source)
    all_ids=[];all_logits=[];sources=[]
    for j,(fit,hold) in enumerate(groups):
        root=args.run/f"fold-{args.fold}-sub-{j}"
        plan=json.loads((root/"plan.json").read_text());record=json.loads((root/"fit.json").read_text())
        if plan["fit_indices"]!=fit or plan["held_indices"]!=hold or plan.get("excluded_indices",plan.get("excluded_inner_validation"))!=excluded or plan.get("scope","inner")!=args.scope:
            raise ValueError("Cross-fit partition failed")
        if record["source"]!=teacher_hash or record["steps"]!=cf.STEPS or record["checkpoint"]!=rc.hash_file(root/"final.pt"):
            raise ValueError("Teacher source or weights changed")
        sources.append(record["checkpoint"])
        pred=teacher_logits(root/"final.pt",images[hold,None],record["shift"])
        all_logits.append(pred.astype("float16"));all_ids.extend(hold)
    ix=np.argsort(all_ids)
    ids=np.asarray(all_ids)[ix]
    if set(ids)!=set(eligible) or len(ids)!=len(eligible):raise ValueError("Incomplete OOF training set")
    logits=np.concatenate(all_logits)[ix]
    filename=args.run/f"oof-{args.fold}.npz"
    np.savez_compressed(filename,ids=ids,logits=logits)
    dice=float(ns.dice(truth[ids].astype(bool),logits>0).mean())
    rc.write_json(args.run/f"oof-{args.fold}.json",{"fold":args.fold,"source":rc.hash_file(__file__),"teachers":sources,
                  "payload":rc.hash_file(filename),"oof_cases":len(eligible),"scope":args.scope,"oof_preference_dice":dice})
    print("OOF",args.fold,len(ids),dice,flush=True)


class Correction(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=rc.Residual(13,32)
        self.down=nn.ModuleList(nn.Sequential(rc.Residual(a,b,2),rc.Residual(b,b)) for a,b in ((32,48),(48,72),(72,96)))
        self.up=nn.ModuleList(rc.Residual(a+b,a) for a,b in ((72,96),(48,72),(32,48)))
        self.shape=nn.Conv2d(32,4,1)
        self.size=nn.Sequential(nn.LayerNorm(2*96+8),nn.Linear(2*96+8,80),nn.SiLU(),nn.Linear(80,4))
        nn.init.zeros_(self.shape.weight);nn.init.zeros_(self.shape.bias)
        nn.init.zeros_(self.size[-1].weight);nn.init.zeros_(self.size[-1].bias)

    def forward(self,image,guide):
        p=guide.sigmoid()
        mid=p.mean(1,keepdim=True)
        core=mid.pow(3); shell=4*mid*(1-mid)
        fg=(image*core).sum((2,3),keepdim=True)/core.sum((2,3),keepdim=True).clamp_min(1)
        dilated=F.max_pool2d(mid,15,1,7)
        ring=(dilated-mid).clamp_min(0)
        bg=(image*ring).sum((2,3),keepdim=True)/ring.sum((2,3),keepdim=True).clamp_min(1)
        span=(fg-bg).clamp_min(.07)
        local=F.avg_pool2d(image,7,1,3,count_include_pad=False)
        coord=(native.coordinates(image.shape[-1],image.device).permute(2,0,1)[None]-64.5)/48
        features=torch.cat((image*2-1,((image-bg)/span).clamp(-1,2),((image-local)/span).clamp(-2,2),
                            coord.expand(len(image),-1,-1,-1),guide.clamp(-8,8)/8,p),1)
        x=self.stem(features.contiguous(memory_format=torch.channels_last))
        saved=[x]
        for block in self.down:
            x=block(x);saved.append(x)
        coarse=x.float()
        def pool(w):
            ww=F.interpolate(w,coarse.shape[-2:],mode="area")
            return (coarse*ww).sum((2,3))/ww.sum((2,3)).clamp_min(.01)
        radius=((guide>0).float().sum((2,3)).clamp_min(1)/math.pi).sqrt()
        soft_radius=(p.sum((2,3)).clamp_min(1)/math.pi).sqrt()
        delta=2.5*self.size(torch.cat((pool(core),pool(shell),radius/12,(soft_radius-radius).clamp(-4,4)),1)).float().tanh()
        for block,skip in zip(self.up,reversed(saved[:-1])):
            x=block(torch.cat((F.interpolate(x,skip.shape[-2:],mode="bilinear",align_corners=False),skip),1))
        disp=4*self.shape(x).float().tanh()
        band=(4*p*(1-p)).clamp_min(.00001)
        # Remove the uniform component of the local displacement and assign it to the area head.
        disp=disp-(disp*band).sum((2,3),keepdim=True)/band.sum((2,3),keepdim=True).clamp_min(1)
        return guide+disp+2*delta[:,:,None,None],(radius+delta).clamp_min(0)


def fit(args):
    ims,labels,_,uids=native.inputs(args.data);_,inner,_=native.splits(uids,args.fold)
    filename=args.run/f"oof-{args.fold}.npz";receipt=json.loads(filename.with_suffix(".json").read_text())
    if rc.hash_file(filename)!=receipt["payload"]:raise ValueError("OOF payload differs")
    with np.load(filename,allow_pickle=False) as z:
        ids=z["ids"].tolist(); stored=z["logits"]
    if ids!=sorted(inner):raise ValueError("Correction fit can only use inner training patients")
    shift=json.loads((args.align/f"fold-{args.fold}-inner-registered/fit.json").read_text())["applied_shift"]
    image=align.registered_ct(torch.from_numpy(ims[ids,None]).cuda(),*shift)
    guide=torch.from_numpy(stored).cuda()
    y=torch.from_numpy(labels[ids]).cuda().float()
    seed=44971+args.fold;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=Correction().cuda().to(memory_format=torch.channels_last)
    ema=copy.deepcopy(net).eval().requires_grad_(False)
    params,avg=list(net.parameters()),list(ema.parameters());opt=torch.optim.AdamW(params,lr=.0005,weight_decay=.00035,fused=True)
    _,inv,n=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True,return_counts=True)
    sample=torch.from_numpy(.5/len(ids)+.5/(n[inv]*len(n))).float().cuda()
    root=args.run/f"correction-{args.fold}";root.mkdir(exist_ok=False)
    for step in range(1,STEPS+1):
        idx=torch.multinomial(sample,24,replacement=True)
        x=image[idx].float()/255;t=y[idx];g=guide[idx].float()
        # Most guides remain real OOF predictions; modest shift/temperature jitter spans calibration differences.
        u=torch.rand(len(idx),4,1,1,device="cuda")
        g=g*(.75+.55*u[:,:1])+((u[:,1:2]-.5)*2)*(u[:,2:3]<.35)
        x=x.clamp_min(1e-6).pow(.85+.3*u[:,3:])
        rot=step%4
        x,g,t=[torch.rot90(v,rot,(-2,-1)) for v in (x,g,t)]
        if (step//4)%2:x,g,t=[v.flip([-1]) for v in (x,g,t)]
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            logits,radius=net(x,g)
        actual=(t.sum((2,3)).clamp_min(1)/math.pi).sqrt()
        loss=vote.mask_loss(logits,t)+.28*F.smooth_l1_loss(radius,actual,beta=.6)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,3,error_if_nonfinite=True)
        opt.param_groups[0]["lr"]=.0005*min(step/100,1)*(.03+.97*(1+math.cos(math.pi*step/STEPS))/2)
        opt.step()
        with torch.no_grad():torch._foreach_lerp_(avg,params,max(.005,1/step))
        if step==1 or step%500==0:print("FIT",args.fold,step,loss.item(),flush=True)
    torch.save(ema.state_dict(),root/"final.pt")
    rc.write_json(root/"fit.json",{"fold":args.fold,"seed":seed,"steps":STEPS,"ids":ids,"source":rc.hash_file(__file__),
                  "oof":receipt["payload"],"checkpoint":rc.hash_file(root/"final.pt"),"shift":shift})


@torch.inference_mode()
def predict(model,im,prior):
    logits=[];radii=[]
    for j in range(0,len(im),16):
        x=torch.as_tensor(im[j:j+16],device="cuda").float()/255
        g=torch.as_tensor(prior[j:j+16],device="cuda").float()
        total=0;r=0
        for flip in (False,True):
            a,b=(x.flip([-1]),g.flip([-1])) if flip else (x,g)
            for k in range(4):
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    z,rad=model(torch.rot90(a,k,(-2,-1)),torch.rot90(b,k,(-2,-1)))
                z=torch.rot90(z,-k,(-2,-1));total+=z.flip([-1]) if flip else z;r+=rad
        logits.append((total/8).cpu().numpy());radii.append((r/8).cpu().numpy())
    return np.concatenate(logits),np.concatenate(radii)


def check(args):
    ims,y,_,uids=native.inputs(args.data);_,train,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})()
    lib,check=polar.calibration_split(group,valid,args.fold);ids=lib+check;n=len(lib)
    folder=args.run/f"correction-{args.fold}";state=json.loads((folder/"fit.json").read_text())
    if state["ids"]!=sorted(train) or state["checkpoint"]!=rc.hash_file(folder/"final.pt"):raise ValueError("Invalid correction checkpoint")
    cache=args.texture/f"features-{args.fold}-inner.npz"
    receipt=json.loads(cache.with_suffix(".json").read_text())
    if rc.hash_file(cache)!=receipt["payload"] or receipt["indices"]!=ids:raise ValueError("Invalid baseline cache")
    with np.load(cache,allow_pickle=False) as z:
        original=np.unpackbits(z["old"],axis=-1).astype(bool)
        base=tex.anchor_logits(z["base"].astype(np.float32),original)
    model=Correction().cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
    im=align.registered_ct(torch.from_numpy(ims[ids,None]),*state["shift"]).numpy()
    out,r=predict(model,im,base)
    truth=y[ids].astype(bool);target=np.sqrt(truth.sum((2,3))/np.pi)
    off=.5*np.median(target[:n]-r[:n],axis=0)
    options=[]
    def apply(indices,w,a):
        q=(1-w)*base[indices]+w*out[indices]
        return np.stack([area.apply_radii(z,m,p,a,off) for z,m,p in zip(q,q>0,r[indices])])
    for w in (0.,.25,.5,.75,1.):
        for a in (0.,.4,.8):
            pred=apply(np.arange(n),w,a)
            options.append({"shape":w,"area":a,"score":float(ns.dice(truth[:n],pred).mean())})
    best=max(options,key=lambda o:o["score"])
    pred=apply(np.arange(n,len(ids)),best["shape"],best["area"])
    res={"reference":float(ns.dice(truth[n:],original[n:]).mean()),"correction_only":float(ns.dice(truth[n:],out[n:]>0).mean()),
         "candidate":float(ns.dice(truth[n:],pred).mean())}
    rc.write_json(args.run/f"inner-{args.fold}.json",{"fold":args.fold,"source":rc.hash_file(__file__),"best":best,"grid":options,
                  "check":res,"radius_rmse":float(np.mean((target[n:]-r[n:])**2)**.5),"offset":off.tolist()})
    np.savez_compressed(args.run/f"corrected-{args.fold}-inner.npz",logits=out.astype("float16"),radius=r,ids=ids)
    print("CHECK",args.fold,best,res,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","fit","check"))
    for k in ("run","align","texture","data"):p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--scope",choices=("inner","full"),default="inner")
    p.add_argument("--fold",required=True,type=int,choices=range(4));a=p.parse_args()
    if a.phase!="prepare" and a.scope!="inner":p.error("Residual correction fitting and checks use the inner scope")
    cv2.setNumThreads(1);torch.set_num_threads(3)
    {"prepare":prepare,"fit":fit,"check":check}[a.phase](a)
