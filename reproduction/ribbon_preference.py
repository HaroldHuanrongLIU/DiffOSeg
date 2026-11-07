"""Closed-curve CT ribbon model for aligned LIDC area-ranked contours."""
import argparse
import copy
import json
import math
from pathlib import Path
import socket
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure import find_contours
from torch import nn

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import rank_contours as rc
from reproduction import rater_area as area

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/ribbon_protocol.md"
A=128
T=np.linspace(-6,6,49,dtype=np.float32)
STEPS,BATCH=6000,40


def interp(image,xy):
    return cv2.remap(image.astype(np.float32),xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def polygon_area_np(p):
    return .5*abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))


def contour(guide):
    curves=find_contours(guide,0,fully_connected="high")
    curves=[v[:,::-1] for v in curves if len(v)>=5 and np.linalg.norm(v[0]-v[-1])<.1]
    if not curves:return None
    curve=max(curves,key=polygon_area_np)
    ds=np.linalg.norm(np.diff(curve,axis=0),axis=1)
    path=np.r_[0,np.cumsum(ds)]
    use=np.r_[True,np.diff(path)>1e-6]
    if path[-1]<5:return None
    u=np.arange(A)*path[-1]/A
    c=np.stack([np.interp(u,path[use],axis[use]) for axis in curve.T],1).astype(np.float32)
    tangent=np.roll(c,-2,axis=0)-np.roll(c,2,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-5)
    n=np.stack((tangent[:,1],-tangent[:,0]),1)
    if interp(guide,(c+0.6*n)[:,None]).mean()>interp(guide,(c-0.6*n)[:,None]).mean():
        n=-n
    return c,n


def sample_ribbon(im,guide,true_fields=None):
    pair=contour(guide)
    if pair is None:return None
    c,n=pair
    xy=c[:,None,:]+n[:,None,:]*T[None,:,None]
    profile=interp(im,xy)
    if true_fields is None:return profile,c,n
    y=np.stack([interp(d,xy) for d in true_fields])
    bracket=(y[:,:,:-1]>=0)&(y[:,:,1:]<0)
    mid=(T[1:]+T[:-1])/2
    ix=(abs(mid)[None,None,:]+100*~bracket).argmin(2)
    left=np.take_along_axis(y,ix[:,:,None],axis=2)[:,:,0]
    right=np.take_along_axis(y,(ix+1)[:,:,None],axis=2)[:,:,0]
    offset=T[ix]+.25*left/(left-right).clip(1e-5)
    offset=np.where(bracket.any(2),offset,np.where(y[:,:,len(T)//2]>0,T[-1],T[0])).clip(-6,6)
    return profile,c,n,offset.astype(np.float32)


class Block(nn.Module):
    def __init__(self,w,d):
        super().__init__()
        self.layers=nn.Sequential(nn.GroupNorm(8,w),nn.SiLU(),nn.Conv1d(w,w,3,padding=d,dilation=d,padding_mode="circular"),
                                  nn.GroupNorm(8,w),nn.SiLU(),nn.Conv1d(w,w,1))
    def forward(self,x):
        return x+self.layers(x)


class Ribbon(nn.Module):
    def __init__(self):
        super().__init__()
        w=96
        self.encoder=nn.Sequential(nn.Conv1d(2*len(T)+7,w,1),*(Block(w,d) for d in (1,2,4,8,16,8,4,2,1)))
        self.offset=nn.Sequential(nn.Conv1d(2*w,w,1),nn.GroupNorm(8,w),nn.SiLU(),nn.Conv1d(w,4,1))

    def forward(self,profile,c,n):
        # profile [B,A,R]; compute physical-window and locally standardized contrast.
        x=profile/255
        inside=x[:,:,:7].mean((1,2),keepdim=True)
        outside=x[:,:,-7:].mean((1,2),keepdim=True)
        contrast=((x-outside)/(inside-outside).clamp_min(.06)).clamp(-1,2)
        cc=c-c.mean(1,keepdim=True)
        dist=cc.norm(dim=2,keepdim=True)
        curvature=(torch.roll(n,-2,1)-torch.roll(n,2,1)).norm(dim=2,keepdim=True)
        scale=dist.mean(1,keepdim=True).expand_as(dist)
        geom=torch.cat((cc/16,n,dist/16,curvature,scale/16),2)
        feat=self.encoder(torch.cat((3*(x-.25),contrast,geom),2).transpose(1,2).contiguous())
        context=feat.mean(2,keepdim=True).expand_as(feat)
        return 7*self.offset(torch.cat((feat,context),1)).tanh()


def polygon_area(points):
    following=torch.roll(points,-1,dims=-2)
    return .5*((points[...,0]*following[...,1])-(points[...,1]*following[...,0])).sum(-1).abs()


def train(args):
    torch.set_num_threads(3);cv2.setNumThreads(1)
    ims,masks,_,uids=native.inputs(args.data)
    outer,inner,valid=native.splits(uids,args.fold)
    mode="inner" if args.mode=="smoke" else args.mode
    ids=inner if mode=="inner" else outer["train_indices"]
    shift,surface=align.estimate_shift(ims[ids if args.mode!="smoke" else ids[:48]],
                                     masks[ids if args.mode!="smoke" else ids[:48]],
                                     [str(uids[i]).split("_")[0] for i in (ids if args.mode!="smoke" else ids[:48])])
    steps=STEPS
    if args.mode=="smoke":ids,steps=ids[:32],6
    seed=245031+args.fold
    rng=np.random.default_rng(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    args.out.mkdir(parents=True,exist_ok=False)
    meta={"fold":args.fold,"mode":args.mode,"steps":steps,"seed":seed,"shift":list(shift),"data":rc.DATA_HASH,
          "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"host":socket.gethostname(),"gpu":torch.cuda.get_device_name()}
    rc.write_json(args.out/"plan.json",{**meta,"train_indices":ids,"inner_valid_indices":valid if mode=="inner" else []})
    prof=[]; curves=[]; normals=[]; labels=[]; sizes=[]; names=[]
    for i in ids:
        im=area.register_image(ims[i],shift)
        truth=masks[i].astype(bool);pixel=sum(truth)
        targets=np.stack([area.sdf(m) for m in truth])
        median=.5*(area.sdf(pixel>=2)+area.sdf(pixel>=3))
        for rep in range(10 if args.mode!="smoke" else 2):
            if rep==0: guide=cv2.GaussianBlur(median,(0,0),.65)
            else:
                coeff=rng.uniform(.2,.9)
                start=coeff*median+(1-coeff)*targets[rng.integers(4)]
                dx,dy=rng.normal(0,1.1,2)
                warp=cv2.warpAffine(start,np.float32([[1,0,dx],[0,1,dy]]),(128,128),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                noise=cv2.resize(rng.normal(size=(14,14)).astype("float32"),(128,128),interpolation=cv2.INTER_CUBIC)
                guide=cv2.GaussianBlur(warp,(0,0),rng.uniform(.65,1.6))+noise*rng.uniform(.45,1.75)+rng.uniform(-2.2,2.2)
            item=sample_ribbon(im,guide,targets)
            if item is None:continue
            x,c,n,y=item
            prof.append(x.astype("float16"));curves.append(c);normals.append(n);labels.append(y)
            sizes.append(truth.sum((1,2)));names.append(str(uids[i]).split("_")[0])
    if not prof:raise ValueError("No usable training guides")
    print("RIBBONS",len(prof),flush=True)
    x,c,n,y,s=[torch.from_numpy(np.stack(z)).cuda() for z in (prof,curves,normals,labels,sizes)]
    _,idx,num=np.unique(names,return_inverse=True,return_counts=True)
    w=torch.from_numpy(.5/len(names)+.5/(num[idx]*len(num))).float().cuda()
    net=Ribbon().cuda();ema=copy.deepcopy(net).eval().requires_grad_(False)
    all_params=list(net.parameters());avg=list(ema.parameters())
    opt=torch.optim.AdamW(all_params,lr=.0007,fused=True,weight_decay=1e-4)
    started=time.monotonic()
    with (args.out/"train.jsonl").open("x",buffering=1) as log:
        for step in range(1,steps+1):
            ix=torch.multinomial(w,BATCH,replacement=True);profile=x[ix].float()
            # Rotate or reflect the CT and guide together, conserving offset and area targets.
            a=torch.rand(BATCH,device="cuda")*2*math.pi
            flip=torch.where(torch.rand_like(a)<.5,-1.,1.)
            mat=native.matrix_for(torch.ones_like(a),a,flip)
            ctr=c[ix].mean(1,keepdim=True);pts=(c[ix]-ctr)@mat+ctr;normal=n[ix]@mat
            if step%2:normal=normal.flip([1]);pts=pts.flip([1]);profile=profile.flip([1]);target=y[ix].flip([2])
            else:target=y[ix]
            jitter=torch.rand(BATCH,3,1,1,device="cuda")
            profile=(255*(profile/255).clamp_min(1e-6).pow(.85+.3*jitter[:,0])*(.85+.3*jitter[:,1])+16*(jitter[:,2]-.5)).clamp(0,255)
            offset=net(profile,pts,normal)
            polygons=pts[:,None]+offset[:,:,:,None]*normal[:,None]
            radius=(polygon_area(polygons).clamp_min(1)/math.pi).sqrt()
            actual=(s[ix].float()/math.pi).sqrt()
            loss=F.smooth_l1_loss(offset,target,beta=.65)+.7*F.smooth_l1_loss(radius,actual,beta=.5)
            loss=loss+.12*F.smooth_l1_loss(offset-torch.roll(offset,2,2),target-torch.roll(target,2,2),beta=.5)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params,4,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=.0007*min(1,step/100)*(.02+.98*(1+math.cos(math.pi*step/steps))/2)
            opt.step()
            with torch.no_grad():torch._foreach_lerp_(avg,all_params,max(.005,1/step))
            if step==1 or step%250==0 or step==steps:
                rec={"step":step,"loss":loss.item(),"seconds":time.monotonic()-started}
                log.write(json.dumps(rec,allow_nan=False)+"\n");print(rec,flush=True)
    torch.save(ema.state_dict(),args.out/"final.pt")
    rc.write_json(args.out/"fit.json",{**meta,"ribbons":len(prof),"checkpoint":rc.hash_file(args.out/"final.pt"),
                  "seconds":time.monotonic()-started})


def polygon_sdf(points):
    # Evaluate at native pixel centers after supersampled contour rasterization.
    low=np.maximum(0,np.floor(points.min(0)-10).astype(int)); high=np.minimum(127,np.ceil(points.max(0)+10).astype(int))
    if np.any(high<low):return np.full((128,128),-8,np.float32)
    ss=4;wh=high-low+1
    tile=np.zeros((ss*wh[1],ss*wh[0]),np.uint8)
    cv2.fillPoly(tile,[np.rint((points-low)*ss).astype(np.int32)],1)
    signed=area.sdf(tile.astype(bool))[::ss,::ss]/ss
    out=np.full((128,128),-8,np.float32)
    out[low[1]:high[1]+1,low[0]:high[0]+1]=signed.clip(-8,8)
    return out


@torch.inference_mode()
def predict(net,images,masks,shift):
    result=[]
    for im,m in zip(images,masks):
        im=area.register_image(im,shift)
        votes=m.sum(0); fields=np.stack([rc.signed(mask) for mask in m]);guide=.5*(area.sdf(votes>=2)+area.sdf(votes>=3))
        parts=[]
        for jitter in (-0.75,0.,.75):
            item=sample_ribbon(im,cv2.GaussianBlur(guide,(0,0),.75)+jitter)
            if item is None:continue
            profile,c,n=item
            xs,cs,ns=[torch.from_numpy(v[None]).cuda() for v in (profile,c,n)]
            offset=(net(xs,cs,ns)+net(xs.flip([1]),cs.flip([1]),ns.flip([1])).flip([2]))[0].cpu().numpy()/2
            refined=np.stack([polygon_sdf(c+delta[:,None]*n) for delta in offset])
            # Only replace the narrow neighborhood of the guide being refined.
            pts=np.rint(c).astype(int).clip(0,127)
            line=np.ones((128,128),np.uint8);line[pts[:,1],pts[:,0]]=0
            dist=cv2.distanceTransform(line,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            w=np.clip((8-dist)/2,0,1)
            parts.append(w*refined+(1-w)*fields)
        result.append(np.mean(parts,axis=0) if parts else fields)
    return np.stack(result)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True);p.add_argument("--mode",choices=("inner","full","smoke"),required=True)
    train(p.parse_args())
