"""Estimate plausible rater areas from registered CT and deliberately inexact guides."""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from reproduction import native_local as native
from reproduction import native_study as ns
from reproduction import rank_contours as rc

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_90/area_protocol.md"
QUANT=(.1,.3,.5,.7,.9)
RADII=np.array([-6.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,6.],np.float32)


def register_image(im,shift):
    x,y=shift
    return im[np.clip(np.arange(im.shape[0])-y,0,im.shape[0]-1)[:,None],np.clip(np.arange(im.shape[1])-x,0,im.shape[1]-1)[None,:]]


def sdf(mask):
    a=mask.astype("uint8");di=cv2.distanceTransform(a,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    dout=cv2.distanceTransform(1-a,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    if not a.any():return np.full(a.shape,-20,np.float32)
    if a.all():return np.full(a.shape,20,np.float32)
    return np.where(a,di-.5,.5-dout)


def describe(im,mask):
    area=int(mask.sum())
    if area==0:
        mask=mask.copy();mask[64,64]=True
    d=sdf(mask); smooth=cv2.GaussianBlur(d,(0,0),1)
    gy,gx=np.gradient(smooth)
    inner=mask&(d<=1); yy,xx=np.where(inner)
    if len(xx)>180:
        j=np.linspace(0,len(xx)-1,180,dtype=int);yy,xx=yy[j],xx[j]
    if not len(xx): yy=np.array([64]);xx=np.array([64])
    norm=np.maximum(np.hypot(gx[yy,xx],gy[yy,xx]),1e-4)
    x=(xx[:,None]+gx[yy,xx,None]/norm[:,None]*RADII).astype("float32")
    y=(yy[:,None]+gy[yy,xx,None]/norm[:,None]*RADII).astype("float32")
    prof=cv2.remap(im.astype("float32"),x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    core=im[d>=max(0.,d.max()*.32)]
    shell=im[(d<-.5)&(d>-7)]
    if not len(shell):shell=im.ravel()
    fg=np.quantile(core,.62); bg=np.quantile(shell,.24); span=max(fg-bg,12)
    localprof=(prof-bg)/span
    deriv=np.diff(prof,axis=1)/np.diff(RADII)
    peak=((RADII[1:]+RADII[:-1])/2)[np.argmax(deriv,axis=1)]
    contrast=prof[:,-2:].mean(1)-prof[:,:2].mean(1)
    wy,wx=np.where(mask)
    c=np.cov(np.stack((wx,wy)),bias=True) if len(wx)>1 else np.eye(2)
    eig=np.linalg.eigvalsh(c).clip(.1)
    shape=[np.sqrt(area/np.pi),np.sqrt(eig[0]),np.sqrt(eig[1]),len(xx)/max(np.sqrt(area),1),
           max(wx)-min(wx),max(wy)-min(wy),np.mean(wx)-64.5,np.mean(wy)-64.5,
           fg,bg,span,im.mean(),im.std(),d.max()]
    signals=[np.quantile(core,QUANT),np.quantile(shell,QUANT),
             np.quantile(prof,[.15,.5,.85],axis=0).ravel(),
             np.quantile(localprof,[.15,.5,.85],axis=0).ravel(),
             np.mean(deriv,axis=0),np.quantile(deriv.max(1),QUANT),
             np.quantile(contrast,QUANT),np.quantile(peak,QUANT)]
    band=(d>-6)&(d<6)
    alpha=((im.astype(float)-bg)/span).clip(0,1)
    for threshold in (.2,.35,.5,.65,.8):
        estimate=(d>=6)|(band&(alpha>threshold))
        shape.append(np.sqrt(estimate.sum()/np.pi)-np.sqrt(area/np.pi))
    return np.concatenate([np.asarray(shape),*signals]).astype(np.float32)


def make_guides(masks,rng):
    light=masks.sum(0)
    fields=np.stack([sdf(x) for x in [*masks,light>=1,light>=2,light>=3,light>=4]])
    xx,yy=np.meshgrid(np.arange(128,dtype=np.float32),np.arange(128,dtype=np.float32))
    for i in range(10):
        first,second=rng.integers(len(fields),size=2);mix=rng.uniform()
        f=mix*fields[first]+(1-mix)*fields[second]
        dx,dy=rng.normal(size=2)*.9
        g=cv2.remap(f,xx+np.float32(dx),yy+np.float32(dy),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        noise=cv2.resize(rng.normal(size=(12,12)).astype(np.float32),(128,128),interpolation=cv2.INTER_CUBIC)
        yield cv2.GaussianBlur(g,(0,0),rng.uniform(.5,1.1))+noise*rng.uniform(.3,1.3)+rng.uniform(-2.2,2.2)>0


def fit(args):
    cv2.setNumThreads(1)
    images,truth,_,uids=native.inputs(args.data)
    outer,train,valid=native.splits(uids,args.fold)
    ids=train if args.mode=="inner" else outer["train_indices"]
    shift_plan=json.loads((args.align/f"fold-{args.fold}-{args.mode}-registered/plan.json").read_text())
    if ids!=shift_plan["train_indices"]:raise ValueError("Area training partition")
    shift=shift_plan["applied_shift"]; rng=np.random.default_rng(76111+args.fold)
    xs=[];ys=[];names=[]
    for i in ids:
        im=register_image(images[i],shift); masks=truth[i].astype(bool)
        target=np.sqrt(masks.sum((1,2))/np.pi)
        for g in make_guides(masks,rng):
            if not g.any():continue
            xs.append(describe(im,g));ys.append(target-np.sqrt(g.sum()/np.pi));names.append(str(uids[i]).split("_")[0])
    _,inv,n=np.unique(names,return_inverse=True,return_counts=True)
    model=ExtraTreesRegressor(n_estimators=180,max_features=.65,max_depth=19,min_samples_leaf=9,n_jobs=5,random_state=76111+args.fold)
    model.fit(np.stack(xs),np.stack(ys),sample_weight=.5+.5*len(inv)/(len(n)*n[inv]))
    dst=args.out;dst.mkdir(parents=True,exist_ok=False)
    joblib.dump(model,dst/"model.joblib",compress=3)
    rc.write_json(dst/"fit.json",{"fold":args.fold,"mode":args.mode,"train_indices":ids,"source":rc.hash_file(__file__),
                  "protocol":rc.hash_file(PROTOCOL),"shift":shift,"rows":len(xs),"model":rc.hash_file(dst/"model.joblib")})
    print("AREA_FIT",args.fold,args.mode,len(xs),flush=True)


def predict_areas(model,images,logits,shift):
    feats=[];counts=[]
    for im,z in zip(images,logits):
        image=register_image(im,shift);m=z>0;votes=m.sum(0)
        guides=[m[1],m[2],votes>=2,votes>=3]
        center=sdf(votes>=2)
        guides.extend((center>1.2,center>-1.3))
        for g in guides:
            feats.append(describe(image,g));counts.append(np.sqrt(g.sum()/np.pi))
    pred=model.predict(np.stack(feats))+np.array(counts)[:,None]
    return np.sort(np.median(pred.reshape(len(images),6,4),axis=1).clip(0),axis=1)


def apply_radii(logit,original,radius,w,offset):
    n=original.reshape(4,-1).sum(1);r=np.sqrt(n/np.pi)
    goal=np.maximum(0,r+w*np.clip(radius-r+offset,-2.5,2.5))
    area=np.minimum(logit.shape[-2]*logit.shape[-1],np.rint(np.pi*goal**2).astype(int))
    result=original.copy()
    for j in range(4):
        # Only add/remove pixels from the incumbent; order the changed boundary by its image score.
        delta=int(area[j]-n[j])
        if not delta:continue
        possible=np.flatnonzero(original[j].ravel() if delta<0 else ~original[j].ravel())
        ix=np.argsort(logit[j].ravel()[possible],kind="stable")
        chosen=possible[ix[: -delta]] if delta<0 else possible[ix[-delta:]]
        result[j].flat[chosen]=(delta>0)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("data","align","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    p.add_argument("--mode",choices=("inner","full"),required=True)
    fit(p.parse_args())
