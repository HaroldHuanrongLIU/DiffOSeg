"""Conditional native-pixel foreground posterior from patient-out-of-fit CT guides."""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier

from reproduction import native_local as native
from reproduction import native_study as ns
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import rater_area as area
from reproduction import texture_study as tex

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/edge_protocol.md"
WEIGHTS=(0.,.25,.5,.75,1.)


def blurred(a,sigma):
    return cv2.GaussianBlur(a.astype("float32"),(0,0),sigma,borderType=cv2.BORDER_REPLICATE)


def local_stats(im,w,sigma):
    weight=w.astype("float32")
    denom=max(weight.sum(),1.)
    mean=(im*weight).sum()/denom
    second=(im*im*weight).sum()/denom
    mass=blurred(weight,sigma)+.025
    m=(blurred(weight*im,sigma)+.025*mean)/mass
    v=(blurred(weight*im*im,sigma)+.025*second)/mass-m*m
    return m,np.maximum(v,3.5**2).astype("float32")


def samples(im,raw,subsample=None,rng=None):
    """Pixels outside the uncertain band retain their original hard classification."""
    im=im.astype("float32");masks=raw>0
    s=np.stack([area.sdf(m) for m in masks]);votes=masks.sum(0)
    mid=(votes>=2).astype("uint8")
    fg=cv2.erode(mid,np.ones((3,3),np.uint8)).astype(bool)
    outdist=-area.sdf(votes>0)
    bg=(outdist>=1.5)&(outdist<=14)
    if not fg.any():fg=mid.astype(bool)
    if not bg.any():bg=~mid.astype(bool)
    # Use masks to select tissue, not to assign CT intensities to classes.
    fgmed=float(np.median(im[fg])) if fg.any() else float(np.median(im))
    bgmed=float(np.quantile(im[bg],.30)) if bg.any() else float(np.median(im))
    span=max(fgmed-bgmed,10.)
    ly,lx=np.gradient(blurred(im,.8))
    high=(im-blurred(im,1.3))/span
    common=[im,(im-bgmed)/span,high,(im-blurred(im,3))/span,np.hypot(lx,ly)/span]
    for sigma in (2.2,5.0,11.):
        a,va=local_stats(im,fg,sigma);b,vb=local_stats(im,bg,sigma)
        za=(im-a)/np.sqrt(va);zb=(im-b)/np.sqrt(vb)
        bf=2*np.log1p(zb*zb/3)-2*np.log1p(za*za/3)+.5*np.log(vb/va)
        common.extend([(im-b)/np.maximum(a-b,8.),(a-b)/span,za,zb,bf,np.sqrt(va)/span,np.sqrt(vb)/span])
    radii=np.sqrt(masks.sum((1,2))/np.pi)
    rows=[];pixels=[]
    for rank in range(4):
        band=abs(s[rank])<=4.6
        yy,xx=np.where(band)
        if subsample is not None and len(yy)>subsample:
            idx=rng.choice(len(yy),size=subsample,replace=False); yy,xx=yy[idx],xx[idx]
        gy,gx=np.gradient(blurred(s[rank],1))
        norm=np.maximum(np.hypot(gx[yy,xx],gy[yy,xx]),.05)
        nx,ny=gx[yy,xx]/norm,gy[yy,xx]/norm
        v=[a[yy,xx] for a in common]+[s[i,yy,xx] for i in range(4)]
        # Confidence scales vary between segmentation models; include signed geometry as a separate input.
        v+=[raw[i,yy,xx].clip(-12,12) for i in range(4)]
        v.extend([s[rank,yy,xx],raw[rank,yy,xx].clip(-12,12),votes[yy,xx],nx,ny,(lx[yy,xx]*nx+ly[yy,xx]*ny)/span])
        if len(yy):
            off=np.array([-5.,-3.,-1.5,0.,1.5,3.,5.],np.float32)
            line=cv2.remap(im,(xx[:,None]+nx[:,None]*off).astype("float32"),(yy[:,None]+ny[:,None]*off).astype("float32"),
                           cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            v.extend(((line[:,j]-bgmed)/span for j in range(7)))
            v.extend(((line[:,j+1]-line[:,j])/span for j in range(6)))
        else:v.extend([np.empty(0)]*13)
        scalars=[rank,radii[rank],radii.mean(),radii[-1]-radii[0],fgmed,bgmed,span]
        v.extend(np.full(len(xx),k) for k in scalars)
        rows.append(np.clip(np.stack(v,axis=1),-1000,1000).astype(np.float32))
        pixels.append((yy,xx))
    return rows,pixels


def predict(model,im,raw):
    rows,pixels=samples(im,raw)
    p=(raw>0).astype(np.float32)
    for r,(yy,xx) in enumerate(pixels):
        if len(yy):p[r,yy,xx]=model.predict_proba(rows[r])[:,1]
    return p


def run(args):
    f=args.fold; root=args.crossfit
    ims,labels,_,uid=native.inputs(args.data);_,train,valid=native.splits(uid,f)
    group=type("Group",(),{"series_uid":uid})()
    lib,check=polar.calibration_split(group,valid,f)
    of=root/f"oof-{f}.npz"; meta=json.loads(of.with_suffix(".json").read_text())
    if rc.hash_file(of)!=meta["payload"]:raise ValueError("OOF identity")
    with np.load(of,allow_pickle=False) as z:
        ids=z["ids"].tolist(); oof=z["logits"].astype(np.float32)
    if ids!=sorted(train):raise ValueError("OOF includes ineligible patients")
    val=args.texture/f"features-{f}-inner.npz"
    with np.load(val,allow_pickle=False) as z:
        if z["ids"].tolist()!=lib+check:raise ValueError("validation identity")
        original=np.unpackbits(z["old"],axis=-1).astype(bool)
        raw=tex.anchor_logits(z["base"].astype(np.float32),original)
    shift=json.loads((args.align/f"fold-{f}-inner-registered/fit.json").read_text())["applied_shift"]
    im=np.stack([area.register_image(x,shift) for x in ims])
    y=labels.astype(bool);rng=np.random.default_rng(117327+f)
    xs=[];ys=[];sw=[]
    for id_list,guide,importance in ((ids,oof,1.),(lib,raw[:len(lib)],2.)):
        names=[str(uid[i]).split("_")[0] for i in id_list]
        _,inv,nn=np.unique(names,return_inverse=True,return_counts=True)
        patient_weights=.55+.45*len(id_list)/(len(nn)*nn[inv])
        for j,idx in enumerate(id_list):
            rows,pixels=samples(im[idx],guide[j],128,rng)
            for rank,(yy,xx) in enumerate(pixels):
                if not len(yy):continue
                xs.append(rows[rank]);ys.append(y[idx,rank,yy,xx]);sw.append(np.full(len(yy),importance*patient_weights[j]))
    model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=23,min_samples_leaf=50,l2_regularization=4,
                                         learning_rate=.085,max_bins=127,early_stopping=False,random_state=117327+f)
    model.fit(np.concatenate(xs),np.concatenate(ys),sample_weight=np.concatenate(sw))
    output=args.run/f"fit-{f}";output.mkdir(exist_ok=False)
    joblib.dump(model,output/"model.joblib",compress=3)
    del xs,ys,sw,oof
    all_ids=lib+check;n=len(lib)
    probs=np.stack([predict(model,im[i],z) for i,z in zip(all_ids,raw)])
    post=logit(probs.clip(.0001,.9999))
    new_radius=np.sqrt(probs.sum((2,3))/np.pi)
    target=np.sqrt(y[all_ids].sum((2,3))/np.pi)
    offset=.5*np.median(target[:n]-new_radius[:n],axis=0)
    def decision(indices,w,t):
        z=(1-w)*raw[indices]+w*post[indices]
        return np.stack([area.apply_radii(a,m,p,t,offset) for a,m,p in zip(z,z>0,new_radius[indices])])
    grid=[]
    for w in WEIGHTS:
        for t in (0.,.4,.8):
            seg=decision(np.arange(n),w,t)
            grid.append({"w":w,"t":t,"dice":float(ns.dice(y[lib],seg).mean())})
    best=max(grid,key=lambda row:row["dice"])
    selected=decision(np.arange(n,len(all_ids)),best["w"],best["t"])
    summary={"rank_dice":ns.dice(y[check],selected).mean(0).tolist(),"candidate":float(ns.dice(y[check],selected).mean()),
             "reference":float(ns.dice(y[check],original[n:]).mean()),"raw":float(ns.dice(y[check],probs[n:]>0.5).mean())}
    rc.write_json(args.run/f"inner-{f}.json",{"fold":f,"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),
                  "oof_payload":meta["payload"],"model":rc.hash_file(output/"model.joblib"),"fit":ids,"calibration":lib,"check":check,
                  "best":best,"grid":grid,"scores":summary})
    print("EDGE_CHECK",f,best,summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("run","data","texture","crossfit","align"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--fold",type=int,required=True,choices=range(4))
    a=p.parse_args();cv2.setNumThreads(1);run(a)
