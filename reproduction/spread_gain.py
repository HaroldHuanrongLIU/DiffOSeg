"""Predict case-specific Dice utility for changes to the spatial rank spread."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge

from reproduction import component_utility as utility
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import nested_guides as nested
from reproduction import panel_spread as panel
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/spread_gain_protocol.md"
MODES=("mean","shape","ct")


def design():
    t=np.log(panel.SCALES)
    return np.column_stack((t,-t*t))


def action(coefficients):
    bc=np.array(coefficients,copy=True)
    bc[...,1]=np.maximum(bc[...,1],0)
    gain=bc@design().T
    idx=np.argmax(gain,axis=-1)
    return np.where(gain.max(axis=-1)>=.05,idx,0)


def features(image,masks):
    base=masks[0]
    radii=np.sqrt(base.sum((1,2))/np.pi)
    perimeters=np.array([panel.mean_perimeter(x[None].repeat(4,axis=0)) for x in base])
    denom=panel.mean_perimeter(base)
    widths=np.array([panel.spread(m,denom) for m in masks])
    pair=np.array([(base[i]^base[j]).sum()/denom for i in range(4) for j in range(i+1,4)])
    geometry=np.r_[radii,perimeters,pair,widths-widths[0]]
    ct=utility.normalized_ct(image,base)
    appearance=[]
    for m in base:
        distance=sdf(m)
        for lower,upper in ((-4,-1),(-1,0),(0,1),(1,4)):
            loc=(distance>=lower)&(distance<upper)
            appearance.extend(ct[:,loc].mean(1) if loc.any() else np.zeros(4))
    return geometry,np.r_[geometry,appearance]


def regress(x,values,ti,hi,w):
    mean=np.average(x[ti],weights=w,axis=0)
    sd=np.sqrt(np.average((x[ti]-mean)**2,weights=w,axis=0));sd=np.where(sd>1e-6,sd,1.)
    model=Ridge(alpha=30.,solver="svd").fit((x[ti]-mean)/sd,values[ti],sample_weight=w)
    return model.predict((x[hi]-mean)/sd).clip(-8,8)


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,q,old=nested.base_bank(args,uids,args.fold)
    bank=np.empty((len(ids),len(panel.SCALES),4,128,16),np.uint8)
    row={i:j for j,i in enumerate(ids)}
    names=np.array([str(uids[i]).split("_")[0] for i in ids]);shift={}
    for j,g in enumerate(groups):
        fit=json.loads((args.original/f"fold-{args.fold}-sub-{j}/fit.json").read_text())
        shift.update({i:fit["shift"] for i in g})
    shape=[];ct=[];curves=[]
    for j,i in enumerate(ids):
        m=panel.actions(q[j]);bank[j]=np.packbits(m,axis=-1)
        x,z=features(register_image(images[i],shift[i]),m);shape.append(x);ct.append(z)
        dice=metrics.dice(truth[i][None].astype(bool),m).mean(1)
        curves.append(100*(dice-dice[0]))
    shape=np.stack(shape);ct=np.stack(ct);curves=np.stack(curves)
    coeff=np.linalg.lstsq(design(),curves.T,rcond=None)[0].T.clip(-8,8)
    result={name:np.full(len(ids),-1,int) for name in MODES};members=[]
    for g,group in enumerate(groups):
        for hold in harmonic.case_partition(group,uids,57121+37*g+args.fold):
            fit=sorted(set(group)-set(hold))
            if nested.patients(hold,uids)&nested.patients(fit,uids):raise ValueError("Patient crossover")
            ti=np.array([row[i] for i in fit]);hi=np.array([row[i] for i in hold])
            _,inv,num=np.unique(names[ti],return_inverse=True,return_counts=True)
            w=.5+.5*len(ti)/(len(num)*num[inv])
            result["mean"][hi]=action(np.average(coeff[ti],weights=w,axis=0))
            for name,x in (("shape",shape),("ct",ct)):
                result[name][hi]=action(regress(x,coeff,ti,hi,w))
            members.append({"upstream":g,"fit":fit,"held":hold})
    result["truth_curve"]=action(coeff);result["truth_best"]=np.argmax(curves,axis=1)
    arrays={"reference":bank[:,0]}
    for name,index in result.items():
        if np.any(index<0):raise ValueError("Missing patient prediction")
        arrays[name]=bank[np.arange(len(ids)),index];arrays["index_"+name]=index.astype(np.uint8)
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":members,"base":old["payload"],
        "teachers":old["teachers"],"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("SPREAD_GAIN_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("data","run","original"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",choices=(2,3),type=int,required=True)
    cv2.setNumThreads(1);lock(p.parse_args())
