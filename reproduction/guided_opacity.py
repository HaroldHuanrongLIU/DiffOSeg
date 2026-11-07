"""Probe a patient-excluded, contrast-only guided correction on native CT boundaries."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit

from reproduction import component_utility as utility
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/guided_opacity_protocol.md"
MODES=("contrast","guided","spatial")
WEIGHTS=(0.,.25,.5,1.)


def box(x):
    return cv2.boxFilter(x,-1,(9,9),normalize=True,borderType=cv2.BORDER_REFLECT_101)


def guided(image,p):
    weight=.1+.9*(2*p-1)**2
    norm=box(weight)
    mean_p=box(weight*p)/norm
    mean_x=box(weight*image)/norm
    var=np.maximum(box(weight*image*image)/norm-mean_x**2,0.)
    cov=box(weight*image*p)/norm-mean_x*mean_p
    slope=cov/(var+(12/255)**2)
    return box(slope)*image+box(mean_p-slope*mean_x),box(mean_p)


def proposals(image,q):
    ref=q>0
    p=expit(q.astype(np.float64))
    margin=.5*np.tanh(q.astype(np.float64)/2)
    guided_values,spatial=zip(*(guided(image.astype(np.float64)/255,v) for v in p))
    h=np.stack(guided_values);s=np.stack(spatial)
    band=np.stack([abs(sdf(m))<=3 for m in ref])
    return {mode:np.stack([np.packbits(np.where(band,margin+w*delta>0,ref),axis=-1) for w in WEIGHTS])
            for mode,delta in (("contrast",h-s),("guided",h-p),("spatial",s-p))}


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,old=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz";meta=json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=utility.QUALITY or
            rc.hash_file(path)!=meta["payload"] or meta["base"]!=old["payload"]):raise ValueError("Unverified nested teachers")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Incorrect fitting order")
        q=.5*(u+z["logits"].astype(np.float32))
    row={i:j for j,i in enumerate(ids)}
    source={m:np.empty((len(ids),len(WEIGHTS),4,128,16),np.uint8) for m in MODES}
    ref=np.packbits(q>0,axis=-1)
    for j,subset in enumerate(groups):
        folder=args.original/f"fold-{args.fold}-sub-{j}"
        shift=json.loads((folder/"fit.json").read_text())["shift"]
        for i in subset:
            k=row[i];masks=proposals(register_image(images[i],shift),q[k])
            for m in MODES:
                if not np.array_equal(masks[m][0],ref[k]):raise ValueError("Zero change must reproduce the reference")
                source[m][k]=masks[m]
    final={m:np.empty_like(ref) for m in MODES};assignments=[]
    for j,subset in enumerate(groups):
        for hold in harmonic.case_partition(subset,uids,57121+37*j+args.fold):
            use=sorted(set(subset)-set(hold))
            if nested.patients(use,uids)&nested.patients(hold,uids):raise ValueError("Calibration patient crossover")
            train=np.array([row[i] for i in use]);held=np.array([row[i] for i in hold])
            choices={}
            for m in MODES:
                grid=[float(metrics.dice(truth[use],np.unpackbits(source[m][train,w],axis=-1)).mean()) for w in range(len(WEIGHTS))]
                index=int(np.argmax(grid));final[m][held]=source[m][held,index]
                choices[m]={"strength":WEIGHTS[index],"grid":grid}
            assignments.append({"upstream":j,"fit":use,"held":hold,"choices":choices})
    if sorted(i for g in assignments for i in g["held"])!=ids:raise ValueError("Incomplete patient coverage")
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,reference=ref,**final)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":assignments,"base":old["payload"],
        "nested":meta["payload"],"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("GUIDED_OPACITY_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","original","nested"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",choices=(0,1),type=int,required=True)
    args=p.parse_args();cv2.setNumThreads(1);lock(args)
