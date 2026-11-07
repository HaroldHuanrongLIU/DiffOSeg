"""Calibrate the within-panel contour spread predicted from a single nodule CT."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from reproduction import attribute_signal as area
from reproduction import component_utility as utility
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/panel_spread_protocol.md"
SCALES=(1.,.875,1.125,.75,1.25,.5,1.5,2.)
DEPLOYABLE=("mean","geometry","ct")


def mean_perimeter(masks):
    padded=np.pad(masks,((0,0),(1,1),(1,1)))
    horizontal=np.count_nonzero(padded[:,:,1:]!=padded[:,:,:-1])
    vertical=np.count_nonzero(padded[:,1:]!=padded[:,:-1])
    return max((horizontal+vertical)/4,1.)


def spread(masks,denominator):
    votes=masks.sum(0)
    return float((votes*(4-votes)).sum())/(6*denominator)


def actions(q):
    masks=q>0
    field=harmonic.continuous_field(q)
    deviation=field-field.mean(0,keepdims=True)
    band=np.stack([abs(sdf(m))<=3 for m in masks])
    result=[]
    for scale in SCALES:
        shift=np.clip((scale-1)*deviation,-2,2)
        result.append(np.where(band,field+shift>0,masks))
    if not np.array_equal(result[0],masks):raise ValueError("Scale one changed the source panel")
    return np.stack(result)


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,old=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz";rec=json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=utility.QUALITY or rec["base"]!=old["payload"]
            or rc.hash_file(path)!=rec["payload"]):raise ValueError("Unverified nested teacher")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Teacher case order differs")
        s=z["logits"].astype(np.float32)
    shifts={}
    for g,group in enumerate(groups):
        fit=json.loads((args.original/f"fold-{args.fold}-sub-{g}/fit.json").read_text())
        shifts.update({i:fit["shift"] for i in group})
    row={i:j for j,i in enumerate(ids)};names=np.array([str(uids[i]).split("_")[0] for i in ids])
    bank=np.empty((len(ids),len(SCALES),4,128,16),np.uint8)
    spreads=np.empty((len(ids),len(SCALES)));targets=[];geom=[];ct=[];best=[]
    for j,i in enumerate(ids):
        q=.5*(u[j]+s[j]);m=actions(q)
        denom=mean_perimeter(m[0])
        spreads[j]=[spread(v,denom) for v in m]
        label=truth[i].astype(bool)
        targets.append(np.clip(spread(label,denom)-spreads[j,0],-3,3))
        best.append(np.argmax(metrics.dice(label[None],m).mean(1)))
        image=register_image(images[i],shifts[i])
        features=area.shape_features(image,u[j],s[j],q)
        extra=[spread(v>0,denom) for v in (u[j],s[j],q)]
        # First 20 entries of the existing descriptor are the radii and model disagreement.
        geom.append(np.r_[features[:20],extra]);ct.append(np.r_[features,extra])
        bank[j]=np.packbits(m,axis=-1)
    geom=np.stack(geom);ct=np.stack(ct);target=np.array(targets)
    prediction={name:np.full(len(ids),np.nan) for name in DEPLOYABLE}
    membership=[]
    for g,group in enumerate(groups):
        for hold in harmonic.case_partition(group,uids,57121+37*g+args.fold):
            use=sorted(set(group)-set(hold))
            if nested.patients(use,uids)&nested.patients(hold,uids):raise ValueError("Patient overlap")
            ti=np.array([row[i] for i in use]);hi=np.array([row[i] for i in hold])
            _,inv,num=np.unique(names[ti],return_inverse=True,return_counts=True)
            w=.5+.5*len(ti)/(len(num)*num[inv])
            prediction["mean"][hi]=np.average(target[ti],weights=w)
            for name,x in (("geometry",geom),("ct",ct)):
                prediction[name][hi]=area.regress(x,target,ti,hi,names)
            membership.append({"upstream":g,"fit":use,"held":hold})
    out={"reference":bank[:,0],"truth_best":bank[np.arange(len(ids)),best]}
    for name,offset in {**prediction,"truth_spread":target}.items():
        if not np.isfinite(offset).all():raise ValueError("Missing held offset")
        index=np.abs(spreads-(spreads[:,0]+.5*offset)[:,None]).argmin(1)
        out[name]=bank[np.arange(len(ids)),index];out["index_"+name]=index.astype(np.uint8)
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**out)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":membership,"base":old["payload"],
        "nested":rec["payload"],"prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("PANEL_SPREAD_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","original","nested"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    cv2.setNumThreads(1);lock(p.parse_args())
