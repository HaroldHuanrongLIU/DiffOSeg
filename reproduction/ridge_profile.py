"""Test outward intensity continuity with leave-patient-out residual calibration."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from sklearn.ensemble import HistGradientBoostingRegressor

from reproduction import component_utility as utility
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/ridge_profile_protocol.md"
MODES=("tangent","normal")


def directional_values(image,coords,unit):
    side=np.stack((-unit[1],unit[0]))
    def sample(pos):
        return map_coordinates(image,pos,order=1,mode="nearest",prefilter=False)
    columns=[]
    for distance in (2,4,7):
        pos=coords+distance*unit
        mid=sample(pos)
        columns.extend((mid,mid-.5*(sample(pos+2*side)+sample(pos-2*side)),sample(coords-distance*unit)))
    return np.column_stack(columns)


@dataclass
class Case:
    reference:np.ndarray
    index:np.ndarray
    margin:np.ndarray
    features:dict
    residual:np.ndarray
    weights:np.ndarray


def example(image,u,s,truth):
    q=.5*(u+s)
    reference=q>0
    ct=utility.normalized_ct(image,reference)
    coords=[];margins=[];ys=[];weights=[]
    xs={m:[] for m in MODES}
    for rank in range(4):
        d=sdf(reference[rank])
        iy,ix=np.nonzero(abs(d)<=2.5)
        if not len(iy):continue
        points=np.stack((iy,ix))
        normals=-np.stack(np.gradient(d))[:,iy,ix]
        normals/=np.maximum(np.linalg.norm(normals,axis=0),1.e-6)
        tangent=np.stack((-normals[1],normals[0]))
        left,right=u[rank,iy,ix],s[rank,iy,ix]
        margin=.5*np.tanh(q[rank,iy,ix]/2)
        area=int(reference[rank].sum())
        basic=np.column_stack((np.full(len(iy),rank/3),np.full(len(iy),np.sqrt(area/np.pi)/16),
            margin,d[iy,ix],left.clip(-12,12)/8,right.clip(-12,12)/8,(left>0)!=(right>0),
            reference[:,iy,ix].mean(0),ct[:,iy,ix].T))
        for mode,unit in (("normal",normals),("tangent",tangent)):
            xs[mode].append(np.column_stack((basic,directional_values(ct[1],points,unit))).astype(np.float32))
        coords.append(rank*d.size+iy*d.shape[1]+ix)
        margins.append(margin)
        ys.append(truth[rank,iy,ix]-(.5+margin))
        weights.append(np.full(len(iy),1./(area+1)))
    # Empty whole panels carry no eligible boundary pixels.
    return Case(np.packbits(reference,axis=-1),np.concatenate(coords) if coords else np.empty(0,int),
                np.concatenate(margins) if coords else np.empty(0),
                {m:np.concatenate(v) if v else np.empty((0,21),np.float32) for m,v in xs.items()},
                np.concatenate(ys) if coords else np.empty(0),np.concatenate(weights) if coords else np.empty(0))


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,old=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz"
    receipt=json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=utility.QUALITY or rc.hash_file(path)!=receipt["payload"]
            or receipt["base"]!=old["payload"]):
        raise ValueError("Nested patient exclusion audit differs")
    with np.load(path,allow_pickle=False) as z:
        if ids!=z["ids"].tolist():raise ValueError("Unexpected teacher order")
        s=z["logits"].astype(np.float32)
    row={i:j for j,i in enumerate(ids)}
    examples={}
    for j,subset in enumerate(groups):
        shift=json.loads((args.original/f"fold-{args.fold}-sub-{j}/fit.json").read_text())["shift"]
        for i in subset:
            k=row[i]
            examples[i]=example(register_image(images[i],shift),u[k],s[k],truth[i])
    ref=np.stack([examples[i].reference for i in ids])
    masks={m:ref.copy() for m in MODES}
    assignments=[]
    for j,subset in enumerate(groups):
        for hold in harmonic.case_partition(subset,uids,57121+37*j+args.fold):
            use=sorted(set(subset)-set(hold))
            if nested.patients(use,uids)&nested.patients(hold,uids):
                raise ValueError("Held patient in correction fit")
            _,inv,counts=np.unique([str(uids[i]).split("_")[0] for i in use],return_inverse=True,return_counts=True)
            case_weights=.5+.5*len(use)/(len(counts)*counts[inv])
            w=np.concatenate([examples[i].weights*wi for i,wi in zip(use,case_weights,strict=True)])
            target=np.concatenate([examples[i].residual for i in use])
            for mode in MODES:
                model=HistGradientBoostingRegressor(max_iter=70,max_leaf_nodes=9,min_samples_leaf=80,l2_regularization=10.,
                            learning_rate=.07,early_stopping=False,random_state=38931+j)
                x=np.concatenate([examples[i].features[mode] for i in use if len(examples[i].index)])
                model.fit(x,target,sample_weight=w/w.mean())
                xc=[examples[i].features[mode] for i in hold if len(examples[i].index)]
                predictions=model.predict(np.concatenate(xc)) if xc else np.empty(0)
                cursor=0
                for i in hold:
                    case=examples[i];n=len(case.index)
                    final=np.unpackbits(case.reference,axis=-1)
                    final.ravel()[case.index]=case.margin+.75*predictions[cursor:cursor+n]>0
                    cursor+=n
                    masks[mode][row[i]]=np.packbits(final,axis=-1)
                if cursor!=len(predictions):raise ValueError("Pixel assignment differs")
            assignments.append({"upstream":j,"fit":use,"held":hold})
    if sorted(i for a in assignments for i in a["held"])!=ids:raise ValueError("Incomplete patient exclusion")
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,reference=ref,**masks)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":assignments,"base":old["payload"],
        "nested":receipt["payload"],"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("RIDGE_PROFILE_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("data","run","original","nested"):p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    cv2.setNumThreads(1);lock(p.parse_args())
