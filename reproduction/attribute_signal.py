"""Training-only, patient-excluded test of privileged LIDC morphology ratings."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge

from reproduction import component_utility as components
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/attribute_signal_protocol.md"
RATINGS=("texture","margin","sphericity","lobulation","spiculation","subtlety","malignancy")
MODES=("reference","image","privileged")


def attributes(entries):
    entries=entries or []
    output=[]
    for key in RATINGS:
        vals=[a.get(key) for a in entries if isinstance(a,dict)]
        ratings=[(v-1)/4 for v in vals if isinstance(v,(int,float,np.number)) and 1<=v<=5]
        output.extend((np.mean(ratings) if ratings else .5,np.std(ratings) if ratings else 0.,
                       len(ratings)/max(len(entries),1)))
    return np.asarray(output,dtype=np.float64)


def shape_features(image,u,s,q):
    masks=[z>0 for z in (u,s,q)]
    radii=np.stack([np.sqrt(m.sum((1,2))/np.pi) for m in masks])
    portions=(masks[0]!=masks[1]).sum((1,2))/np.maximum(masks[2].sum((1,2)),1)
    ct=components.normalized_ct(image,masks[2])
    local=[]
    for m in masks[2]:
        d=sdf(m)
        for lo,hi in ((-4,-1),(-1,0),(0,1),(1,4)):
            roi=(d>=lo)&(d<hi)
            local.extend(ct[:,roi].mean(1) if roi.any() else np.zeros(4))
    return np.r_[radii.ravel(),radii[1]-radii[0],portions,local]


def regress(inputs,target,train,held,names):
    _,inverse,sizes=np.unique(names[train],return_inverse=True,return_counts=True)
    weight=.5+.5*len(train)/(len(sizes)*sizes[inverse])
    x=inputs[train]
    mean=np.average(x,weights=weight,axis=0)
    scale=np.sqrt(np.average((x-mean)**2,weights=weight,axis=0))
    scale=np.where(scale>1e-6,scale,1.)
    estimator=Ridge(alpha=30.,solver="svd").fit((x-mean)/scale,target[train],sample_weight=weight)
    return estimator.predict((inputs[held]-mean)/scale).clip(-3,3)


def adjust(q,offset):
    mask=q>0
    radius=np.sqrt(mask.sum((1,2))/np.pi)
    count=np.rint(np.pi*np.maximum(radius+.5*offset,0)**2).astype(int).clip(0,q[0].size)
    result=np.zeros_like(q,dtype=bool).reshape(4,-1)
    for j in range(4):
        order=np.argsort(-q[j].ravel(),kind="stable")
        result[j,order[:count[j]]]=True
    return result.reshape(q.shape)


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,base=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz"
    meta=json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=components.QUALITY or
            meta["payload"]!=rc.hash_file(path) or meta["base"]!=base["payload"]):
        raise ValueError("Nested OOF provenance")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:
            raise ValueError("Wrong fitting patient order")
        s=z["logits"].astype(np.float32)
    with args.data.open("rb") as st:
        rows=list(pickle.load(st).values())
    names=np.array([str(uids[i]).split("_")[0] for i in ids])
    q=.5*(u+s)
    ratings=np.stack([attributes(rows[i].get("annotations")) for i in ids])
    row={i:j for j,i in enumerate(ids)}
    shift={}
    for g,subset in enumerate(groups):
        record=json.loads((args.original/f"fold-{args.fold}-sub-{g}/fit.json").read_text())
        shift.update({i:record["shift"] for i in subset})
    image=np.stack([shape_features(register_image(images[i],shift[i]),u[j],s[j],q[j]) for j,i in enumerate(ids)])
    target=(np.sqrt(truth[ids].sum((2,3))/np.pi)-np.sqrt((q>0).sum((2,3))/np.pi)).clip(-3,3)
    inputs={"image":image,"privileged":np.column_stack((image,ratings))}
    predictions={name:np.full(target.shape,np.nan) for name in inputs}
    receipt=[]
    for g,subset in enumerate(groups):
        for j,held in enumerate(harmonic.case_partition(subset,uids,57121+37*g+args.fold)):
            train=sorted(set(subset)-set(held))
            fit_idx=np.array([row[i] for i in train])
            held_idx=np.array([row[i] for i in held])
            if nested.patients(train,uids)&nested.patients(held,uids):
                raise ValueError("Patient crosses regression holdout")
            for name,x in inputs.items():
                predictions[name][held_idx]=regress(x,target,fit_idx,held_idx,names)
            receipt.append({"upstream_group":g,"held":held,"fit":train})
    arrays={"reference":np.packbits(q>0,axis=-1)}
    for name,offset in predictions.items():
        if not np.isfinite(offset).all():
            raise ValueError("Incomplete patient exclusion")
        masks=np.stack([adjust(logits,dx) for logits,dx in zip(q,offset,strict=True)])
        arrays[name]=np.packbits(masks,axis=-1)
    dst=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":receipt,"base":base["payload"],
          "nested":meta["payload"],"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),
          "protocol":rc.hash_file(PROTOCOL),"cases_with_any":int((ratings[:,2::3]>0).any(1).sum()),
          "cases_with_four_scores":sum(len(rows[i].get("annotations") or [])==4 for i in ids)})
    print("ATTRIBUTE_SIGNAL_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","original","nested"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    cv2.setNumThreads(1)
    lock(p.parse_args())
