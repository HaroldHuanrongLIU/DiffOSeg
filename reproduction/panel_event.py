"""Predict any panel error first, then the affected ranks on unanimous boundaries."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import HistGradientBoostingClassifier

from reproduction import component_utility as components
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/panel_event_protocol.md"


def unanimous_band(mask):
    votes=mask.sum(0)
    distance=np.zeros(votes.shape,np.float32)
    for side in (votes==0,votes==4):
        if side.any() and not side.all():
            distance[side]=distance_transform_edt(side)[side]-.5
    return (distance>0)&(distance<=2.5),distance


@dataclass
class Example:
    original:np.ndarray
    points:np.ndarray
    shared:np.ndarray
    rank:np.ndarray
    errors:np.ndarray
    weights:np.ndarray
    take:np.ndarray


def example(image,u,s,truth,index):
    q=.5*(u+s);ref=q>0
    band,distance=unanimous_band(ref);y,x=np.where(band)
    ct=components.normalized_ct(image,ref)
    radii=np.sqrt(ref.sum((1,2))/np.pi)
    cy,cx=np.argwhere(ref.sum(0)>=2).mean(0) if ref.any() and (ref.sum(0)>=2).any() else (63.5,63.5)
    scale=max(1.,float(radii.mean()))
    shared=np.column_stack((ref[0,y,x],distance[y,x],(x-cx)/scale,(y-cy)/scale,
                            np.broadcast_to(radii/16,(len(y),4)),u[:,y,x].T.clip(-12,12)/8,
                            s[:,y,x].T.clip(-12,12)/8,ct[:,y,x].T)).astype(np.float32)
    features=[]
    for r in range(4):
        features.append(np.column_stack((shared,np.full(len(y),r/3),q[r,y,x].clip(-12,12)/8,
                                         np.full(len(y),radii[r]/16))).astype(np.float32))
    errors=(truth[:,y,x].astype(bool)!=ref[:,y,x]).T
    w=1/(ref.sum((1,2))+1)
    return Example(np.packbits(ref,axis=-1),np.flatnonzero(band),shared,np.stack(features,axis=1),errors,w,(x+y+index)%2==0)


def flip(case,posterior):
    out=np.unpackbits(case.original,axis=-1).astype(bool)
    orig=out.reshape(4,-1)[:,case.points].T
    change=posterior>np.where(orig,.60,.50)
    for rank in range(4):
        out[rank].ravel()[case.points[change[:,rank]]]^=True
    return np.packbits(out,axis=-1)


def classifier(x,y,w,seed):
    model=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=9,min_samples_leaf=100,l2_regularization=10.,
          learning_rate=.07,early_stopping=False,random_state=seed)
    return model.fit(x,y,sample_weight=w/w.mean())


def fit_and_predict(train,held,names,seed):
    _,inv,n=np.unique(names,return_inverse=True,return_counts=True)
    weights=.5+.5*len(train)/(len(n)*n[inv])
    x=np.concatenate([case.shared[case.take] for case in train])
    xr=np.concatenate([case.rank[case.take] for case in train])
    y=np.concatenate([case.errors[case.take] for case in train]);event=y.any(1)
    w=np.concatenate([np.broadcast_to(wi*case.weights,(case.take.sum(),4)) for wi,case in zip(weights,train,strict=True)])
    gate=classifier(x,event,w.mean(1),seed)
    direct=classifier(xr.reshape(-1,xr.shape[-1]),y.ravel(),w.ravel(),seed)
    conditional=classifier(xr[event].reshape(-1,xr.shape[-1]),y[event].ravel(),w[event].ravel(),seed)
    result={key:[] for key in ("direct","joint","true_event")}
    probs=[]
    for case in held:
        if not len(case.points):
            for key in result:result[key].append(case.original.copy())
            probs.append(np.empty((0,1)))
            continue
        shared=gate.predict_proba(case.shared)[:,1,None]
        flat=case.rank.reshape(-1,case.rank.shape[-1])
        given=conditional.predict_proba(flat)[:,1].reshape(-1,4)
        result["direct"].append(flip(case,direct.predict_proba(flat)[:,1].reshape(-1,4)))
        result["joint"].append(flip(case,shared*given))
        result["true_event"].append(flip(case,case.errors.any(1)[:,None]*given))
        probs.append(shared)
    return result,probs


def lock(args):
    images,truth,_,uids=native.inputs(args.data);_,_,groups=nested.split_groups(uids,args.fold)
    ids,u,old=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz";rec=nested.read(path.with_suffix(".json"))
    if rc.hash_file(args.nested/"independent_quality.json")!=components.QUALITY or old["payload"]!=rec["base"] or rc.hash_file(path)!=rec["payload"]:
        raise ValueError("Unexpected nested teacher provenance")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong case order")
        s=z["logits"].astype(np.float32)
    row={i:j for j,i in enumerate(ids)};cases={}
    for g,group in enumerate(groups):
        shift=nested.read(args.original/f"fold-{args.fold}-sub-{g}/fit.json")["shift"]
        for i in group:
            j=row[i];cases[i]=example(register_image(images[i],shift),u[j],s[j],truth[i],i)
    ref=np.stack([cases[i].original for i in ids]);out={k:ref.copy() for k in ("direct","joint","true_event")}
    memberships=[];pieces={}
    for g,group in enumerate(groups):
        for hold in harmonic.case_partition(group,uids,57121+37*g+args.fold):
            fit=sorted(set(group)-set(hold))
            if nested.patients(fit,uids)&nested.patients(hold,uids):raise ValueError("Patient crossed subfold")
            values,probs=fit_and_predict([cases[i] for i in fit],[cases[i] for i in hold],[str(uids[i]).split("_")[0] for i in fit],842967+g)
            for k in out:out[k][[row[i] for i in hold]]=values[k]
            pieces.update(zip(hold,probs,strict=True))
            memberships.append({"upstream":g,"fit":fit,"held":hold})
    if sorted(pieces)!=ids:raise ValueError("Incomplete patient cover")
    path=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(path,ids=ids,reference=ref,event=np.concatenate([pieces[i] for i in ids]).ravel(),**out)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":memberships,"base":old["payload"],"nested":rec["payload"],
         "prediction":rc.hash_file(path),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("PANEL_EVENT_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("run","data","original","nested"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    cv2.setNumThreads(1);lock(p.parse_args())
