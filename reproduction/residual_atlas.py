"""Patient-excluded low-rank residual atlas in nodule-relative coordinates."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge

from reproduction import component_utility as utility
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf
from reproduction.reader_error_transfer import anatomy,rotate

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/residual_atlas_protocol.md"
MODES=("mean","shape","ct","truth_projection")


def atlas_maps(mask):
    center,radius,angle=anatomy(mask)
    rotation=rotate(angle)
    axis=np.linspace(-2,2,33)
    yy,xx=np.meshgrid(axis,axis,indexing="ij")
    forward=np.stack((xx,yy),-1)@rotation.T*radius+center
    yy,xx=np.indices(mask.shape[-2:])
    inverse=((np.stack((xx,yy),-1)-center)@rotation/radius+2)*8
    return forward.astype(np.float32),inverse.astype(np.float32)


def sample(fields,grid):
    return np.stack([cv2.remap(v.astype(np.float32),grid[:,:,0],grid[:,:,1],cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE) for v in fields])


def down(fields):
    return np.stack([cv2.resize(v,(8,8),interpolation=cv2.INTER_AREA) for v in fields])


@dataclass
class Example:
    geom:np.ndarray
    ct:np.ndarray
    radii:np.ndarray
    target:np.ndarray
    field:np.ndarray
    support:np.ndarray
    inverse:np.ndarray


def example(image,u,s,truth):
    q=.5*(u+s);reference=q>0
    forward,inverse=atlas_maps(reference)
    sd=np.stack([sdf(m) for m in reference])
    truth_sd=np.stack([sdf(m) for m in truth])
    radii=np.sqrt(np.stack([(z>0).sum((1,2)) for z in (u,s,q)])/np.pi)
    delta=np.stack([sdf(v)-sdf(w) for v,w in zip(s>0,u>0)])
    g=down(sample(np.concatenate((sd.clip(-5,5),delta.clip(-3,3))),forward))
    ct=down(sample(utility.normalized_ct(image,reference)[[1,3]],forward))
    target=sample((truth_sd-sd).clip(-3,3),forward)
    inside=((inverse>=0)&(inverse<=32)).all(2)
    return Example(g,ct,radii.ravel(),target,harmonic.continuous_field(q),(abs(sd)<=3)&inside,inverse)


def both(fields):
    return np.stack((fields,np.flip(fields,(-2,-1)))).reshape(2,-1)


def blocks(case):
    return np.stack((case.radii,case.radii)),both(case.geom),both(case.ct)


def fit_basis(data,w):
    mean=np.average(data,axis=0,weights=w)
    centered=data-mean
    scaled=centered*np.sqrt(w[:,None])
    values,vectors=np.linalg.eigh(scaled@scaled.T)
    keep=np.arange(len(values)-1,max(-1,len(values)-7),-1)
    keep=keep[values[keep]>1e-8]
    basis=(vectors[:,keep].T@scaled)/np.sqrt(values[keep,None])
    return mean,basis


def fit_predict(train,held,names):
    _,inv,counts=np.unique(names,return_inverse=True,return_counts=True)
    w=np.repeat(.5+.5*len(train)/(len(counts)*counts[inv]),2)
    train_y=np.concatenate([both(x.target) for x in train]).astype(np.float64)
    mean,basis=fit_basis(train_y,w)
    coefficients=(train_y-mean)@basis.T
    xtrain=[];xtest=[]
    for xs,zs in zip(zip(*(blocks(c) for c in train)),zip(*(blocks(c) for c in held))):
        x=np.concatenate(xs).astype(np.float64);z=np.concatenate(zs).astype(np.float64)
        mu=np.average(x,axis=0,weights=w);sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=w))
        scale=np.maximum(sd,.01)*np.sqrt(x.shape[1])
        xtrain.append((x-mu)/scale);xtest.append((z-mu)/scale)
    estimates={"mean":np.broadcast_to(mean,(2*len(held),len(mean)))}
    for mode,n in (("shape",2),("ct",3)):
        reg=Ridge(alpha=10.,solver="cholesky").fit(np.column_stack(xtrain[:n]),coefficients,sample_weight=w)
        estimates[mode]=mean+reg.predict(np.column_stack(xtest[:n]))@basis
    # Evaluate the representable error separately, without using this quantity for prediction.
    actual=np.concatenate([both(x.target) for x in held])
    estimates["truth_projection"]=mean+(actual-mean)@basis.T@basis
    result={}
    for mode,raw in estimates.items():
        paired=raw.reshape(len(held),2,4,33,33)
        canonical=.5*(paired[:,0]+np.flip(paired[:,1],(-2,-1)))
        masks=[]
        for c,patch in zip(held,canonical,strict=True):
            shift=.5*sample(patch,c.inverse).clip(-2,2)*c.support
            masks.append(np.packbits(c.field+shift>0,axis=-1))
        result[mode]=masks
    return result,len(basis)


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,old=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz"
    rec=json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=utility.QUALITY or
            old["payload"]!=rec["base"] or rc.hash_file(path)!=rec["payload"]):raise ValueError("Invalid nested teacher identity")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("OOF order mismatch")
        s=z["logits"].astype(np.float32)
    row={i:j for j,i in enumerate(ids)}
    cases={}
    for g,group in enumerate(groups):
        fit=json.loads((args.original/f"fold-{args.fold}-sub-{g}/fit.json").read_text())
        for i in group:
            j=row[i];cases[i]=example(register_image(images[i],fit["shift"]),u[j],s[j],truth[i])
    ref=np.packbits((u+s)>0,axis=-1)
    masks={mode:ref.copy() for mode in MODES}
    assignments=[]
    for g,group in enumerate(groups):
        for hold in harmonic.case_partition(group,uids,57121+37*g+args.fold):
            use=sorted(set(group)-set(hold))
            if nested.patients(use,uids)&nested.patients(hold,uids):raise ValueError("Training patient overlap")
            out,rank=fit_predict([cases[i] for i in use],[cases[i] for i in hold],[str(uids[i]).split("_")[0] for i in use])
            for mode in MODES:masks[mode][[row[i] for i in hold]]=out[mode]
            assignments.append({"upstream":g,"fit":use,"held":hold,"basis_rank":rank})
    if sorted(i for g in assignments for i in g["held"])!=ids:raise ValueError("Incomplete patient coverage")
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,reference=ref,**masks)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":assignments,"base":old["payload"],
      "nested":rec["payload"],"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("ATLAS_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","original","nested"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    cv2.setNumThreads(1);lock(p.parse_args())
