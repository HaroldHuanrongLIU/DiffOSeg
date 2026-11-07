"""Select a guide/image mixing proportion separately for each predicted area rank."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import Ridge

from reproduction import component_utility as tissue
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/adaptive_pool_protocol.md"
CAPACITY="1db88676a76a01bf501723f6059989b289f66f6a3342f94607f58da336bdb72a"
WEIGHTS=(.25,0.,.125,.375,.5,.625,.75,.875,1.)
MODES=("fixed_rank","case","rank")


def features(image,guide,members,reference):
    # Continuous branch values only occur at disputed positions; elsewhere all signs equal E.
    g=guide>0;mean=members.mean(0,dtype=np.float32);m=mean>0
    ct=tissue.normalized_ct(image,reference)
    out=[]
    for k in range(4):
        left=g[k]&~m[k];right=m[k]&~g[k]
        disputed=(members[:,k]>0)!=g[k]
        region=disputed.any(0)
        size=reference[k].sum()+1
        geom=[np.sqrt(size/np.pi)/12,(g[k].sum()-m[k].sum())/size,
              left.sum()/size,right.sum()/size,np.mean(disputed.sum((1,2)))/size,
              np.std(disputed.sum((1,2)))/size]
        for values in (abs(guide[k]),abs(mean[k]),members[:,k].astype(np.float32).std(0)):
            geom.append(float(values[region].mean()) if region.any() else 0.)
        for support in (left,right):
            geom.extend(ct[:,support].mean(1) if support.any() else np.zeros(4))
        out.append(np.r_[np.eye(4)[k],geom])
    return np.stack(out)


def select(pred):
    # Reference weight is the first column, including for exact ties.
    best=np.argmax(pred,axis=-1)
    return np.where(pred.max(-1)>.05,best,0)


def fit_predict(x,delta,names):
    n=len(names);_,inv,counts=np.unique(names,return_inverse=True,return_counts=True)
    weights=np.repeat(.5+.5*n/(len(counts)*counts[inv]),4)
    train=x[:n].reshape(4*n,-1)
    mu=np.average(train,weights=weights,axis=0)
    sd=np.sqrt(np.average((train-mu)**2,weights=weights,axis=0))
    sd=np.where(sd>1e-6,sd,1.)
    curve=delta.clip(-5,5)
    model=Ridge(alpha=30.,solver="svd").fit((train-mu)/sd,curve.reshape(4*n,-1),sample_weight=weights)
    pred=model.predict((x.reshape(-1,x.shape[-1])-mu)/sd).reshape(len(x),4,len(WEIGHTS))
    control=np.average(delta,axis=0,weights=weights.reshape(n,4)[:,0])
    return {"rank":select(pred),"case":np.repeat(select(pred.mean(1))[:,None],4,axis=1),
            "fixed_rank":np.broadcast_to(select(control),(len(x),4))}


def lock(args):
    if rc.hash_file(args.screen/"library.json")!=CAPACITY or not nested.read(args.screen/"library.json")["qualifies"]:
        raise ValueError("Missing library headroom")
    if rc.hash_file(args.pool/"summary.json")!="d554afb52b0db929fa7aa8bcf04885da9e451d74db7984ceae4063d222474abf":
        raise ValueError("Invalid replay provenance")
    images,truth,_,uids=native.inputs(args.data)
    seal=nested.read(args.pool/f"seal-{args.fold}.json")
    path=args.pool/f"scores-{args.fold}.npz";ref=args.stable/f"pred-{args.fold}.npz"
    if seal["inputs"]!=rc.hash_file(path) or seal["reference"]!=rc.hash_file(ref):
        raise ValueError("Unexpected cached source")
    lib=seal["library"];check=seal["check"];ids=lib+check;n=len(lib)
    with np.load(path,allow_pickle=False) as z,np.load(ref,allow_pickle=False) as old:
        if ids!=z["ids"].tolist() or ids!=old["ids"].tolist():raise ValueError("Wrong patient order")
        reference=np.unpackbits(old["adapted"],axis=-1).astype(bool)
        loc=z["index"];guide=z["guide"];members=z["members"]
    votes=members.mean(0,dtype=np.float32)
    ref_sign=np.where(reference,1.,-1.).astype(np.float32)
    g=ref_sign.copy();g.ravel()[loc]=guide
    # Agreed pixels contribute only their signs to features; the numerical summaries use disputed pixels.
    predictions=np.broadcast_to(ref_sign,(3,*ref_sign.shape)).astype(np.float16).copy()
    for j,m in enumerate(members):predictions[j].ravel()[loc]=m
    x=np.stack([features(register_image(images[i],(-1,-1)),g[j],predictions[:,j],reference[j]) for j,i in enumerate(ids)])
    area=reference[:n].sum((2,3))+truth[lib].sum((2,3))
    hits=(reference[:n]&truth[lib]).sum((2,3))
    initial=np.divide(2*hits,area,out=np.ones(area.shape),where=area>0)
    curves=[];banks=[]
    for w in WEIGHTS:
        m=reference.copy();m.ravel()[loc]=(1-w)*guide+w*votes>0
        if w==.25 and not np.array_equal(m,reference):raise ValueError("Base must reproduce E")
        curves.append(100*(metrics.dice(truth[lib],m[:n])-initial))
        banks.append(np.packbits(m,axis=-1))
    delta=np.stack(curves,axis=-1)
    selected=fit_predict(x,delta,[str(uids[i]).split("_")[0] for i in lib])
    banks=np.stack(banks)
    arrays={"reference":np.packbits(reference,axis=-1)}
    for key,idx in selected.items():
        arrays[key]=banks[idx,np.arange(len(ids))[:,None],np.arange(4)]
        arrays["index_"+key]=idx.astype(np.uint8)
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**arrays)
    design=args.run/f"features-{args.fold}.npz";np.savez_compressed(design,ids=ids,x=x)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,
        "pool":seal["inputs"],"reference":seal["reference"],"features":rc.hash_file(design),"prediction":rc.hash_file(dst),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("ADAPTIVE_POOL_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","pool","stable","screen"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    cv2.setNumThreads(1);lock(p.parse_args())
