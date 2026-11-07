"""Test within-slice tissue prototypes on strictly patient-excluded teacher disagreements."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from reproduction import component_utility as blocks
from reproduction import gated_hierarchy as cache_owner
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image,sdf
from reproduction.shell_occupancy import unpack_level

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/slice_prototypes_protocol.md"
MODES=("population","slice")


def seed_stats(descriptors,guide,second,reference):
    counts=np.zeros((4,2),int)
    moments=np.zeros((4,2,2,len(descriptors)),np.float64)
    for rank in range(4):
        d=sdf(reference[rank])
        foreground=(guide[rank]>0)&(second[rank]>0)&(d>=1)
        background=(guide[rank]<=0)&(second[rank]<=0)&(d<=-1)&(d>=-7)
        for label,mask in enumerate((foreground,background)):
            counts[rank,label]=int(mask.sum())
            if counts[rank,label]:
                pixels=descriptors[:,mask]
                moments[rank,label,0]=pixels.mean(1,dtype=np.float64)
                moments[rank,label,1]=np.square(pixels).mean(1,dtype=np.float64)
    return counts,moments


def adaptive_moments(population,local,count):
    factor=count[...,None,None]/(count[...,None,None]+16.)
    return population+factor*(local-population)


def distances(first,second,model):
    mean=model[:,0]
    var=np.maximum(model[:,1]-np.square(mean),.035)
    squared=np.maximum(second[:,None]-2*first[:,None]*mean+np.square(mean),0.)/var
    values=[]
    for channels in (slice(0,4),slice(4,None)):
        d=squared[:,:,channels].mean(2)
        likelihood=(squared[:,:,channels]+np.log(var[:,channels])).mean(2)
        values.extend([d[:,0],d[:,1],likelihood[:,1]-likelihood[:,0]])
    return np.column_stack(values).clip(-60,60)


@dataclass
class RankBlocks:
    rank:int
    components:object
    mean:np.ndarray
    second:np.ndarray
    utility:np.ndarray
    weight:np.ndarray


@dataclass
class Example:
    seed_count:np.ndarray
    seed_moments:np.ndarray
    reference:np.ndarray
    blocks:list


def example(image,u,s,packed,truth):
    mask=(u+s)>0
    ct=blocks.normalized_ct(image,mask)
    high=unpack_level(packed[512:],4)
    normalized=high.astype(np.float32)
    normalized/=np.sqrt(np.square(normalized).mean(0)+1e-4)
    desc=np.concatenate((ct,normalized))
    count,moments=seed_stats(desc,u,s,mask)
    parts=[]
    for rank in range(4):
        comp=blocks.components(u,s,mask,ct,high,rank,False)
        if comp is None:continue
        point=desc.reshape(len(desc),-1)[:,comp.points].T
        first=blocks.aggregate(point,comp.group,comp.count)
        second=blocks.aggregate(point**2,comp.group,comp.count)
        target=blocks.flip_utility(comp,mask[rank],truth[rank]) if truth is not None else None
        parts.append(RankBlocks(rank,comp,first,second,target,comp.count/(mask[rank].sum()+1)))
    return Example(count,moments,np.packbits(mask,axis=-1),parts)


def pooled(train,names):
    _,inverse,n=np.unique(names,return_inverse=True,return_counts=True)
    weights=.5/len(train)+.5/(len(n)*n[inverse])
    source=np.stack([case.seed_moments for case in train])
    present=np.stack([case.seed_count for case in train])>0
    mass=weights[:,None,None]*present
    if (mass.sum(0)==0).any():raise ValueError("Missing training prototypes for a reader or tissue")
    return (source*mass[:,:,:,None,None]).sum(0)/mass.sum(0)[:,:,None,None],weights


def design(case,part,population,mode):
    proto=population if mode=="population" else adaptive_moments(population,case.seed_moments,case.seed_count)
    values=distances(part.mean,part.second,proto[part.rank])
    return np.column_stack((part.components.context,values)).astype(np.float32)


def fit_and_apply(train,held,names,seed):
    population,case_weights=pooled(train,names)
    result={}
    for mode in MODES:
        xs=[];ys=[];ws=[]
        for case,w in zip(train,case_weights,strict=True):
            for part in case.blocks:
                xs.append(design(case,part,population,mode));ys.append(part.utility);ws.append(w*part.weight)
        weights=np.concatenate(ws)
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=12,min_samples_leaf=40,
            learning_rate=.06,l2_regularization=8.,early_stopping=False,random_state=seed)
        model.fit(np.concatenate(xs),np.concatenate(ys),sample_weight=weights/weights.mean())
        result[mode]=[]
        design_parts=[design(case,part,population,mode) for case in held for part in case.blocks]
        predictions=model.predict(np.concatenate(design_parts)) if design_parts else np.empty(0)
        cursor=0
        for case in held:
            final=np.unpackbits(case.reference,axis=-1).astype(bool)
            for part in case.blocks:
                n=len(part.components.count)
                positive=predictions[cursor:cursor+n]>.05
                change=positive[part.components.group]
                final[part.rank].ravel()[part.components.points[change]]^=True
                cursor+=n
            result[mode].append(np.packbits(final,axis=-1))
        if cursor!=len(predictions):raise ValueError("Unassigned disagreement blocks")
    return result


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,base=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz"
    rec=json.loads(path.with_suffix(".json").read_text())
    if (rc.hash_file(args.nested/"independent_quality.json")!=blocks.QUALITY or
            rec["payload"]!=rc.hash_file(path) or rec["base"]!=base["payload"]):
        raise ValueError("Wrong nested OOF pair")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Different fitting cases")
        s=z["logits"].astype(np.float32)
    cache=cache_owner.verified_parent(args)
    row={i:j for j,i in enumerate(ids)}
    registrations={}
    for j,group in enumerate(groups):
        shift=json.loads((args.original/f"fold-{args.fold}-sub-{j}/fit.json").read_text())["shift"]
        if shift!=[-1,-1]:raise ValueError("The public single-slice feature registration differs")
        registrations.update({i:shift for i in group})
    all_cases=[example(register_image(images[i],registrations[i]),u[j],s[j],cache.features[i,0],truth[i])
               for j,i in enumerate(ids)]
    masks={"reference":np.stack([c.reference for c in all_cases])}
    for mode in MODES:masks[mode]=np.empty_like(masks["reference"])
    membership=[]
    for j,group in enumerate(groups):
        for hold in harmonic.case_partition(group,uids,57121+37*j+args.fold):
            use=sorted(set(group)-set(hold))
            if nested.patients(use,uids)&nested.patients(hold,uids):raise ValueError("Meta-fit patient overlap")
            fitted=[all_cases[row[i]] for i in use];held=[all_cases[row[i]] for i in hold]
            names=[str(uids[i]).split("_")[0] for i in use]
            results=fit_and_apply(fitted,held,names,93181+j)
            for mode in MODES:masks[mode][[row[i] for i in hold]]=results[mode]
            membership.append({"teacher_group":j,"fit":use,"held":hold})
    if sorted(i for r in membership for i in r["held"])!=ids:raise ValueError("Incomplete patient-excluded predictions")
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":membership,"base":base["payload"],
        "nested":rec["payload"],"features":cache.receipt,"prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),
        "protocol":rc.hash_file(PROTOCOL)})
    print("SLICE_PROTOTYPES_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","original","nested","parent","public"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    p.set_defaults(backbone="sam2.1-tiny",image_size=512)
    cv2.setNumThreads(1);lock(p.parse_args())
