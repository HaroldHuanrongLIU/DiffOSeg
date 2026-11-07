"""Use independently trained patient subsets as witnesses only on disputed components."""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor

from reproduction import component_utility as utility
from reproduction import gated_hierarchy as cache_owner
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction.rater_area import register_image
from reproduction.shell_occupancy import unpack_level

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/subset_witness_protocol.md"
MODES=("base","mean","witness")


def witness_features(first,second,comp):
    a=first.ravel()[comp.points].astype(np.float32)
    b=second.ravel()[comp.points].astype(np.float32)
    direction=comp.initial[comp.group]
    opposite_a=(a>0)!=direction
    opposite_b=(b>0)!=direction
    mid=(a+b)/2
    mean=np.column_stack((mid.clip(-12,12)/8,expit(mid),((mid>0)!=direction).astype(float)))
    extra=np.column_stack((abs(expit(a)-expit(b)),np.minimum(a,b).clip(-12,12)/8,
          np.maximum(a,b).clip(-12,12)/8,(a>0)!=(b>0),opposite_a&opposite_b))
    aggregate=lambda x:utility.aggregate(x,comp.group,comp.count)
    both=np.bincount(comp.group,weights=(opposite_a&opposite_b),minlength=len(comp.count))==comp.count
    return aggregate(mean),aggregate(extra),both


@dataclass
class Example:
    reference:np.ndarray
    parts:list
    x:dict
    y:np.ndarray
    w:np.ndarray
    direct:np.ndarray


def example(image,u,s,first,second,packed,truth):
    ref=(u+s)>0
    ct=utility.normalized_ct(image,ref);high=unpack_level(packed[512:],4)
    xs={k:[] for k in MODES};parts=[];ys=[];weights=[];direct=ref.copy()
    for rank in range(4):
        block=utility.components(u,s,ref,ct,high,rank,False)
        if block is None:continue
        avg,extra,both=witness_features(first[rank],second[rank],block)
        for mode,addition in (("base",np.zeros_like(np.column_stack((avg,extra)))),
                              ("mean",np.column_stack((avg,np.zeros_like(extra)))),
                              ("witness",np.column_stack((avg,extra)))):
            xs[mode].append(np.column_stack((block.context,addition)).astype(np.float32))
        parts.append((rank,block))
        direct[rank].ravel()[block.points[both[block.group]]]^=True
        ys.append(utility.flip_utility(block,ref[rank],truth[rank]))
        weights.append(block.count/(ref[rank].sum()+1))
    return Example(np.packbits(ref,axis=-1),parts,{k:np.concatenate(v) if v else None for k,v in xs.items()},
          np.concatenate(ys) if ys else np.empty(0),np.concatenate(weights) if weights else np.empty(0),
          np.packbits(direct,axis=-1))


def fit_and_apply(train,test,names,seed):
    _,inv,n=np.unique(names,return_inverse=True,return_counts=True)
    case_weight=.5+.5*len(train)/(len(n)*n[inv])
    target=np.concatenate([x.y for x in train])
    weights=np.concatenate([w*x.w for w,x in zip(case_weight,train,strict=True)])
    output={}
    for mode in MODES:
        features=np.concatenate([x.x[mode] for x in train if x.parts])
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=12,min_samples_leaf=40,
            learning_rate=.06,l2_regularization=8.,early_stopping=False,random_state=seed)
        model.fit(features,target,sample_weight=weights/weights.mean())
        held=[x.x[mode] for x in test if x.parts]
        pred=model.predict(np.concatenate(held)) if held else np.empty(0)
        output[mode]=[];cursor=0
        for case in test:
            mask=np.unpackbits(case.reference,axis=-1).astype(bool)
            for rank,block in case.parts:
                size=len(block.count)
                act=pred[cursor:cursor+size]>.05;cursor+=size
                mask[rank].ravel()[block.points[act[block.group]]]^=True
            output[mode].append(np.packbits(mask,axis=-1))
        if cursor!=len(pred):raise ValueError("Incomplete component assignment")
    return output


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,base=nested.base_bank(args,uids,args.fold)
    strong=args.nested/"nested-training"/f"oof-{args.fold}.npz"
    rec=nested.read(strong.with_suffix(".json"))
    if rc.hash_file(args.nested/"independent_quality.json")!=utility.QUALITY or rec["payload"]!=rc.hash_file(strong) or rec["base"]!=base["payload"]:
        raise ValueError("Invalid nested teachers")
    with np.load(strong,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong case order")
        s=z["logits"].astype(np.float32)
    ownership={i:g for g,group in enumerate(groups) for i in group}
    by_case={i:{} for i in ids};subbanks=[]
    provenance=args.nested/"source.sha256"
    if rc.hash_file(provenance)!="6d52c2c77a45f85e55729bf3730dd248c3ddf174d8e2240df3a927bda4a7fc3c":
        raise ValueError("Unexpected original nested teacher source")
    original_source=args.nested/"source"
    subprocess.run(["sha256sum","--quiet","-c",str(provenance)],cwd=original_source,check=True)
    # The original verifier binds the bank to its own source file and teacher dependencies.
    verifier="""import json,sys
from pathlib import Path
from types import SimpleNamespace
from reproduction import nested_guides as n
a=SimpleNamespace(run=Path(sys.argv[1]),fold=int(sys.argv[2]))
_,_,_,uids=n.native.inputs(Path(sys.argv[3]))
records=[]
for g in range(3):
 folder,ids,r=n.verify_subbank(a,uids,g)
 records.append([str(folder),ids,r])
print(json.dumps(records))"""
    result=subprocess.run([sys.executable,"-B","-c",verifier,str(args.nested),str(args.fold),str(args.data)],
        cwd=original_source,env={**os.environ,"PYTHONPATH":str(original_source)},check=True,capture_output=True,text=True)
    for g,(directory,eligible,info) in enumerate(json.loads(result.stdout)):
        folder=Path(directory)
        if info["payload"]!=rc.hash_file(folder/f"oof-{args.fold}.npz"):
            raise ValueError("Bank changed after archived verification")
        subbanks.append(info["payload"])
        with np.load(folder/f"oof-{args.fold}.npz",allow_pickle=False) as z:
            for i,donor,pred in zip(eligible,z["donors"],z["logits"],strict=True):
                if donor in (g,ownership[i]) or donor in by_case[i]:
                    raise ValueError("Wrong or repeated witness")
                by_case[i][int(donor)]=pred
    view=cache_owner.verified_parent(args)
    cases={}
    for j,i in enumerate(ids):
        donors=sorted(by_case[i])
        if donors!=sorted(set(range(3))-{ownership[i]}):raise ValueError("Missing complementary witnesses")
        cases[i]=example(register_image(images[i],[-1,-1]),u[j],s[j],by_case[i][donors[0]],by_case[i][donors[1]],
                         view.features[i,0],truth[i])
    ref=np.stack([cases[i].reference for i in ids]);outputs={k:ref.copy() for k in MODES}
    row={i:j for j,i in enumerate(ids)};membership=[]
    for g,group in enumerate(groups):
        for held in harmonic.case_partition(group,uids,57121+37*g+args.fold):
            fit=sorted(set(group)-set(held))
            if nested.patients(fit,uids)&nested.patients(held,uids):raise ValueError("Patient crossover")
            result=fit_and_apply([cases[i] for i in fit],[cases[i] for i in held],[str(uids[i]).split("_")[0] for i in fit],231571+g)
            for mode in MODES:outputs[mode][[row[i] for i in held]]=result[mode]
            membership.append({"upstream":g,"fit":fit,"held":held})
    if sorted(i for g in membership for i in g["held"])!=ids:raise ValueError("Incomplete patient holdout")
    out=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(out,ids=ids,reference=ref,direct=np.stack([cases[i].direct for i in ids]),**outputs)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":membership,"base":base["payload"],"nested":rec["payload"],
        "subbanks":subbanks,"features":view.receipt,"prediction":rc.hash_file(out),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("WITNESS_LOCK",args.fold,len(ids),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("run","data","original","nested","parent","public"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True);p.set_defaults(backbone="sam2.1-tiny",image_size=512)
    cv2.setNumThreads(1);lock(p.parse_args())
