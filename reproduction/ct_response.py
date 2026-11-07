"""Look for cross-patient area information in a single CT's sharpening response."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.special import expit

from reproduction import aligned_preference as align
from reproduction import attribute_signal as area
from reproduction import component_utility as components
from reproduction import harmonic_boundary as harmonic
from reproduction import native_local as native
from reproduction import nested_guides as nested
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/ct_response_protocol.md"
MODES=("base","unsigned","signed")


def perturb(image):
    x=image.astype(np.float32)
    detail=x-cv2.GaussianBlur(x,(0,0),1.2)
    return tuple(np.rint(np.clip(x+a*detail,0,255)).astype(np.uint8) for a in (-.6,.6))


@torch.inference_mode()
def fixed_centers(model,images,shift):
    centers=[]
    for x in torch.from_numpy(images[:,None]).cuda().split(16):
        x=align.registered_ct(x,*shift).float()/255
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            z=model(x)
        centers.append(native.rater_center(z[:,:4]>0).cpu().numpy())
    c=np.concatenate(centers)
    distance=abs(c-64.5).max(1,keepdims=True)
    return np.where((distance>4)&(distance<48),c,64.5)


def response(original,blur,sharp):
    p=expit(original);m=original>0
    roi=abs(original)<=4
    n=roi.sum((1,2)).clip(1)
    radii=np.sqrt(m.sum((1,2))/np.pi)
    features=[]
    for v in (blur,sharp):
        mask=v>0
        features.extend((np.sqrt(mask.sum((1,2))/np.pi)-radii,
            (mask!=m).sum((1,2))/m.sum((1,2)).clip(1),
            ((expit(v)-p)*roi).sum((1,2))/n))
    return np.concatenate(features)


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,_,groups=nested.split_groups(uids,args.fold)
    ids,u,base=nested.base_bank(args,uids,args.fold)
    path=args.nested/"nested-training"/f"oof-{args.fold}.npz"
    rec=json.loads(path.with_suffix(".json").read_text())
    if rc.hash_file(args.nested/"independent_quality.json")!=components.QUALITY or rec["base"]!=base["payload"] or rec["payload"]!=rc.hash_file(path):
        raise ValueError("Invalid nested teacher pair")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Different fitting order")
        s=z["logits"].astype(np.float32)
    row={i:j for j,i in enumerate(ids)}
    q=.5*(u+s)
    base_features=[None]*len(ids);new_features=[None]*len(ids)
    voting=np.zeros((len(ids),4,128,16),np.uint8)
    replay_errors=0;positives=0
    for g,subset in enumerate(groups):
        folder=args.original/f"fold-{args.fold}-sub-{g}"
        fit=json.loads((folder/"fit.json").read_text());shift=fit["shift"]
        model=vote.load_net(folder/"final.pt").eval()
        originals=images[subset]
        center=fixed_centers(model,originals,shift)
        changes=np.stack([perturb(x) for x in originals])
        def predict(arr):
            val=align.predict(model,arr[:,None],center,shift)[:,:4].astype(np.float16).astype(np.float32)
            if not np.isfinite(val).all():raise ValueError("Nonfinite response field")
            return val
        neutral=predict(originals);blur=predict(changes[:,0]);sharp=predict(changes[:,1])
        for j,i in enumerate(subset):
            k=row[i]
            replay_errors+=int(np.count_nonzero((neutral[j]>0)!=(u[k]>0)))
            positives+=int(np.count_nonzero(u[k]>0))
            aligned=area.register_image(images[i],shift)
            base_features[k]=area.shape_features(aligned,u[k],s[k],q[k])
            new_features[k]=response(neutral[j],blur[j],sharp[j])
            voting[k]=np.packbits((s[k]+(neutral[j]+blur[j]+sharp[j])/3)>0,axis=-1)
    if replay_errors/max(positives,1)>.001:raise ValueError(f"Teacher replay mismatch: {replay_errors}/{positives}")
    x=np.stack(base_features);z=np.stack(new_features)
    inputs={"base":x,"signed":np.column_stack((x,z)),"unsigned":np.column_stack((x,np.abs(z)))}
    target=(np.sqrt(truth[ids].sum((2,3))/np.pi)-np.sqrt((q>0).sum((2,3))/np.pi)).clip(-3,3)
    names=np.array([str(uids[i]).split("_")[0] for i in ids])
    offsets={name:np.full(target.shape,np.nan) for name in MODES};identity=[]
    for g,group in enumerate(groups):
        for hold in harmonic.case_partition(group,uids,57121+37*g+args.fold):
            use=sorted(set(group)-set(hold))
            if nested.patients(use,uids)&nested.patients(hold,uids):raise ValueError("Patient crossover")
            ti=np.array([row[i] for i in use]);hi=np.array([row[i] for i in hold])
            for key in MODES:offsets[key][hi]=area.regress(inputs[key],target,ti,hi,names)
            identity.append({"upstream":g,"fit":use,"held":hold})
    masks={"reference":np.packbits(q>0,axis=-1),"averaged":voting}
    for key,values in offsets.items():
        if not np.isfinite(values).all():raise ValueError("Missing held-patient offsets")
        masks[key]=np.stack([np.packbits(area.adjust(logit,dx),axis=-1) for logit,dx in zip(q,values,strict=True)])
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"seal-{args.fold}.json",{"fold":args.fold,"groups":identity,"base":base["payload"],
        "nested":rec["payload"],"replay_errors":replay_errors,"positive_pixels":positives,"teachers":base["teachers"],
        "prediction":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("CT_RESPONSE_LOCK",args.fold,replay_errors,positives,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("data","run","original","nested"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=(0,1),required=True)
    args=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(3);lock(args)
