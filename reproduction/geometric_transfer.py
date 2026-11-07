"""Transfer median primary contour motion to the boundary of the fixed E segmentation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from reproduction import aligned_preference as align
from reproduction import harmonic_boundary as harmonic
from reproduction import native_study as metrics
from reproduction import native_views as views
from reproduction import rank_contours as rc
from reproduction import stratified_errors as upstream
from reproduction import vote_contours as vote
from reproduction.rater_area import sdf

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/geometric_transfer_protocol.md"
METHODS=("logit","direct","boundary")
STRENGTHS=(0.,.25,.5,1.)


def boundary_motion(surface,motion):
    band=(np.abs(surface)<=.5).astype(np.float32)
    density=gaussian_filter(band,2.,mode="constant",truncate=3.5)
    numerator=gaussian_filter(motion*band,2.,mode="constant",truncate=3.5)
    return np.divide(numerator,density,out=np.zeros_like(numerator),where=density>.001)


def geometric_fields(anchor,initial,predictions):
    """Return E's signed field and two displacement choices in native pixel units."""
    efield=harmonic.continuous_field(anchor)
    support=(np.stack([np.abs(sdf(mask)) for mask in anchor>0])<=3)
    direct=np.zeros_like(efield);boundary=np.zeros_like(efield)
    for rank in range(4):
        original=sdf(initial[rank])
        distances=np.stack([sdf(m[rank])-original for m in predictions])
        delta=np.median(distances,axis=0).clip(-3,3)
        direct[rank]=np.where(support[rank],delta,0.)
        boundary[rank]=np.where(support[rank],boundary_motion(original,delta),0.)
    return efield,direct,boundary


def lock(args):
    images,truth,train,lib,check,anchor,reference,previous,centers=views.eval_context(args)
    ids=lib+check;n=len(lib)
    folder,warm=views.warm_identity(args,train)
    seal=views.read(args.upstream/f"selection-{args.fold}.json")
    earlier=args.upstream/f"pred-{args.fold}.npz"
    if (seal["prediction"]!=rc.hash_file(earlier) or seal["library"]!=lib or seal["check"]!=check
            or seal["reference"]!=previous["prediction"] or seal["warm"]!=warm["checkpoint"]
            or seal["source"]!=rc.hash_file(upstream.__file__)):
        raise ValueError("Underlying model or patient identities differ")
    def predict(path):
        result=align.predict(vote.load_net(path),images[ids,None],centers,warm["applied_shift"])[:,:4]
        if not np.isfinite(result).all():raise ValueError("Nonfinite primary logits")
        return result
    original=predict(folder/"final.pt")
    with np.load(earlier,allow_pickle=False) as old:
        if old["ids"].tolist()!=ids:raise ValueError("Incorrect source-prediction order")
        np.testing.assert_array_equal(reference,np.unpackbits(old["reference"],axis=-1))
        initial=original>0
        np.testing.assert_array_equal(np.packbits(initial,axis=-1),old["initial"])
        members=[]
        for mode in upstream.MODES:
            path=args.upstream/f"fold-{args.fold}-{mode}/final.pt"
            if rc.hash_file(path)!=seal["members"][mode]:raise ValueError("Underlying model checkpoint changed")
            prediction=predict(path)
            np.testing.assert_array_equal(np.packbits(prediction>0,axis=-1),old["standalone_"+mode])
            members.append(prediction)
    raw=np.median(np.stack(members)-original[None],axis=0)
    binary=np.stack([z>0 for z in members]);del members
    signed=[];direct=[];boundary=[]
    for i in range(len(ids)):
        e,d,b=geometric_fields(anchor[i],initial[i],binary[:,i])
        signed.append(e);direct.append(d);boundary.append(b)
    signed=np.stack(signed);direct=np.stack(direct);boundary=np.stack(boundary)
    if not np.array_equal(signed>0,reference):raise ValueError("Zero motion does not reproduce E")
    # Preserve actual arithmetic inputs for independent reconstruction of the three selected grids.
    state=args.run/f"inputs-{args.fold}.npz"
    np.savez_compressed(state,ids=ids,continuous_e=anchor,logit_delta=raw)
    masks={"reference":np.packbits(reference,axis=-1)}
    choices={}
    for mode,current,change in (("logit",anchor,raw),("direct",signed,direct),("boundary",signed,boundary)):
        grid=[float(metrics.dice(truth[lib],current[:n]+w*change[:n]>0).mean()) for w in STRENGTHS]
        w=STRENGTHS[int(np.argmax(grid))]
        choices[mode]={"weight":w,"library_scores":grid}
        masks[mode]=np.packbits(current+w*change>0,axis=-1)
    path=args.run/f"pred-{args.fold}.npz";np.savez_compressed(path,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,
        "reference":previous["prediction"],"upstream":seal["prediction"],"members":seal["members"],"choices":choices,
        "inputs":rc.hash_file(state),"prediction":rc.hash_file(path),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("GEOMETRIC_TRANSFER_LOCK",args.fold,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("run","data","upstream","align","stable","formal","parent","ensemble","public","local","texture"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4),required=True)
    args=p.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1);lock(args)
