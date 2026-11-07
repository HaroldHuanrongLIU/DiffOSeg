"""Rebuild the frozen 90% comparator and check single-slice candidates on training patients."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from reproduction import native_local as native
from reproduction import native_study as ns
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import single90_study as old
from reproduction import texture_preference as texture

REFERENCE = "2024e556c9cf003d037701d458d2da3eb527deb0192bf8666cf8383651a220d0"
WEIGHTS = (0.,0.25,0.5,0.75,1.)
read=old.read


def base_args(args):
    return SimpleNamespace(**{**vars(args),"run":args.incumbent})


def comparator(args, mode):
    if rc.hash_file(args.incumbent/"assessment/preference_report.json")!=REFERENCE:
        raise ValueError("90% comparator identity differs")
    a=base_args(args)
    locks,selected=old.selection(a)
    if selected!=read(a.run/"selection.json"):raise ValueError("90% selection changed")
    rec=locks[a.fold]
    data,ids,lib,check,images,_,base,z,prior,window,shift=old.fold_data(a,mode)
    combined=old.blend(base,z,window,rec["zoom_weight"])
    mask=old.masks(combined,prior)
    if selected["primary"]=="area":
        sizes=old.area.predict_areas(old.area_model(a,mode),images,combined,shift)
        mask=old.area_correct(combined,mask,sizes,rec["area_weight"],np.array(rec["offset"]))
    elif selected["primary"]=="base":
        combined=base; mask=old.masks(combined,prior)
    if mode=="full":
        folder=a.run/f"assessment/fold-{a.fold}"
        for i in range(len(ids)):
            with np.load(folder/f"{i:05d}.npz",allow_pickle=False) as d:
                mask[i]=np.unpackbits(d["full"],axis=-1).astype(bool)
    return data,ids,lib,check,images,combined,mask,shift


@torch.inference_mode()
def centers(args,mode,images):
    f=args.fold
    lpath=args.local/f"fold-{f}-{mode}-locator"
    fit=read(lpath/"fit.json")
    if fit["checkpoint"]!=rc.hash_file(lpath/"final.pt"):raise ValueError("Locator identity")
    net=native.Locator().cuda().to(memory_format=torch.channels_last).eval()
    net.load_state_dict(torch.load(lpath/"final.pt",map_location="cpu",weights_only=True))
    result=[]
    for block in torch.from_numpy(images[:,None]).cuda().split(16):
        x=block.float()/255
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            h=(net(x)+net(x.flip([-1])).flip([-1]))/2
        result.append(native.detect(h).cpu().numpy())
    c=np.concatenate(result)
    return np.where(np.abs(c-64.5).max(1,keepdims=True)>4.,c,64.5)


def cache(args,mode):
    path=args.run/f"features-{args.fold}-{mode}.npz"
    if path.exists():raise FileExistsError(path)
    data,ids,lib,check,im,base,mask,shift=comparator(args,mode)
    folder=args.run/f"fold-{args.fold}-{mode}"
    fit=read(folder/"fit.json");plan=read(folder/"plan.json")
    outer,inner,_=rc.split_for(data,args.fold)
    expected=inner if mode=="inner" else outer["train_indices"]
    if (plan["train_indices"]!=expected or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or
            fit["steps"]!=texture.STEPS or fit["source"]!=rc.hash_file(texture.__file__) or fit["shift"]!=shift):
        raise ValueError("Texture checkpoint does not match its split")
    c=centers(args,mode,im)
    new=texture.predict(texture.load_model(folder/"final.pt"),im[:,None],c,shift)
    np.savez_compressed(path,ids=np.array(ids),base=base.astype("float16"),old=np.packbits(mask,axis=-1),
                        new=new.astype("float16"),centers=c)
    rc.write_json(args.run/f"features-{args.fold}-{mode}.json",{"fold":args.fold,"mode":mode,"indices":ids,
                  "lib":lib,"check":check,"checkpoint":fit["checkpoint"],"comparator":REFERENCE,
                  "source":rc.hash_file(__file__),"payload":rc.hash_file(path)})
    print("FEATURES",mode,args.fold,len(ids),flush=True)


def anchor_logits(raw,mask):
    return np.where(mask,1.,-1.).astype(np.float32)*np.maximum(np.abs(raw),.05)


def inner(args):
    _,truth,_,uids=native.inputs(args.data)
    _,_,valid=native.splits(uids,args.fold)
    group=SimpleNamespace(series_uid=uids)
    lib,check=polar.calibration_split(group,valid,args.fold)
    path=args.run/f"features-{args.fold}-inner.npz"
    receipt=read(path.with_suffix(".json"))
    if receipt["payload"]!=rc.hash_file(path):raise ValueError("Feature identity")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=lib+check:raise ValueError("Inner order differs")
        mask=np.unpackbits(z["old"],axis=-1).astype(bool)
        base=anchor_logits(z["base"].astype(np.float32),mask)
        new=z["new"][:,:4].astype(np.float32)
    # Read reference annotations solely after prediction cache has been completed.
    y=truth[lib+check].astype(bool);n=len(lib)
    scores=[]
    for w in WEIGHTS:
        scores.append(float(ns.dice(y[:n],(1-w)*base[:n]+w*new[:n]>0).mean()))
    w=WEIGHTS[int(np.argmax(scores))]
    pred=(1-w)*base[n:]+w*new[n:]>0
    record={"fold":args.fold,"comparator":REFERENCE,"source":rc.hash_file(__file__),"weights":list(WEIGHTS),
            "fit_scores":scores,"weight":w,"fit_indices":lib,"check_indices":check,
            "check":{"reference":float(ns.dice(y[n:],mask[n:]).mean()),"texture_only":float(ns.dice(y[n:],new[n:]>0).mean()),
                     "candidate":float(ns.dice(y[n:],pred).mean())}}
    rc.write_json(args.run/f"inner-{args.fold}.json",record);print("INNER",args.fold,w,record["check"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","check"))
    for name in ("data","run","incumbent","local","align","zoom","area","v1","v2","v3","v4","v5","v7","v8","v9"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",choices=range(4),type=int,required=True)
    parser.add_argument("--mode",choices=("inner","full"),default="inner")
    args=parser.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(3)
    if args.phase=="prepare":cache(args,args.mode)
    else:inner(args)
