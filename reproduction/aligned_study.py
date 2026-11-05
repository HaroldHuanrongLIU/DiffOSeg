"""Inner-only check of native-axis registration before image augmentation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from reproduction import aligned_preference as ap
from reproduction import native_local as loc
from reproduction import native_study as ns
from reproduction import order_study as v9
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import vote_contours as vote


def read(p):return json.loads(p.read_text())


def check(args):
    f=args.fold
    oldargs=SimpleNamespace(**{**vars(args),"run":args.v9,"fixed":args.v5,"ordinal":args.v8,"canonical":args.v7})
    data,_,valid,old,weights,previous,parent,lock=v9.setup(oldargs,"inner")
    lib,sel=polar.calibration_split(data,valid,f);ids=lib+sel;n=len(lib)
    probs,_,_,cart=v9.preference_features(oldargs,data,ids,"inner",old,weights,previous,parent,lock)
    raw=ns.logit(probs.clip(1e-6,1-1e-6))
    truth=np.array([data.mask_labels[i] for i in ids],bool)
    img=np.stack([data.images[i] for i in ids])[:,None]
    folder=args.run/f"fold-{f}-inner-{args.variant}"
    fit=read(folder/"fit.json");plan=read(folder/"plan.json")
    if (fit["source"]!=rc.hash_file(ap.__file__) or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or
            plan["train_indices"]!=rc.split_for(data,f)[1] or fit["steps"]!=ap.STEPS):
        raise ValueError("fit identity")
    locator=loc.Locator().cuda().to(memory_format=torch.channels_last).eval()
    locator_path=args.native_root/f"fold-{f}-inner-locator"
    if read(locator_path/"plan.json")["train_indices"]!=plan["train_indices"]:raise ValueError("locator partition differs")
    locator.load_state_dict(torch.load(locator_path/"final.pt",map_location="cpu",weights_only=True))
    centers=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        for x in torch.from_numpy(img).cuda().split(16):
            im=x.float()/255;heat=(locator(im)+locator(im.flip([-1])).flip([-1]))/2
            centers.append(loc.detect(heat).cpu().numpy())
    ctr=np.concatenate(centers)
    ctr=np.where(np.abs(ctr-64.5).max(1,keepdims=True)>4.,ctr,64.5)
    new=ap.predict(vote.load_net(folder/"final.pt"),img,ctr,fit["applied_shift"])
    ctrl=ns.preference(raw,new[:,:4],ctr,cart,0.,0.)
    grid=[]
    for w in ns.MIX:
        for s in ns.SHRINK:
            m=ns.preference(raw[:n],new[:n,:4],ctr[:n],cart[:n],w,s)
            grid.append({"w":w,"s":s,"dice":float(ns.dice(truth[:n],m).mean())})
    best=max(grid,key=lambda z:z["dice"])
    m=ns.preference(raw[n:],new[n:,:4],ctr[n:],cart[n:],best["w"],best["s"])
    pred=ns.dice(truth[n:],m)
    rr={"fold":f,"variant":args.variant,"source":rc.hash_file(__file__),"model":fit["checkpoint"],
        "data":rc.DATA_HASH,"shift":fit["applied_shift"],"fit_indices":lib,"check_indices":sel,"grid":grid,"best":best,
        "v9":float(ns.dice(truth[n:],ctrl[n:]).mean()),"candidate":float(pred.mean()),"each":pred.mean(0).tolist(),
        "image_only":float(ns.dice(truth[n:],new[n:,:4]>0).mean())}
    out=args.run/f"check-{f}-{args.variant}.json"
    if out.exists():raise FileExistsError(out)
    np.savez_compressed(args.run/f"features-{f}-{args.variant}.npz",ids=ids,center=ctr,base=raw.astype("float16"),
                        new=new.astype("float16"),cart=np.packbits(cart,axis=-1))
    rc.write_json(out,rr);print(f,args.variant,best,rr["v9"],rr["candidate"],rr["image_only"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fold",type=int,choices=range(4),required=True)
    for k in ("data","run","v1","v2","v3","v4","v5","v7","v8","v9","native-root"):
        p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--variant",choices=("registered","identity"),required=True)
    cv2.setNumThreads(1);torch.set_num_threads(3);check(p.parse_args())
