"""Single-slice input check and fixed-center intervention for the locked 90% model."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.special import logit
import torch

from reproduction import aligned_preference as ap
from reproduction import native_local as loc
from reproduction import native_study as ns
from reproduction import order_study as v9
from reproduction import rank_contours as rc
from reproduction import rater_area as radii
from reproduction import registered_zoom as zoom
from reproduction import single90_study as study
from reproduction import vote_contours as vote


def predict(args,imgs,prior,fixed,ref,models,params):
    locator,align_net,zoom_net,regressor=models
    w,s,zw,aw,bias,shift,primary=params
    centers=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
        for block in torch.from_numpy(imgs).cuda().split(16):
            x=block.float()/255
            heat=(locator(x)+locator(x.flip([-1])).flip([-1]))/2
            centers.append(loc.detect(heat).cpu().numpy())
    centers=np.concatenate(centers)
    centers=np.where(np.abs(centers-64.5).max(1,keepdims=True)>4,centers,64.5)
    if fixed: centers=np.full_like(centers,64.5)
    new=ap.predict(align_net,imgs,centers,shift)[:,:4].astype("float16").astype("float32")
    z=zoom.predict(zoom_net,imgs,centers,shift)[:,:4].astype("float16").astype("float32")
    base=np.concatenate([ns.moment_project(x,c,s) for x,c in zip(np.array_split((1-w)*ref+w*new,max(1,len(ref)//12)),
                                                                  np.array_split(centers,max(1,len(ref)//12)))])
    yy,xx=np.indices((128,128),dtype="float32")
    win=np.clip((34-np.maximum(abs(xx[None]-centers[:,0,None,None]),abs(yy[None]-centers[:,1,None,None])))/8,0,1)[:,None]
    merged=study.blend(base,z,win,zw)
    if primary=="base":return study.masks(base,prior)
    masks=study.masks(merged,prior)
    if primary=="zoom":return masks
    ar=radii.predict_areas(regressor,imgs[:,0],merged,shift)
    return study.area_correct(merged,masks,ar,aw,bias)


def run(args):
    sel=study.read(args.run/"selection.json")
    fold_rows=[]
    for f in range(4):
        args.fold=f
        a,(data,outer,_,old,weights,previous,parent,original)=study.previous(args,"full")
        ids=outer["test_indices"]
        mode="full";ad=args.align/f"fold-{f}-{mode}-registered";zd=args.zoom/f"fold-{f}-{mode}"
        ld=args.local/f"fold-{f}-{mode}-locator"; lock=study.read(args.run/f"lock-{f}.json")
        locnet=loc.Locator().cuda().to(memory_format=torch.channels_last).eval()
        locnet.load_state_dict(torch.load(ld/"final.pt",map_location="cpu",weights_only=True))
        models=(locnet,vote.load_net(ad/"final.pt"),zoom.model_from(zd/"final.pt"),study.area_model(args,"full"))
        before=study.read(args.align/f"check-{f}-registered.json")["best"]
        params=(before["w"],before["s"],lock["zoom_weight"],lock["area_weight"],np.array(lock["offset"]),
                study.read(ad/"fit.json")["applied_shift"],sel["primary"])
        images=np.stack([data.images[i] for i in ids])[:,None]
        prior=[]
        for i in range(len(ids)):
            with np.load(args.v1/f"assessment/fold-{f}/{i:05d}_masks.npz") as d:prior.append(np.unpackbits(d["pref"],axis=-1).astype(bool))
        prior=np.stack(prior)
        def baseline(model_input,indices):
            probs,*_=v9.preference_features(a,model_input,indices,mode,old,weights,previous,parent,original)
            return logit(probs.clip(1e-6,1-1e-6)).astype("float16").astype("float32")
        base=baseline(data,ids)
        pred=predict(args,images,prior,True,base,models,params)
        truth=np.array([data.mask_labels[i] for i in ids],bool)
        # Force a batch neighbor change while keeping the exact same target CT and proposal.
        class ImageOnly:
            def __init__(self,values):self.images=values
        test_cases=[]
        for other in (images[1,0],np.zeros_like(images[1,0])):
            im=np.stack((images[0,0],other))[:,None]
            ref=baseline(ImageOnly(im[:,0]),[0,1])
            test_cases.append(predict(args,im,prior[:2],False,ref,models,params)[0])
        if not np.array_equal(*test_cases):raise ValueError("Unexpected dependence on another image")
        fold_rows.append({"fold":f,"cases":len(ids),"fixed_center_dice":float(ns.dice(truth,pred).mean()),
                          "image_only_neighbor_invariance":True})
    rc.write_json(args.run/"assessment/single_slice_control.json",{"folds":fold_rows,
                  "fixed_center_fourfold":float(np.mean([r["fixed_center_dice"] for r in fold_rows])),
                  "full":study.read(args.run/"assessment/preference_report.json")["means"]["full"]})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("run","data","local","align","zoom","area","v1","v2","v3","v4","v5","v7","v8","v9"):
        p.add_argument("--"+key,type=Path,required=True)
    cv2.setNumThreads(1);torch.set_num_threads(3);run(p.parse_args())
