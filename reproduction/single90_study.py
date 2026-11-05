"""Lock single-slice preference variants before scoring the complete original folds."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np
from scipy.special import expit,logit
import torch

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as ns
from reproduction import order_study as v9
from reproduction import polar_contours as polar
from reproduction import polar_eval as v2
from reproduction import rank_contours as rc
from reproduction import rater_area as area
from reproduction import registered_zoom as zoom
from reproduction import vote_contours as vote

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_90/decision.md"
ARMS=("base","zoom","area")
read=lambda p:json.loads(p.read_text())


def require(x,msg):
    if not x:raise ValueError(msg)


def identity():
    return {"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL),"v9":ns.V9,
            "aligned":rc.hash_file(align.__file__),"zoom":rc.hash_file(zoom.__file__),"area":rc.hash_file(area.__file__)}


def previous(args,mode):
    a=SimpleNamespace(**{**vars(args),"run":args.v9,"fixed":args.v5,"ordinal":args.v8,"canonical":args.v7})
    return a,v9.setup(a,mode)


def fold_data(args,mode):
    a,(data,outer,valid,old,weights,prev,parent,original)=previous(args,mode)
    f=args.fold;lib,check=polar.calibration_split(data,valid,f)
    ids=lib+check if mode=="inner" else outer["test_indices"]
    expected=rc.split_for(data,f)[1] if mode=="inner" else outer["train_indices"]
    images=np.stack([data.images[i] for i in ids])[:,None]
    path=args.align/f"fold-{f}-{mode}-registered"; zpath=args.zoom/f"fold-{f}-{mode}"; lpath=args.local/f"fold-{f}-{mode}-locator"
    for d,src in ((path,align.__file__),(zpath,zoom.__file__),(lpath,native.__file__)):
        meta=read(d/"fit.json")
        require(meta["source"]==rc.hash_file(src) and meta["checkpoint"]==rc.hash_file(d/"final.pt") and
                read(d/"plan.json")["train_indices"]==expected and meta["fold"]==f and meta["mode"]==mode,"model verification "+str(d))
    require(read(zpath/"fit.json")["parent"]==read(path/"fit.json")["checkpoint"],"wrong zoom parent")
    shift=read(path/"fit.json")["applied_shift"]
    first=read(args.align/f"check-{f}-registered.json")
    require((lib,check)==(first["fit_indices"],first["check_indices"]) and
            first["source"]==rc.hash_file(args.align/"eval-source/reproduction/aligned_study.py") and
            first["model"]==rc.hash_file(args.align/f"fold-{f}-inner-registered/final.pt") and
            rc.hash_file(args.v9/"assessment/fourfold_comparison.json")==ns.V9,"inner lock")
    if mode=="inner":
        with np.load(args.align/f"features-{f}-registered.npz",allow_pickle=False) as d:
            require(d["ids"].tolist()==ids,"cached inner IDs")
            base,new,ctr=d["base"].astype("float32"),d["new"][:,:4].astype("float32"),d["center"]
            prior=np.unpackbits(d["cart"],axis=-1).astype(bool)
    else:
        probs,_,_,_=v9.preference_features(a,data,ids,mode,old,weights,prev,parent,original)
        prior=[]
        for i in range(len(ids)):
            with np.load(args.v1/f"assessment/fold-{f}/{i:05d}_masks.npz") as d:
                prior.append(np.unpackbits(d["pref"],axis=-1).astype(bool))
        prior=np.stack(prior)
        base=logit(probs.clip(1e-6,1-1e-6)).astype("float16").astype("float32")
        locator=native.Locator().cuda().to(memory_format=torch.channels_last).eval()
        locator.load_state_dict(torch.load(lpath/"final.pt",map_location="cpu",weights_only=True))
        centers=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            for block in torch.from_numpy(images).cuda().split(16):
                x=block.float()/255;heat=(locator(x)+locator(x.flip([-1])).flip([-1]))/2
                centers.append(native.detect(heat).cpu().numpy())
        ctr=np.concatenate(centers)
        ctr=np.where(np.abs(ctr-64.5).max(1,keepdims=True)>4,ctr,64.5)
        new=align.predict(vote.load_net(path/"final.pt"),images,ctr,shift)[:,:4].astype("float16").astype("float32")
    z=zoom.predict(zoom.model_from(zpath/"final.pt"),images,ctr,shift)[:,:4].astype("float16").astype("float32")
    w,s=first["best"]["w"],first["best"]["s"]
    merged=np.concatenate([ns.moment_project(x,c,s) for x,c in zip(np.array_split((1-w)*base+w*new,max(1,len(base)//12)),np.array_split(ctr,max(1,len(base)//12)))])
    yy,xx=np.indices((128,128),dtype="float32")
    radius=np.maximum(abs(xx[None]-ctr[:,0,None,None]),abs(yy[None]-ctr[:,1,None,None]))
    window=np.clip((34-radius)/8,0,1)[:,None]
    return data,ids,lib,check,images[:,0],base,merged,z,prior,window,shift


def blend(base,z,window,w):
    return base+w*window*(z-base)


def masks(logits,prior):
    from reproduction.joint_rater import preference
    return np.stack([preference(x,y) for x,y in zip(expit(logits),prior)])


def area_correct(logits,original,pred,w,offset):
    return np.stack([area.apply_radii(z,m,p,w,offset) for z,m,p in zip(logits,original,pred)])


def area_model(args,mode):
    d=args.area/f"fold-{args.fold}-{mode}"
    fit=read(d/"fit.json")
    plan=read(args.align/f"fold-{args.fold}-{mode}-registered/plan.json")
    require(fit["train_indices"]==plan["train_indices"] and fit["source"]==rc.hash_file(area.__file__) and
            fit["model"]==rc.hash_file(d/"model.joblib"),"radius model provenance")
    return joblib.load(d/"model.joblib")


def calibrate(args):
    data,ids,lib,check,images,old,base,z,prior,window,shift=fold_data(args,"inner")
    n=len(lib);truth=np.array([data.mask_labels[i] for i in ids],bool)
    baseline=masks(old,prior);base_mask=masks(base,prior)
    scores=[]
    for w in ns.MIX:
        pred=masks(blend(base[:n],z[:n],window[:n],w),prior[:n])
        scores.append(float(ns.dice(truth[:n],pred).mean()))
    w=ns.MIX[int(np.argmax(scores))]
    logits=blend(base,z,window,w);merged=masks(logits,prior)
    pred=area.predict_areas(area_model(args,"inner"),images,logits,shift)
    exact=np.sqrt(truth.sum((2,3))/np.pi);inc=np.sqrt(merged.sum((2,3))/np.pi)
    offset=.5*np.median(exact[:n]-pred[:n],axis=0)
    area_scores=[]
    for t in ns.MIX:
        v=area_correct(logits[:n],merged[:n],pred[:n],t,offset)
        area_scores.append(float(ns.dice(truth[:n],v).mean()))
    t=ns.MIX[int(np.argmax(area_scores))]
    improved=area_correct(logits[n:],merged[n:],pred[n:],t,offset)
    methods={"v9":baseline[n:],"base":base_mask[n:],"zoom":merged[n:],"area":improved}
    check_scores={k:float(ns.dice(truth[n:],m).mean()) for k,m in methods.items()}
    record={**identity(),"fold":args.fold,"fit_indices":lib,"check_indices":check,"zoom_weight":w,"area_weight":t,
            "offset":offset.tolist(),"zoom_fit":scores,"area_fit":area_scores,"check":check_scores,
            "radius_rmse":{"existing":float(np.sqrt(np.mean((inc[n:]-exact[n:])**2))),
                           "regressor":float(np.sqrt(np.mean((pred[n:]+offset-exact[n:])**2)))},
            "align_lock":rc.hash_file(args.align/f"check-{args.fold}-registered.json")}
    rc.write_json(args.run/f"lock-{args.fold}.json",record)
    print("CHECK",args.fold,w,t,check_scores,record["radius_rmse"],flush=True)


def selection(args):
    rows=[read(args.run/f"lock-{f}.json") for f in range(4)]
    for f,r in enumerate(rows):
        require(r["fold"]==f and all(r[k]==v for k,v in identity().items()) and
                r["align_lock"]==rc.hash_file(args.align/f"check-{f}-registered.json"),"candidate lock identity")
    choice="base"
    for m in ("zoom","area"):
        gain=np.array([r["check"][m]-r["check"][choice] for r in rows])
        if gain.mean()>=.00015 and sum(gain>0)>=2:choice=m
    means={m:float(np.mean([r["check"][m] for r in rows])) for m in ("v9",*ARMS)}
    require(means[choice]>=.9 and means[choice]>means["v9"],f"inner promotion failed: {means}")
    return rows,{"primary":choice,"check_means":means,"lock_hashes":[rc.hash_file(args.run/f"lock-{f}.json") for f in range(4)]}


def gate(args):
    _,value=selection(args);rc.write_json(args.run/"selection.json",value);print(value,flush=True)


def infer(args):
    rows,choice=selection(args);require(choice==read(args.run/"selection.json"),"selection changed")
    f=args.fold;r=rows[f]
    _,ids,lib,check,images,_,base,z,prior,window,shift=fold_data(args,"full")
    require(r["fit_indices"]==lib and r["check_indices"]==check,"calibration partition changed")
    logits=blend(base,z,window,r["zoom_weight"])
    b,z_mask=masks(base,prior),masks(logits,prior)
    sizes=area.predict_areas(area_model(args,"full"),images,logits,shift)
    a=area_correct(logits,z_mask,sizes,r["area_weight"],np.array(r["offset"]))
    banks={"base":b,"zoom":z_mask,"area":a};banks["full"]=banks[choice["primary"]]
    d=args.run/f"assessment/fold-{f}";d.mkdir(parents=True,exist_ok=False)
    for i in range(len(ids)):
        np.savez_compressed(d/f"{i:05d}.npz",**{k:np.packbits(v[i],axis=-1) for k,v in banks.items()})
    rc.write_json(d/"sealed.json",{**identity(),"fold":f,"cases":len(ids),"bank":v2.digest_bank(d,len(ids)),
                  "selection":rc.hash_file(args.run/"selection.json")})


def seals(args):
    _,s=selection(args);require(s==read(args.run/"selection.json"),"unlocked selection")
    for f in range(4):
        row=read(args.run/f"assessment/fold-{f}/sealed.json")
        require(row["fold"]==f and row["selection"]==rc.hash_file(args.run/"selection.json") and
                all(row[k]==v for k,v in identity().items()),"unsealed bank")


def score(args):
    seals(args); f=args.fold
    _,(data,outer,*_)=previous(args,"full");ids=outer["test_indices"]
    d=args.run/f"assessment/fold-{f}";v=args.v9/f"assessment/fold-{f}"
    require(v2.digest_bank(d,len(ids))==read(d/"sealed.json")["bank"] and
            v2.digest_bank(v,len(ids))==read(v/"sealed.json")["bank_sha256"],"mask provenance")
    old=read(v/"paired.json")["rows"];rows=[]
    for i,idx in enumerate(ids):
        truth=np.array(data.mask_labels[idx],bool);patient=str(data.series_uid[idx]).split("_")[0]
        row={"index":idx,"patient":patient}
        require((idx,patient)==(old[i]["index"],old[i]["patient"]),"paired order")
        with np.load(v/f"{i:05d}.npz") as z: ref=np.unpackbits(z["full"],axis=-1).astype(bool)
        with np.load(d/f"{i:05d}.npz") as z:
            masks_i={"v9":ref,**{k:np.unpackbits(z[k],axis=-1).astype(bool) for k in ("full",*ARMS)}}
        for k,m in masks_i.items():
            e=ns.dice(truth[None],m[None])[0]
            row[k]={"Dice_mean":float(e.mean()),"Dice_each":e.tolist()}
        require(abs(row["v9"]["Dice_mean"]-old[i]["full"]["Dice_mean"])<1e-6,"v9 per-case parity")
        rows.append(row)
    rc.write_json(d/"paired.json",{"rows":rows,"seal":rc.hash_file(d/"sealed.json")})


def report(args):
    seals(args); folds=[];boot=[];patients=set();indices=set();rng=np.random.default_rng(212903)
    for f in range(4):
        d=args.run/f"assessment/fold-{f}";table=read(d/"paired.json");rows=table["rows"]
        require(table["seal"]==rc.hash_file(d/"sealed.json"),"scored bank identity")
        names,inv=np.unique([r["patient"] for r in rows],return_inverse=True)
        ids=[r["index"] for r in rows];split=read(args.v1/f"baseline/fold_{f}/lidc/fold_{f}/stage_2/split.json")
        require(ids==split["test_indices"] and set(names)==set(split["test_patients"]) and
                not indices.intersection(ids) and not patients.intersection(names),"incomplete folds")
        indices.update(ids);patients.update(names)
        vals={m:np.array([r[m]["Dice_each"] for r in rows]) for m in ("v9","full",*ARMS)}
        means={m:{"Dice_mean":float(a.mean()),"Dice_each":a.mean(0).tolist()} for m,a in vals.items()}
        n=np.bincount(inv);draw=rng.multinomial(len(n),np.full(len(n),1/len(n)),size=5000)
        by=np.zeros((len(n),2))
        np.add.at(by,inv,np.stack((vals["full"].mean(1),(vals["full"]-vals["v9"]).mean(1)),1))
        boot.append(draw@by/(draw@n)[:,None]);folds.append({"fold":f,"patients":len(n),"cases":len(ids),"means":means})
    require(len(patients)==214 and len(indices)==1609,"patient coverage")
    means={m:float(np.mean([f["means"][m]["Dice_mean"] for f in folds])) for m in vals}
    require(abs(means["v9"]-read(args.v9/"assessment/fourfold_comparison.json")["metrics"]["Dice_mean"]["full"])<1e-6,"v9 summary parity")
    ci=np.percentile(np.mean(boot,axis=0),[2.5,97.5],axis=0).T.tolist()
    rc.write_json(args.run/"assessment/preference_report.json",{**identity(),"selection":read(args.run/"selection.json"),
        "patients":len(patients),"cases":len(indices),"folds":folds,"means":means,"absolute_ci95":ci[0],"delta_ci95":ci[1],
        "threshold":.9,"target_pass":means["full"]>=.9})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("calibrate","gate","infer","score","report"))
    for key in ("run","data","local","align","zoom","area","v1","v2","v3","v4","v5","v7","v8","v9"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(4))
    a=p.parse_args()
    if a.phase in ("calibrate","infer","score") and a.fold is None:p.error("fold required")
    cv2.setNumThreads(1);torch.set_num_threads(3)
    {"calibrate":calibrate,"gate":gate,"infer":infer,"score":score,"report":report}[a.phase](a)
