"""Train with correlated real reader-boundary residuals from other fitting patients."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import nested_guides as nested
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_field import cpu_signed
from reproduction.trajectory_sam import select_weight

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/reader_error_transfer_protocol.md"
MODES=("real","shuffled","joint")


def anatomy(mask):
    yy,xx=np.where(mask.sum(0)>=2)
    if not len(xx): return np.array([63.5,63.5]),10.,0.
    center=np.array([xx.mean(),yy.mean()])
    radius=max(np.sqrt(len(xx)/np.pi),1.)
    x,y=xx-center[0],yy-center[1]
    angle=.5*np.arctan2(2*np.mean(x*y),np.mean(x*x-y*y))
    return center,radius,angle


def rotate(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


def candidate_donors(index,names,radii):
    closest={}
    for j,name in enumerate(names):
        if name!=names[index]:
            key=(abs(radii[index]-radii[j]),j)
            if name not in closest or key<closest[name]:closest[name]=key
    ids=[j for _,j in sorted(closest.values())[:16]]
    if len(ids)<4:raise ValueError("Four distinct fitting donor patients are required")
    return ids


def transplant_grid(target,donor,signs):
    ct,rt,at=target;cd,rd,ad=donor
    matrix=rotate(ad)@np.diag(signs)@rotate(at).T*(rd/rt)
    yy,xx=np.indices((128,128),dtype=np.float64)
    coords=(np.stack((xx,yy),axis=-1)-ct)@matrix.T+cd
    return coords.astype(np.float32)


def choose_guide(bank,indices,draws,mode):
    if mode=="joint":return bank[indices,draws].float()
    ranks=torch.arange(4,device=indices.device)[None]
    return bank[indices[:,None],(draws[:,None]+ranks)%4,ranks].float()


def prepare(args):
    _,truth,_,uids=native.inputs(args.data)
    ids,oof,old=nested.base_bank(args,uids,args.fold)
    y=truth[ids].astype(bool);pred=oof>0
    names=np.array([str(uids[i]).split("_")[0] for i in ids])
    geom=[anatomy(m) for m in y];radii=np.array([g[1] for g in geom])
    true_fields=np.empty(y.shape,dtype=np.float32);error=np.empty_like(true_fields)
    ratios=np.full((len(ids),4,2),np.nan)
    for i in range(len(ids)):
        for r in range(4):
            ys=cpu_signed(y[i,r]);ps=cpu_signed(pred[i,r]);true_fields[i,r]=ys
            error[i,r]=np.clip(ps-ys,-4,4)
            for sign in (0,1):
                use=(pred[i,r]==bool(sign))&(np.abs(ps)<=3)
                if use.any():ratios[i,r,sign]=np.median(np.abs(oof[i,r][use])/np.maximum(abs(ps[use]),.5))
    per_patient=np.stack([np.nanmedian(ratios[names==n],axis=0) for n in sorted(set(names))])
    slope=np.nan_to_num(np.nanmedian(per_patient,axis=0),nan=2.).clip(.5,8)
    rng=np.random.default_rng(891311+args.fold)
    path=args.run/f"bank-{args.fold}.npy"
    partial=path.with_suffix(".partial.npy")
    if path.exists() or partial.exists():raise FileExistsError(path)
    store=np.lib.format.open_memmap(partial,mode="w+",dtype=np.float16,shape=(len(ids),4,4,128,128))
    donors=np.empty((len(ids),4),np.int32);errors=[];case_dice=[]
    for i in range(len(ids)):
        selected=rng.choice(candidate_donors(i,names,radii),4,replace=False)
        donors[i]=np.asarray(ids)[selected]
        if names[i] in names[selected] or len(set(names[selected]))!=4:raise ValueError("Donors must be other patients")
        for k,j in enumerate(selected):
            xy=transplant_grid(geom[i],geom[j],rng.choice([-1.,1.],size=2))
            alpha=rng.uniform(.6,1.15)*radii[i]/radii[j]
            for r in range(4):
                shift=cv2.remap(error[j,r],xy[:,:,0],xy[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                sd=true_fields[i,r]+np.clip(alpha*shift,-4,4)
                store[i,k,r]=np.clip(sd*np.where(sd>0,slope[r,1],slope[r,0]),-12,12).astype(np.float16)
        generated=store[i]>0
        errors.append(np.sqrt(generated.sum((2,3))/np.pi)-np.sqrt(y[i].sum((1,2))/np.pi))
        case_dice.append(metrics.dice(y[i][None],generated).mean())
    store.flush();del store
    partial.replace(path)
    donorfile=args.run/f"donors-{args.fold}.npy";np.save(donorfile,donors)
    err=np.stack(errors)
    shuffled=np.stack([np.roll(err[:,:,r],-r,axis=1) for r in range(4)],axis=2)
    def corr(values):
        return float(np.median(np.corrcoef(values.reshape(-1,4).T)[np.triu_indices(4,1)]))
    mean=float(np.mean(case_dice));joint_corr,shuffled_corr=corr(err),corr(shuffled)
    rc.write_json(path.with_suffix(".json"),{"fold":args.fold,"ids":ids,"donors":rc.hash_file(donorfile),"oof":old["payload"],
        "slope":slope.tolist(),"prompt_dice":mean,"joint_correlation":joint_corr,"shuffled_correlation":shuffled_corr,
        "eligible":bool(.85<=mean<=.96 and joint_corr>shuffled_corr),"payload":rc.hash_file(path),"source":rc.hash_file(__file__),
        "protocol":rc.hash_file(PROTOCOL)})
    print("TRANSFER_PREPARE",args.fold,mean,joint_corr,shuffled_corr,flush=True)


def banks(args):
    records=[]
    for f in (0,1):
        path=args.run/f"bank-{f}.npy";r=json.loads(path.with_suffix(".json").read_text())
        if (not r["eligible"] or r["source"]!=rc.hash_file(__file__) or r["protocol"]!=rc.hash_file(PROTOCOL)
                or r["donors"]!=rc.hash_file(args.run/f"donors-{f}.npy") or r["payload"]!=rc.hash_file(path)):
            raise ValueError("Both synthetic banks must pass the pretraining checks")
        records.append(r)
    return records


def train(args):
    meta=banks(args)[args.fold];file=args.run/f"bank-{args.fold}.npy"
    opt=argparse.Namespace(**vars(args));opt.crossfit=args.original
    ids=meta["ids"][:32] if args.smoke else meta["ids"]
    data=np.load(file,mmap_mode="r",allow_pickle=False)
    bank=torch.from_numpy(data[:len(ids)].copy()).cuda()
    generator=torch.Generator(device="cuda").manual_seed(887101+args.fold)
    folder=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.variant}")
    def update(real,ix,step):
        if step==1:
            sample={"cases":[ids[i] for i in ix.tolist()],"seed":175171+args.fold}
            if args.variant!="real" and not args.smoke and sample!=json.loads((args.run/f"fold-{args.fold}-real/first.json").read_text()):
                raise ValueError("Unmatched initial patient sampling")
            rc.write_json(folder/"first.json",sample)
        if step%2:return real
        draws=torch.randint(4,(len(ix),),generator=generator,device="cuda")
        if args.variant=="real":return real
        return choose_guide(bank,ix,draws,args.variant)
    view=cache_owner.verified_parent(args)
    prompt.train(opt,decoder_type=hierarchy.HierarchicalRanks,variant=args.variant,study_protocol=PROTOCOL,view=view,
                 public_weights=hierarchy.WEIGHT,seed_offset=125000,guide_update=update)
    rc.write_json(folder/"transfer.json",{"fit":rc.hash_file(folder/"fit.json"),"bank":meta["payload"],
                  "donors":meta["donors"],"source":rc.hash_file(__file__)})


def lock(args):
    images,truth,training,lib,checked,guide,reference,previous,view,model=nested.e_guidance(args)
    ids=lib+checked;n=len(lib)
    masks={"reference":np.packbits(reference,axis=-1)};choices={}
    bank=banks(args)[args.fold]
    for name in MODES:
        folder=args.run/f"fold-{args.fold}-{name}";fit=json.loads((folder/"fit.json").read_text())
        transfer=json.loads((folder/"transfer.json").read_text())
        if (fit["steps"]!=5000 or fit["seed"]!=175171+args.fold or fit["train_indices"]!=sorted(training) or fit["variant"]!=name
                or fit["oof"]!=bank["oof"] or fit["scope"]!="inner" or fit["public_weights"]!=hierarchy.WEIGHT
                or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT or fit["protocol"]!=rc.hash_file(PROTOCOL)
                or fit["source"]!=rc.hash_file(prompt.__file__) or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")
                or transfer["bank"]!=bank["payload"] or transfer["fit"]!=rc.hash_file(folder/"fit.json")):
            raise ValueError("Synthetic training identity differs")
        model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
        logits=prompt.predict(model,view.features,ids,images,guide)
        if not np.isfinite(logits).all():raise ValueError("Nonfinite donor-refinement output")
        weight,grid=select_weight(truth[lib],guide[:n],logits[:n])
        masks[name]=np.packbits((1-weight)*guide+weight*logits>0,axis=-1)
        choices[name]={"weight":weight,"grid":grid,"checkpoint":fit["checkpoint"]}
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":checked,"reference":previous["prediction"],
        "bank":bank["payload"],"choices":choices,"prediction":rc.hash_file(dst),"protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("TRANSFER_LOCK",args.fold,{k:v["weight"] for k,v in choices.items()},flush=True)


def score(args):
    seals=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    if any(r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz") for f,r in enumerate(seals)):
        raise ValueError("Both donor-study prediction seals are required")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Unregistered donor score order")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference",*MODES)}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores});print("TRANSFER_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("prepare","fit","lock","score"))
    for key in ("run","parent","ensemble","stable","formal","data","public","original"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0);parser.add_argument("--variant",choices=MODES,default="real")
    parser.add_argument("--smoke",action="store_true");parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=parser.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2)
    {"prepare":prepare,"fit":train,"lock":lock,"score":score}[args.phase](args)
