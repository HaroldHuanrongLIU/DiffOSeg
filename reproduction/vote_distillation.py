"""Distill patient-excluded rank probabilities constrained by the training vote count."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit
import torch
from torch.nn import functional as F

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import nested_guides as nested
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import OFFSETS,mean_saved_logits

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/vote_distillation_protocol.md"
QUALITY="3f536c2278d9bcaabe3a2db374d26c1d27b3964a1551fdb40d58ce536f5312d6"
MODES=("uniform","weak","nested")
TEMPERATURE=2.


def projected_targets(logits,labels):
    votes=labels.sum(0)
    target=np.broadcast_to((votes/4)[None],labels.shape).copy().astype(np.float32)
    uncertain=(votes>0)&(votes<4)
    if uncertain.any():
        z=logits[:,uncertain].astype(np.float64).clip(-12,12)/TEMPERATURE
        k=votes[uncertain]
        low=np.full(k.shape,-18.);high=np.full(k.shape,18.)
        for _ in range(25):
            mid=(low+high)/2
            excess=expit(z+mid).sum(0)>k
            high=np.where(excess,mid,high);low=np.where(excess,low,mid)
        target[:,uncertain]=expit(z+(low+high)/2).astype(np.float32)
    return target


def vote_kl(logits,target,labels):
    z=logits.float()/TEMPERATURE
    ce=F.binary_cross_entropy_with_logits(z,target,reduction="none")
    entropy_term=torch.xlogy(target,target)+torch.xlogy(1-target,1-target)
    votes=labels.sum(1,keepdim=True);ambiguous=(votes>0)&(votes<4)
    return TEMPERATURE**2*(((ce+entropy_term)*ambiguous).sum((-2,-1))/ambiguous.sum((-2,-1)).clamp_min(1)).mean()


def prepare(args):
    _,truth,_,uids=native.inputs(args.data)
    ids,original,base=nested.base_bank(args,uids,args.fold)
    if rc.hash_file(args.nested/"independent_quality.json")!=QUALITY:raise ValueError("Nested patient audit changed")
    new=args.nested/"nested-training"/f"oof-{args.fold}.npz";rec=json.loads(new.with_suffix(".json").read_text())
    if rec["payload"]!=rc.hash_file(new) or rec["base"]!=base["payload"]:raise ValueError("Nested OOF identity differs")
    with np.load(new,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong projected-target cases")
        stronger=z["logits"]
    arrays={};errors={};briers={}
    for name,teacher in (("weak",original),("nested",stronger)):
        targets=np.stack([projected_targets(logits,truth[i]) for logits,i in zip(teacher,ids,strict=True)])
        votes=truth[ids].sum(1);max_error=float(np.abs(targets.sum(1)-votes).max())
        if max_error>2e-6 or not np.isfinite(targets).all():raise ValueError("Teacher probabilities violate the four-reader vote")
        arrays[name]=targets;errors[name]=max_error
        amb=(votes>0)&(votes<4)
        sq=np.square(targets-truth[ids])
        per=(sq*amb[:,None]).sum((1,2,3))/(4*amb.sum((1,2)).clip(1))
        briers[name]=float(per[amb.any((1,2))].mean())
    path=args.run/f"targets-{args.fold}.npz";np.savez_compressed(path,ids=ids,**arrays)
    rc.write_json(path.with_suffix(".json"),{"fold":args.fold,"ids":ids,"base":base["payload"],"nested":rec["payload"],
        "quality":QUALITY,"brier":briers,"vote_error":errors,"payload":rc.hash_file(path),"protocol":rc.hash_file(PROTOCOL),
        "source":rc.hash_file(__file__)})
    print("VOTE_TARGETS",args.fold,briers,errors,flush=True)


def train(args):
    folder=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.variant}-{args.seed_index}")
    path=args.run/f"targets-{args.fold}.npz";meta=json.loads(path.with_suffix(".json").read_text())
    if meta["payload"]!=rc.hash_file(path) or meta["base"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz"):
        raise ValueError("Clinical target or OOF input changed")
    with np.load(path,allow_pickle=False) as z:
        ids=z["ids"].tolist()
        if ids!=meta["ids"]:raise ValueError("Wrong projected-target order")
        take=32 if args.smoke else len(ids)
        target=None if args.variant=="uniform" else torch.from_numpy(z[args.variant][:take].copy()).cuda()
    view=cache_owner.verified_parent(args)

    def objective(model,embedding,guide,image,labels,ix,k,step):
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            logits,raw=model(embedding,guide,image)
        probability=(labels.mean(1,keepdim=True).expand_as(labels) if target is None else
                     torch.rot90(target[ix],k,(-2,-1)))
        standard=prompt.segmentation_loss(logits,raw,labels)
        extra=vote_kl(logits,probability,labels)
        if step==1:
            first={"ids":[ids[i] for i in ix.tolist()],"rotation":k,"supervised":standard.item(),"vote_kl":extra.item()}
            if args.smoke:
                grad=torch.autograd.grad(extra,logits,retain_graph=True)[0]
                if not torch.isfinite(grad).all() or not grad.abs().sum().item():raise ValueError("Missing vote-distillation gradient")
            else:
                owner=(args.parent/f"fold-{args.fold}-pyramid" if args.seed_index==0 and args.fold<2 else
                       args.ensemble/f"fold-{args.fold}-seed-{args.seed_index}")
                with (owner/"train.jsonl").open() as stream:previous=json.loads(stream.readline())["loss"]
                if not np.isclose(first["supervised"],previous,rtol=2e-4,atol=4e-5):
                    raise ValueError(f"The unchanged first supervised loss differs: {first['supervised']} vs {previous}")
                if args.variant!="uniform":
                    control=json.loads((args.run/f"fold-{args.fold}-uniform-{args.seed_index}/first.json").read_text())
                    if first["ids"]!=control["ids"] or k!=control["rotation"]:raise ValueError("Unmatched initial batch")
            rc.write_json(folder/"first.json",first)
        coefficient=.30*min(1.,(step-1)/299)
        return standard+coefficient*extra

    name=f"{args.variant}-{args.seed_index}"
    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=name,study_protocol=PROTOCOL,view=view,
                 public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index],paired_loss=objective)
    rc.write_json(folder/"distill.json",{"target":meta["payload"],"fit":rc.hash_file(folder/"fit.json"),
                  "source":rc.hash_file(__file__),"temperature":TEMPERATURE})


def lock(args):
    if (rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or
            rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT):
        raise ValueError("Formal E identity differs")
    images,_,_,uids=native.inputs(args.data);_,training,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})();lib,check=polar.calibration_split(group,valid,args.fold)
    ids=lib+check
    field,fr=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(field,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Wrong deterministic field order")
        guide=z["adapted"]
    rec=json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    file=args.stable/f"pred-{args.fold}.npz"
    if rec["prediction"]!=rc.hash_file(file) or rec["field"]!=fr["payload"]:raise ValueError("Previous masks changed")
    with np.load(file,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Previous E order differs")
        masks={"reference":z["adapted"]}
    w=rec["choices"]["adapted"]["weight"];view=cache_owner.verified_parent(args)
    target=json.loads((args.run/f"targets-{args.fold}.json").read_text())
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval();identities={}
    for name in MODES:
        copies=[];weights=[]
        for i,offset in enumerate(OFFSETS):
            variant=f"{name}-{i}";folder=args.run/f"fold-{args.fold}-{variant}"
            fit=json.loads((folder/"fit.json").read_text());extra=json.loads((folder/"distill.json").read_text())
            if (fit["steps"]!=5000 or fit["seed"]!=50171+args.fold+offset or fit["scope"]!="inner"
                    or fit["variant"]!=variant or fit["train_indices"]!=sorted(training) or fit["oof"]!=target["base"]
                    or fit["public_weights"]!=hierarchy.WEIGHT or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT
                    or fit["source"]!=rc.hash_file(prompt.__file__) or fit["protocol"]!=rc.hash_file(PROTOCOL)
                    or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or extra["fit"]!=rc.hash_file(folder/"fit.json")
                    or extra["target"]!=target["payload"] or extra["source"]!=rc.hash_file(__file__)):
                raise ValueError("Distilled seed identity differs")
            model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
            copies.append(prompt.predict(model,view.features,ids,images,guide).astype(np.float16));weights.append(fit["checkpoint"])
        mean=mean_saved_logits(copies)
        if not np.isfinite(mean).all():raise ValueError("Nonfinite distilled ensemble")
        masks[name]=np.packbits((1-w)*guide+w*mean>0,axis=-1);identities[name]=weights
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**masks)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,"weight":w,
        "reference":rec["prediction"],"targets":target["payload"],"members":identities,"prediction":rc.hash_file(dst),
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("VOTE_DISTILL_LOCK",args.fold,w,flush=True)


def score(args):
    seals=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in (0,1)]
    if any(r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz") for f,r in enumerate(seals)):
        raise ValueError("Incomplete vote-distillation seals")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Distillation patient order differs")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference",*MODES)}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores});print("VOTE_DISTILL_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("prepare","fit","lock","score"))
    for key in ("run","parent","ensemble","formal","stable","nested","data","public","crossfit","original"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0)
    parser.add_argument("--variant",choices=MODES,default="uniform");parser.add_argument("--seed-index",type=int,choices=range(3),default=0)
    parser.add_argument("--smoke",action="store_true");parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    {"prepare":prepare,"fit":train,"lock":lock,"score":score}[args.phase](args)
