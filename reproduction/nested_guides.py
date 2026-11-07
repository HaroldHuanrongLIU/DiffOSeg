"""Patient-excluded second-stage prompts for the existing LIDC single-slice pipeline."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import aligned_preference as align
from reproduction import canonical_guide
from reproduction import crossfit_contours as cf
from reproduction import crossfit_residual as prediction
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import mirror_views
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent
from reproduction.outer_preference_audit import mask_dice
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits
from reproduction.trajectory_sam import select_weight

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/nested_guides_protocol.md"


def read(path):
    return json.loads(path.read_text())


def patients(indices,uids):
    return {str(uids[i]).split("_")[0] for i in indices}


def split_groups(uids,fold):
    outer,fitting,valid=native.splits(uids,fold)
    groups=[held for _,held in cf.partitions(fitting,uids,fold)]
    if sorted(i for group in groups for i in group)!=sorted(fitting):
        raise ValueError("Incomplete nested patient groups")
    if any(patients(groups[i],uids)&patients(groups[j],uids) for i in range(3) for j in range(i)):
        raise ValueError("Nested groups must be patient-disjoint")
    if patients(fitting,uids)&patients(valid+outer["test_indices"],uids):
        raise ValueError("The original outer/validation patients must remain excluded")
    return fitting,valid,groups


def donors_for(held,groups):
    train=sorted(i for j,group in enumerate(groups) if j!=held for i in group)
    ownership={i:j for j,group in enumerate(groups) for i in group}
    donors=[3-held-ownership[i] for i in train]
    if any(k==held or i in groups[k] for i,k in zip(train,donors)):
        raise ValueError("A training-guide teacher includes an ineligible group")
    return train,donors


def verify_teacher(folder,used,held,excluded,source,sub,fold,single):
    plan,fit=read(folder/"plan.json"),read(folder/"fit.json")
    if (plan["fold"]!=fold or plan["sub"]!=sub or plan.get("scope","inner")!="inner"
            or fit["steps"]!=cf.STEPS or fit["source"]!=source or plan["fit_indices"]!=used or plan["held_indices"]!=held
            or plan.get("excluded_indices",plan.get("excluded_inner_validation"))!=excluded
            or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")
            or (single and (fit.get("role")!="single_patient_group" or fit["protocol"]!=rc.hash_file(PROTOCOL)))):
        raise ValueError("Teacher identity or patient exposure differs")
    return fit


def base_bank(args,uids,fold):
    fitting,valid,groups=split_groups(uids,fold)
    path=args.original/f"oof-{fold}.npz";record=read(path.with_suffix(".json"))
    if record["payload"]!=rc.hash_file(path):
        raise ValueError("Original OOF checksum changed")
    for j,held in enumerate(groups):
        held_set=set(held)
        used=[i for i in fitting if i not in held_set]
        fit=verify_teacher(args.original/f"fold-{fold}-sub-{j}",used,held,valid,
                           rc.hash_file(args.original/"source/reproduction/crossfit_contours.py"),j,fold,False)
        if fit["checkpoint"]!=record["teachers"][j] or patients(used,uids)&patients(held,uids):
            raise ValueError("Original teacher must exclude the held patient")
    with np.load(path,allow_pickle=False) as z:
        ids=z["ids"].tolist()
        if ids!=sorted(fitting):raise ValueError("Original OOF case order differs")
        logits=z["logits"].astype(np.float32)
    return ids,logits,record


def prepare(args):
    images,truth,_,uids=native.inputs(args.data)
    fitting,valid,groups=split_groups(uids,args.fold)
    base_bank(args,uids,args.fold)
    predictions=[]
    checks=[]
    for j,used in enumerate(groups):
        excluded_set=set(used)
        held=[i for i in fitting if i not in excluded_set]
        folder=args.run/f"singleton-{args.fold}-{j}"
        fit=verify_teacher(folder,used,held,valid,rc.hash_file(cf.__file__),j,args.fold,True)
        if patients(used,uids)&patients(held,uids):raise ValueError("Singleton training/hold overlap")
        values=prediction.teacher_logits(folder/"final.pt",images[held,None],fit["shift"]).astype(np.float16)
        if not np.isfinite(values).all():raise ValueError("Nonfinite first-level guide")
        predictions.append(dict(zip(held,values,strict=True)))
        checks.append(fit["checkpoint"])
    for held in range(3):
        ids,donors=donors_for(held,groups)
        if patients(ids,uids)&patients(groups[held],uids):
            raise ValueError("Second-level training patients overlap their held group")
        for i,k in zip(ids,donors,strict=True):
            if str(uids[i]).split("_")[0] in patients(groups[k],uids) or patients(groups[k],uids)&patients(groups[held],uids):
                raise ValueError("Indirect teacher exposure")
        shift,_=align.estimate_shift(images[ids],truth[ids],[str(uids[i]).split("_")[0] for i in ids])
        if tuple(shift)!=(-1,-1):
            raise ValueError("This subgroup requires its own registration-specific public image cache")
        directory=args.run/f"subbank-{args.fold}-{held}"
        directory.mkdir(exist_ok=False)
        path=directory/f"oof-{args.fold}.npz"
        logits=np.stack([predictions[k][i] for i,k in zip(ids,donors,strict=True)])
        np.savez_compressed(path,ids=ids,logits=logits,donors=np.array(donors,dtype=np.uint8))
        rc.write_json(path.with_suffix(".json"),{"fold":args.fold,"held_group":held,"excluded_group_ids":groups[held],
            "training":ids,"singleton_checkpoints":checks,"shift":list(shift),"payload":rc.hash_file(path),
            "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
        print("NESTED_BANK",args.fold,held,len(ids),len(groups[held]),flush=True)


def verify_subbank(args,uids,j):
    _,valid,groups=split_groups(uids,args.fold)
    ids,donors=donors_for(j,groups)
    folder=args.run/f"subbank-{args.fold}-{j}"
    path=folder/f"oof-{args.fold}.npz";rec=read(path.with_suffix(".json"))
    if (rec["payload"]!=rc.hash_file(path) or rec["training"]!=ids or rec["excluded_group_ids"]!=groups[j]
            or rec["source"]!=rc.hash_file(__file__) or rec["protocol"]!=rc.hash_file(PROTOCOL) or rec["shift"]!=[-1,-1]):
        raise ValueError("Nested training bank identity differs")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids or z["donors"].tolist()!=donors:
            raise ValueError("Nested donor assignment changed")
    fitting=cf.training_partition(uids,args.fold,"inner")[0]
    for k in set(donors):
        used=groups[k];used_set=set(used)
        fit=verify_teacher(args.run/f"singleton-{args.fold}-{k}",used,[i for i in fitting if i not in used_set],
                           valid,rc.hash_file(cf.__file__),k,args.fold,True)
        if fit["checkpoint"]!=rec["singleton_checkpoints"][k] or patients(used,uids)&patients(groups[j],uids):
            raise ValueError("A second-level teacher had indirect held-patient exposure")
    return folder,ids,rec


def intermediate(args):
    _,_,_,uids=native.inputs(args.data)
    folder,ids,rec=verify_subbank(args,uids,args.sub)
    opt=copy.copy(args);opt.crossfit=folder
    view=cache_owner.verified_parent(args)
    prompt.train(opt,decoder_type=hierarchy.HierarchicalRanks,variant=f"middle-{args.sub}",study_protocol=PROTOCOL,
                 view=view,public_weights=hierarchy.WEIGHT,seed_offset=105000+317*args.sub,fit_subset=ids)


def refine(args):
    images,_,_,uids=native.inputs(args.data)
    ids,base,brec=base_bank(args,uids,args.fold)
    fitting,_,groups=split_groups(uids,args.fold)
    views=cache_owner.verified_parent(args)
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    buffer=np.empty_like(base,dtype=np.float16)
    original_positions={i:k for k,i in enumerate(ids)}
    covered=set();teachers=[];subbanks=[]
    for j,held in enumerate(groups):
        folder,used,rec=verify_subbank(args,uids,j)
        directory=args.run/f"fold-{args.fold}-middle-{j}"
        fit=read(directory/"fit.json")
        if (fit["steps"]!=5000 or fit["scope"]!="inner" or fit["seed"]!=155171+args.fold+317*j or fit["train_indices"]!=used
                or fit["oof"]!=rec["payload"] or fit["source"]!=rc.hash_file(prompt.__file__) or fit["protocol"]!=rc.hash_file(PROTOCOL)
                or fit["variant"]!=f"middle-{j}" or fit["public_weights"]!=hierarchy.WEIGHT or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT
                or fit["checkpoint"]!=rc.hash_file(directory/"final.pt") or patients(used,uids)&patients(held,uids)
                or not set(held).issubset(fit["excluded_indices"]) or covered&set(held)):
            raise ValueError("Second-stage teacher identity or exclusion differs")
        model.load_state_dict(torch.load(directory/"final.pt",weights_only=True,map_location="cpu"))
        positions=np.array([original_positions[i] for i in held])
        predicted=prompt.predict(model,views.features,held,images,base[positions])
        buffer[positions]=(.5*base[positions]+.5*predicted).astype(np.float16)
        covered.update(held);teachers.append(fit["checkpoint"]);subbanks.append(rec["payload"])
    if covered!=set(fitting) or not np.isfinite(buffer).all():raise ValueError("Incomplete or nonfinite nested OOF")
    directory=args.run/"nested-training";directory.mkdir(exist_ok=True)
    dst=directory/f"oof-{args.fold}.npz"
    np.savez_compressed(dst,ids=ids,logits=buffer)
    rc.write_json(dst.with_suffix(".json"),{"fold":args.fold,"teachers":teachers,"subbanks":subbanks,"base":brec["payload"],
        "payload":rc.hash_file(dst),"source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("NESTED_OOF",args.fold,len(covered),rc.hash_file(dst),flush=True)


def quality(args):
    _,truth,_,uids=native.inputs(args.data)
    rows=[];count=0;max_error=0.
    for f in (0,1):
        ids,base,old=base_bank(args,uids,f)
        path=args.run/"nested-training"/f"oof-{f}.npz";info=read(path.with_suffix(".json"))
        if info["base"]!=old["payload"] or info["payload"]!=rc.hash_file(path):
            raise ValueError("Nested fitting-only OOF identity differs")
        with np.load(path,allow_pickle=False) as z:
            if z["ids"].tolist()!=ids:raise ValueError("Training OOF order differs")
            nested=z["logits"]>0
        scores={}
        for name,masks in (("base",base>0),("nested",nested)):
            independent=[]
            for i,m in zip(ids,masks,strict=True):
                independent.extend(mask_dice(x,y) for x,y in zip(m,truth[i],strict=True))
            mean=float(np.mean(independent));count+=len(independent)
            max_error=max(max_error,abs(mean-float(metrics.dice(truth[ids],masks).mean())))
            scores[name]=mean
        rows.append({"fold":f,"fitting_cases":len(ids),"scores":scores,"gain_pp":100*(scores["nested"]-scores["base"]),
                     "receipt":rc.hash_file(path.with_suffix(".json"))})
    if max_error>1e-12:raise ValueError("Independent fitting-only Dice differs")
    avg=float(np.mean([r["gain_pp"] for r in rows]))
    passed=all(r["gain_pp"]>0 for r in rows) and avg>=.05
    rc.write_json(args.run/"quality.json",{"scope":"OOF teacher diagnosis on fitting patients only","folds":rows,
        "mean_gain_pp":avg,"gate_passed":bool(passed),"rank_scores_checked":count,"maximum_error":max_error,
        "source":rc.hash_file(__file__),"protocol":rc.hash_file(PROTOCOL)})
    print("NESTED_QUALITY",avg,passed,rows,flush=True)


def gate(args):
    q=read(args.run/"quality.json")
    if (not q["gate_passed"] or q["mean_gain_pp"]<.05 or q["source"]!=rc.hash_file(__file__)
            or not all(r["gain_pp"]>0 for r in q["folds"])):
        raise ValueError("Nested training quality did not pass the registered gate")
    for r in q["folds"]:
        if r["receipt"]!=rc.hash_file(args.run/"nested-training"/f"oof-{r['fold']}.json"):
            raise ValueError("Nested quality bank changed")


def final_fit(args):
    gate(args)
    opt=copy.copy(args);opt.crossfit=(args.run/"nested-training") if args.variant=="nested" else args.original
    view=cache_owner.verified_parent(args)
    prompt.train(opt,decoder_type=hierarchy.HierarchicalRanks,variant=args.variant,study_protocol=PROTOCOL,
                 view=view,public_weights=hierarchy.WEIGHT,seed_offset=139000)


def e_guidance(args):
    if rc.hash_file(args.stable/"summary.json")!=mirror_views.E_INNER or rc.hash_file(args.formal/"assessment/outer_report.json")!=mirror_views.E_REPORT:
        raise ValueError("Formal E reference changed")
    images,truth,_,uids=native.inputs(args.data);_,fitting,valid=native.splits(uids,args.fold)
    group=type("Groups",(),{"series_uid":uids})();lib,check=polar.calibration_split(group,valid,args.fold)
    ids=lib+check;n=len(lib)
    field,_=verified_field(args.stable,"inner",args.fold,mirror_views.STABLE)
    with np.load(field,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("Stable field order differs")
        guide=z["adapted"]
    record=read(args.stable/f"selection-{args.fold}.json");fp=args.stable/f"pred-{args.fold}.npz"
    if record["prediction"]!=rc.hash_file(fp):raise ValueError("Original E masks differ")
    with np.load(fp,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:raise ValueError("E order differs")
        baseline=np.unpackbits(z["adapted"],axis=-1).astype(bool)
    view=cache_owner.verified_parent(args);model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    members=[]
    for directory,_ in canonical_guide.model_sources(args,fitting,args.fold):
        model.load_state_dict(torch.load(directory/"final.pt",weights_only=True,map_location="cpu"))
        members.append(prompt.predict(model,view.features,ids,images,guide).astype(np.float16))
    weight=record["choices"]["adapted"]["weight"]
    newguide=incumbent.anchor_logits((1-weight)*guide+weight*mean_saved_logits(members),baseline)
    return images,truth,fitting,lib,check,newguide,baseline,record,view,model


def lock(args):
    gate(args)
    images,truth,fitting,lib,check,newguide,baseline,record,view,model=e_guidance(args)
    ids=lib+check;n=len(lib)
    arrays={"reference":np.packbits(baseline,axis=-1)};choices={}
    for name in ("original","nested"):
        folder=args.run/f"fold-{args.fold}-{name}";fit=read(folder/"fit.json")
        owner=args.original if name=="original" else args.run/"nested-training"
        if (fit["train_indices"]!=sorted(fitting) or fit["scope"]!="inner" or fit["variant"]!=name or fit["steps"]!=5000
                or fit["seed"]!=189171+args.fold or fit["oof"]!=rc.hash_file(owner/f"oof-{args.fold}.npz")
                or fit["public_weights"]!=hierarchy.WEIGHT or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT
                or fit["source"]!=rc.hash_file(prompt.__file__) or fit["protocol"]!=rc.hash_file(PROTOCOL)
                or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
            raise ValueError("Final student identity differs")
        model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
        logits=prompt.predict(model,view.features,ids,images,newguide)
        if not np.isfinite(logits).all():raise ValueError("Nonfinite nested prompt prediction")
        w,grid=select_weight(truth[lib],newguide[:n],logits[:n])
        arrays[name]=np.packbits((1-w)*newguide+w*logits>0,axis=-1)
        choices[name]={"weight":w,"grid":grid,"checkpoint":fit["checkpoint"]}
    dst=args.run/f"pred-{args.fold}.npz";np.savez_compressed(dst,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":lib,"check":check,
        "reference":record["prediction"],"choices":choices,"quality":rc.hash_file(args.run/"quality.json"),
        "prediction":rc.hash_file(dst),"source":rc.hash_file(__file__)})
    print("NESTED_LOCK",args.fold,{k:v["weight"] for k,v in choices.items()},flush=True)


def score(args):
    seals=[read(args.run/f"selection-{f}.json") for f in (0,1)]
    for f,r in enumerate(seals):
        if r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz"):
            raise ValueError("Both nested predictions must be sealed")
    _,truth,_,_=native.inputs(args.data)
    for f,r in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+r["check"]:raise ValueError("Wrong nested case order")
            scores={k:float(metrics.dice(truth[r["check"]],np.unpackbits(z[k][len(r["library"]):],axis=-1).astype(bool)).mean())
                    for k in ("reference","original","nested")}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores});print("NESTED_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("prepare","middle","refine","quality","fit","lock","score"))
    for key in ("run","parent","ensemble","formal","stable","data","public","original"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0);parser.add_argument("--sub",type=int,choices=range(3),default=0)
    parser.add_argument("--variant",choices=("original","nested"),default="original");parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=parser.parse_args();cv2.setNumThreads(1);torch.set_num_threads(2)
    {"prepare":prepare,"middle":intermediate,"refine":refine,"quality":quality,"fit":final_fit,"lock":lock,"score":score}[args.phase](args)
