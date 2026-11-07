"""Locked original patient four-fold evaluation of the independent SAM2 seed ensemble."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.seed_ensemble import OFFSETS, mean_saved_logits

PROTOCOL=Path(__file__).resolve().parents[1]/"docs/lidc_91/outer_ensemble_protocol.md"
INNER="1eb13a03183e78b509bc51d537564300ed5b601f1c60201d06f0f954e771422b"
PREVIOUS="799311c1c176f2d8239c619e4c6f35a67fa0ec9300c60c3194cb87d97ade58a3"
E11="d27211bd93bc8eb2eccd728bd9a04840dfb0e7864ec0d15cd7826c782edeb2a0"


def read(path):
    return json.loads(path.read_text())


def selection(args):
    if (rc.hash_file(args.data)!=rc.DATA_HASH or rc.hash_file(args.inner/"summary.json")!=INNER
            or rc.hash_file(args.crossfit/"assessment/outer_report.json")!=PREVIOUS
            or rc.hash_file(args.base/"fourfold_summary.json")!=E11):
        raise ValueError("Formal identities differ")
    summary=read(args.inner/"summary.json")
    if not summary["gate_passed"] or not all(r["gain_reference_pp"]>0 for r in summary["folds"][2:]):
        raise ValueError("The registered replication gate did not pass")
    folds=[]
    for f,row in enumerate(summary["folds"]):
        lockpath=args.inner/f"selection-{f}.json";lock=read(lockpath)
        if (row["fold"]!=f or rc.hash_file(args.inner/f"inner-{f}.json")!=row["receipt"]
                or row["weights"]["ensemble"]!=lock["choices"]["ensemble"]["weight"]):
            raise ValueError("Inner fold-local weight identity changed")
        folds.append({"fold":f,"weight":row["weights"]["ensemble"],"seeds":[50171+f+s for s in OFFSETS],
                      "inner_lock":rc.hash_file(lockpath)})
    return {"target":0.91,"dataset":rc.DATA_HASH,"previous_report":PREVIOUS,"inner":INNER,"e11":E11,
            "public_weights":hierarchy.WEIGHT,"feature_receipt":cache_owner.FEATURE_RECEIPT,
            "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__),"folds":folds}


def lock(args):
    record=selection(args)
    path=args.run/"selection.json"
    if path.exists():
        if read(path)!=record:raise ValueError("Existing formal selection differs")
    else:rc.write_json(path,record)
    print("FORMAL_B_LOCKED",[(r["fold"],r["weight"],r["seeds"]) for r in record["folds"]],flush=True)


def fixed(args):
    record=read(args.run/"selection.json")
    if record!=selection(args):raise ValueError("Formal B lock changed")
    return record


def inputs(args):
    if rc.hash_file(args.data)!=rc.DATA_HASH:raise ValueError("Wrong LIDC data")
    with args.data.open("rb") as stream:cases=list(pickle.load(stream).values())
    return np.stack([case["image"] for case in cases]),[case["series_uid"] for case in cases]


def split_for(args,uids,fold):
    stage=args.base/f"fold_{fold}/lidc/fold_{fold}/stage_2"
    expected=read(stage/"split.json")
    split,_,_=native.splits(uids,fold)
    if any(expected[k]!=value for k,value in split.items()):raise ValueError("Original patient split changed")
    return split,stage


def training_bank(args,split,uids):
    f=args.fold; path=args.crossfit/f"oof-{f}.npz";info=read(path.with_suffix(".json"))
    if info["scope"]!="full" or info["fold"]!=f or info["payload"]!=rc.hash_file(path):
        raise ValueError("Full training-only OOF guide differs")
    with np.load(path,allow_pickle=False) as z:
        if z["ids"].tolist()!=sorted(split["train_indices"]):raise ValueError("OOF patient scope changed")
    train=set(split["train_indices"]);covered=set()
    patient=lambda ids:{str(uids[i]).split("_")[0] for i in ids}
    for sub in range(3):
        folder=args.crossfit/f"fold-{f}-sub-{sub}"
        plan=read(folder/"plan.json");fit=read(folder/"fit.json")
        held=set(plan["held_indices"]);used=set(plan["fit_indices"])
        if (plan["scope"]!="full" or plan["fold"]!=f or plan["sub"]!=sub or held|used!=train
                or patient(held)&patient(used) or covered&held or plan["excluded_indices"]!=split["test_indices"]
                or fit["checkpoint"]!=info["teachers"][sub] or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
            raise ValueError("Teacher exposure or identity differs")
        covered.update(held)
    if covered!=train:raise ValueError("Incomplete held-patient teacher bank")
    return info


def train(args):
    fixed(args)
    _,uids=inputs(args);split,_=split_for(args,uids,args.fold)
    training_bank(args,split,uids)
    view=cache_owner.verified_parent(args)
    label=f"seed-{args.seed_index}"
    folder=args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{label}")
    if (folder/"fit.json").exists():
        fit=read(folder/"fit.json")
        expected=sorted(split["train_indices"])[:32] if args.smoke else sorted(split["train_indices"])
        if (fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or fit["train_indices"]!=expected
                or fit["seed"]!=50171+args.fold+OFFSETS[args.seed_index] or fit["variant"]!=label
                or fit["steps"]!=(5 if args.smoke else prompt.STEPS) or fit["protocol"]!=rc.hash_file(PROTOCOL)
                or fit["scope"]!="full" or fit["public_weights"]!=hierarchy.WEIGHT
                or fit["source"]!=rc.hash_file(prompt.__file__) or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")):
            raise ValueError("Previously finished full fit differs")
        return
    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=label,study_protocol=PROTOCOL,
                 view=view,public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index])


def verify_seal(args,f):
    folder=args.run/f"assessment/fold-{f}"
    seal=read(folder/"sealed.json")
    if (seal["fold"]!=f or seal["selection"]!=rc.hash_file(args.run/"selection.json")
            or seal["source"]!=rc.hash_file(__file__) or seal["payload"]!=rc.hash_file(folder/"predictions.npz")):
        raise ValueError("Prediction seal changed")
    return seal


def infer(args):
    record=fixed(args);f=args.fold
    folder=args.run/f"assessment/fold-{f}"
    if (folder/"sealed.json").exists():
        verify_seal(args,f)
        return
    images,uids=inputs(args);split,_=split_for(args,uids,f)
    bank=training_bank(args,split,uids)
    guide_path=args.run/f"guide-{f}.npz";identity=read(guide_path.with_suffix(".json"))
    if (identity["fold"]!=f or identity["selection"]!=rc.hash_file(args.run/"selection.json")
            or identity["payload"]!=rc.hash_file(guide_path) or identity["replay_different_pixels"]!=0):
        raise ValueError("Replayed previous guide identity differs")
    with np.load(guide_path,allow_pickle=False) as z:
        ids=z["ids"].tolist();guide=z["guide"];reference=z["reference"]
    if ids!=split["test_indices"] or guide.shape!=(len(ids),4,128,128):
        raise ValueError("Wrong inference scope")
    view=cache_owner.verified_parent(args)
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    logits=[];checkpoints=[]
    for i,seed in enumerate(record["folds"][f]["seeds"]):
        name=f"seed-{i}";trained=args.run/f"fold-{f}-{name}";fit=read(trained/"fit.json")
        if (fit["steps"]!=prompt.STEPS or fit["seed"]!=seed or fit["variant"]!=name or fit["scope"]!="full"
                or fit["architecture"]!=hierarchy.HierarchicalRanks.__name__ or fit["train_indices"]!=sorted(split["train_indices"])
                or fit["excluded_indices"]!=split["test_indices"] or fit["source"]!=rc.hash_file(prompt.__file__)
                or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["public_weights"]!=hierarchy.WEIGHT
                or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT or fit["oof"]!=bank["payload"]
                or fit["checkpoint"]!=rc.hash_file(trained/"final.pt")):
            raise ValueError("Full seed model identity differs")
        model.load_state_dict(torch.load(trained/"final.pt",weights_only=True,map_location="cpu"))
        z=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
        if not np.isfinite(z).all():raise ValueError("Nonfinite outer seed logits")
        logits.append(z);checkpoints.append(fit["checkpoint"])
    weight=record["folds"][f]["weight"]
    final=(1-weight)*guide+weight*mean_saved_logits(logits)>0
    folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/"predictions.partial.npz";path=folder/"predictions.npz"
    np.savez_compressed(tmp,ids=ids,masks=np.packbits(final,axis=-1),reference=reference)
    tmp.replace(path)
    rc.write_json(folder/"sealed.json",{"fold":f,"cases":len(ids),"weight":weight,"checkpoints":checkpoints,
                  "guide":identity["payload"],"payload":rc.hash_file(path),"selection":rc.hash_file(args.run/"selection.json"),
                  "source":rc.hash_file(__file__)})
    print("FORMAL_B_SEALED",f,len(ids),flush=True)


def score(args):
    fixed(args)
    seals=[verify_seal(args,f) for f in range(4)]
    score_sealed(args,seals,rc.hash_file(__file__))


def score_sealed(args,seals,source_digest,method="formal B"):
    if len(seals)!=4:
        raise ValueError("All four seals are required before scoring")
    _,truth,_,uids=native.inputs(args.data)
    folds=[];bootstrap=[];covered=set();patients=set()
    rng=np.random.default_rng(9102337)
    names=("e11","reference","candidate")
    for f,seal in enumerate(seals):
        split,stage=split_for(args,uids,f)
        folder=args.run/f"assessment/fold-{f}"
        with np.load(folder/"predictions.npz",allow_pickle=False) as z:
            ids=z["ids"].tolist();pred=np.unpackbits(z["masks"],axis=-1).astype(bool)
            previous=np.unpackbits(z["reference"],axis=-1).astype(bool)
        if ids!=split["test_indices"] or len(ids)!=seal["cases"]:raise ValueError("Wrong sealed order")
        with np.load(args.crossfit/f"assessment/fold-{f}/predictions.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=ids or not np.array_equal(previous,np.unpackbits(z["masks"],axis=-1)):
                raise ValueError("Formal A comparator changed")
        e11=[]
        for j,i in enumerate(ids):
            with np.load(stage/f"cases/{j:05d}_masks.npz",allow_pickle=False) as z:
                e11.append(np.unpackbits(z["predictions"])[:4*128*128].reshape(4,128,128).astype(bool))
        values=np.stack([metrics.dice(truth[ids],m) for m in (np.stack(e11),previous,pred)],axis=1)
        group,inverse=np.unique([str(uids[i]).split("_")[0] for i in ids],return_inverse=True)
        if set(group)!=set(split["test_patients"]) or patients&set(group) or covered&set(ids):
            raise ValueError("Patients or cases repeat")
        patients.update(group);covered.update(ids)
        rows=[{"index":i,"patient":str(uids[i]).split("_")[0],
               **{name:values[j,k].tolist() for k,name in enumerate(names)}} for j,i in enumerate(ids)]
        rc.write_json(folder/"paired.json",{"seal":rc.hash_file(folder/"sealed.json"),"rows":rows})
        means=values.mean((0,2))
        folds.append({"fold":f,"cases":len(ids),"patients":len(group),"means":dict(zip(names,means.tolist())),
                      "rank_means":dict(zip(names,values.mean(0).tolist()))})
        sums=np.zeros((len(group),3));np.add.at(sums,inverse,values.mean(2))
        counts=np.bincount(inverse);w=rng.multinomial(len(group),np.full(len(group),1/len(group)),size=5000)
        bootstrap.append((w@sums)/(w@counts)[:,None])
    if len(patients)!=214 or covered!=set(range(1609)):raise ValueError("Incomplete LIDC coverage")
    means={k:float(np.mean([f["means"][k] for f in folds])) for k in names}
    prior=read(args.crossfit/"assessment/outer_report.json")["means"]
    np.testing.assert_allclose([means["e11"],means["reference"]],[prior["e11"],prior["candidate"]],atol=1e-12,rtol=0)
    draws=np.mean(bootstrap,axis=0)
    report={"folds":folds,"patients":len(patients),"cases":len(covered),"means":means,
            "delta_vs_reference_pp":100*(means["candidate"]-means["reference"]),
            "delta_vs_e11_pp":100*(means["candidate"]-means["e11"]),
            "absolute_ci95":np.percentile(draws[:,2],[2.5,97.5]).tolist(),
            "paired_reference_ci95_pp":np.percentile(100*(draws[:,2]-draws[:,1]),[2.5,97.5]).tolist(),
            "paired_e11_ci95_pp":np.percentile(100*(draws[:,2]-draws[:,0]),[2.5,97.5]).tolist(),
            "target":0.91,"target_pass":means["candidate"]>=0.91,"selection":rc.hash_file(args.run/"selection.json"),
            "source":source_digest,"method":method,"scope":"sealed original LIDC four-fold repeated-cohort assessment"}
    rc.write_json(args.run/"assessment/outer_report.json",report)
    print("ORIGINAL_FOURFOLD_RESULT",method,means,report["target_pass"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("lock","fit","infer","score"))
    for field in ("run","inner","parent","data","public","crossfit","base"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    parser.add_argument("--seed-index",type=int,choices=range(3),default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="full",image_size=512,backbone="sam2.1-tiny")
    cv2.setNumThreads(1);torch.set_num_threads(2)
    args=parser.parse_args()
    {"lock":lock,"fit":train,"infer":infer,"score":score}[args.phase](args)
