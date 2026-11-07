"""Seal the original four folds for deterministic CT refinement and the fixed seed ensemble."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import outer_seed_ensemble as original
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import OFFSETS, mean_saved_logits

PROTOCOL = Path(__file__).resolve().parents[1]/"docs/lidc_91/outer_stable_refinement_protocol.md"
FIELD_PROTOCOL = PROTOCOL.with_name("stable_refinement_protocol.md")
INNER = "5f30f6c9500bd196c4f4f9cf8dbe8b0153f27bd8135d88254d1b079d6c9f189f"
WEIGHT_LOCK = "1c319f7c7302abcb87f8468a61f73511b635d101348e6e87db73baa2d639b270"
WEIGHT_SOURCE = "bd17fe40e947db131b29a0df0c6dd0b78dea84e0fbbaa72ec7f07f4f6cff7826"


def selection(args):
    if (rc.hash_file(args.data)!=rc.DATA_HASH or rc.hash_file(args.inner/"summary.json")!=INNER
            or rc.hash_file(args.crossfit/"assessment/outer_report.json")!=original.PREVIOUS
            or rc.hash_file(args.base/"fourfold_summary.json")!=original.E11
            or rc.hash_file(args.weights/"selection.json")!=WEIGHT_LOCK
            or rc.hash_file(args.weights/"source.sha256")!=WEIGHT_SOURCE):
        raise ValueError("Formal data, prior run or training provenance differs")
    if rc.hash_file(args.inner/"source/reproduction/registered_seed_field.py") != rc.hash_file(
            Path(__file__).with_name("registered_seed_field.py")):
        raise ValueError("Full and inner field implementations differ")
    summary=original.read(args.inner/"summary.json")
    if (not summary["gate_passed"] or summary["gain_pp"]<.05 or not all(r["gain_pp"]>0 for r in summary["folds"][2:])):
        raise ValueError("The prespecified inner replication gate did not pass")
    _,uids=original.inputs(args)
    folds=[]
    for f,row in enumerate(summary["folds"]):
        choice=original.read(args.inner/f"selection-{f}.json");w=row["weights"]["adapted"]
        if (row["fold"]!=f or rc.hash_file(args.inner/f"inner-{f}.json")!=row["receipt"]
                or choice["choices"]["adapted"]["weight"]!=w):
            raise ValueError("Inner fold-local fusion weight differs")
        verified_field(args.inner,"inner",f,FIELD_PROTOCOL)
        split,_=original.split_for(args,uids,f)
        args.fold=f
        bank=original.training_bank(args,split,uids)
        members=[]
        for i,offset in enumerate(OFFSETS):
            folder=args.weights/f"fold-{f}-seed-{i}";fit=original.read(folder/"fit.json")
            if (fit["train_indices"]!=sorted(split["train_indices"]) or fit["excluded_indices"]!=split["test_indices"]
                    or fit["scope"]!="full" or fit["variant"]!=f"seed-{i}" or fit["steps"]!=5000
                    or fit["seed"]!=50171+f+offset or fit["oof"]!=bank["payload"]
                    or fit["architecture"]!=hierarchy.HierarchicalRanks.__name__ or fit["public_weights"]!=hierarchy.WEIGHT
                    or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")
                    or fit["source"]!=rc.hash_file(args.weights/"source/reproduction/prompt_preference.py")
                    or fit["protocol"]!=rc.hash_file(args.weights/"source/docs/lidc_91/outer_ensemble_protocol.md")):
                raise ValueError("Outer seed identity, patient scope or bank changed")
            members.append({"seed":fit["seed"],"checkpoint":fit["checkpoint"],"receipt":rc.hash_file(folder/"fit.json")})
        folds.append({"fold":f,"weight":w,"members":members,"inner_selection":rc.hash_file(args.inner/f"selection-{f}.json")})
    return {"method":"formal E deterministic CPU refinement and three SAM2 seeds","target":.91,"dataset":rc.DATA_HASH,
            "previous_report":original.PREVIOUS,"e11":original.E11,"inner":INNER,"training_source":WEIGHT_SOURCE,
            "training_lock":WEIGHT_LOCK,"field_protocol":rc.hash_file(FIELD_PROTOCOL),"protocol":rc.hash_file(PROTOCOL),
            "field_source":rc.hash_file(Path(__file__).with_name("registered_seed_field.py")),
            "source":rc.hash_file(__file__),"folds":folds}


def fixed(args):
    fold=args.fold
    try:
        record=selection(args)
    finally:
        args.fold=fold
    path=args.run/"selection.json"
    if path.exists():
        if original.read(path)!=record:raise ValueError("Formal E lock changed")
    elif args.phase=="lock":
        rc.write_json(path,record)
    else:
        raise ValueError("Formal E must be locked before inference")
    return record


def lock(args):
    record=fixed(args)
    print("FORMAL_E_LOCK",[(r["fold"],r["weight"]) for r in record["folds"]],rc.hash_file(args.run/"selection.json"),flush=True)


def verify_seal(args,f):
    folder=args.run/f"assessment/fold-{f}";seal=original.read(folder/"sealed.json")
    _,field=verified_field(args.run,"full",f,FIELD_PROTOCOL)
    if (seal["fold"]!=f or seal["selection"]!=rc.hash_file(args.run/"selection.json")
            or seal["source"]!=rc.hash_file(__file__) or seal["payload"]!=rc.hash_file(folder/"predictions.npz")
            or seal["guide"]!=field["payload"] or seal["guide_file"]!=f"field-full-{f}.npz"):
        raise ValueError("Formal E seal changed")
    return seal


def infer(args):
    record=fixed(args);f=args.fold
    folder=args.run/f"assessment/fold-{f}"
    if (folder/"sealed.json").exists():
        verify_seal(args,f)
        return
    images,uids=original.inputs(args)
    split,_=original.split_for(args,uids,f)
    path,field=verified_field(args.run,"full",f,FIELD_PROTOCOL)
    if field["source"]!=record["field_source"]:
        raise ValueError("Full field implementation differs from registered inner study")
    with np.load(path,allow_pickle=False) as z:
        ids=z["ids"].tolist();guide=z["adapted"];reference=z["reference"]
    if ids!=split["test_indices"] or not np.array_equal(guide>0,np.unpackbits(reference,axis=-1)):
        raise ValueError("Wrong full guide or patient scope")
    previous=args.crossfit/f"assessment/fold-{f}"
    if original.read(previous/"sealed.json")["payload"]!=rc.hash_file(previous/"predictions.npz"):
        raise ValueError("Historical A seal changed")
    with np.load(previous/"predictions.npz",allow_pickle=False) as z:
        if z["ids"].tolist()!=ids or not np.array_equal(reference,z["masks"]):
            raise ValueError("Canonical previous binary reference differs")
    view=cache_owner.verified_parent(args)
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    outputs=[]
    for i,m in enumerate(record["folds"][f]["members"]):
        file=args.weights/f"fold-{f}-seed-{i}/final.pt"
        if rc.hash_file(file)!=m["checkpoint"]:raise ValueError("Changed formal seed checkpoint")
        model.load_state_dict(torch.load(file,map_location="cpu",weights_only=True))
        logits=prompt.predict(model,view.features,ids,images,guide).astype(np.float16)
        if not np.isfinite(logits).all():raise ValueError("Nonfinite formal E output")
        outputs.append(logits)
    w=record["folds"][f]["weight"]
    masks=(1-w)*guide+w*mean_saved_logits(outputs)>0
    folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/"predictions.partial.npz";dst=folder/"predictions.npz"
    np.savez_compressed(tmp, ids=ids, masks=np.packbits(masks,axis=-1), reference=reference)
    tmp.replace(dst)
    rc.write_json(folder/"sealed.json",{"fold":f,"cases":len(ids),"weight":w,"guide_file":path.name,"guide":field["payload"],
                  "members":record["folds"][f]["members"],"selection":rc.hash_file(args.run/"selection.json"),
                  "source":rc.hash_file(__file__),"payload":rc.hash_file(dst)})
    print("FORMAL_E_SEALED",f,len(ids),flush=True)


def score(args):
    fixed(args)
    seals=[verify_seal(args,f) for f in range(4)]
    original.score_sealed(args,seals,rc.hash_file(__file__),"formal E deterministic CPU refinement and three SAM2 seeds")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("lock","infer","score"))
    for name in ("run","weights","inner","parent","data","public","crossfit","base"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    args=parser.parse_args();{"lock":lock,"infer":infer,"score":score}[args.phase](args)
