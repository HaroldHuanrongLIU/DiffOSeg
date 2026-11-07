"""Evaluate a three-seed image-only SAM2 ensemble against the accepted four-fold method."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import gated_hierarchy as parent
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.prefix_utility import accepted_inner_guide
from reproduction.trajectory_sam import select_weight

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/seed_ensemble_protocol.md"
OFFSETS = (0,17000,34000)


def mean_saved_logits(logits):
    if len(logits)!=len(OFFSETS) or any(z.dtype!=np.float16 for z in logits):
        raise ValueError("The ensemble requires three packed half-precision logit arrays")
    return np.mean(logits,axis=0,dtype=np.float32)


def train(args):
    if args.fold<2 and args.seed_index==0:
        raise ValueError("The registered first seed uses the frozen original fit in folds 0 and 1")
    view = parent.verified_parent(args)
    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=f"seed-{args.seed_index}",
                 study_protocol=PROTOCOL,view=view,public_weights=hierarchy.WEIGHT,seed_offset=OFFSETS[args.seed_index])


def archive_member(args, ids, library, checked):
    path = args.parent/f"inner-{args.fold}-pyramid.json"
    rec = json.loads(path.read_text())
    audit = json.loads((args.parent/"pilot_summary.json").read_text())["rows"][args.fold]["pyramid"]
    bank = args.parent/f"pred-{args.fold}-pyramid.npz"
    if (rc.hash_file(path)!=audit["receipt"] or rec["library"]!=library or rec["check"]!=checked
            or rc.hash_file(bank)!=rec["prediction"]):
        raise ValueError("Historical first-seed result changed")
    with np.load(bank,allow_pickle=False) as z:
        if z["ids"].tolist()!=ids:
            raise ValueError("Historical first-seed population differs")
        logits = z["logits"]
        masks = z["masks"]
    identity={"weight":rec["weight"],"grid":rec["grid"],"checkpoint":rec["checkpoint"],"receipt":rc.hash_file(path)}
    return logits,masks,identity


def lock(args):
    images,truth,_,uids=native.inputs(args.data)
    _,training,valid=native.splits(uids,args.fold)
    groups=type("Groups",(),{"series_uid":uids})()
    library,checked=polar.calibration_split(groups,valid,args.fold)
    ids=library+checked
    n=len(library)
    original,guide,previous,_=accepted_inner_guide(args,library,checked)
    view=parent.verified_parent(args)
    model=hierarchy.HierarchicalRanks(args.public).cuda().eval()
    arrays={"reference":np.packbits(original,axis=-1)}
    choices={}
    for i,offset in enumerate(OFFSETS):
        name=f"seed-{i}"
        if i==0 and args.fold<2:
            z,masks,identity=archive_member(args,ids,library,checked)
        else:
            folder=args.run/f"fold-{args.fold}-{name}"
            fit=json.loads((folder/"fit.json").read_text())
            if (fit["train_indices"]!=sorted(training) or fit["steps"]!=prompt.STEPS or fit["scope"]!="inner"
                    or fit["seed"]!=50171+args.fold+offset or fit["variant"]!=name
                    or fit["architecture"]!=hierarchy.HierarchicalRanks.__name__
                    or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["source"]!=rc.hash_file(prompt.__file__)
                    or fit["public_weights"]!=hierarchy.WEIGHT or fit["view_receipt"]!=parent.FEATURE_RECEIPT
                    or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                    or fit["checkpoint"]!=rc.hash_file(folder/"final.pt")):
                raise ValueError("Independent seed training identity differs")
            model.load_state_dict(torch.load(folder/"final.pt",weights_only=True,map_location="cpu"))
            values=prompt.predict(model,view.features,ids,images,guide)
            weight,grid=select_weight(truth[library],guide[:n],values[:n])
            masks=np.packbits((1-weight)*guide+weight*values>0,axis=-1)
            z=values.astype(np.float16)
            identity={"weight":weight,"grid":grid,"checkpoint":fit["checkpoint"],"receipt":rc.hash_file(folder/"fit.json")}
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite member logits")
        arrays[name]=masks
        arrays["logit_"+name]=z
        choices[name]=identity
    mean=mean_saved_logits([arrays[f"logit_seed-{i}"] for i in range(3)])
    weight,grid=select_weight(truth[library],guide[:n],mean[:n])
    choices["ensemble"]={"weight":weight,"grid":grid}
    arrays["ensemble"]=np.packbits((1-weight)*guide+weight*mean>0,axis=-1)
    bank=args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(bank,ids=ids,**arrays)
    rc.write_json(args.run/f"selection-{args.fold}.json",{"fold":args.fold,"library":library,"check":checked,
                  "choices":choices,"previous":previous["prediction"],"prediction":rc.hash_file(bank),
                  "protocol":rc.hash_file(PROTOCOL),"source":rc.hash_file(__file__)})
    print("SEED_LOCK",args.fold,{k:v["weight"] for k,v in choices.items()},flush=True)


def score(args):
    locks=[json.loads((args.run/f"selection-{f}.json").read_text()) for f in range(4)]
    for f,r in enumerate(locks):
        if (r["fold"]!=f or r["prediction"]!=rc.hash_file(args.run/f"pred-{f}.npz")
                or r["protocol"]!=rc.hash_file(PROTOCOL)):
            raise ValueError("Incomplete four-fold ensemble locks")
    _,truth,_,_=native.inputs(args.data)
    for r in locks:
        f=r["fold"];n=len(r["library"]);checked=r["check"]
        with np.load(args.run/f"pred-{f}.npz",allow_pickle=False) as z:
            if z["ids"].tolist()!=r["library"]+checked:
                raise ValueError("Ensemble prediction order changed")
            scores={name:float(metrics.dice(truth[checked],np.unpackbits(z[name][n:],axis=-1).astype(bool)).mean())
                    for name in ("reference","seed-0","seed-1","seed-2","ensemble")}
        rc.write_json(args.run/f"inner-{f}.json",{**r,"scores":scores})
        print("SEED_CHECK",f,scores,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","lock","score"))
    for field in ("run","parent","data","public","crossfit","formal","encoder","ct","texture"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),default=0)
    parser.add_argument("--seed-index",type=int,choices=range(3),default=0)
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args=parser.parse_args()
    {"fit":train,"lock":lock,"score":score}[args.phase](args)
