"""Replay the frozen formal A in its original Python source tree for formal B guidance."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from reproduction import outer_preference as prior

FROZEN_SOURCE = "0d14e0b2641678c899f97693081cb0f9e8cd4af6ac33809a3899b9291a400abd"
A_REPORT = "799311c1c176f2d8239c619e4c6f35a67fa0ec9300c60c3194cb87d97ade58a3"


def export(args):
    snapshot = args.run/"checked-source"
    if (Path(prior.__file__).resolve()!=snapshot/"reproduction/outer_preference.py"
            or prior.rc.hash_file(args.run/"checked-source.sha256")!=FROZEN_SOURCE
            or prior.rc.hash_file(args.run/"assessment/outer_report.json")!=A_REPORT):
        raise ValueError("The exact previous formal runtime is required")
    selection = json.loads((args.output/"selection.json").read_text())
    if selection["previous_report"]!=A_REPORT or selection["protocol"]!=prior.rc.hash_file(args.protocol):
        raise ValueError("Next formal lock differs")
    f = args.fold
    for key,name in prior.HISTORY.items():
        setattr(args,key,args.run.parent/name)
    args.scope="full";args.crossfit=args.run;args.previous=args.run;args.cache_run=args.encoder
    args.feature_cache=args.public;args.image_size=512;args.adapt=True
    choice=prior.locked(args)["folds"][f]
    args.ct_adapter=choice["variant"]=="ct-adapted"
    dst=args.output/f"guide-{f}.npz"
    receipt=dst.with_suffix(".json")
    if receipt.exists():
        r=prior.read(receipt)
        if r["payload"]!=prior.rc.hash_file(dst) or r["selection"]!=prior.rc.hash_file(args.output/"selection.json"):
            raise ValueError("Existing exported guide differs")
        return
    data,ids,_,_,_,raw,reference,shift=prior.incumbent.comparator(args,"full")
    split,_,_=prior.rc.split_for(data,f)
    source_folder=args.run/f"assessment/fold-{f}"
    previous=prior.verify_bank(source_folder,f,prior.rc.hash_file(args.run/"selection.json"))
    if ids!=split["test_indices"] or shift!=[-1,-1]:
        raise ValueError("Previous patient order or registration changed")
    old=prior.incumbent.anchor_logits(raw.astype(np.float16).astype(np.float32),reference)
    old_hash=hashlib.sha256(np.ascontiguousarray(old).tobytes()).hexdigest()
    if old_hash!=previous["guide_sha256"]:
        raise ValueError(f"Formal A input hash differs in fold {f}: {old_hash}")
    model,fit,choice=prior.check_fit(args,split["train_indices"])
    cache,_=prior.encoder.open_prefix(args)
    logits=prior.encoder.predict(model,cache,ids,np.stack(data.images),old)
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite historical logits")
    with np.load(source_folder/"predictions.npz",allow_pickle=False) as archive:
        if archive["ids"].tolist()!=ids:
            raise ValueError("Wrong historical prediction order")
        archived=np.unpackbits(archive["masks"],axis=-1).astype(bool)
    w=choice["weight"]
    replay=(1-w)*old+w*logits>0
    differences=int(np.count_nonzero(replay!=archived))
    if differences:
        raise ValueError(f"Historical formal replay differs in {differences} pixels in fold {f}")
    guide=prior.incumbent.anchor_logits((1-w)*old+w*logits.astype(np.float16).astype(np.float32),archived)
    partial=dst.with_suffix(".partial.npz")
    np.savez_compressed(partial,ids=ids,guide=guide,reference=np.packbits(archived,axis=-1))
    partial.replace(dst)
    prior.rc.write_json(receipt,{"fold":f,"cases":len(ids),"source_snapshot":FROZEN_SOURCE,
                        "previous_seal":prior.rc.hash_file(source_folder/"sealed.json"),
                        "previous_input":old_hash,"previous_model":fit["checkpoint"],"replay_different_pixels":differences,
                        "selection":prior.rc.hash_file(args.output/"selection.json"),"payload":prior.rc.hash_file(dst)})
    print("VERIFIED_PREVIOUS_GUIDE",f,len(ids),differences,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("run","output","data","protocol"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    export(parser.parse_args())
