"""Compare five registered EMA checkpoints with the final and library-selected model."""
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

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/trajectory_sam_protocol.md"
TIMES = (1000,2000,3000,4000,5000)
WEIGHTS = (0.,0.25,0.5,0.75,1.)


def select_weight(truth, guide, logits):
    grid = [float(metrics.dice(truth,(1-w)*guide+w*logits>0).mean()) for w in WEIGHTS]
    return WEIGHTS[int(np.argmax(grid))], grid


def final_identity(folder, step):
    last = torch.load(folder/f"ema-{step}.pt",map_location="cpu",weights_only=True)
    final = torch.load(folder/"final.pt",map_location="cpu",weights_only=True)
    if last.keys()!=final.keys() or not all(torch.equal(last[key],final[key]) for key in final):
        raise ValueError("Last trajectory snapshot differs from the final model")


def train(args):
    view = parent.verified_parent(args)
    times = (1,5) if args.smoke else TIMES
    prompt.train(args, decoder_type=hierarchy.HierarchicalRanks, variant="trajectory", study_protocol=PROTOCOL,
                 view=view, public_weights=hierarchy.WEIGHT, snapshot_steps=times)
    folder = args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-trajectory")
    final_identity(folder,times[-1])
    if not args.smoke:
        with (args.parent/f"fold-{args.fold}-pyramid/train.jsonl").open() as stream:
            old_first = json.loads(stream.readline())["loss"]
        with (folder/"train.jsonl").open() as stream:
            first = json.loads(stream.readline())["loss"]
        if abs(first-old_first)>2e-5:
            raise ValueError(f"First loss of the repeated trajectory differs: {first} vs {old_first}")
    rc.write_json(folder/"snapshots.json",{"times":list(times),"fit":rc.hash_file(folder/"fit.json"),
                  "models":{str(t):rc.hash_file(folder/f"ema-{t}.pt") for t in times},
                  "last_equals_final":True,"source":rc.hash_file(__file__)})


def check(args):
    images, truth, _, uids = native.inputs(args.data)
    _, training, valid = native.splits(uids,args.fold)
    group = type("Groups",(),{"series_uid":uids})()
    library, checked = polar.calibration_split(group,valid,args.fold)
    ids = library + checked
    original, guide, previous, _ = accepted_inner_guide(args,library,checked)
    folder = args.run/f"fold-{args.fold}-trajectory"
    fit = json.loads((folder/"fit.json").read_text())
    trail = json.loads((folder/"snapshots.json").read_text())
    if (fit["steps"] != prompt.STEPS or fit["seed"] != 50171+args.fold or fit["scope"]!="inner"
            or fit["train_indices"] != sorted(training) or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
            or fit["variant"] != "trajectory" or fit["architecture"]!=hierarchy.HierarchicalRanks.__name__
            or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["source"]!=rc.hash_file(prompt.__file__)
            or fit["public_weights"]!=hierarchy.WEIGHT or fit["view_receipt"]!=parent.FEATURE_RECEIPT
            or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or trail["fit"]!=rc.hash_file(folder/"fit.json")
            or trail["source"]!=rc.hash_file(__file__) or tuple(trail["times"])!=TIMES):
        raise ValueError("Trajectory training identity differs")
    final_identity(folder,TIMES[-1])
    n = len(library)
    view = parent.verified_parent(args)
    model = hierarchy.HierarchicalRanks(args.public).cuda().eval()
    stages = {}
    for t in TIMES:
        file = folder/f"ema-{t}.pt"
        if trail["models"][str(t)] != rc.hash_file(file):
            raise ValueError("Trajectory checkpoint changed")
        model.load_state_dict(torch.load(file,map_location="cpu",weights_only=True))
        stages[str(t)] = prompt.predict(model,view.features,ids,images,guide)
    stages["average"] = sum(stages.values()) / len(TIMES)
    grids, weights = {}, {}
    for name,logits in stages.items():
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite trajectory prediction")
        weights[name],grids[name] = select_weight(truth[library],guide[:n],logits[:n])
    early = str(TIMES[int(np.argmax([max(grids[str(t)]) for t in TIMES]))])
    lock = {"fold":args.fold,"library":library,"weights":weights,"grids":grids,"early":early,
            "protocol":rc.hash_file(PROTOCOL),"previous":previous["prediction"],
            "snapshots":rc.hash_file(folder/"snapshots.json")}
    rc.write_json(args.run/f"selection-{args.fold}.json",lock)
    scores, packed = {}, {}
    for name,logits in stages.items():
        weight = weights[name]
        masks = (1-weight)*guide+weight*logits>0
        scores[name] = {"standalone":float(metrics.dice(truth[checked],logits[n:]>0).mean()),
                        "candidate":float(metrics.dice(truth[checked],masks[n:]).mean())}
        packed[name] = np.packbits(masks,axis=-1)
    payload = args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(payload,ids=ids,reference=np.packbits(original,axis=-1),**packed)
    reference = float(metrics.dice(truth[checked],original[n:]).mean())
    rc.write_json(args.run/f"inner-{args.fold}.json",{**lock,"check":checked,"scores":scores,
                  "reference":reference,"prediction":rc.hash_file(payload)})
    print("TRAJECTORY_CHECK",args.fold,"reference",reference,"final",scores["5000"],"mean",scores["average"],
          "early",early,scores[early],flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","check"))
    for field in ("run","parent","data","public","crossfit","formal","encoder","ct","texture"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    args = parser.parse_args()
    {"fit":train,"check":check}[args.phase](args)
