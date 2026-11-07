"""Compare two supervised orientations with a matched contour-consistency loss."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import seed_pair_study as pair_study
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/reflection_consistency_protocol.md"
MIRROR_SUMMARY = "3993851279626255464632cd4e39390a7aedbe4de1e2f8a4a634ef42fe7dec2e"


def reflect_rotated(value, rotation):
    """Apply a horizontal flip in native coordinates to an already rotated tensor."""
    return value.flip([-1 if rotation % 2 == 0 else -2])


def boundary_loss(left, right, labels):
    outer = F.max_pool2d(labels, 7, 1, 3)
    inner = -F.max_pool2d(-labels, 7, 1, 3)
    band = outer > inner
    difference = left.float().sigmoid() - right.float().sigmoid()
    return ((band * difference.square()).sum((-2,-1)) / band.sum((-2,-1)).clamp_min(1)).mean()


def reflected_features(args):
    if rc.hash_file(args.mirror/"summary.json") != MIRROR_SUMMARY:
        raise ValueError("Mirror study identity differs")
    owner = copy.copy(args)
    owner.run = args.mirror
    return hierarchy.feature_cache(owner, producer=args.mirror/"source/reproduction/hierarchical_sam.py", reflected=True)


def train(args):
    view = cache_owner.verified_parent(args)
    path = args.crossfit / f"oof-{args.fold}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["payload"] != rc.hash_file(path):
        raise ValueError("Out-of-fit guide archive changed")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
    if args.smoke:
        ids = ids[:32]
    reflected = torch.from_numpy(reflected_features(args)[ids].copy()).cuda()
    name = args.variant
    directory = args.run/(f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{name}")
    mirror_hash = rc.hash_file(args.mirror/"features.json")

    def paired(model, embedding, guide, image, labels, ix, rotation, step):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
            left, raw_left = model(embedding, guide, image)
            right, raw_right = model(reflected[ix,rotation].float(),reflect_rotated(guide,rotation),
                                     reflect_rotated(image,rotation))
        right = reflect_rotated(right, rotation)
        raw_right = reflect_rotated(raw_right, rotation)
        supervision = (prompt.segmentation_loss(left, raw_left, labels) +
                       prompt.segmentation_loss(right, raw_right, labels)) * 0.5
        consistency = boundary_loss(left, right, labels)
        if step == 1:
            first = {"indices":[ids[j] for j in ix.tolist()], "rotation":rotation,
                     "supervised_loss":supervision.item(), "consistency":consistency.item()}
            if args.smoke:
                gradients = torch.autograd.grad(consistency, (left, right), retain_graph=True)
                if not all(g.abs().sum().item()>0 for g in gradients):
                    raise ValueError("The consistency loss must act on both image views")
            elif name == "consistent":
                control = json.loads((args.run/f"fold-{args.fold}-paired/first.json").read_text())
                if (first["indices"]!=control["indices"] or first["rotation"]!=control["rotation"]
                        or not np.isclose(first["supervised_loss"],control["supervised_loss"],rtol=2e-4,atol=4e-5)):
                    raise ValueError("The initial supervision and training batch must match the control")
            rc.write_json(directory/"first.json",first)
        coefficient = min(1.,(step-1)/499) if name=="consistent" else 0.
        return supervision + coefficient*consistency

    prompt.train(args,decoder_type=hierarchy.HierarchicalRanks,variant=name,study_protocol=PROTOCOL,
                 view=view,public_weights=hierarchy.WEIGHT,seed_offset=60000,paired_loss=paired)
    rc.write_json(directory/"pair.json",{"fit":rc.hash_file(directory/"fit.json"),
                  "mirrored_features":mirror_hash,"source":rc.hash_file(__file__)})


def lock(args):
    reflected=reflected_features(args)
    def member(model,original,ids,images,guide,name,training):
        directory=args.run/f"fold-{args.fold}-{name}"
        fit=json.loads((directory/"fit.json").read_text())
        pair=json.loads((directory/"pair.json").read_text())
        if (fit["steps"]!=5000 or fit["seed"]!=110171+args.fold or fit["scope"]!="inner" or fit["variant"]!=name
                or fit["train_indices"]!=sorted(training) or fit["oof"]!=rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
                or fit["view_receipt"]!=cache_owner.FEATURE_RECEIPT or fit["public_weights"]!=hierarchy.WEIGHT
                or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["source"]!=rc.hash_file(prompt.__file__)
                or fit["checkpoint"]!=rc.hash_file(directory/"final.pt") or pair["fit"]!=rc.hash_file(directory/"fit.json")
                or pair["mirrored_features"]!=rc.hash_file(args.mirror/"features.json") or pair["source"]!=rc.hash_file(__file__)):
            raise ValueError("Paired model identity differs")
        model.load_state_dict(torch.load(directory/"final.pt",map_location="cpu",weights_only=True))
        left=prompt.predict(model,original,ids,images,guide)
        right=prompt.predict(model,reflected,ids,images,guide,reflected=True)
        single=(left+right)*.5
        return single,fit["checkpoint"]
    pair_study.lock(args,("paired","consistent"),PROTOCOL,__file__,member)


def score(args):
    pair_study.score(args,("paired","consistent"))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("fit","lock","score"))
    for name in ("run","parent","ensemble","formal","stable","mirror","data","public","crossfit"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=(0,1),default=0)
    parser.add_argument("--variant",choices=("paired","consistent"),default="paired")
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner",image_size=512,backbone="sam2.1-tiny")
    args=parser.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
    {"fit":train,"lock":lock,"score":score}[args.phase](args)
