"""Adapt spatial SAM2 skip weights starting from the trained coarse decoder."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.prefix_utility import accepted_inner_guide

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/gated_hierarchy_protocol.md"
PARENT = "f994203a9fa7d2b207ae83850c9e31789451f90b8497937b2d27ec1e1f7e6ec1"
FEATURE_RECEIPT = "f088f22b5409d332791044849fcc0a898e5e88f37a64231f262cf00c790807d7"
SEED_OFFSET = 10000


class SpatialSkipGate(nn.Module):
    def __init__(self):
        super().__init__()
        # Adding zero-initialized parameters must not advance the paired sampling stream.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(4, 2, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features, guide, image):
        probability = guide.sigmoid().mean(1, keepdim=True)
        uncertainty = 4 * probability * (1-probability)
        low = F.avg_pool2d(image, 5, stride=1, padding=2, count_include_pad=False)
        padded = F.pad(image, (1,1,1,1), mode="replicate")
        dx = padded[:,:,1:-1,2:] - padded[:,:,1:-1,:-2]
        dy = padded[:,:,2:,1:-1] - padded[:,:,:-2,1:-1]
        gradient = torch.sqrt(dx.square()+dy.square())
        inputs = torch.cat((probability, uncertainty, (8*(image-low).abs()).clamp_max(1),
                            (4*gradient).clamp_max(1)), dim=1)
        gates = self.projection(inputs).tanh()
        return [feat * F.interpolate(gates[:,i:i+1], feat.shape[-2:], mode="bilinear", align_corners=False)
                for i,feat in enumerate(features)]


class GatedRanks(hierarchy.HierarchicalRanks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_gate = SpatialSkipGate()

    def spatial_features(self, embedding, guide, image):
        coarse, high = hierarchy.unpack(embedding, True)
        return coarse, self.spatial_gate(high, guide, image)


def verified_parent(args):
    if (rc.hash_file(args.parent/"pilot_summary.json") != PARENT
            or rc.hash_file(args.parent/"features.json") != FEATURE_RECEIPT):
        raise ValueError("The parent image-feature study changed")
    parent_args = copy.copy(args)
    parent_args.run = args.parent
    source = args.parent / "source/reproduction/hierarchical_sam.py"
    return hierarchy.AlreadyRotatedView(parent_args, producer=source)


def warm_state(args, training):
    folder = args.parent / f"fold-{args.fold}-coarse"
    fit = json.loads((folder/"fit.json").read_text())
    if (fit["steps"] != prompt.STEPS or fit["variant"] != "coarse" or fit["train_indices"] != sorted(training)
            or fit["view_receipt"] != FEATURE_RECEIPT or fit["public_weights"] != hierarchy.WEIGHT
            or fit["oof"] != rc.hash_file(args.crossfit/f"oof-{args.fold}.npz")
            or fit["checkpoint"] != rc.hash_file(folder/"final.pt")):
        raise ValueError("Warm coarse decoder identity differs")
    return torch.load(folder/"final.pt", weights_only=True, map_location="cpu"), fit["checkpoint"]


def train(args):
    _, _, _, uids = native.inputs(args.data)
    _, training, _ = native.splits(uids, args.fold)
    state, parent_hash = warm_state(args, training)
    view = verified_parent(args)
    cls = GatedRanks if args.variant == "gated" else hierarchy.CoarseRanks

    def initialize(folder, **kwargs):
        model = cls(folder, **kwargs)
        differences = model.load_state_dict(state, strict=False)
        missing = {"spatial_gate.projection.weight", "spatial_gate.projection.bias"} if cls is GatedRanks else set()
        if differences.unexpected_keys or set(differences.missing_keys) != missing:
            raise ValueError("Warm coarse decoder is incomplete")
        return model

    initialize.__name__ = cls.__name__
    prompt.train(args, decoder_type=initialize, variant=args.variant, study_protocol=PROTOCOL, view=view,
                 public_weights=hierarchy.WEIGHT, seed_offset=SEED_OFFSET)
    out = args.run / (f"smoke-{args.fold}" if args.smoke else f"fold-{args.fold}-{args.variant}")
    first = json.loads((out/"train.jsonl").read_text().splitlines()[0])["loss"]
    if args.variant == "gated":
        final = torch.load(out/"final.pt", weights_only=True, map_location="cpu")
        if final["spatial_gate.projection.weight"].abs().sum().item() == 0:
            raise ValueError("Spatial gates never learned")
        if not args.smoke:
            control = args.run / f"fold-{args.fold}-continued/train.jsonl"
            paired = json.loads(control.read_text().splitlines()[0])["loss"]
            if abs(paired-first) > 2e-5:
                raise ValueError(f"Initial paired losses differ: {paired} vs {first}")
    rc.write_json(out/"continuation.json", {"warm_checkpoint":parent_hash,"fit":rc.hash_file(out/"fit.json"),
                  "source":rc.hash_file(__file__),"first_loss":first,"seed_offset":SEED_OFFSET})


def check(args):
    images, truth, _, uids = native.inputs(args.data)
    _, training, valid = native.splits(uids, args.fold)
    group = type("Groups",(),{"series_uid":uids})()
    library, checked = polar.calibration_split(group,valid,args.fold)
    original, guide, previous, _ = accepted_inner_guide(args,library,checked)
    folder = args.run/f"fold-{args.fold}-{args.variant}"
    fit = json.loads((folder/"fit.json").read_text())
    warm = json.loads((folder/"continuation.json").read_text())
    _, expected = warm_state(args,training)
    cls = GatedRanks if args.variant=="gated" else hierarchy.CoarseRanks
    if (fit["steps"] != prompt.STEPS or fit["train_indices"] != sorted(training) or fit["variant"] != args.variant
            or fit["architecture"] != cls.__name__ or fit["scope"]!="inner" or fit["seed"]!=50171+args.fold+SEED_OFFSET
            or fit["public_weights"] != hierarchy.WEIGHT or fit["view_receipt"]!=FEATURE_RECEIPT
            or fit["protocol"]!=rc.hash_file(PROTOCOL) or fit["source"]!=rc.hash_file(prompt.__file__)
            or fit["checkpoint"]!=rc.hash_file(folder/"final.pt") or warm["fit"]!=rc.hash_file(folder/"fit.json")
            or warm["warm_checkpoint"]!=expected or warm["source"]!=rc.hash_file(__file__)):
        raise ValueError("Gated continuation identity differs")
    view = verified_parent(args)
    model = cls(args.public).cuda().eval()
    model.load_state_dict(torch.load(folder/"final.pt",map_location="cpu",weights_only=True))
    logits = prompt.predict(model,view.features,library+checked,images,guide)
    hierarchy.record_check(args,truth,library,checked,original,guide,previous,fit,logits,PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("fit","check"))
    for field in ("run","parent","data","public","crossfit","formal","encoder","ct","texture"):
        parser.add_argument("--"+field,type=Path,required=True)
    parser.add_argument("--fold",type=int,choices=range(4),required=True)
    parser.add_argument("--variant",choices=("continued","gated"),required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.set_defaults(scope="inner", image_size=512, backbone="sam2.1-tiny")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args = parser.parse_args()
    {"fit":train,"check":check}[args.phase](args)
