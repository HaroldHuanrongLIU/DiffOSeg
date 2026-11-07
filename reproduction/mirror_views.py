"""Test dihedral averaging of the three fixed formal-E image decoders."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import canonical_guide as canonical
from reproduction import gated_hierarchy as cache_owner
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.registered_seed_study import verified_field
from reproduction.seed_ensemble import mean_saved_logits

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/mirror_views_protocol.md"
STABLE = PROTOCOL.with_name("stable_refinement_protocol.md")
E_REPORT = "7131bf880b3fb533bfea48172a2de6e2901697198b74f9c8367a51af1cb19c44"
E_AUDIT = "a69d4606c50ee03f08ca399cc1d36bab12f2c6e16fc15cf8056f9ad90bae757f"
E_INNER = "5f30f6c9500bd196c4f4f9cf8dbe8b0153f27bd8135d88254d1b079d6c9f189f"


def paired_predictions(models, ordinary, reflected, ids, images, guide, weight):
    views = {"rotations": [], "mirror": [], "eight": []}
    for model in models:
        regular = prompt.predict(model, ordinary, ids, images, guide)
        mirrored = prompt.predict(model, reflected, ids, images, guide, reflected=True)
        if not np.isfinite(regular).all() or not np.isfinite(mirrored).all():
            raise ValueError("Nonfinite orientation prediction")
        views["rotations"].append(regular.astype(np.float16))
        views["mirror"].append(mirrored.astype(np.float16))
        views["eight"].append(((regular + mirrored) * 0.5).astype(np.float16))
    return {name: np.packbits((1-weight)*guide + weight*mean_saved_logits(values) > 0, axis=-1)
            for name, values in views.items()}


def lock(args):
    if (rc.hash_file(args.formal/"assessment/outer_report.json") != E_REPORT or
            rc.hash_file(args.formal/"assessment/independent_audit.json") != E_AUDIT or
            rc.hash_file(args.stable/"summary.json") != E_INNER or rc.hash_file(args.data) != rc.DATA_HASH):
        raise ValueError("Formal E or LIDC identity changed")
    with args.data.open("rb") as stream:
        examples = list(pickle.load(stream).values())
    images = np.stack([case["image"] for case in examples])
    uids = [case["series_uid"] for case in examples]
    _, fitting, valid = native.splits(uids, args.fold)
    group = type("Groups", (), {"series_uid": uids})()
    library, checked = polar.calibration_split(group, valid, args.fold)
    ids = library + checked
    path, field = verified_field(args.stable, "inner", args.fold, STABLE)
    with np.load(path, allow_pickle=False) as bank:
        if bank["ids"].tolist() != ids:
            raise ValueError("Guide is not in the registered patient order")
        guide = bank["adapted"]
        if not np.array_equal(guide > 0, np.unpackbits(bank["reference"], axis=-1)):
            raise ValueError("Invalid guide signs")
    record = json.loads((args.stable/f"selection-{args.fold}.json").read_text())
    reference_path = args.stable/f"pred-{args.fold}.npz"
    if (record["prediction"] != rc.hash_file(reference_path) or record["field"] != field["payload"] or
            record["library"] != library or record["check"] != checked):
        raise ValueError("Frozen E inner selection differs")
    with np.load(reference_path, allow_pickle=False) as bank:
        if bank["ids"].tolist() != ids:
            raise ValueError("Archived E patient order differs")
        reference = bank["adapted"]
    weight = record["choices"]["adapted"]["weight"]
    with (args.formal/"selection.json").open() as stream:
        if weight != json.load(stream)["folds"][args.fold]["weight"]:
            raise ValueError("Inner and formal E use different weights")
    seeds = canonical.model_sources(args, fitting, args.fold)
    if [sha for _,sha in seeds] != record["checkpoints"]:
        raise ValueError("Wrong seed checkpoints")
    ordinary = cache_owner.verified_parent(args).features
    reflected = hierarchy.feature_cache(args, reflected=True)
    def members():
        model = hierarchy.HierarchicalRanks(args.public).cuda().eval()
        for folder, _ in seeds:
            model.load_state_dict(torch.load(folder/"final.pt", map_location="cpu", weights_only=True))
            yield model
    masks = paired_predictions(members(), ordinary, reflected, ids, images, guide, weight)
    payload = args.run/f"pred-{args.fold}.npz"
    temporary = payload.with_suffix(".partial.npz")
    np.savez_compressed(temporary, ids=ids, reference=reference, **masks)
    temporary.replace(payload)
    rc.write_json(args.run/f"selection-{args.fold}.json", {"fold": args.fold, "library": library,
        "check": checked, "weight": weight, "members": record["checkpoints"], "reference": record["prediction"],
        "field": field["payload"], "mirror_cache": rc.hash_file(args.run/"features.json"),
        "prediction": rc.hash_file(payload), "source": rc.hash_file(__file__), "protocol": rc.hash_file(PROTOCOL)})
    print("MIRROR_LOCK", args.fold, len(checked), weight, flush=True)


def score(args):
    seals = [json.loads((args.run/f"selection-{f}.json").read_text()) for f in range(4)]
    for f, seal in enumerate(seals):
        if seal["fold"] != f or rc.hash_file(args.run/f"pred-{f}.npz") != seal["prediction"]:
            raise ValueError("Incomplete mirror seals")
    _, truth, _, _ = native.inputs(args.data)
    for f, seal in enumerate(seals):
        with np.load(args.run/f"pred-{f}.npz", allow_pickle=False) as bank:
            if bank["ids"].tolist() != seal["library"]+seal["check"]:
                raise ValueError("Changed case order")
            scores = {}
            for name in ("reference","rotations","mirror","eight"):
                masks = np.unpackbits(bank[name][len(seal["library"]):], axis=-1).astype(bool)
                scores[name] = float(metrics.dice(truth[seal["check"]], masks).mean())
        rc.write_json(args.run/f"inner-{f}.json", {**seal, "scores": scores})
        print("MIRROR_CHECK", f, scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("encode","lock","score"))
    for name in ("run","parent","ensemble","formal","stable","data","public","align"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.phase == "encode":
        hierarchy.encode(args, reflected=True)
    else:
        {"lock":lock,"score":score}[args.phase](args)
