"""Evaluate the saved seed models after independently reproducing the CT-only fields."""
import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from reproduction import canonical_guide as canonical
from reproduction import gated_hierarchy as public_cache
from reproduction import hierarchical_sam as hierarchy
from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours as polar
from reproduction import prompt_preference as prompt
from reproduction import rank_contours as rc
from reproduction.seed_ensemble import mean_saved_logits
from reproduction.trajectory_sam import select_weight

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/registered_guide_protocol.md"


def verified_field(root, scope, fold, protocol):
    stem = f"field-{scope}-{fold}"
    path = root / (stem + ".npz")
    record = json.loads((root / (stem + ".json")).read_text())
    replay = json.loads((root / f"replay-{scope}-{fold}.json").read_text())
    if (record["scope"] != scope or record["fold"] != fold or record["protocol"] != rc.hash_file(protocol)
            or rc.hash_file(path) != record["payload"] or replay != {**record, "byte_equal": True}):
        raise ValueError("Registered fields require a matching independent replay")
    return path, record


def lock(args):
    if rc.hash_file(args.data) != rc.DATA_HASH:
        raise ValueError("Wrong LIDC data")
    with args.data.open("rb") as stream:
        cases = list(pickle.load(stream).values())
    images = np.stack([c["image"] for c in cases])
    uids = [c["series_uid"] for c in cases]
    _, fitting, valid = native.splits(uids, args.fold)
    group = type("Groups", (), {"series_uid": uids})()
    library, checked = polar.calibration_split(group, valid, args.fold)
    ids = library + checked
    labels = np.stack([sorted(cases[i]["masks"], key=np.count_nonzero) for i in library]).astype(bool)
    protocol = PROTOCOL.with_name("stable_refinement_protocol.md") if args.stable_refinement else PROTOCOL
    path, field = verified_field(args.run, "inner", args.fold, protocol)
    seeds = canonical.model_sources(args, fitting, args.fold)
    view = public_cache.verified_parent(args)
    model = hierarchy.HierarchicalRanks(args.public).cuda().eval()
    choices, packed = {}, {}
    with np.load(path, allow_pickle=False) as bank:
        if bank["ids"].tolist() != ids:
            raise ValueError("Registered field library and check order differs")
        packed["reference"] = bank["reference"]
        original = np.unpackbits(packed["reference"], axis=-1).astype(bool)
        for name in ("adapted", "direct"):
            guide = bank[name]
            if not np.array_equal(guide > 0, original):
                raise ValueError("Previous binary reference signs differ")
            logits = []
            for directory, _ in seeds:
                model.load_state_dict(torch.load(directory/"final.pt", map_location="cpu", weights_only=True))
                logits.append(prompt.predict(model, view.features, ids, images, guide).astype(np.float16))
            average = mean_saved_logits(logits)
            if not np.isfinite(average).all():
                raise ValueError("Nonfinite registered prediction")
            n = len(library)
            weight, grid = select_weight(labels, guide[:n], average[:n])
            packed[name] = np.packbits((1-weight)*guide + weight*average > 0, axis=-1)
            choices[name] = {"weight": weight, "grid": grid}
    dst = args.run/f"pred-{args.fold}.npz"
    np.savez_compressed(dst, ids=ids, **packed)
    rc.write_json(args.run/f"selection-{args.fold}.json", {"fold": args.fold, "library": library, "check": checked,
        "choices": choices, "field": field["payload"], "checkpoints": [h for _, h in seeds],
        "source": rc.hash_file(__file__), "protocol": rc.hash_file(protocol), "prediction": rc.hash_file(dst)})
    print("REGISTERED_LOCK", args.fold, {k:v["weight"] for k,v in choices.items()}, flush=True)


def score(args):
    locks = [json.loads((args.run/f"selection-{f}.json").read_text()) for f in range(4)]
    for f, r in enumerate(locks):
        if r["fold"] != f or rc.hash_file(args.run/f"pred-{f}.npz") != r["prediction"]:
            raise ValueError("Incomplete registered inner seals")
    _, truth, _, _ = native.inputs(args.data)
    for f, r in enumerate(locks):
        with np.load(args.run/f"pred-{f}.npz", allow_pickle=False) as z:
            if z["ids"].tolist() != r["library"] + r["check"]:
                raise ValueError("Wrong registered IDs")
            scores = {name: float(metrics.dice(truth[r["check"]],
                np.unpackbits(z[name][len(r["library"]):], axis=-1).astype(bool)).mean())
                for name in ("reference", "adapted", "direct")}
        rc.write_json(args.run/f"inner-{f}.json", {**r, "scores": scores})
        print("REGISTERED_CHECK", f, scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("lock", "score"))
    for name in ("run", "parent", "ensemble", "data", "public"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--stable-refinement",action="store_true")
    parser.add_argument("--fold", type=int, choices=range(4), default=0)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    args = parser.parse_args()
    {"lock": lock, "score": score}[args.phase](args)
