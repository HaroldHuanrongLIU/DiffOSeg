"""Evaluate a bounded component and hole control on frozen single-slice predictions."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import texture_study as reference

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/topology_protocol.md"


def clean_mask(mask, fraction, hole_limit):
    result = mask.astype(np.uint8).copy()
    if fraction > 0 and result.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        sizes = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(count, dtype=bool)
        keep[1:] = sizes >= fraction * sizes.max()
        result = keep[labels].astype(np.uint8)
    if hole_limit > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(1-result, connectivity=8)
        outside = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        fill = stats[:, cv2.CC_STAT_AREA] <= hole_limit
        fill[0] = False
        fill[outside] = False
        result[fill[labels]] = 1
    return result.astype(bool)


def run(args):
    _, truth, _, uids = native.inputs(args.data)
    _, _, valid = native.splits(uids, args.fold)
    group = type("Groups", (), {"series_uid": uids})()
    library, checked = polar.calibration_split(group, valid, args.fold)
    ids = library + checked
    path = args.texture / f"features-{args.fold}-inner.npz"
    meta = reference.read(path.with_suffix(".json"))
    if meta["indices"] != ids or meta["payload"] != rc.hash_file(path):
        raise ValueError("Frozen prediction identity differs")
    with np.load(path, allow_pickle=False) as archive:
        masks = np.unpackbits(archive["old"], axis=-1).astype(bool)
    candidates, grid = [], []
    for fraction in (0, 0.05, 0.15, 1):
        for limit in (0, 4, 16, 16384):
            result = np.stack([[clean_mask(mask, fraction, limit) for mask in case] for case in masks])
            candidates.append(result)
            grid.append({"component_fraction": fraction, "hole_limit": limit,
                         "library_score": float(scoring.dice(truth[library], result[:len(library)]).mean())})
    selected = int(np.argmax([row["library_score"] for row in grid]))
    choice = {"fold": args.fold, "source": rc.hash_file(__file__), "protocol": rc.hash_file(PROTOCOL),
              "library": library, "grid": grid, "selected": grid[selected], "comparator": meta["comparator"]}
    rc.write_json(args.run / f"selection-{args.fold}.json", choice)
    scores = [float(scoring.dice(truth[checked], mask[len(library):]).mean()) for mask in candidates]
    payload = args.run / f"pred-inner-{args.fold}.npz"
    np.savez_compressed(payload, ids=ids, masks=np.packbits(candidates[selected], axis=-1))
    rc.write_json(args.run / f"inner-{args.fold}.json", {
        **choice, "check_indices": checked, "all_check_scores": scores,
        "check": {"reference": scores[0], "candidate": scores[selected]},
        "prediction": rc.hash_file(payload),
    })
    print("TOPOLOGY_CHECK", args.fold, grid[selected], scores[0], scores[selected], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "data", "texture"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    cv2.setNumThreads(1)
    run(parser.parse_args())
