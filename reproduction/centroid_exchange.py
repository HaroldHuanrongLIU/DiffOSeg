"""Constrain the average binary centroid by exchanging uncertain boundary pixels."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from reproduction import native_local as native
from reproduction import native_study as metrics
from reproduction import polar_contours
from reproduction import rank_contours as rc
from reproduction.prefix_utility import accepted_inner_guide

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/centroid_exchange_protocol.md"
BUDGETS = (0, 1, 2, 4, 8, 16)
STRENGTHS = (0, 0.25, 0.5, 0.75, 1.0)
NEIGHBORS = ndimage.generate_binary_structure(2, 1)


def mean_centroid(masks):
    points = [np.argwhere(mask) for mask in masks]
    centers = [region.mean(0) for region in points if len(region)]
    return np.mean(centers, axis=0) if centers else np.array([np.nan, np.nan])


def distance_squared(center, target):
    return np.square(np.maximum(np.abs(center - target) - 0.5, 0.0)).sum(axis=-1)


def boundary_candidates(mask, scores):
    edge_in = mask & ~ndimage.binary_erosion(mask, structure=NEIGHBORS)
    edge_out = ndimage.binary_dilation(mask, structure=NEIGHBORS) & ~mask

    def cheapest(edge):
        coords = np.argwhere(edge)
        values = np.abs(scores[edge])
        order = np.argsort(values, kind="stable")[:24]
        return coords[order], values[order]

    return cheapest(edge_in), cheapest(edge_out)


def exchange_path(scores, target, weighted):
    current = scores > 0
    areas = current.sum(axis=(1, 2))
    center = mean_centroid(current)
    snapshots = [current.copy()]
    step_counts = [0]
    movable = areas.min() > 0 and np.max(np.abs(center - target)) < 3
    applied = 0
    for step in range(1, max(BUDGETS) + 1):
        if not movable or distance_squared(center, target) < 1e-12:
            break
        best_gain = 0.0
        move = None
        for rank, (mask, size) in enumerate(zip(current, areas)):
            (remove, remove_cost), (add, add_cost) = boundary_candidates(mask, scores[rank])
            if not len(remove) or not len(add):
                continue
            centers = center + (add[None, :, :] - remove[:, None, :]) / (4 * size)
            gain = distance_squared(center, target) - distance_squared(centers, target)
            if weighted:
                gain /= 1 + remove_cost[:, None] + add_cost[None, :]
            ri, ai = np.unravel_index(gain.argmax(), gain.shape)
            if gain[ri, ai] > best_gain + 1e-12:
                best_gain = gain[ri, ai]
                move = rank, tuple(remove[ri]), tuple(add[ai]), centers[ri, ai]
        if move is None:
            break
        rank, drop, take, center = move
        current[rank][drop] = False
        current[rank][take] = True
        applied = step
        if step in BUDGETS:
            snapshots.append(current.copy())
            step_counts.append(step)
    # If the path ends between registered budgets, every later budget uses the terminal mask.
    while len(snapshots) < len(BUDGETS):
        snapshots.append(current.copy())
        step_counts.append(applied)
    return np.stack(snapshots), np.asarray(step_counts)


def run(args):
    _, truth, _, uids = native.inputs(args.data)
    _, training, valid = native.splits(uids, args.fold)
    groups = type("Groups", (), {"series_uid": uids})()
    library, checked = polar_contours.calibration_split(groups, valid, args.fold)
    original, guide, receipt, variant = accepted_inner_guide(args, library, checked)
    fitting_centers = np.stack([mean_centroid(truth[i]) for i in training])
    target = np.rint(2 * np.nanmedian(fitting_centers, axis=0)) / 2
    n = len(library)
    lock = {"fold": args.fold, "target_yx": target.tolist(), "training": training, "library": library,
            "protocol": rc.hash_file(PROTOCOL), "source": rc.hash_file(__file__),
            "previous_variant": variant, "previous_prediction": receipt["prediction"], "choices": {}}
    masks = {"reference": original}
    log_counts = {}
    for label in ("weighted", "unweighted"):
        paths, counts = zip(*(exchange_path(z, target, label == "weighted") for z in guide))
        path = np.stack(paths)
        grid = [float(metrics.dice(truth[library], path[:n, j]).mean()) for j in range(len(BUDGETS))]
        j = int(np.argmax(grid))
        chosen = path[:, j]
        if not np.array_equal(chosen.sum((-1, -2)), original.sum((-1, -2))):
            raise ValueError("Exchange changed a rank area")
        masks[label] = chosen
        lock["choices"][label] = {"budget": BUDGETS[j], "scores": grid}
        log_counts[label] = np.stack(counts)[n:, j].tolist()
    locations = np.nan_to_num(np.stack([mean_centroid(case)[::-1] for case in original]), nan=64.5)
    soft = []
    for strength in STRENGTHS:
        arrays = [metrics.moment_project(guide[i:i+8], locations[i:i+8], strength) > 0
                  for i in range(0, len(guide), 8)]
        soft.append(np.concatenate(arrays))
    values = [float(metrics.dice(truth[library], m[:n]).mean()) for m in soft]
    j = int(np.argmax(values))
    lock["choices"]["continuous"] = {"strength": STRENGTHS[j], "scores": values}
    masks["continuous"] = soft[j]
    rc.write_json(args.run / f"selection-{args.fold}.json", lock)
    payload = args.run / f"pred-{args.fold}.npz"
    np.savez_compressed(payload, ids=library + checked,
                        **{label: np.packbits(m, axis=-1) for label, m in masks.items()})
    scores = {label: float(metrics.dice(truth[checked], m[n:]).mean()) for label, m in masks.items()}
    rc.write_json(args.run / f"inner-{args.fold}.json",
                  {**lock, "check": checked, "scores": scores, "steps_checked": log_counts,
                   "prediction": rc.hash_file(payload), "area_error": 0})
    print("CENTROID_CHECK", args.fold, lock["target_yx"], scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for field in ("run", "data", "formal", "encoder", "ct", "texture"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    run(parser.parse_args())
