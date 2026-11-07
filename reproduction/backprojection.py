"""Recover native-grid masks using the forward bilinear sampling operator."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
import torch
import torch.nn.functional as F

from reproduction import aligned_preference as align
from reproduction import native_local as native
from reproduction import native_study as scoring
from reproduction import polar_contours as polar
from reproduction import rank_contours as rc
from reproduction import texture_study as incumbent
from reproduction import vote_contours as vote

PROTOCOL = Path(__file__).resolve().parents[1] / "docs/lidc_91/backprojection_protocol.md"
VIEWS = [(scale, k * math.pi / 2, reflection, 0.0, 0.0)
         for scale in (0.74, 0.94, 1.10)
         for k in range(4) for reflection in (-1.0, 1.0)]
VIEWS += [(1.0, math.pi / 8, 1.0, dx, dy)
          for dx in (-0.35, 0.35) for dy in (-0.35, 0.35)]


def sampling_operator(grid, shape):
    """Exact align_corners=True bilinear sampler, with zero padding."""
    height, width = shape
    xy = np.asarray(grid, dtype=np.float64).reshape(-1, 2)
    x = (xy[:, 0] + 1) * ((width - 1) / 2)
    y = (xy[:, 1] + 1) * ((height - 1) / 2)
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - left, y - top
    xx = np.stack((left, left + 1, left, left + 1), axis=1)
    yy = np.stack((top, top, top + 1, top + 1), axis=1)
    values = np.stack(((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy), axis=1)
    valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
    rows = np.broadcast_to(np.arange(len(x))[:, None], xx.shape)
    operator = sparse.coo_matrix(
        (values[valid], (rows[valid], (yy * width + xx)[valid])),
        shape=(len(x), height * width),
    )
    return operator.tocsr()


def solve_occupancy(normal, rhs, prior, strength, shape):
    """Projected least squares, followed by exact four-color binary descent."""
    diagonal = normal.diagonal()
    mass = np.asarray(normal.sum(axis=1)).ravel()
    anchor = strength * mass
    target = rhs + anchor[:, None] * prior
    step = 1.0 / max(float((mass + anchor).max()), 1e-6)
    values = prior.copy()
    for _ in range(60):
        gradient = normal @ values + anchor[:, None] * values - target
        values = np.clip(values - step * gradient, 0, 1)
    continuous = values.copy()
    binary = (values >= 0.5).astype(np.float64)
    yy, xx = np.indices(shape)
    color = ((yy % 2) * 2 + xx % 2).ravel()
    for _ in range(6):
        for phase in range(4):
            off_diagonal = normal @ binary - diagonal[:, None] * binary
            delta = diagonal[:, None] + 2 * off_diagonal - 2 * rhs
            delta += anchor[:, None] * (1 - 2 * prior)
            active = (color == phase) & (mass > 0)
            binary[active] = delta[active] < 0
    return continuous, binary.astype(bool)


def grids(center, device):
    count = len(VIEWS)
    centers = torch.as_tensor(center, dtype=torch.float32, device=device).expand(count, -1).clone()
    centers += torch.tensor([(v[3], v[4]) for v in VIEWS], device=device)
    scale, angle, reflection = [
        torch.tensor([v[k] for v in VIEWS], device=device) for k in range(3)
    ]
    return align.transform_grids(centers, scale, angle, reflection)


def partitions(args):
    images, truth, _, patients = native.inputs(args.data)
    _, train, valid = native.splits(patients, args.fold)
    groups = type("Groups", (), {"series_uid": patients})()
    library, check = polar.calibration_split(groups, valid, args.fold)
    path = args.texture / f"features-{args.fold}-inner.npz"
    receipt = incumbent.read(path.with_suffix(".json"))
    ids = library + check
    if receipt["payload"] != rc.hash_file(path) or receipt["indices"] != ids:
        raise ValueError("Incumbent feature identity or patient order differs")
    if receipt["comparator"] != incumbent.REFERENCE:
        raise ValueError("Incumbent differs from the frozen 90% run")
    with np.load(path, allow_pickle=False) as cached:
        original = np.unpackbits(cached["old"], axis=-1).astype(bool)
        base = incumbent.anchor_logits(cached["base"].astype(np.float32), original)
        centers = cached["centers"].copy()
    return images, truth, patients, train, library, check, original, base, centers


@torch.inference_mode()
def prepare(args):
    images, _, _, train, library, check, _, _, centers = partitions(args)
    ids = library + check
    folder = args.align / f"fold-{args.fold}-inner-registered"
    plan = incumbent.read(folder / "plan.json")
    fit = incumbent.read(folder / "fit.json")
    if plan["train_indices"] != train or fit["checkpoint"] != rc.hash_file(folder / "final.pt"):
        raise ValueError("Aligned model training identity differs")
    net = vote.load_net(folder / "final.pt")
    path = args.run / f"views-{args.fold}.npy"
    if path.exists():
        raise FileExistsError(path)
    predictions = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(len(ids), len(VIEWS), 4, 128, 128)
    )
    standard = []
    for position, case in enumerate(ids):
        image = torch.as_tensor(images[case][None, None], device="cuda")
        image = align.registered_ct(image, *fit["applied_shift"]).float() / 255
        forward, backward = grids(centers[position], "cuda")
        outputs = []
        for grid in forward.split(8):
            view = F.grid_sample(image.expand(len(grid), -1, -1, -1), grid,
                                 padding_mode="border", align_corners=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_bf16_supported()):
                outputs.append(net(view)[:, :4].float())
        logits = torch.cat(outputs)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite view prediction")
        predictions[position] = logits.cpu().numpy()
        restored = F.grid_sample(logits, backward, padding_mode="border", align_corners=True)
        weight = (1 - backward.abs().amax(-1)).clamp(0, 0.05)[:, None]
        total = weight.sum(0)
        averaged = torch.where(total > 0, (restored * weight).sum(0) / total.clamp_min(1e-8), -16)
        standard.append(averaged.cpu().numpy())
        if position % 50 == 0:
            print("VIEWS", args.fold, position, len(ids), flush=True)
    predictions.flush()
    del predictions
    standard_path = args.run / f"ordinary-{args.fold}.npz"
    np.savez_compressed(standard_path, ids=ids, logits=np.stack(standard).astype(np.float16))
    rc.write_json(args.run / f"views-{args.fold}.json", {
        "ids": ids, "source": rc.hash_file(__file__), "protocol": rc.hash_file(PROTOCOL),
        "checkpoint": fit["checkpoint"], "views": VIEWS,
        "payload": rc.hash_file(path), "ordinary": rc.hash_file(standard_path),
    })


def calibrate_link(predictions, centers, labels):
    """Fit rank-wise fractional Bernoulli targets on library cases only."""
    generator = np.random.default_rng(61031)
    inputs, targets = [[] for _ in range(4)], [[] for _ in range(4)]
    for raw, center, truth in zip(predictions, centers, labels):
        grid, _ = grids(center, "cpu")
        operator = sampling_operator(grid.numpy(), (128, 128))
        sampled = operator @ truth.reshape(4, -1).T
        logits = raw.transpose(0, 2, 3, 1).reshape(-1, 4).astype(np.float64)
        for rank in range(4):
            eligible = np.flatnonzero(abs(logits[:, rank]) < 6)
            indices = generator.choice(eligible, min(len(eligible), 1024), replace=False)
            inputs[rank].append(logits[indices, rank])
            targets[rank].append(sampled[indices, rank])
    parameters = []
    for rank in range(4):
        x, y = np.concatenate(inputs[rank]), np.concatenate(targets[rank])
        def objective(parameter):
            value = parameter[0] * x + parameter[1]
            residual = expit(value) - y
            loss = np.mean(np.logaddexp(0, value) - y * value)
            gradient = np.array([np.mean(residual * x), residual.mean()])
            return loss, gradient
        fitted = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                          bounds=((0.1, 4.0), (-2.0, 2.0)))
        if not fitted.success:
            raise RuntimeError(f"Observation link did not converge: {fitted.message}")
        parameters.append(fitted.x)
    return np.asarray(parameters)


def check(args):
    _, truth, _, _, library, checked, original, base, centers = partitions(args)
    ids = library + checked
    path = args.run / f"views-{args.fold}.npy"
    receipt = incumbent.read(path.with_suffix(".json"))
    ordinary_path = args.run / f"ordinary-{args.fold}.npz"
    if (receipt["payload"] != rc.hash_file(path) or receipt["ordinary"] != rc.hash_file(ordinary_path)
            or receipt["source"] != rc.hash_file(__file__) or receipt["ids"] != ids):
        raise ValueError("View cache provenance differs")
    predictions = np.load(path, mmap_mode="r", allow_pickle=False)
    with np.load(ordinary_path, allow_pickle=False) as cache:
        ordinary = cache["logits"] > 0
    count = len(library)
    link = calibrate_link(predictions[:count], centers[:count], truth[library])
    rc.write_json(args.run / f"link-{args.fold}.json", {
        "parameters": link.tolist(), "fit_indices": library, "source": rc.hash_file(__file__)
    })
    candidates = {"reference": original, "ordinary": ordinary, "adjoint": []}
    for strength in (0.0, 0.1, 0.3, 1.0):
        candidates[f"soft-{strength}"] = []
        candidates[f"binary-{strength}"] = []
    axis = np.minimum(np.arange(128), np.arange(128)[::-1])
    window = np.minimum(axis[:, None], axis[None, :]).clip(0, 6) / 6
    for position, (raw, center) in enumerate(zip(predictions, centers)):
        grid, _ = grids(center, "cpu")
        grid = grid.numpy()
        row_weights = ((abs(grid).max(-1) <= 1) * window[None]).reshape(-1) / len(VIEWS)
        operator = sampling_operator(grid, (128, 128))
        normal = operator.T @ operator.multiply(row_weights[:, None])
        logits = raw.transpose(0, 2, 3, 1).reshape(-1, 4).astype(np.float64)
        observations = expit(logits * link[:, 0] + link[:, 1])
        rhs = operator.T @ (observations * row_weights[:, None])
        mass = np.asarray(normal.sum(axis=1)).ravel()
        prior = expit(2 * base[position].reshape(4, -1).T)
        normalized = np.divide(rhs, mass[:, None], out=prior.copy(), where=mass[:, None] > 0)
        candidates["adjoint"].append((normalized.T.reshape(4, 128, 128) > 0.5))
        for strength in (0.0, 0.1, 0.3, 1.0):
            soft, binary = solve_occupancy(normal, rhs, prior, strength, (128, 128))
            candidates[f"soft-{strength}"].append(soft.T.reshape(4, 128, 128) > 0.5)
            candidates[f"binary-{strength}"].append(binary.T.reshape(4, 128, 128))
        if position % 50 == 0:
            print("INVERSE", args.fold, position, len(ids), flush=True)
    candidates = {name: np.asarray(mask) for name, mask in candidates.items()}
    library_scores = {
        name: float(scoring.dice(truth[library], mask[:count]).mean())
        for name, mask in candidates.items()
    }
    selected = max(library_scores, key=library_scores.get)
    selection = {"selected": selected, "library_scores": library_scores,
                 "library": library, "source": rc.hash_file(__file__)}
    rc.write_json(args.run / f"selection-{args.fold}.json", selection)
    check_scores = {
        name: float(scoring.dice(truth[checked], mask[count:]).mean())
        for name, mask in candidates.items()
    }
    rc.write_json(args.run / f"inner-{args.fold}.json", {
        **selection, "fold": args.fold, "check_indices": checked, "all_check_scores": check_scores,
        "check": {"reference": check_scores["reference"], "candidate": check_scores[selected]},
        "observations": receipt["payload"], "protocol": rc.hash_file(PROTOCOL),
    })
    np.savez_compressed(args.run / f"pred-inner-{args.fold}.npz",
                        ids=ids, masks=np.packbits(candidates[selected], axis=-1))
    print("INVERSE_CHECK", args.fold, selected, check_scores, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "check"))
    for name in ("run", "data", "align", "texture"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    {"prepare": prepare, "check": check}[args.phase](args)
