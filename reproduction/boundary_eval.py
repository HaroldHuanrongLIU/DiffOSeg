"""Locked paired evaluation for the rank-contour DiffOSeg extension."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from datasets.lidc import BaseDataset
from reproduction import rank_contours as rc
from reproduction.metrics import consensus_scores, preference_scores


SUMMARY_HASH = "d27211bd93bc8eb2eccd728bd9a04840dfb0e7864ec0d15cd7826c782edeb2a0"
COUNTS = (10, 30, 50)
MODELS = ("baseline", "image_only", "corrected")
PRIMARY = ("GED@50", "Dice_soft@50", "Dice_mean")


def read(path):
    return json.loads(path.read_text())


def unpack(path, name, expected):
    with np.load(path, allow_pickle=False) as z:
        shape_key = "prediction_shape" if name == "predictions" else "truth_shape"
        shape = tuple(z[shape_key])
        if shape != expected or z[name].size != (np.prod(shape) + 7) // 8:
            raise ValueError(f"Invalid {name} shape or payload at {path}")
        return np.unpackbits(z[name], count=int(np.prod(shape))).reshape(shape).astype(bool)


def correct(consensus, preference, image_masks):
    """Takes only predicted contours. Inference never accepts a reference label."""
    image_field = np.stack([rc.signed(mask) for mask in image_masks])
    pref_field = np.stack([rc.signed(mask) for mask in preference])
    corrected_pref = (0.6 * image_field + 0.4 * pref_field) > 0
    samples = np.stack([rc.signed(mask) for mask in consensus])
    area = consensus.sum(axis=(-1, -2))
    banks = {}
    for k in COUNTS:
        order = np.argsort(area[:k], kind="stable")
        # Use a centered empirical CDF so rounding is symmetric for the four ranks.
        groups = ((np.arange(k) + 0.5) * 4 / k).astype(int)
        corrected = np.empty_like(consensus[:k])
        for rank in range(4):
            ids = order[groups == rank]
            center = samples[ids].mean(0)
            fields = 0.6 * image_field[rank] + 0.4 * center + 0.8 * (samples[ids] - center)
            corrected[ids] = fields > 0
        banks[k] = corrected
    return banks, corrected_pref


def metrics(truth, banks, preference):
    row = {}
    for k in COUNTS:
        row.update({f"{key}@{k}": value for key, value in consensus_scores(truth, banks[k]).items()})
    aligned = preference_scores(truth, preference)
    row["Dice_mean"] = aligned["Dice_mean"]
    row.update({f"Dice_A{i+1}": float(d) for i, d in enumerate(aligned["Dice_each"])})
    return row


def inference(args):
    if rc.hash_file(args.data) != rc.DATA_HASH or rc.hash_file(args.baseline / "fourfold_summary.json") != SUMMARY_HASH:
        raise ValueError("Input hash differs")
    data = BaseDataset(str(args.data))
    outer, inner_train, valid = rc.split_for(data, args.fold)
    frozen = args.baseline / f"fold_{args.fold}/lidc/fold_{args.fold}"
    for stage in (1, 2):
        split = read(frozen / f"stage_{stage}/split.json")
        if any(split[key] != value for key, value in outer.items()):
            raise ValueError("Baseline and current split differ")
    inner = args.models / f"fold-{args.fold}-inner"
    full = args.models / f"fold-{args.fold}-full"
    gate, selected = read(inner / "result.json"), read(full / "result.json")
    code = rc.hash_file(rc.__file__)
    protocol = rc.hash_file(Path(rc.__file__).resolve().parents[1] / "docs/lidc_boundary/protocol.md")
    for path, mode, result, ids in ((inner, "inner", gate, inner_train), (full, "full", selected, outer["train_indices"])):
        plan = read(path / "plan.json")
        if (plan["mode"], plan["fold"], plan["training_source"], plan["protocol_sha256"], plan["dataset_sha256"],
                plan["train_indices"], plan["inner_valid_indices"], result["steps"]) != (
                mode, args.fold, code, protocol, rc.DATA_HASH, ids, valid if mode == "inner" else [], rc.STEPS):
            raise ValueError("Training provenance or holdout differs")
    if not gate["passed"] or gate["aligned_dice"] < 0.8 or rc.hash_file(full / "final.pt") != selected["checkpoint_sha256"]:
        raise ValueError("Inner gate or checkpoint identity differs")
    model = rc.RankContours().cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(full / "final.pt", map_location="cpu", weights_only=True))
    args.out.mkdir(parents=True, exist_ok=False)
    imgs = torch.from_numpy(np.stack([data.images[i] for i in outer["test_indices"]])[:, None]).cuda()
    image_masks = torch.cat([rc.predict(model, chunk) for chunk in imgs.split(12)]).cpu().numpy()
    path = args.out / "image_masks.npz"
    np.savez_compressed(path, masks=np.packbits(image_masks, axis=-1))
    start = time.monotonic()
    for i, auxiliary in enumerate(image_masks):
        name = f"{i:05d}_masks.npz"
        bank = unpack(frozen / "stage_1/cases" / name, "predictions", (50, 128, 128))
        expert = unpack(frozen / "stage_2/cases" / name, "predictions", (4, 128, 128))
        banks, prefs = correct(bank, expert, auxiliary)
        np.savez_compressed(args.out / name, pref=np.packbits(prefs, axis=-1),
                            **{f"k{k}": np.packbits(banks[k], axis=-1) for k in COUNTS})
    rc.write_json(args.out / "inference.json", {"fold": args.fold, "cases": len(imgs), "training_source": code,
                  "evaluation_source": rc.hash_file(__file__), "protocol": protocol, "summary": SUMMARY_HASH,
                  "checkpoint": selected["checkpoint_sha256"], "image_masks": rc.hash_file(path),
                  "inner_dice": gate["aligned_dice"], "correction_seconds": time.monotonic() - start})


def score(args):
    receipt = read(args.out / "inference.json")
    if (receipt["evaluation_source"] != rc.hash_file(__file__) or rc.hash_file(args.data) != rc.DATA_HASH or
            rc.hash_file(args.baseline / "fourfold_summary.json") != SUMMARY_HASH):
        raise ValueError("Scoring inputs or method changed")
    data = BaseDataset(str(args.data))
    frozen = args.baseline / f"fold_{args.fold}/lidc/fold_{args.fold}"
    indices = rc.split_for(data, args.fold)[0]["test_indices"]
    if (receipt["fold"], receipt["cases"], receipt["image_masks"]) != (
            args.fold, len(indices), rc.hash_file(args.out / "image_masks.npz")):
        raise ValueError("Prediction bank is not complete")
    with np.load(args.out / "image_masks.npz") as packed:
        aux = np.unpackbits(packed["masks"], axis=-1).astype(bool)
    rows = []
    for i, idx in enumerate(indices):
        truth = np.array(data.mask_labels[idx], dtype=bool)
        name = f"{i:05d}_masks.npz"
        f1, f2 = (frozen / f"stage_{n}/cases" / name for n in (1, 2))
        for path in (f1, f2):
            if not np.array_equal(truth, unpack(path, "truth", (4, 128, 128))):
                raise ValueError(f"Misaligned case {args.fold}/{i}")
        baseline = unpack(f1, "predictions", (50, 128, 128))
        pref = unpack(f2, "predictions", (4, 128, 128))
        with np.load(args.out / name) as z:
            fixed = {k: np.unpackbits(z[f"k{k}"], axis=-1).astype(bool) for k in COUNTS}
            fixed_pref = np.unpackbits(z["pref"], axis=-1).astype(bool)
        image = aux[i]
        image_bank = {k: image[((np.arange(k) + 0.5) * 4 / k).astype(int)] for k in COUNTS}
        rows.append({"index": idx, "patient": str(data.series_uid[idx]).split("_")[0],
                     "baseline": metrics(truth, {k: baseline[:k] for k in COUNTS}, pref),
                     "corrected": metrics(truth, fixed, fixed_pref),
                     "image_only": metrics(truth, image_bank, image)})
    # Reproduce the frozen per-fold metrics from the exact same masks before exposing new scores.
    original = read(args.baseline / "fourfold_summary.json")["folds"][args.fold]["stages"]
    target = {f"{metric}@{k}": val for k, block in original["1"]["metrics"].items() for metric, val in block.items()}
    target["Dice_mean"] = original["2"]["metrics"]["Dice_mean"]
    target.update({f"Dice_A{i+1}": v for i, v in enumerate(original["2"]["metrics"]["Dice_each"])})
    drift = max(abs(np.mean([r["baseline"][key] for r in rows]) - v) for key, v in target.items())
    if drift > 2e-6:
        raise ValueError(f"Reference scoring differs by {drift}")
    rc.write_json(args.out / "paired.json", {"receipt": receipt, "baseline_max_abs_error": drift, "rows": rows})


def report(args):
    if rc.hash_file(args.baseline / "fourfold_summary.json") != SUMMARY_HASH:
        raise ValueError("Comparison baseline differs")
    tables = [read(args.out / f"fold-{f}/paired.json") for f in range(4)]
    summ = read(args.baseline / "fourfold_summary.json")
    patients, indices = set(), set()
    keys = list(tables[0]["rows"][0]["baseline"])
    folds, boot = [], []
    rng = np.random.default_rng(212903)
    for f, table in enumerate(tables):
        rows = table["rows"]
        split = read(args.baseline / f"fold_{f}/lidc/fold_{f}/stage_1/split.json")
        ids = [r["index"] for r in rows]
        groups = sorted({r["patient"] for r in rows})
        if (ids != split["test_indices"] or set(groups) != set(split["test_patients"]) or
                patients.intersection(groups) or indices.intersection(ids) or table["receipt"]["fold"] != f or
                table["receipt"]["evaluation_source"] != rc.hash_file(__file__)):
            raise ValueError("Incomplete or overlapping comparison fold")
        patients.update(groups)
        indices.update(ids)
        values = {m: np.array([[r[m][k] for k in keys] for r in rows]) for m in MODELS}
        means = {m: dict(zip(keys, arr.mean(0))) for m, arr in values.items()}
        folds.append({"fold": f, "patients": len(groups), "slices": len(rows), "means": means,
                      "inner_dice": table["receipt"]["inner_dice"], "baseline_max_abs_error": table["baseline_max_abs_error"]})
        group_idx = np.array([groups.index(r["patient"]) for r in rows])
        n = np.bincount(group_idx)
        sums = np.zeros((len(groups), len(keys)))
        np.add.at(sums, group_idx, values["corrected"] - values["baseline"])
        times = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=5000)
        boot.append((times @ sums) / (times @ n)[:, None])
    if len(indices) != 1609 or len(patients) != 214:
        raise ValueError("Incomplete LIDC coverage")
    comparison = {}
    draws = np.stack(boot).mean(0)
    for j, k in enumerate(keys):
        scores = {m: float(np.mean([f["means"][m][k] for f in folds])) for m in MODELS}
        scores["delta"] = scores["corrected"] - scores["baseline"]
        scores["paired_patient_percentile95"] = np.percentile(draws[:, j], [2.5, 97.5]).tolist()
        comparison[k] = scores
    paper = summ["paper_comparison"]
    thresholds = {x["metric"]+"@50": x["observed"] for x in paper["lidc_stage_1"]["metrics"] if x["samples"] == 50}
    thresholds["Dice_mean"] = next(x["observed"] for x in paper["lidc_stage_2"]["metrics"] if x["metric"] == "Dice_mean")
    passes = {k: (comparison[k]["corrected"] < v if k.startswith("GED") else comparison[k]["corrected"] > v)
              for k, v in thresholds.items()}
    rc.write_json(args.out / "fourfold_comparison.json", {"protocol": tables[0]["receipt"]["protocol"], "baseline": SUMMARY_HASH,
                  "evaluation_source": rc.hash_file(__file__), "slices": len(indices), "patients": len(patients),
                  "folds": folds, "metrics": comparison, "primary_pass": passes, "all_three_improved": all(passes.values())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["infer", "score", "report"])
    parser.add_argument("--fold", type=int, choices=range(4))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    {"infer": inference, "score": score, "report": report}[args.phase](args)
