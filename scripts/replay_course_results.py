"""Recompute the six P/A results from frozen masks, without model training/inference."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np

from check_course_project import check, classify, digest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261, 20262, 20263)
RUN = "20261025_table3_pa_claim_v1"


def read(path):
    return json.loads(path.read_text())


def score_masks(truth, predictions):
    """Integer set intersections and unions; empty/empty IoU is zero."""
    labels = np.asarray(truth, dtype=bool).reshape(len(truth), -1)
    samples = np.asarray(predictions, dtype=bool).reshape(len(predictions), -1)
    population = np.concatenate((labels, samples))
    packed = np.packbits(population, axis=1)
    bit_count = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    intersection = bit_count[packed[:, None, :] & packed[None, :, :]].sum(
        axis=-1, dtype=np.int64
    )
    area = population.sum(axis=1, dtype=np.int64)
    union = area[:, None] + area[None, :] - intersection
    distance = 1 - np.divide(
        intersection, union, out=np.zeros_like(union, dtype=float), where=union != 0
    )
    n = len(labels)
    ged = (
        2 * distance[:n, n:].mean() - distance[:n, :n].mean() - distance[n:, n:].mean()
    )
    iou32 = np.divide(
        intersection.astype(np.float32),
        union.astype(np.float32),
        out=np.ones_like(union, dtype=np.float32),
        where=union != 0,
    )
    distance32 = 1 - iou32
    ged32 = (
        2 * distance32[:n, n:].mean()
        - distance32[:n, :n].mean()
        - distance32[n:, n:].mean()
    )
    ged32 -= ((~labels.any(1)).mean() - (~samples.any(1)).mean()) ** 2
    soft = []
    soft32 = []
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        a, b = labels.mean(0) > threshold, samples.mean(0) > threshold
        denominator = int(a.sum() + b.sum())
        soft.append(
            2 * int(np.count_nonzero(a & b)) / denominator if denominator else 1.0
        )
        soft32.append(
            float(np.float32(2 * np.count_nonzero(a & b)) / np.float32(denominator))
            if denominator
            else 1.0
        )
    return [float(ged32), float(np.mean(soft32))], [float(ged), float(np.mean(soft))]


def main(archive_root, data_path, output):
    if output.exists():
        raise ValueError("Choose a new replay directory; earlier evidence is retained")
    started = time.monotonic()
    check(archive_root=archive_root)
    evidence = ROOT / "results/course_project"
    manifest = read(evidence / "manifest.json")
    if digest(data_path) != manifest["dataset_sha256"]:
        raise ValueError("Dataset identity differs")
    # The user-provided archived pickle was hash-verified before deserialization.
    with data_path.open("rb") as stream:
        dataset = list(pickle.load(stream).values())
    run = archive_root / RUN
    split = read(run / "split.json")
    if set(split["train_patients"]) & set(split["test_patients"]):
        raise ValueError("Training and held-out patients overlap")
    patients = {p: i for i, p in enumerate(sorted(set(split["test_patient_ids"])))}
    patient_ids = np.array([patients[p] for p in split["test_patient_ids"]])
    with (evidence / "pa_cases.csv").open(newline="") as stream:
        portable = {
            (r["condition"], int(r["seed"]), int(r["case_index"])): r
            for r in csv.DictReader(stream)
        }
    output.mkdir(parents=True)
    arrays, records, run_checks = {}, [], []
    max_error = 0.0
    max_float64_difference = 0.0
    checkpoints = []
    for condition in ("P", "A"):
        for seed in SEEDS:
            folder = (
                run / "runs" / condition / ("seed_" + str(seed)) / "lidc/fold_0/stage_1"
            )
            result = read(folder / "results.json")
            checkpoint_hash = digest(folder / "final.pt")
            if checkpoint_hash != result["checkpoint_sha256"]:
                raise ValueError("Checkpoint identity differs")
            checkpoints.append(checkpoint_hash)
            worker_split = read(folder / "split.json")
            if worker_split["test_indices"] != split["test_indices"]:
                raise ValueError("Worker patient fold differs")
            rows = []
            for i, index in enumerate(split["test_indices"]):
                record = read(folder / "cases" / ("%05d.json" % i))
                with np.load(
                    folder / "cases" / record["masks"], allow_pickle=False
                ) as z:
                    truth = (
                        np.unpackbits(z["truth"])[: np.prod(z["truth_shape"])]
                        .reshape(z["truth_shape"])
                        .astype(bool)
                    )
                    predictions = (
                        np.unpackbits(z["predictions"])[
                            : np.prod(z["prediction_shape"])
                        ]
                        .reshape(z["prediction_shape"])
                        .astype(bool)
                    )
                if truth.shape != (4, 128, 128) or predictions.shape != (30, 128, 128):
                    raise ValueError("Unexpected truth or sample count")
                original = np.asarray(dataset[index]["masks"], dtype=bool)
                fingerprints = lambda masks: sorted(
                    hashlib.sha256(m.tobytes()).hexdigest() for m in masks
                )
                if fingerprints(truth) != fingerprints(original):
                    raise ValueError("Saved truth differs from original dataset")
                actual, float64_scores = score_masks(truth, predictions)
                max_float64_difference = max(
                    max_float64_difference,
                    float(np.max(np.abs(np.array(actual) - float64_scores))),
                )
                previous = [record["scores"]["30"][k] for k in ("GED", "Dice_soft")]
                exported = portable[condition, seed, i]
                if int(exported["patient_cluster"]) != int(patient_ids[i]):
                    raise ValueError("Portable patient pairing differs")
                if not np.allclose(
                    previous,
                    [float(exported[k]) for k in ("GED30", "DiceSoft30")],
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError("Portable scores differ from archive")
                error = float(np.max(np.abs(np.array(actual) - previous)))
                if error > 1e-12:
                    raise ValueError(
                        "Recomputed metrics differ beyond floating-point tolerance"
                    )
                max_error = max(max_error, error)
                rows.append(actual)
                records.append([condition, seed, i, int(patient_ids[i]), *actual])
            arrays[condition, seed] = np.array(rows)
            if not np.allclose(
                np.mean(rows, axis=0),
                [result["metrics"]["30"][k] for k in ("GED", "Dice_soft")],
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("Recomputed run mean differs")
            run_checks.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "cases": len(rows),
                    "checkpoint_sha256": checkpoint_hash,
                    "recomputed_mean": np.mean(rows, axis=0).tolist(),
                }
            )
            print(condition, seed, "450 masks scored", flush=True)
    p = np.array([arrays["P", s] for s in SEEDS])
    a = np.array([arrays["A", s] for s in SEEDS])
    differences = p - a
    counts = np.bincount(patient_ids, minlength=54)
    sums = np.stack(
        [differences[:, patient_ids == i].sum(1) for i in range(54)], axis=1
    )
    rng = np.random.default_rng(271828)
    seed_weights = rng.multinomial(3, np.full(3, 1 / 3), size=10000)
    patient_weights = rng.multinomial(54, np.full(54, 1 / 54), size=10000)
    draws = np.array(
        [
            (sums * s[:, None, None] * w[None, :, None]).sum((0, 1))
            / (3 * (w @ counts))
            for s, w in zip(seed_weights, patient_weights)
        ]
    )
    ci = np.quantile(draws, [0.025, 0.975], axis=0).T
    decision = classify(ci)
    expected = read(evidence / "pa_results.json")["primary"]
    if decision != expected["decision"] or not np.allclose(
        ci, expected["difference_ci95"], atol=1e-7, rtol=0
    ):
        raise ValueError("Replayed inference disagrees with historical conclusion")
    with (output / "recomputed_cases.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "condition",
                "seed",
                "case_index",
                "patient_cluster",
                "GED30",
                "DiceSoft30",
            ]
        )
        writer.writerows(records)
    summary = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "executor": "Codex, delegated by user",
        "scope": "Full metric/statistical replay of six frozen P/A prediction banks",
        "training": "NOT_RUN",
        "model_inference": "NOT_RUN",
        "independent_classmate": False,
        "dataset_sha256": manifest["dataset_sha256"],
        "source_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "runs": run_checks,
        "case_groups": len(records),
        "generated_masks_scored": len(records) * 30,
        "maximum_metric_error": max_error,
        "float64_diagnostic_max_difference": max_float64_difference,
        "metric_precision": "Integer intersections; primary uses frozen float32 division/means, with separate float64 diagnostic",
        "P_mean": p.mean((0, 1)).tolist(),
        "A_mean": a.mean((0, 1)).tolist(),
        "difference": differences.mean((0, 1)).tolist(),
        "difference_ci95": ci.tolist(),
        "decision": decision,
        "bootstrap_replicates": 10000,
        "elapsed_seconds": time.monotonic() - started,
        "case_csv_sha256": digest(output / "recomputed_cases.csv"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("REPLAY_PASS", decision, "max metric error", max_error, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.archive_root.resolve(), args.data.resolve(), args.output.resolve())
