"""CPU evidence smoke and optional bootstrap verification; reads inputs only."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/course_project"
SEEDS = (20261, 20262, 20263)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(intervals):
    ged, dice = intervals
    if ged[1] < 0 and dice[0] > 0:
        return "supported"
    if ged[0] > 0 or dice[1] < 0:
        return "not_supported"
    return "inconclusive"


def validate_cases(rows, slices=450, patients=54):
    grouped = {(c, s): {} for c in ("P", "A") for s in SEEDS}
    clusters = {}
    for row in rows:
        key = (row["condition"], int(row["seed"]))
        if key not in grouped:
            raise ValueError("Unexpected condition or seed")
        index, patient = int(row["case_index"]), int(row["patient_cluster"])
        scores = [float(row["GED30"]), float(row["DiceSoft30"])]
        if index in grouped[key] or not 0 <= index < slices:
            raise ValueError("Duplicate or invalid case index")
        if not 0 <= patient < patients or not all(math.isfinite(x) for x in scores):
            raise ValueError("Invalid patient cluster or nonfinite metric")
        if index in clusters and clusters[index] != patient:
            raise ValueError("Paired patient mapping differs")
        clusters[index] = patient
        grouped[key][index] = scores
    if (
        any(len(v) != slices for v in grouped.values())
        or len(set(clusters.values())) != patients
    ):
        raise ValueError("Incomplete paired cohort")
    arrays = {
        key: [values[i] for i in range(slices)] for key, values in grouped.items()
    }
    return arrays, [clusters[i] for i in range(slices)]


def check(root=EVIDENCE, bootstrap=False, archive_root=None):
    manifest = json.loads((root / "manifest.json").read_text())
    for key, item in manifest["artifacts"].items():
        if digest(root / item["file"]) != item["sha256"]:
            raise ValueError("Evidence hash mismatch: " + key)
        if archive_root is not None and "archive_relative" in item:
            if digest(archive_root / item["archive_relative"]) != item["sha256"]:
                raise ValueError("Archive hash mismatch: " + key)
    with tarfile.open(root / "pa_source.tar.gz", "r:gz") as bundle:
        members = bundle.getmembers()
        if any(
            not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts
            for m in members
        ):
            raise ValueError("Unexpected source bundle member")
        h = hashlib.sha256()
        for member in sorted(members, key=lambda m: m.name):
            h.update(member.name.encode())
            h.update(bundle.extractfile(member).read())
        if h.hexdigest() != manifest["pa_source_id"]:
            raise ValueError("Frozen source content identity differs")
    report = json.loads((root / "pa_results.json").read_text())
    primary = report["primary"]
    with (root / "pa_cases.csv").open(newline="") as stream:
        arrays, patients = validate_cases(list(csv.DictReader(stream)))
    for c in ("P", "A"):
        computed = [
            [math.fsum(row[m] for row in arrays[c, s]) / 450 for m in range(2)]
            for s in SEEDS
        ]
        expected = primary[c + "_per_seed"]
        if any(
            abs(x - y) > 1e-12 for a, b in zip(computed, expected) for x, y in zip(a, b)
        ):
            raise ValueError("Per-seed result differs")
        means = [math.fsum(row[m] for row in computed) / 3 for m in range(2)]
        if any(abs(x - y) > 1e-12 for x, y in zip(means, primary[c + "_mean"])):
            raise ValueError("Condition mean differs")
    differences = [primary["P_mean"][m] - primary["A_mean"][m] for m in range(2)]
    if any(abs(x - y) > 1e-12 for x, y in zip(differences, primary["difference"])):
        raise ValueError("Paired difference differs")
    if classify(primary["difference_ci95"]) != primary["decision"]:
        raise ValueError("Claim decision violates frozen rule")
    runs = json.loads((root / "pa_runs.json").read_text())
    if len(runs) != 6 or {(r["condition"], r["seed"]) for r in runs} != set(arrays):
        raise ValueError("Missing or duplicate run")
    for seed in SEEDS:
        pair = [r for r in runs if r["seed"] == seed]
        if pair[0]["initialization_sha256"] != pair[1]["initialization_sha256"]:
            raise ValueError("Paired initialization differs")
        if any(
            r["steps"] != 120000 or r["cases"] != 450 or r["batch_size"] != 12
            for r in pair
        ):
            raise ValueError("Training contract differs")
    print(
        "SMOKE_PASS: frozen source; hashes; 6 runs; 2700 records; 54 paired patient clusters; decision=inconclusive"
    )
    if bootstrap:
        import numpy as np

        differences = np.array([arrays["P", s] for s in SEEDS]) - np.array(
            [arrays["A", s] for s in SEEDS]
        )
        ids = np.asarray(patients)
        totals = np.stack([differences[:, ids == p].sum(1) for p in range(54)], axis=1)
        counts = np.bincount(ids, minlength=54)
        rng = np.random.default_rng(primary["bootstrap_seed"])
        sw = rng.multinomial(3, np.full(3, 1 / 3), size=primary["nboot"])
        pw = rng.multinomial(54, np.full(54, 1 / 54), size=primary["nboot"])
        samples = np.array(
            [
                (totals * s[:, None, None] * p[None, :, None]).sum((0, 1))
                / (3 * (p @ counts))
                for s, p in zip(sw, pw)
            ]
        )
        ci = np.quantile(samples, [0.025, 0.975], axis=0).T
        if not np.allclose(ci, primary["difference_ci95"], atol=1e-12, rtol=0):
            raise ValueError("Recomputed bootstrap differs")
        print("BOOTSTRAP_PASS:", json.dumps(ci.tolist()))
    return primary["decision"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Requires NumPy; repeat all 10000 draws",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        help="Also compare bundled evidence with original archived bytes",
    )
    args = parser.parse_args()
    check(bootstrap=args.bootstrap, archive_root=args.archive_root)
