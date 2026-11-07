"""Independent original-mask audit, run after all outer predictions and scores are sealed."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def mask_dice(p, y):
    predicted, labelled = np.flatnonzero(p), np.flatnonzero(y)
    total = len(predicted) + len(labelled)
    return 2 * len(np.intersect1d(predicted, labelled, assume_unique=True)) / total if total else 1.0


def run(root, data, runs_root):
    base = runs_root/"20261106_rank_boundary_v1/baseline"
    incumbent = runs_root/"20261106_preference90_comparisonb/assessment"
    report = json.loads((root/"assessment/outer_report.json").read_text())
    assert digest(data) == "6c401c96b0e1737d43fdccfc446cb0c3601473d3b7668ad4a68d22ec0c1ee81b"
    assert digest(base/"fourfold_summary.json") == "d27211bd93bc8eb2eccd728bd9a04840dfb0e7864ec0d15cd7826c782edeb2a0"
    assert digest(incumbent/"preference_report.json") == "2024e556c9cf003d037701d458d2da3eb527deb0192bf8666cf8383651a220d0"
    assert digest(root/"selection.json") == report["selection"]
    with data.open("rb") as stream:
        cases = list(pickle.load(stream).values())
    names = ("e11","reference","candidate")
    folds, covered, patients = [], set(), set()
    largest_error = 0.0
    comparisons = 0
    for f in range(4):
        folder = root/f"assessment/fold-{f}"
        stage = base/f"fold_{f}/lidc/fold_{f}/stage_2"
        manifest = json.loads((stage/"split.json").read_text())
        seal = json.loads((folder/"sealed.json").read_text())
        table = json.loads((folder/"paired.json").read_text())
        assert table["seal"] == digest(folder/"sealed.json")
        assert seal["payload"] == digest(folder/"predictions.npz") and seal["selection"] == report["selection"]
        with np.load(folder/"predictions.npz", allow_pickle=False) as z:
            ids = z["ids"].tolist()
            masks = np.unpackbits(z["masks"], axis=-1).astype(bool)
        assert ids == manifest["test_indices"] and len(table["rows"]) == len(ids)
        assert not covered.intersection(ids) and not patients.intersection(manifest["test_patients"])
        covered.update(ids); patients.update(manifest["test_patients"])
        totals = []
        for j, idx in enumerate(ids):
            case = cases[idx]
            annotations = np.asarray(case["masks"]).astype(bool)
            areas = np.count_nonzero(annotations, axis=(1,2))
            truth = annotations[np.argsort(areas, kind="stable")]
            with np.load(stage/f"cases/{j:05d}_masks.npz", allow_pickle=False) as z:
                old = np.unpackbits(z["predictions"])[:4*128*128].reshape(4,128,128).astype(bool)
            with np.load(incumbent/f"fold-{f}/{j:05d}.npz", allow_pickle=False) as z:
                prior = np.unpackbits(z["full"], axis=-1).astype(bool)
            row = table["rows"][j]
            assert row["index"] == idx and row["patient"] == str(case["series_uid"]).split("_")[0]
            values = np.asarray([[mask_dice(pred, label) for pred,label in zip(method,truth)]
                                 for method in (old,prior,masks[j])])
            largest_error = max(largest_error, float(np.abs(values-np.asarray([row[name] for name in names])).max()))
            comparisons += values.size
            totals.append(values)
        means = np.asarray(totals).mean((0,2))
        np.testing.assert_allclose(means, [report["folds"][f]["means"][name] for name in names], rtol=0, atol=1e-12)
        folds.append(means)
    assert covered == set(range(1609)) and len(patients) == 214 and comparisons == 19308
    aggregate = np.mean(folds, axis=0)
    np.testing.assert_allclose(aggregate, [report["means"][name] for name in names], rtol=0, atol=1e-12)
    assert largest_error < 1e-12 and report["target"] == 0.91
    assert report["target_pass"] == bool(aggregate[2] >= 0.91)
    result = {"scope":"independent complete original four-fold set-intersection audit",
              "cases":len(covered),"patients":len(patients),"rank_scores_recalculated":comparisons,
              "maximum_error":largest_error,"means":dict(zip(names, aggregate.tolist())),
              "target_pass":bool(aggregate[2]>=0.91),"report":digest(root/"assessment/outer_report.json"),
              "source":digest(Path(__file__))}
    path = root/"assessment/independent_audit.json"
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("run","data","runs-root"):
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    run(args.run,args.data,args.runs_root)
