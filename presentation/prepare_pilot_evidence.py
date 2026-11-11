"""Extract slide pilot numbers from existing archives without training or inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("GED@50", "Dice_soft@50", "Dice_mean")


def prepare(archive_root):
    sources = []

    def read(relative):
        path = archive_root / relative
        raw = path.read_bytes()
        sources.append({"archive_relative": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    evidence = {
        "scope": "Historical inner pilots on outer-training patients; no new experiments",
        "aggregation": "Equal mean of the four fold-specific inner results; not outer-test metrics",
        "units": "Raw Dice values are fractions; slides multiply Dice deltas by 100 for percentage points",
        "sources": sources,
    }
    for key, run, field, threshold in (
        ("boundary", "20261106_rank_boundary_v1", "aligned_dice", 0.80),
        ("polar", "20261106_polar_residual_v2", "inner_dice", 0.78),
    ):
        rows = []
        for fold in range(4):
            record = read(f"{run}/fold-{fold}-inner/result.json")
            value = record[field]
            assert record["fold"] == fold and math.isfinite(value)
            passed = value >= threshold
            assert record["passed"] == passed
            rows.append({"fold": fold, "dice": value, "passed": passed})
        evidence[key] = {
            "folds": rows,
            "threshold": threshold,
            "dice_min": min(r["dice"] for r in rows),
            "dice_max": max(r["dice"] for r in rows),
            "all_passed": all(r["passed"] for r in rows),
            "gate_type": "Fit sanity gate; also requires finite outputs, not evidence of outer improvement",
        }

    rows = []
    for fold in range(4):
        record = read(f"20261106_vote_transport_v3/lock-{fold}.json")
        assert record["fold"] == fold and len(record["candidates"]) == 12
        selected = next(c for c in record["candidates"] if
                        c["hardness"] == record["hardness"] and c["strength"] == record["strength"])
        assert selected == max(record["candidates"], key=lambda c: c["objective"])
        rows.append({
            "fold": fold,
            "candidate_count": len(record["candidates"]),
            "control": dict(zip(METRICS[:2], record["surrogate_control"])),
            "selected": {"GED@50": selected["ged50"], "Dice_soft@50": selected["soft50"]},
            "hardness": selected["hardness"],
            "strength": selected["strength"],
            "selection_objective": selected["objective"],
        })
    evidence["votes"] = {
        "folds": rows,
        "selection_rule": "Maximize the minimum of relative GED reduction and relative DiceSoft gain over 12 options per fold; listed-order tie break",
        "gate_type": "Engineering checks plus inner parameter selection; no fixed numerical superiority cutoff",
        "mean_delta": {k: math.fsum(r["selected"][k] - r["control"][k] for r in rows) / 4 for k in METRICS[:2]},
    }

    evidence["v4"] = {
        "gate": "All three fold-equal inner point estimates improve over the locked v3 surrogate: GED50 delta < 0, DiceSoft50 delta > 0, preference Dice delta > 0",
        "significance_gate": False,
        "check_set_reused": True,
    }
    for key, run in (("decoder_only", "20261106_joint_rater_v4"), ("energy", "20261106_joint_energy_v4")):
        rows = []
        for fold in range(4):
            record = read(f"{run}/lock-{fold}.json")
            assert record["fold"] == fold
            delta = {k: record["inner_v4"][k] - record["inner_v3"][k] for k in METRICS}
            assert all(math.isfinite(v) for v in delta.values())
            rows.append({"fold": fold, "control": record["inner_v3"], "candidate": record["inner_v4"], "delta": delta})
        mean_delta = {k: math.fsum(r["delta"][k] for r in rows) / 4 for k in METRICS}
        passed = mean_delta["GED@50"] < 0 and mean_delta["Dice_soft@50"] > 0 and mean_delta["Dice_mean"] > 0
        evidence["v4"][key] = {
            "folds": rows, "mean_delta": mean_delta, "passed": passed,
            "outer_evaluation": "COMPLETED" if key == "energy" else "NOT_RUN",
        }
    assert not evidence["v4"]["decoder_only"]["passed"] and evidence["v4"]["energy"]["passed"]
    outer = json.loads((ROOT / "results/course_project/v4.json").read_text())
    assert all(abs(evidence["v4"]["energy"]["mean_delta"][k] - outer["inner_deltas"][k]) < 1e-12 for k in METRICS)
    evidence["protocols"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("docs/lidc_boundary/protocol.md", "docs/lidc_polar/protocol.md", "docs/lidc_vote/protocol.md", "docs/lidc_joint/protocol.md", "docs/lidc_joint/energy_protocol.md")
    }
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    evidence = prepare(args.archive_root)
    (ROOT / "presentation/pilot_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("PILOT_EVIDENCE_PASS: 20 archived fold records; H1/H2 gates; H3 selection; H4 fail-to-pass revision")


if __name__ == "__main__":
    main()
