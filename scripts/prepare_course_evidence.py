"""Export existing evidence without changing archived runs or launching models."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/course_project"
PA = "20261025_table3_pa_claim_v1"
SELECTED = {
    "e11": ("20261106_rank_boundary_v1", "baseline/fourfold_summary.json"),
    "v1": ("20261106_rank_boundary_v1", "assessment/fourfold_comparison.json"),
    "v2": ("20261106_polar_residual_v2", "assessment/fourfold_comparison.json"),
    "v3": ("20261106_vote_transport_v3", "assessment/fourfold_comparison.json"),
    "v4": ("20261106_joint_energy_v4", "assessment/fourfold_comparison.json"),
    "v9": ("20261106_order_preference_v9", "assessment/fourfold_comparison.json"),
    "preference90": (
        "20261106_preference90_comparisonb",
        "assessment/preference_report.json",
    ),
    "formal_a": ("20261106_preference91_formal_a", "assessment/outer_report.json"),
    "formal_e": ("20261106_preference91_formal_e", "assessment/outer_report.json"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def export_timings(archive):
    rows = []
    for run in sorted(archive.iterdir()):
        if not run.is_dir() or run.name == PA:
            continue
        for parent, directories, files in os.walk(run):
            depth = len(Path(parent).relative_to(run).parts)
            directories[:] = sorted(
                d
                for d in directories
                if depth < 4
                and d
                not in {
                    "source",
                    "evaluation-source",
                    "release-source",
                    "baseline",
                    "cases",
                    "public",
                    "runtime",
                    "features",
                    "cache",
                    ".git",
                }
            )
            for name in ("result.json", "fit.json"):
                if name not in files:
                    continue
                path = Path(parent) / name
                record = read(path)
                if (
                    isinstance(record, dict)
                    and isinstance(record.get("seconds"), (int, float))
                    and ("steps" in record or "step" in record)
                ):
                    rows.append(
                        [
                            str(path.relative_to(archive)),
                            digest(path),
                            record["seconds"],
                            record.get("steps", record.get("step")),
                            record.get("gpu", "NOT_RECORDED"),
                        ]
                    )
    with (OUT / "compute_receipts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["receipt", "sha256", "recorded_seconds", "steps", "recorded_gpu"]
        )
        writer.writerows(rows)
    write_json(
        OUT / "compute_coverage.json",
        {
            "receipt_count": len(rows),
            "receipts_with_gpu_name": sum(r[4] != "NOT_RECORDED" for r in rows),
            "scope": "Leaf fit/result timers at up to four directory levels; source/cache/baseline copies excluded",
            "interpretation": "Timer inventory only. Overlap, missing stages and GPU allocation are not reconciled. Do not sum as total GPU-hours.",
        },
    )


def export(archive):
    pa = archive / PA
    freeze = read(pa / "freeze.json")
    for name in ("protocol", "split", "base_config"):
        if digest(pa / (name + ".json")) != freeze[name + "_sha256"]:
            raise ValueError("Changed frozen input: " + name)
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    def copy_evidence(key, source, destination):
        shutil.copyfile(source, OUT / destination)
        artifacts[key] = {
            "file": destination,
            "sha256": digest(source),
            "archive_relative": str(source.relative_to(archive)),
        }

    copy_evidence("pa", pa / "final_results.json", "pa_results.json")
    copy_evidence("pa_protocol", pa / "protocol.json", "pa_protocol.json")
    copy_evidence(
        "pa_source",
        pa / ("source-" + freeze["source_id"][:12] + ".tar.gz"),
        "pa_source.tar.gz",
    )
    copy_evidence(
        "pa_metrics_dependency", pa / "source/metrics_set.py", "pa_metrics_set.py"
    )
    for key, (run_id, relative) in SELECTED.items():
        copy_evidence(key, archive / run_id / relative, key + ".json")
    split = read(pa / "split.json")
    patients = {p: i for i, p in enumerate(sorted(set(split["test_patient_ids"])))}
    rows, runs = [], []
    for condition in ("P", "A"):
        for seed in (20261, 20262, 20263):
            folder = (
                pa / "runs" / condition / ("seed_" + str(seed)) / "lidc/fold_0/stage_1"
            )
            config, result, execution = (
                read(folder / (n + ".json")) for n in ("config", "results", "execution")
            )
            last = json.loads((folder / "training.jsonl").read_text().splitlines()[-1])
            if last["step"] != 120000 or result["cases"] != 450:
                raise ValueError("Incomplete P/A run")
            for i in range(450):
                record = read(folder / "cases" / ("%05d.json" % i))
                scores = record["scores"]["30"]
                rows.append(
                    [
                        condition,
                        seed,
                        i,
                        patients[split["test_patient_ids"][i]],
                        scores["GED"],
                        scores["Dice_soft"],
                    ]
                )
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "steps": last["step"],
                    "batch_size": config["batch_size"],
                    "cases": result["cases"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "config_sha256": digest(folder / "config.json"),
                    "initialization_sha256": read(folder / "initial_weights.json")[
                        "sha256"
                    ],
                    "gpu": execution["gpu"],
                    "torch": execution["torch"],
                    "source_id": execution["source_id"],
                    "metrics": result["metrics"]["30"],
                }
            )
    with (OUT / "pa_cases.csv").open("w", newline="") as stream:
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
        writer.writerows(rows)
    artifacts["pa_cases"] = {
        "file": "pa_cases.csv",
        "sha256": digest(OUT / "pa_cases.csv"),
        "derivation": "Frozen case JSON; sorted local integer clusters replace patient identifiers",
    }
    write_json(OUT / "pa_runs.json", runs)
    artifacts["pa_runs"] = {
        "file": "pa_runs.json",
        "sha256": digest(OUT / "pa_runs.json"),
        "derivation": "Six frozen configs, results, execution and training records",
    }
    registry = []
    for run in sorted(archive.iterdir()):
        if not run.is_dir():
            continue
        known = [k for k, (name, _) in SELECTED.items() if name == run.name]
        summary_paths = [
            p
            for p in [
                run / "final_results.json",
                run / "summary.json",
                run / "assessment/outer_report.json",
                run / "assessment/fourfold_comparison.json",
            ]
            if p.is_file()
        ]
        summary = summary_paths[0] if summary_paths else None
        source = run / "source.sha256"
        claim = run.name == PA
        registry.append(
            [
                run.name,
                ";".join(known)
                or ("original_claim" if claim else "additional_archive"),
                "complete_inconclusive" if claim else "indexed_archive",
                freeze["git_head"] if claim else "UNRECORDED",
                (
                    freeze["source_id"]
                    if claim
                    else (digest(source) if source.exists() else "UNRECORDED")
                ),
                str(summary.relative_to(archive)) if summary else "UNINDEXED",
                digest(summary) if summary else "UNRECORDED",
                "20261;20262;20263" if claim else "SEE_RUN_CONFIG",
                "RTX 5090" if claim else "SEE_RUN_RECEIPTS",
                (
                    str(read(pa / "final_results.json")["gpu_hours"])
                    if claim
                    else "UNACCOUNTED"
                ),
                "training+preflight+evaluation" if claim else "not aggregated",
                "not separately metered; no billing receipt",
            ]
        )
    with (OUT / "registry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run_id",
                "role",
                "evidence_status",
                "git_commit",
                "source_identity",
                "result_relative",
                "result_sha256",
                "seeds",
                "hardware",
                "gpu_hours",
                "gpu_hours_scope",
                "cost",
            ]
        )
        writer.writerows(registry)
    artifacts["registry"] = {
        "file": "registry.csv",
        "sha256": digest(OUT / "registry.csv"),
        "derivation": "Directory inventory; indexed_archive is not a completion claim",
    }
    export_timings(archive)
    for name in ("compute_receipts.csv", "compute_coverage.json"):
        artifacts[name] = {
            "file": name,
            "sha256": digest(OUT / name),
            "derivation": "Explicit fit/result timing fields; coverage limitations recorded",
        }
    manifest = {
        "schema": 1,
        "assembled_date": "2026-11-08",
        "release_state": "candidate_with_recorded_team_instructions",
        "assembly_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "pa_source_id": freeze["source_id"],
        "pa_original_freeze_sha256": digest(pa / "freeze.json"),
        "dataset_sha256": read(pa / "inputs.json")["lidc_attributes.pkl"],
        "train_patients": len(split["train_patients"]),
        "test_patients": len(split["test_patients"]),
        "test_slices": len(split["test_indices"]),
        "artifacts": artifacts,
    }
    write_json(OUT / "manifest.json", manifest)
    print(
        "EXPORTED",
        len(runs),
        "P/A runs;",
        len(rows),
        "case records;",
        len(registry),
        "archive entries",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    export(parser.parse_args().archive_root.resolve())
