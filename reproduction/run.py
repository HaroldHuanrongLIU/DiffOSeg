"""Run fixed-budget experiments using the repository's datasets, UNet and Trainer."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tarfile
import time
import traceback
from importlib.metadata import distributions

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ddpm.models.one_hot_categorical import OneHotCategoricalBCHW
from ddpm.polyak import PolyakAverager
from ddpm.trainer import Trainer, _build_model
from ddpm.utils import expanduservars, worker_init_fn
from reproduction.data import datasets_for_run
from reproduction.metrics import consensus_scores, preference_scores


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path, source_id, config_id=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if config_id is not None and state["config_id"] != config_id:
        raise ValueError(f"Checkpoint config differs: {path}")
    if state["source_id"] != source_id:
        raise ValueError(f"Checkpoint source differs: {path}")
    return state


def save_masks(path, truth, predictions):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, truth=np.packbits(truth), predictions=np.packbits(predictions),
                            truth_shape=np.array(truth.shape), prediction_shape=np.array(predictions.shape))
    temporary.replace(path)


def load_masks(path):
    with np.load(path, allow_pickle=False) as data:
        truth_shape = tuple(data["truth_shape"])
        prediction_shape = tuple(data["prediction_shape"])
        truth = np.unpackbits(data["truth"])[:np.prod(truth_shape)].reshape(truth_shape).astype(bool)
        predictions = np.unpackbits(data["predictions"])[:np.prod(prediction_shape)].reshape(prediction_shape).astype(bool)
    return truth, predictions


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def snapshot(output):
    paths = []
    for directory in ("ddpm", "datasets", "reproduction", "configs", "tests", "scripts"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.suffix in (".py", ".yml", ".yaml", ".sh", ".sbatch"))
    paths.extend(ROOT / name for name in ("uv.lock", "pyproject.toml", "INSTALL.md"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    source_id = digest.hexdigest()
    archive = output / f"source-{source_id[:12]}.tar.gz"
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(paths):
                tar.add(path, arcname=str(path.relative_to(ROOT)))
    return source_id


def model_for(params, train):
    sample = train[0]
    shapes = [sample[0].shape, sample[-1].shape]
    return _build_model(params, shapes, shapes[0])


def save_checkpoint(path, model, average, optimizer, scheduler, step, config_id, source_id, samples_seen=None):
    state = {
        "model": model.unet.state_dict(), "average_model": average.unet.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "step": step, "config_id": config_id, "source_id": source_id,
        "samples_seen": samples_seen,
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_experiment(params, train, directory, source_id, stage1_path, interrupt_after=None):
    config_id = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    final_path = directory / "final.pt"
    last_path = directory / "last.pt"
    if final_path.exists():
        state = read_checkpoint(final_path, source_id, config_id)
        if state["step"] != params["steps"]:
            raise ValueError(f"Completed checkpoint step differs: {final_path}")
        return True
    seed_all(params["run_seed"])
    model = model_for(params, train)
    if params["stage"] == 2 and not last_path.exists():
        state = read_checkpoint(stage1_path, source_id)
        current = model.unet.state_dict()
        loaded = {key: value for key, value in state["model"].items()
                  if key in current and value.shape == current[key].shape}
        missing, unexpected = model.unet.load_state_dict(loaded, strict=False)
        allowed_missing = lambda key: key.startswith("prompt_block.") or key == "input_blocks.0.0.weight"
        if not loaded or unexpected or any(not allowed_missing(key) for key in missing):
            raise ValueError(f"Unexpected stage transfer mismatch: {missing}, {unexpected}")
        atomic_json(directory / "initialization.json", {
            "stage1_checkpoint": str(stage1_path), "stage1_sha256": sha256(stage1_path),
            "loaded_tensors": len(loaded), "new_or_resized_tensors": missing,
            "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
        })
        del state, current, loaded
    average = copy.deepcopy(model)
    polyak = PolyakAverager(model, average, alpha=params["polyak_alpha"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    minimum = params["minimum_learning_rate"] / params["learning_rate"]
    steps = params["steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: minimum + (1 - minimum) * max(0, 1 - step / max(steps - 1, 1))
    )
    trainer = Trainer(polyak, optimizer, scheduler, torch.ones(2, device="cuda"),
                      None, None, params)
    start = 0
    samples_seen = 0
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config_id"] != config_id or state["source_id"] != source_id:
            raise ValueError(f"Resume provenance differs: {last_path}")
        model.unet.load_state_dict(state["model"])
        average.unet.load_state_dict(state["average_model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["step"]
        samples_seen = state["samples_seen"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        del state
    loader = DataLoader(train, batch_size=params["batch_size"], shuffle=True,
                        num_workers=params["num_workers"], pin_memory=True,
                        persistent_workers=params["num_workers"] > 0,
                        worker_init_fn=worker_init_fn,
                        generator=torch.Generator().manual_seed(params["run_seed"] + start))
    iterator = iter(loader)
    started = time.monotonic()
    initial_samples = samples_seen
    print(f"TRAIN {directory} from {start} to {steps}; samples={len(train)}; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    with (directory / "training.jsonl").open("a", buffering=1) as log:
        for step in range(start + 1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            result = trainer.train_step(None, batch)
            samples_seen += len(batch[0])
            if not math.isfinite(result["loss"]):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            if step == start + 1 or step % params["log_every"] == 0 or step == steps:
                elapsed = time.monotonic() - started
                rate = (step - start) / elapsed
                record = {
                    "state": "training", "step": step, "target_steps": steps,
                    "loss": result["loss"], "lr": result["lr"], "steps_per_second": rate,
                    "eta_hours": (steps - step) / rate / 3600,
                    "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "classification": params["classification"], "time": time.time(),
                    "batch_size": params["batch_size"], "samples_seen": samples_seen,
                    "images_per_second": (samples_seen - initial_samples) / elapsed,
                    "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                }
                log.write(json.dumps(record) + "\n")
                atomic_json(directory / "status.json", record)
                report(directory.parents[2])
                print(json.dumps(record), flush=True)
            if step % params["checkpoint_every"] == 0 or step == steps:
                save_checkpoint(last_path, model, average, optimizer, scheduler,
                                step, config_id, source_id, samples_seen)
            if interrupt_after is not None and step == interrupt_after and step < steps:
                if step % params["checkpoint_every"]:
                    save_checkpoint(last_path, model, average, optimizer, scheduler,
                                    step, config_id, source_id, samples_seen)
                atomic_json(directory / "status.json", {
                    "state": "interrupted", "step": step, "target_steps": steps,
                    "classification": params["classification"], "time": time.time(),
                })
                return False
    # Last and final are the same completed fixed-budget checkpoint, without duplicating GBs.
    os.link(last_path, final_path)
    atomic_json(directory / "status.json", {
        "state": "trained", "step": steps, "target_steps": steps,
        "classification": params["classification"], "time": time.time(),
        "samples_seen": samples_seen,
    })
    del iterator, loader, trainer, optimizer, scheduler, polyak, average, model
    torch.cuda.empty_cache()
    return True


@torch.inference_mode()
def sample_foreground(model, image, count, batch_size, expert=None):
    parts = []
    device = next(model.parameters()).device
    for offset in range(0, count, batch_size):
        size = min(batch_size, count - offset)
        hint = image[None].to(device).expand(size, -1, -1, -1).contiguous()
        logits = torch.zeros(size, 2, *image.shape[-2:], device=device)
        noise = OneHotCategoricalBCHW(logits=logits).sample()
        condition = {"hint": hint}
        if expert is not None:
            condition["txt"] = (expert[offset:offset + size].to(device) if isinstance(expert, torch.Tensor)
                                else torch.full((size,), expert, device=device, dtype=torch.long))
        probability = model(noise, condition)["diffusion_out"]
        parts.append(probability[:, 1].cpu().numpy())
    return np.concatenate(parts)


def sample_experts(model, image, count, batch_size):
    experts = torch.arange(4).repeat_interleave(count)
    return sample_foreground(model, image, 4 * count, batch_size, experts).reshape(4, count, *image.shape[-2:])


def evaluate_experiment(params, train, dataset, directory, source_id, sampling_batch_size=None):
    if (directory / "results.json").exists():
        return
    model = model_for(params, train)
    state = read_checkpoint(directory / "final.pt", source_id)
    training_source_id = state["source_id"]
    model.unet.load_state_dict(state["average_model"])
    model.eval()
    del state
    counts = params["consensus_samples"]
    sampling_batch_size = sampling_batch_size or params["sampling_batch_size"]
    total = min(len(dataset), params.get("evaluation_cases", len(dataset)))
    case_directory = directory / "cases"
    case_directory.mkdir(exist_ok=True)
    records = []
    for index in range(total):
        result_path = case_directory / f"{index:05d}.json"
        if result_path.exists():
            records.append(json.loads(result_path.read_text()))
            continue
        seed_all(params["run_seed"] + index)
        image, labels = dataset[index]
        volume = image.ndim == 4
        images = image.permute(1, 0, 2, 3) if volume else image[None]
        truth = labels.argmax(2 if volume else 1).numpy().astype(bool)
        if "evaluation_slices" in params and volume:
            images = images[:params["evaluation_slices"]]
            truth = truth[:, :params["evaluation_slices"]]
        started = time.monotonic()
        if params["stage"] == 1:
            predictions = np.stack([
                sample_foreground(model, frame, max(counts), sampling_batch_size)
                for frame in images
            ], axis=1)
            if not volume:
                predictions = predictions[:, 0]
            hard_predictions = predictions > 0.5
            scores = {str(count): consensus_scores(truth, hard_predictions[:count])
                      for count in counts}
        else:
            hard_predictions = np.stack([
                sample_experts(model, frame, params["preference_samples"], sampling_batch_size).mean(1) > 0.5
                for frame in images
            ], axis=1)
            if not volume:
                hard_predictions = hard_predictions[:, 0]
            scores = preference_scores(truth, hard_predictions)
        mask_path = case_directory / f"{index:05d}_masks.npz"
        save_masks(mask_path, truth, hard_predictions)
        record = {"index": index, "scores": scores, "seconds": time.monotonic() - started,
                  "masks": mask_path.name, "mask_packing": "numpy.packbits, flattened, big-endian bits"}
        atomic_json(result_path, record)
        records.append(record)
        atomic_json(directory / "status.json", {
            "state": "evaluating", "completed_cases": len(records), "total_cases": total,
            "classification": params["classification"], "time": time.time(),
            "sampling_batch_size": sampling_batch_size,
        })
        report(directory.parents[2])
        print(f"EVAL {directory} {index + 1}/{total}: {json.dumps(scores)}", flush=True)
    if params["stage"] == 1:
        means = {str(count): {metric: float(np.mean([r["scores"][str(count)][metric] for r in records]))
                              for metric in records[0]["scores"][str(count)]}
                 for count in counts}
    else:
        each = np.mean([r["scores"]["Dice_each"] for r in records], axis=0)
        means = {"Dice_each": each.tolist(), "Dice_mean": float(each.mean())}
    atomic_json(directory / "results.json", {
        "dataset": params["dataset"], "stage": params["stage"], "fold": params["fold"],
        "classification": params["classification"], "evaluation_split": params["evaluation_split"],
        "cases": total, "metrics": means, "source_id": source_id,
        "checkpoint_sha256": sha256(directory / "final.pt"),
        "training_source_id": training_source_id, "training_batch_size": params["batch_size"],
        "sampling_batch_size": sampling_batch_size,
        "consensus_samples": counts, "preference_samples": params["preference_samples"],
        "dice_units": "fraction", "time": time.time(),
        "ged_conventions": {"GED": "released reference: empty/empty IoU=0",
                            "GED_empty_match": "alternative: empty/empty IoU=1"},
    })
    atomic_json(directory / "status.json", {
        "state": "complete", "cases": total, "classification": params["classification"], "time": time.time(),
    })
    del model
    torch.cuda.empty_cache()


def paper_comparison(rows):
    comparison = {
        "paper": "https://arxiv.org/abs/2507.13087v2",
        "aggregation": "unweighted mean of four held-out patient folds",
        "scope": "LIDC only; NPC and other paper experiments not run",
        "baselines_and_ablations": "not run by this queue",
    }
    for stage in (1, 2):
        results = [row["results"] for row in rows if "results" in row
                   and row["results"]["dataset"] == "lidc"
                   and row["results"]["stage"] == stage
                   and row["results"]["classification"] == "paper_budget_with_documented_assumptions"]
        key = f"lidc_stage_{stage}"
        if {result["fold"] for result in results} != {0, 1, 2, 3}:
            comparison[key] = {"state": "incomplete", "completed_folds": len(results)}
            continue
        entries = []
        if stage == 1:
            for count, ged, dice in ((10, 0.09154, 0.9121), (30, 0.0773, 0.9220), (50, 0.0745, 0.9237)):
                for metric, target in (("GED", ged), ("Dice_soft", dice)):
                    values = [result["metrics"][str(count)][metric] for result in results]
                    entries.append({"metric": metric, "samples": count, "paper": target,
                                    "observed": float(np.mean(values)), "fold_std": float(np.std(values)),
                                    "difference": float(np.mean(values) - target)})
        else:
            targets = [0.9185, 0.9015, 0.9130, 0.9068]
            values = np.array([result["metrics"]["Dice_each"] for result in results])
            for expert, target in enumerate(targets):
                observed = float(values[:, expert].mean())
                entries.append({"metric": f"Dice_A{expert + 1}", "paper": target,
                                "observed": observed, "difference": observed - target,
                                "fold_std": float(values[:, expert].std())})
            entries.append({"metric": "Dice_mean", "paper": 0.9099,
                            "observed": float(values.mean()), "difference": float(values.mean() - 0.9099)})
        comparison[key] = {"state": "evaluated_with_documented_assumptions",
                           "metrics": entries,
                           "fold_batch_sizes": {str(r["fold"]): r.get("training_batch_size") for r in results}}
    return comparison


def report(output):
    rows = []
    for path in sorted(output.glob("*/fold_*/stage_*/status.json")):
        record = json.loads(path.read_text())
        result_path = path.with_name("results.json")
        if result_path.exists():
            record["results"] = json.loads(result_path.read_text())
        rows.append({"experiment": str(path.parent.relative_to(output)), **record})
    plan_path = output / "run_plan.json"
    if plan_path.exists():
        started = {row["experiment"] for row in rows}
        for job in json.loads(plan_path.read_text())["jobs"]:
            if job["experiment"] not in started:
                rows.append({**job, "state": "queued", "step": 0})
    rows.sort(key=lambda row: row["experiment"])
    atomic_json(output / "experiment_ledger.json", rows)
    comparison = paper_comparison(rows)
    atomic_json(output / "paper_comparison.json", comparison)
    lines = [
        "# DiffOSeg reproduction status", "",
        "Paper-budget runs have documented assumptions; smoke runs are engineering checks.",
        "Scope: LIDC only; NPC and other paper experiments are not run.",
        "No claim of paper-result reproduction is made before held-out evaluation completes.",
        "", "| Experiment | State | Progress | Evidence class |", "|---|---|---|---|",
    ]
    for row in rows:
        progress = f'{row.get("step", row.get("completed_cases", row.get("cases", 0)))}/{row.get("target_steps", row.get("total_cases", row.get("cases", 0)))}'
        lines.append(f'| {row["experiment"]} | {row["state"]} | {progress} | {row.get("classification", "unavailable")} |')
    lines.extend(["", "## Paper comparison", "", "Four completed patient folds are required before aggregating LIDC scores."])
    for key in ("lidc_stage_1", "lidc_stage_2"):
        item = comparison[key]
        if "metrics" not in item:
            lines.append(f'- {key}: {item["completed_folds"]}/4 folds evaluated.')
            continue
        lines.extend(["", f"### {key}", "", "| Metric | Samples | Paper | Observed | Difference |", "|---|---|---|---|---|"])
        for metric in item["metrics"]:
            lines.append(f'| {metric["metric"]} | {metric.get("samples", "-")} | {metric["paper"]:.5f} | {metric["observed"]:.5f} | {metric["difference"]:+.5f} |')
    lines.extend(["", "Dice is expressed as a fraction. GED uses the released empty-mask convention; GED_empty_match is also saved with per-case masks for independent verification.",
                  "NPC and baseline/ablation experiments are outside this run."])
    (output / "reproduction_report.md").write_text("\n".join(lines) + "\n")


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    config["data_root"] = expanduservars(config["data_root"])
    return config


def main():
    parser = argparse.ArgumentParser(description="LIDC patient-level experiments")
    parser.add_argument("--config", default=str(ROOT / "configs/reproduction.yml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, choices=range(4), required=True)
    parser.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--interrupt-after", type=int,
                        help="Smoke only: checkpoint and stop Stage I at this step")
    args = parser.parse_args()
    if args.interrupt_after is not None and (
            not args.smoke or args.phase == "evaluate" or not 1 <= args.interrupt_after < 20):
        parser.error("--interrupt-after requires smoke training and a step in 1..19")

    config = load_config(args.config)
    output = Path(args.output).resolve()
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {"lidc_attributes.pkl": sha256(Path(config["data_root"]) / "lidc_attributes.pkl")}
    torch.set_num_threads(config["torch_threads"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.backends.cudnn.benchmark = True
    source_id = snapshot(output)
    environment_path = output / "environment.json"
    hashes_path = output / "input_hashes.json"
    if environment_path.exists() and json.loads(environment_path.read_text())["source_id"] != source_id:
        raise ValueError("Output directory was created with different source bytes")
    if hashes_path.exists() and json.loads(hashes_path.read_text()) != input_hashes:
        raise ValueError("Output directory was created with different dataset bytes")
    atomic_json(environment_path, {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_id": source_id, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
    })
    atomic_json(hashes_path, input_hashes)
    settings = config["datasets"]["lidc"]
    if args.fold not in settings["folds"]:
        raise ValueError("Requested fold is not configured")
    jobs = [{"experiment": f"lidc/fold_{args.fold}/stage_{stage}",
             "target_steps": 20 if args.smoke else settings["stage_steps"][stage - 1],
             "classification": "engineering_smoke" if args.smoke else "paper_budget_with_documented_assumptions"}
            for stage in (1, 2)]
    existing_plan = output / "run_plan.json"
    if existing_plan.exists():
        prior_jobs = json.loads(existing_plan.read_text())["jobs"]
        jobs = [job for job in prior_jobs if not job["experiment"].startswith(f"lidc/fold_{args.fold}/")] + jobs
    atomic_json(existing_plan, {"jobs": jobs, "total_training_steps": sum(j["target_steps"] for j in jobs)})
    current = None
    atomic_json(output / "pipeline_status.json", {
        "state": "running", "dataset": "lidc", "fold": args.fold, "phase": args.phase,
        "smoke": args.smoke, "source_id": source_id, "time": time.time(),
    })
    try:
        fold_directory = output / "lidc" / f"fold_{args.fold}"
        phases = [(1, "train"), (2, "train"), (1, "evaluate"), (2, "evaluate")]
        if args.phase != "all":
            phases = [(stage, phase) for stage, phase in phases if phase == args.phase]
        for stage, phase in phases:
            current = fold_directory / f"stage_{stage}"
            current.mkdir(parents=True, exist_ok=True)
            train, evaluation, metadata = datasets_for_run(config["data_root"], "lidc", stage, args.fold, config["seed"])
            params = {key: value for key, value in config.items() if key != "datasets"}
            params.update({
                "dataset": "lidc", "dataset_file": "datasets.lidc", "stage": stage,
                "fold": args.fold, "distributed": False, "multigpu": False,
                "steps": settings["stage_steps"][stage - 1],
                "consensus_samples": settings["consensus_samples"],
                "evaluation_vote_strategy": "confidence", "run_seed": config["seed"] + args.fold,
                "evaluation_split": metadata["evaluation_split"],
                "classification": "paper_budget_with_documented_assumptions",
            })
            if args.smoke:
                params.update(steps=20, log_every=10, checkpoint_every=10, num_workers=0,
                              consensus_samples=[4], preference_samples=4, evaluation_cases=1,
                              classification="engineering_smoke")
            config_path = current / "config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != params:
                raise ValueError(f"Existing experiment config differs: {config_path}")
            atomic_json(config_path, params)
            atomic_json(current / "split.json", metadata)
            atomic_json(current / "execution.json", {
                "runtime_source_id": source_id, "sampling_batch_size": config["sampling_batch_size"],
                "training_batch_size": params["batch_size"], "phase": phase,
            })
            if phase == "train":
                finished = train_experiment(params, train, current, source_id,
                                            fold_directory / "stage_1/final.pt",
                                            args.interrupt_after if stage == 1 else None)
                if not finished:
                    atomic_json(output / "pipeline_status.json", {
                        "state": "interrupted", "fold": args.fold, "stage": stage,
                        "smoke": True, "source_id": source_id, "time": time.time(),
                    })
                    return
            else:
                evaluate_experiment(params, train, evaluation, current, source_id,
                                    sampling_batch_size=config["sampling_batch_size"])
            report(output)
            del train, evaluation
        atomic_json(output / "pipeline_status.json", {
            "state": "complete" if args.phase == "all" else f"{args.phase}_complete",
            "fold": args.fold, "smoke": args.smoke, "source_id": source_id, "time": time.time(),
        })
    except BaseException as error:
        status = {"state": "failed", "error": str(error), "traceback": traceback.format_exc(),
                  "experiment": str(current), "time": time.time()}
        if current is not None:
            atomic_json(current / "failure.json", status)
            prior = json.loads((current / "status.json").read_text()) if (current / "status.json").exists() else {}
            atomic_json(current / "status.json", {**prior, "state": "failed", "error": str(error)})
        atomic_json(output / "pipeline_status.json", status)
        raise
    finally:
        report(output)


if __name__ == "__main__":
    main()
