#!/bin/bash
#SBATCH --job-name=diffoseg-lidc-smoke
#SBATCH --partition=gbunchQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

set -euo pipefail
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
output=${1:?Pass a new output directory}
if [ -e "$output/smoke" ]; then
    echo "Refusing to overwrite existing smoke output: $output/smoke" >&2
    exit 1
fi
cd "$project"
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=4
echo "SLURM job: ${SLURM_JOB_ID:?SLURM allocation required}"
.venv/bin/python -u -B -m reproduction.run --output "$output" --fold 0 --smoke --interrupt-after 10
.venv/bin/python - "$output/smoke/lidc/fold_0/stage_1/last.pt" <<'CHECK'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert state["step"] == 10
assert state["optimizer"]["state"]
assert state["scheduler"]["last_epoch"] == 10
assert all(key in state for key in ("rng_python", "rng_numpy", "rng_torch", "rng_cuda"))
for group in ("model", "average_model"):
    assert all(torch.isfinite(v).all() for v in state[group].values() if torch.is_floating_point(v))
print("INTERRUPTED_CHECKPOINT_PASS step=10")
CHECK
.venv/bin/python -u -B -m reproduction.run --output "$output" --fold 0 --smoke
.venv/bin/python - "$output/smoke" <<'CHECK'
import json, sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
assert json.loads((root / "pipeline_status.json").read_text())["state"] == "complete"
fold = root / "lidc/fold_0"
for stage in (1, 2):
    d = fold / f"stage_{stage}"
    result = json.loads((d / "results.json").read_text())
    split = json.loads((d / "split.json").read_text())
    state = torch.load(d / "final.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 20 and result["classification"] == "engineering_smoke"
    assert not set(split["train_patients"]) & set(split["test_patients"])
    assert result["cases"] == 1
    if stage == 2:
        assert len(result["metrics"]["Dice_each"]) == 4
        initialization = json.loads((d / "initialization.json").read_text())
        assert initialization["all_parameters_trainable"]
    print(f"STAGE_{stage}_PASS step=20 cases=1")
training = [json.loads(line) for line in (fold / "stage_1/training.jsonl").read_text().splitlines()]
assert [r["step"] for r in training] == [1, 10, 11, 20]
print("RESUME_PASS observed_steps=1,10,11,20")
print("LIDC_SMOKE_VALIDATION_PASSED")
CHECK
