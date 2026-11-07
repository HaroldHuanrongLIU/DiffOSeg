#!/usr/bin/env bash
set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?GPU allocation required}"
root=$1
base="$HOME/.local/share/diffoseg-runs"
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
"$py" -B -m unittest tests.test_prior_diffusion
args=("$py" -B -u -m reproduction.prior_diffusion fit --run "$root" --data "$project/data/lidc_attributes.pkl"
      --align "$base/20261106_single_slice_90_aligned" --crossfit "$base/20261106_preference91_jointprompt/oof-training" --fold 1)
"${args[@]}" --variant guided --smoke
for variant in uniform guided;do "${args[@]}" --variant "$variant"; done
