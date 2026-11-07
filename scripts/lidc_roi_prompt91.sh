#!/usr/bin/env bash
set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?GPU allocation required}"
root=$1
base="$HOME/.local/share/diffoseg-runs"
public="$base/20261106_preference91_jointprompt"
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
"$py" -B -m unittest tests.test_roi_prompt
run=("$py" -B -u -m reproduction.roi_prompt)
args=(--fold 1 --run "$root" --data "$project/data/lidc_attributes.pkl" --public "$public"
      --crossfit "$public/oof-training" --align "$base/20261106_single_slice_90_aligned")
for variant in global local;do
    "${run[@]}" prepare "${args[@]}" --variant "$variant"
    if [[ "$variant" == local ]];then "${run[@]}" fit "${args[@]}" --variant "$variant" --smoke;fi
    "${run[@]}" fit "${args[@]}" --variant "$variant"
done
