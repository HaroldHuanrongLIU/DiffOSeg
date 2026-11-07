#!/usr/bin/env bash
set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?GPU allocation required}"
host=$1
root=$2
fold=${3:-${SLURM_ARRAY_TASK_ID:?fold required}}
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
if [[ "$host" = e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
"$py" -B -m unittest tests.test_outer_teacher_scope
args=("$py" -B -u -m reproduction.crossfit_contours --scope full --fold "$fold" --data "$project/data/lidc_attributes.pkl")
"${args[@]}" --sub 0 --smoke --out "$root/smoke-teacher-$fold"
for sub in 0 1 2; do
    "${args[@]}" --sub "$sub" --out "$root/fold-$fold-sub-$sub"
done
