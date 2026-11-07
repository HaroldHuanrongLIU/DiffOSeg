#!/usr/bin/env bash
set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?allocation required}"
root=$1
aligned=$2
fold=$3
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$root/source"
sha256sum --quiet -c ../source.sha256
args=("$project/.venv/bin/python" -B -u -m reproduction.preference_gradients fit
      --run "$root" --data "$project/data/lidc_attributes.pkl" --align "$aligned" --fold "$fold")
"$project/.venv/bin/python" -B -m unittest tests.test_preference_gradients
"${args[@]}" --variant pcgrad --smoke
for variant in ordinary pcgrad; do
    "${args[@]}" --variant "$variant"
done
