#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1
r=$2
prior=$3
shift 3
if [[ $machine == lakers ]]; then
    project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
    data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl
else
    : "${SLURM_JOB_ID:?GPU allocation required}"
    project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
    data="$project/data/lidc_attributes.pkl"
    if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi
fi
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$r/source"; sha256sum --quiet -c ../source.sha256
base=("$project/.venv/bin/python" -B -u -m reproduction.vote_contours --data "$data" --prior "$prior")
for view in fullfield zoom; do
    "${base[@]}" --mode smoke --fold "$1" --view "$view" --out "$r/smoke-$machine-$view"
done
for fold in "$@"; do
    for view in fullfield zoom; do
        root="$r/fold-$fold-$view"
        "${base[@]}" --mode inner --fold "$fold" --view "$view" --out "$root-inner"
        "${base[@]}" --mode full --fold "$fold" --view "$view" --out "$root-full" --gate "$root-inner/result.json"
    done
done
