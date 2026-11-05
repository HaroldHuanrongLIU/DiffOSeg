#!/usr/bin/env bash
# Run from the frozen training snapshot. Submit with the cluster's allocation flags.
set -euo pipefail
umask 077
machine=${1:?machine lakers/e11/n21}
fold=${2:?fold number}
run=${3:?private output root}
case "$machine" in
    lakers)
        project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
        data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl
        ;;
    e11|n21)
        : "${SLURM_JOB_ID:?A Slurm compute allocation is required}"
        project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
        data="$project/data/lidc_attributes.pkl"
        if [[ $machine == e11 ]]; then
            source "$HOME/.local/share/diffoseg-runtime/activate.sh"
        fi
        ;;
    *) exit 2 ;;
esac
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$run/source"
sha256sum --quiet -c "$run/source.sha256"
python="$project/.venv/bin/python"
phase=${4:-study}
if [[ $phase == smoke ]]; then
    "$python" -B -u -m reproduction.rank_contours --data "$data" --fold "$fold" --out "$run/smoke-$machine" --mode smoke
else
    for mode in inner full; do
        extra=()
        if [[ $mode == full ]]; then extra=(--gate "$run/fold-$fold-inner/result.json"); fi
        "$python" -B -u -m reproduction.rank_contours --data "$data" --fold "$fold" --out "$run/fold-$fold-$mode" --mode "$mode" "${extra[@]}"
    done
fi
