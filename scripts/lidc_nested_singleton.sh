#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1
root=$2
fold=$3
sub=$4
case "$machine" in
    lakers)
        project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
        data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl
        ;;
    e11|n21)
        : "${SLURM_JOB_ID:?GPU allocation required}"
        project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
        data="$project/data/lidc_attributes.pkl"
        if [[ "$machine" == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi
        ;;
    *) exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source"
sha256sum --quiet -c ../source.sha256
command=("$project/.venv/bin/python" -B -u -m reproduction.crossfit_contours --data "$data"
         --single-group --scope inner --fold "$fold" --sub "$sub")
"${command[@]}" --smoke --out "$root/smoke-single-$fold-$sub"
"${command[@]}" --out "$root/singleton-$fold-$sub"
