#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1; root=$2; init=$3; shift 3
case "$machine" in
lakers)
    project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
    data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl ;;
e11|n21)
    : "${SLURM_JOB_ID:?Slurm GPU allocation required}"
    project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
    data="$project/data/lidc_attributes.pkl"
    if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi ;;
*) exit 2 ;;
esac
cd "$root/source"; sha256sum --quiet -c ../source.sha256
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
base=("$project/.venv/bin/python" -B -u -m reproduction.contour_field --data "$data" --init "$init")
"${base[@]}" --mode smoke --fold "$1" --out "$root/smoke-$1"
for fold in "$@"; do
    "${base[@]}" --mode inner --fold "$fold" --out "$root/fold-$fold-inner"
    "${base[@]}" --mode full --fold "$fold" --out "$root/fold-$fold-full" --gate "$root/fold-$fold-inner/result.json"
done
