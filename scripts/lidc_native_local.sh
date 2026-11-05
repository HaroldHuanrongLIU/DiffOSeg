#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1; root=$2; init=$3; phase=$4; shift 4
case "$machine" in
  lakers)
    project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
    data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl ;;
  e11|n21)
    : "${SLURM_JOB_ID:?Slurm allocation required}"
    project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
    data="$project/data/lidc_attributes.pkl"
    if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi ;;
  *) exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$root/source"; sha256sum --quiet -c ../source.sha256
fit=("$project/.venv/bin/python" -B -u -m reproduction.native_local --data "$data" --init "$init")
if [[ $phase == inner ]]; then
    for stage in locator local; do
        "${fit[@]}" --fold "$1" --stage "$stage" --mode smoke --out "$root/smoke-$1-$stage"
    done
fi
for f in "$@"; do
    for stage in locator local; do
        "${fit[@]}" --fold "$f" --stage "$stage" --mode "$phase" --out "$root/fold-$f-$phase-$stage"
    done
done
