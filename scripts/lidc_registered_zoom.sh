#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1; r=$2; parent=$3; phase=$4; shift 4
case "$machine" in
lakers) project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg; data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl ;;
e11|n21) : "${SLURM_JOB_ID:?Slurm allocation required}"; project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"; data="$project/data/lidc_attributes.pkl"; if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi ;;
*) exit 2 ;;
esac
cd "$r/source"; sha256sum --quiet -c ../source.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
fit=("$project/.venv/bin/python" -B -u -m reproduction.registered_zoom --data "$data" --parent "$parent")
if [[ $phase == inner ]]; then "${fit[@]}" --fold "$1" --mode smoke --out "$r/smoke-$1"; fi
for f in "$@";do "${fit[@]}" --fold "$f" --mode "$phase" --out "$r/fold-$f-$phase";done
