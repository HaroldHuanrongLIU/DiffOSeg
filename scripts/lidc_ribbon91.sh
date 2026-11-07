#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1; root=$2; phase=$3; shift 3
if [[ $machine == lakers ]]; then
    project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
    data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl
else
    : "${SLURM_JOB_ID:?Request a compute allocation}"
    project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"; data="$project/data/lidc_attributes.pkl"
    if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi
fi
cd "$root/source"; sha256sum --quiet -c ../source.sha256
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
cmd=("$project/.venv/bin/python" -u -B -m reproduction.ribbon_preference --data "$data")
if [[ $phase == inner ]]; then "${cmd[@]}" --fold "$1" --mode smoke --out "$root/smoke-$1"; fi
for f in "$@"; do "${cmd[@]}" --fold "$f" --mode "$phase" --out "$root/fold-$f-$phase"; done
