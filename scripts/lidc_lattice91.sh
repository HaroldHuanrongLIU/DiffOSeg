#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1;root=$2;parent=$3;mode=$4;shift 4
if [[ $machine == lakers ]];then project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg;data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl
else : "${SLURM_JOB_ID:?allocation required}";project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg";data="$project/data/lidc_attributes.pkl";if [[ $machine == e11 ]];then source "$HOME/.local/share/diffoseg-runtime/activate.sh";fi
fi
cd "$root/source";sha256sum --quiet -c ../source.sha256
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3
fit=("$project/.venv/bin/python" -u -B -m reproduction.lattice_preference --align "$parent" --data "$data")
if [[ $mode == inner ]];then "${fit[@]}" --fold "$1" --mode smoke --out "$root/smoke-$1";fi
for f in "$@";do "${fit[@]}" --fold "$f" --mode "$mode" --out "$root/fold-$f-$mode";done
