#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1;root=$2;shift 2
if [[ $machine == lakers ]];then project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg;data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl
else : "${SLURM_JOB_ID:?Slurm allocation required}";project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg";data="$project/data/lidc_attributes.pkl";if [[ $machine == e11 ]];then source "$HOME/.local/share/diffoseg-runtime/activate.sh";fi
fi
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source";sha256sum --quiet -c ../source.sha256
cmd=("$project/.venv/bin/python" -B -u -m reproduction.crossfit_contours --data "$data")
"${cmd[@]}" --smoke --fold "$1" --sub 0 --out "$root/smoke-$1"
for f in "$@";do for sub in 0 1 2;do "${cmd[@]}" --fold "$f" --sub "$sub" --out "$root/fold-$f-sub-$sub";done;done
