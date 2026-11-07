#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1; root=$2; init=$3; reg=$4; phase=$5; shift 5
case "$machine" in
 lakers) proj=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg; data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl ;;
 e11|n21) : "${SLURM_JOB_ID:?GPU allocation required}"; proj="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"; data="$proj/data/lidc_attributes.pkl"; if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi ;;
 *) exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source"; sha256sum --quiet -c ../source.sha256
fit=("$proj/.venv/bin/python" -B -u -m reproduction.texture_preference --data "$data" --align "$reg" --init "$init")
if [[ $phase == inner ]]; then "${fit[@]}" --fold "$1" --mode smoke --out "$root/smoke-$1"; fi
for f in "$@"; do "${fit[@]}" --mode "$phase" --fold "$f" --out "$root/fold-$f-$phase"; done
