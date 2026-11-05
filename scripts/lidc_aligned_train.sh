#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1; root=$2; previous=$3; phase=$4; variant=$5; shift 5
case "$machine" in
 lakers) project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg; data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl ;;
 e11|n21) : "${SLURM_JOB_ID:?allocation required}"; project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"; data="$project/data/lidc_attributes.pkl"; if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi ;;
 *) exit 2 ;;
esac
cd "$root/source"; sha256sum --quiet -c ../source.sha256
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
args=("$project/.venv/bin/python" -B -u -m reproduction.aligned_preference --data "$data" --previous "$previous")
case "$variant" in registered) ;; identity) args+=(--identity-shift);; *) exit 2;; esac
if [[ $phase == inner ]]; then
 "${args[@]}" --fold "$1" --mode smoke --out "$root/smoke-$variant-$1"
fi
for f in "$@"; do
 "${args[@]}" --fold "$f" --mode "$phase" --out "$root/fold-$f-$phase-$variant"
done
