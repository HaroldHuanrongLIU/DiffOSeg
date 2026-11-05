#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1; root=$2; previous=$3; shift 3
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
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source"; sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
fit=("$py" -B -u -m reproduction.normal_refiner --data "$data" --seed-base "${LIDC_REFINER_SEED_BASE:-117901}")
check=("$py" -B -u -m reproduction.normal_gate --data "$data" --run "$root" --v3 "$previous")
"${fit[@]}" --fold "$1" --mode smoke --out "$root/smoke-$1"
"${check[@]}" --fold "$1" --smoke
for f in "$@"; do
    "${fit[@]}" --fold "$f" --mode inner --out "$root/fold-$f-inner"
    "${check[@]}" --fold "$f"
    "${fit[@]}" --fold "$f" --mode full --out "$root/fold-$f-full" --gate "$root/fold-$f-inner/result.json"
done
