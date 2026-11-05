#!/usr/bin/env bash
set -eu -o pipefail
umask 077
machine=$1
r=$2
shift 2
case "$machine" in
  lakers)
    project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
    data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl ;;
  e11|n21)
    : "${SLURM_JOB_ID:?use a GPU allocation}"
    project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
    data="$project/data/lidc_attributes.pkl"
    if [[ $machine == e11 ]]; then source "$HOME/.local/share/diffoseg-runtime/activate.sh"; fi ;;
  *) exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$r/source"
sha256sum --quiet -c ../source.sha256
python="$project/.venv/bin/python"
"$python" -B -u -m reproduction.polar_contours --data "$data" --fold "$1" --mode smoke --out "$r/smoke-$machine"
for f in "$@"; do
    "$python" -B -u -m reproduction.polar_contours --data "$data" --fold "$f" --mode inner --out "$r/fold-$f-inner"
    "$python" -B -u -m reproduction.polar_contours --data "$data" --fold "$f" --mode full --out "$r/fold-$f-full" --gate "$r/fold-$f-inner/result.json"
done
