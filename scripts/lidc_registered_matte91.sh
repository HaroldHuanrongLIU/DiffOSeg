#!/usr/bin/env bash
#SBATCH --job-name=lidc-alpha-ranks
#SBATCH --partition=gbunchQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?GPU allocation required}"
run=$1
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
base="$HOME/.local/share/diffoseg-runs"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
cd "$run/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
"$py" -B -m unittest discover -s tests -p test_registered_matte.py
args=("$py" -B -u -m reproduction.registered_matte fit --fold 1 --run "$run"
    --data "$project/data/lidc_attributes.pkl" --align "$base/20261106_single_slice_90_aligned")
"${args[@]}" --mode registered --smoke
for mode in vote raw registered; do "${args[@]}" --mode "$mode"; done
