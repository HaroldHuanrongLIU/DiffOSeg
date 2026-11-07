#!/usr/bin/env bash
#SBATCH --job-name=lidc-contour-flow
#SBATCH --partition=gbunchQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?Requires an allocated GPU}"
run=$1
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
base="$HOME/.local/share/diffoseg-runs"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
cd "$run/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
"$py" -B -m unittest discover -s tests -p 'test_contour_flow.py'
args=("$py" -B -u -m reproduction.contour_flow fit --fold 1 --run "$run"
      --original "$base/20261106_preference91_jointprompt/oof-training" --data "$project/data/lidc_attributes.pkl")
"${args[@]}" --mode smooth --smoke
for mode in white smooth; do
    "${args[@]}" --mode "$mode"
done
