#!/usr/bin/env bash
set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?GPU allocation required}"
root=$1
fold=1
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
base="$HOME/.local/share/diffoseg-runs"
aligned="$base/20261106_single_slice_90_aligned"
crossfit="$base/20261106_preference91_crossfit"
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$root/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
data="$project/data/lidc_attributes.pkl"
"$py" -B -m unittest tests.test_joint_prompt
oof="$root/oof-training"
mkdir -p "$oof"
if [[ ! -e "$oof/source" ]]; then ln -s "$crossfit/source" "$oof/source"; fi
for sub in 0 1 2; do
    folder="fold-$fold-sub-$sub"
    if [[ ! -e "$oof/$folder" ]]; then ln -s "$crossfit/$folder" "$oof/$folder"; fi
done
if [[ ! -e "$oof/oof-$fold.json" ]]; then
    "$py" -B -u -m reproduction.crossfit_residual prepare --data "$data" --run "$oof" --align "$aligned" --texture "$root" --fold "$fold" --scope inner
fi
args=("$py" -B -u -m reproduction.joint_prompt)
options=(--run "$root" --public "$root" --data "$data" --align "$aligned" --crossfit "$oof" --fold "$fold")
if [[ ! -e "$root/image_features.json" ]]; then "${args[@]}" encode "${options[@]}"; fi
"${args[@]}" fit "${options[@]}" --variant joint --smoke
for variant in control joint; do
    "${args[@]}" fit "${options[@]}" --variant "$variant"
done
