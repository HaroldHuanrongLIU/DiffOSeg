#!/usr/bin/env bash
set -eu -o pipefail
umask 077
host=$1
root=$2
fold=$3
case "$host" in
    lakers)
        project=/home/hrl/mohismlab/phd_course/cisc8006/DiffOSeg
        data=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_data/lidc_attributes.pkl
        aligned=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_single_slice_90_aligned
        previous=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_preference91_prompt
        public="$previous"
        crossfit=/mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs/20261106_preference91_crossfit
        warm=()
        ;;
    e11)
        : "${SLURM_JOB_ID:?GPU allocation required}"
        source "$HOME/.local/share/diffoseg-runtime/activate.sh"
        project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
        data="$project/data/lidc_attributes.pkl"
        base="$HOME/.local/share/diffoseg-runs"
        aligned="$base/20261106_single_slice_90_aligned"
        public="$base/20261106_preference91_jointprompt"
        previous="$public"
        crossfit="$public/oof-training"
        warm=(--warm-variant control)
        ;;
    *) exit 2 ;;
esac
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1
cd "$root/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
"$py" -B -m unittest tests.test_early_attention
args=("$py" -B -u -m reproduction.encoder_adaptation fit --run "$root" --data "$data" --public "$public"
      --align "$aligned" --previous "$previous" --crossfit "$crossfit" --texture "$root" --adapt --fold "$fold" "${warm[@]}")
"${args[@]}" --prefix-mode live-lora --smoke
for mode in live-frozen live-lora;do "${args[@]}" --prefix-mode "$mode";done
