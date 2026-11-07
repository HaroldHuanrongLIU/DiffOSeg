#!/usr/bin/env bash
set -eu -o pipefail
umask 077
: "${SLURM_JOB_ID:?GPU allocation required}"
root=$1
aligned=$2
fold=$3
project="$HOME/PhD_Course_Project/CISC8006/DiffOSeg"
source "$HOME/.local/share/diffoseg-runtime/activate.sh"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
cd "$root/source"
sha256sum --quiet -c ../source.sha256
py="$project/.venv/bin/python"
"$py" -B -m unittest tests.test_context_mix
args=("$py" -B -u -m reproduction.context_mix fit --run "$root" --data "$project/data/lidc_attributes.pkl"
      --align "$aligned" --fold "$fold")
"${args[@]}" --variant context --smoke
for variant in control context; do
    "${args[@]}" --variant "$variant"
done
"$py" -B -c 'import json,math,sys; from pathlib import Path; root=Path(sys.argv[1]); fold=sys.argv[2]; values=[json.loads((root/f"fold-{fold}-{arm}/train.jsonl").read_text().splitlines()[0])["loss"] for arm in ("control","context")]; assert math.isclose(*values,rel_tol=1e-6,abs_tol=1e-7),values; print("PAIRED_FIRST_LOSS",values)' "$root" "$fold"
