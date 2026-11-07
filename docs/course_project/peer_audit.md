# Results replay and delegated technical review

The user requested a fresh results replay for this closeout item. **Executor: Codex.** No independent classmate's participation is represented.

## Actual replay

- Date: 2026-11-08 (UTC+8).
- Environment: Python 3.11.15, NumPy 1.26.4, CPU.
- Input: original P/A archive, verified LIDC bytes, all six final-checkpoint hashes and saved mask banks.
- Coverage: 2700 case groups, 81000 generated masks, three paired seeds, 54 held-out patients.
- Recalculation: independent bit-set intersections and unions; frozen float32 metric accumulation; separate float64 diagnostic; 10000 paired seed-by-patient bootstrap draws.
- Result: **PASS**, primary per-case maximum error **0**, outcome **inconclusive**.
- Float64 diagnostic maximum difference: 2.01230581975409e-7.
- Training / model inference: **NOT_RUN**.
- Record: [summary.json](../../results/course_project/replay_20261108_verified/summary.json) and [new case scores](../../results/course_project/replay_20261108_verified/recomputed_cases.csv).
- Initial numerical-precision check failure: [failure.json](../../results/course_project/replay_20261108/failure.json), retained with diagnosis.

## Command

Run from the repository root with a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u -B \
  scripts/replay_course_results.py \
  --archive-root /path/to/DiffOSeg_runs \
  --data /path/to/lidc_attributes.pkl \
  --output /path/to/new_replay_directory
```

The source archive and dataset stay unchanged. The result confirms metric/statistical reproducibility from saved predictions. It does not measure the variability of retraining or demonstrate an outside peer's clean installation.
