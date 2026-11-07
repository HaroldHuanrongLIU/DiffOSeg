# Submitted claim and frozen comparison

The authoritative submitted wording remains in [the submitted claim](../course_project_requirement/CISC8006-submitted-claim.md). This package records its completed empirical test, with the tier and fidelity limitations stated separately from the outcome.

## Contract and execution

| Field | Submitted / frozen setting | Evidence |
|---|---|---|
| Target | DiffOSeg Table 3, Stage I, P versus A | Original claim; pa_protocol.json |
| Data | Public preprocessed LIDC-IDRI, 128 by 128, four annotations | Dataset hash in manifest.json |
| Split | Patient fold 0, split seed 1337, 160 train / 54 test patients, 450 test patches | Original split hash; paired cluster map |
| P | Uniform subset size 1..4, then uniform subset of that size; mean selected masks | Frozen datasets/lidc.py |
| A | Mean all four masks; consume unused P subset draws to match RNG streams | Frozen reproduction/table3_claim.py |
| Common training | Batch 12, 120000 updates, Adam, 250-step cosine diffusion, final EMA | pa_runs.json; frozen configs |
| Repetitions | Seeds 20261, 20262, 20263 for both conditions | Six completed runs |
| Evaluation | 30 generated masks per patch; GED and DiceSoft | 2700 exported case records |
| Inference | Paired seed-by-patient bootstrap, 10000 draws, seed 271828 | Frozen table3_analysis.py and current CPU verifier |
| Budget | Ceiling 240 GPU-hours, at most two RTX 5090; recorded use 12.1568865645 GPU-hours on one RTX 5090 | Original gpu_usage.jsonl and final result |

For P minus A, supported requires the GED CI upper endpoint below zero and the DiceSoft CI lower endpoint above zero. Not supported requires either a significantly reversed metric. All other outcomes are inconclusive. Both condition means also have the expected ordering in this run.

## Outcome

P/A mean GED30: **0.1042761931 / 0.1043579194**.
P/A mean DiceSoft30: **90.3138729% / 90.1985670%**.

GED difference: -0.0000817264, CI [-0.0053970626, +0.0054403074].
DiceSoft difference: +0.1153060 percentage points, CI [-0.2291579, +0.4960257] percentage points.
The declared result is **inconclusive**. It is a completed comparison and does not establish P's superiority.

## Fidelity and prior exposure

The local protocol froze before this six-run comparison on 25 October. Earlier P-only corrected runs had already evaluated the same folds. Fold 0 was the fixed first fold, not selected from a new performance search. This is not a previously unseen cohort or an external registration.

The paper leaves some operational details unspecified. The archive explicitly interprets fractional consensus via a per-pixel categorical draw before diffusion, uses a uniform subset-size sampler, and applies shared intensity multiplication Uniform[0.9,1.1] with clipping. These choices are frozen assumptions. Teacher confirmation was reported by the user on 2026-11-08 and is recorded in team_plan.json; no separate approval document was supplied.

The P/A source is preserved byte-for-byte as pa_source.tar.gz. The snapshot omitted metrics_set.py, so its separately hashed dependency is included as pa_metrics_set.py. They must be used together. The current E11 runner uses a different training implementation and is not the identity of this P/A experiment.
