# Compute and cost accounting

## Verified P/A comparison

The frozen ceiling is 240 GPU-hours with at most two RTX 5090 GPUs. The actual controller used one RTX 5090. Summing the fourteen recorded worker durations in gpu_usage.jsonl gives **12.156886564493439 GPU-hours**, including preflight, training and evaluation. The source-packaging failure occurred before GPU work; its incident record is retained externally.

This is allocated worker elapsed time, not a device-energy estimate. Waiting time is excluded. Shared-host utilization varied, so elapsed time is not a portable FLOP-equivalent budget. Monetary cost was not separately metered and no billing receipt was found.

## Extensions

The [registry](../../results/course_project/registry.csv) inventories 145 archive roots, including exploratory, failed, partial and final work. The label indexed_archive means only that the directory and available summary were indexed.

The selected extensions add training steps, image branches and later public pretrained models. Their per-component timings and machine details are scattered across receipts. The package does **not** add incomplete timings into a supposed whole-project total. Those rows remain UNACCOUNTED; SEE_RUN_CONFIG and SEE_RUN_RECEIPTS are explicit missing aggregation fields, not zero cost.

Consequently the P/A comparison is within its ceiling, while compliance of the full historical programme with a team-wide ceiling is **unverified**. The user has reported teacher confirmation. The statement does not supply a missing total or specify how the historical ceiling applies across all extensions. No new GPU run is needed to resolve the documentation itself.

## Timer coverage inventory

compute_receipts.csv lists 534 existing fit/result timer receipts, including 129 with a recorded GPU name, together with each receipt hash. compute_coverage.json defines the bounded search. Timers may omit stages or overlap and allocation counts are not uniformly recorded. This index supports reconciliation; its arithmetic sum is deliberately not represented as total GPU-hours.
