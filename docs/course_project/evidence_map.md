# Results to artifacts

All paths below are relative to the repository. Original archive-relative paths and SHA-256 values are in results/course_project/manifest.json. Set the archive root when checking the original local copies.

| Claim / table | Portable evidence | Producing implementation / external artifacts |
|---|---|---|
| Original P/A outcome and three-seed table | pa_results.json, pa_runs.json, pa_cases.csv, pa_protocol.json | Frozen pa_source.tar.gz plus pa_metrics_set.py; six runs under 20261025_table3_pa_claim_v1 |
| P/A paired uncertainty | pa_cases.csv, pa_results.json | Frozen table3_analysis.py; scripts/check_course_project.py --bootstrap independently reaggregates |
| E11 four-fold reference | e11.json | 20261106_rank_boundary_v1/baseline/fourfold_summary.json; original E11 checkpoints remain external |
| H1 boundary correction | v1.json | reproduction/rank_contours.py, boundary_eval.py; frozen v1 source and assessment |
| H2 negative polar study | v2.json | reproduction/polar_contours.py, polar_eval.py; frozen v2 assessment |
| H3 vote supervision | v3.json | reproduction/vote_contours.py, vote_eval.py; frozen v3 assessment |
| H4 energy calibration and ablations | v4.json | reproduction/joint_energy.py, joint_study.py; frozen v4 assessment |
| Pilot numbers, promotion gates and v4 selection in slides 9–10 | presentation/pilot_evidence.json | presentation/prepare_pilot_evidence.py; 20 archived fold result/lock records with hashes and five protocol identities |
| Historical branch combination | v9.json | Frozen order_preference_v9 assessment |
| Supplemental 90%, A and E | preference90.json, formal_a.json, formal_e.json | Corresponding archived selection, sealed predictions, training receipts |
| Negative and partial history | registry.csv; docs/lidc_91/results.md | Each listed external run; inventory alone does not prove completion |
| Tables in report and slides | tables.json, tables.md | scripts/render_course_tables.py from the checked JSON |
| Compute coverage | compute_receipts.csv, compute_coverage.json | Explicit archived fit/result timing fields; not a GPU-hour total |
| Full P/A mask replay | replay_20261108_verified/summary.json and recomputed_cases.csv | scripts/replay_course_results.py; all 2700 groups, 81000 masks; Codex executor |
| Team allocation / confirmation | team_plan.json | User-provided 30/30/40 allocation, delegated checking and teacher confirmation |
| This package verification | validation.json | New smoke/tests plus archived-source test rerun; exact commands recorded |

The short filenames in the second column live in results/course_project/; explicitly prefixed presentation/ paths are relative to the repository root. Medical images, masks, trained weights and public-model binaries remain outside Git. The included patient_cluster values are newly assigned integers for paired bootstrap and reveal no original patient identifier.

Historical logs reported 34 tests for the P/A source and 139 for formal E. They refer to those frozen sources, not automatically to the current checkout. New verification is recorded separately.
