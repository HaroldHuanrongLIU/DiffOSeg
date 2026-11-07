# Discrepancy diagnosis

This section distinguishes observations, implementation facts and untested explanations. It does not assign a causal explanation without a controlled experiment.

## What the comparison establishes

| Quantity | Paper Table 3 | Local single-fold three-seed mean |
|---|---:|---:|
| P GED30 | 0.0773 | 0.104276 |
| A GED30 | 0.1029 | 0.104358 |
| P DiceSoft30 (%) | 92.20 | 90.3139 |
| A DiceSoft30 (%) | 90.43 | 90.1986 |

The observed P/A gap is much smaller than the paper's gap. P's GED is better in one of the three paired seeds and worse in two. DiceSoft has the expected ordering in all three, but its patient-and-seed interval includes zero. This is evidence for an inconclusive comparison, not a significant reversal.

## Checks that narrow the explanation

- Six runs share the split, training budget, architecture and evaluation definition. Initialization hashes match within each seed pair.
- The frozen source, protocol, dataset and checkpoint identities passed the local audit.
- Aggregating 2700 case records reproduces the reported means. Repeating all 10000 paired bootstrap draws reproduces the CI endpoints to floating-point precision.
- Independent foreground-mask calculations on 18 case/condition/seed groups differ by at most 4.65e-8 from saved GED/DiceSoft.
- Alternative empty-mask GED and patient-only intervals also leave the claim inconclusive. The primary analysis remains the registered seed-by-patient analysis.

These checks reduce the likelihood of a reporting, pairing or metric-implementation mistake. They do not verify that the local implementation equals the authors' unreleased training procedure.

## Mechanism and remaining uncertainty

The frozen P sampler chooses k uniformly from 1..4 and then a subset uniformly. For any expert j, the probability of inclusion conditional on k is k/4 and its weight is 1/k. Therefore its expected weight is 1/4. P and A have the same expected per-pixel consensus before categorical encoding. P can still change cross-pixel dependence through shared expert subsets. This analytic fact explains why changing aggregation does not guarantee an improvement; it is not a demonstrated cause of the observed gap.

Plausible differences include the weight distribution and consensus encoding, unspecified intensity augmentation, fixed split composition, finite training variability and the single-fold scale. The paper's results average four folds, while this test uses a previously examined first fold. No experiment in this dossier isolates these factors. A future sensitivity study would require its own registered scope and compute allocation.

## Relation to E11 and later improvements

E11 uses four patient folds and one seed recipe (1337 plus fold), preserves soft consensus in its training path and omits the unspecified intensity scaling. The P/A archive samples a categorical target and includes a frozen scaling range. Subsequent extensions use E11's saved outputs and the E11 metric contract, not the P/A checkpoints.

Boundary supervision, count modelling and energy calibration address observed localization/distribution limitations of that separate baseline. Their results support those local comparisons. They do not retrospectively change the P/A claim, explain its gap by themselves, or establish that P exceeds A.
