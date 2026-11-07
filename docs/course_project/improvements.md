# Improvement portfolio and evaluation scope

This document organizes existing work. The hypotheses and historical promotion rules below come from their archived protocols; this assembly is not a new preregistration. Teacher confirmation is recorded separately from the user's 2026-11-08 statement.

## Hypotheses, pilots and selection

| ID | Diagnosed limitation / mechanism | Prediction and pilot | Full evaluation and outcome |
|---|---|---|---|
| H1: boundary correction | Diffusion outputs retain diversity but aligned contours have localization error | A shared four-rank image model can correct geometry; inner fits exclude validation patients and pass a finite-output / 0.80 Dice gate | v1 four-fold evaluation. GED50 0.113739, DiceSoft50 90.136%; aligned Dice 88.658%. Image-only control retained |
| H2: polar residuals | Area rank alone may miss correlated boundary variability | Angular residual sampling may improve the distribution; polar-only inner Dice gate 0.78 | v2 four-fold evaluation fails all three point-estimate improvement directions against v1; negative evidence retained |
| H3: vote supervision | Whole-contour size rank does not directly supervise per-pixel disagreement | Joint rank and cumulative-vote heads plus full/zoom views can improve fidelity and sample frequencies; selection uses inner patients | v3 four-fold evaluation improves the three declared directions; image-only control and prior failed v2 retained |
| H4: energy calibration | Consistent marginal votes do not ensure a good spatial sample distribution | Fit a joint decoder and final distribution readout using energy distance; initial decoder-only pilot fails, energy pilot passes | v4 full evaluation and component-removal controls; GED benefit supported, small Dice increments uncertain |

See [v1](../lidc_boundary/protocol.md), [v2](../lidc_polar/protocol.md), [v3](../lidc_vote/protocol.md), and [v4](../lidc_joint/results.md) for exact procedures, seeds, splits and prior exposure.

The report presents v4 as the complete extension study, with H1-H3 explaining its inherited components and negative results. This is an editorial selection of existing evidence. The user has reported teacher confirmation; no unreported specific rescope or exemption wording is inferred.

## Endpoints and controls

Historical extension gates use GED50, DiceSoft50 and aligned preference Dice. All preserve K=10/30/50 outputs. K=30 results are secondary and connect the metric definition to the original claim without replacing the declared K=50 endpoints.

v4 versus v3: GED50 difference -0.003426 with paired 95% CI [-0.005314,-0.001551]. DiceSoft50 difference +0.093 pp with CI [-0.005,+0.189] pp. Aligned Dice difference +0.009 pp with CI [-0.092,+0.107] pp.

Removing the energy readout raises GED50 to 0.115012. Keeping the old sample order gives 0.103678, better than the complete v4's 0.104515. Thus the evidence supports the energy readout, not a positive contribution from every component. These removals use the same fitted parameters rather than separately retrained controls.

## Fairness and uncertainty limits

All comparisons use matched patient partitions, dataset bytes, metric definitions and saved reference predictions. Additional networks, supervision and inference cost differ. This is not an equal-total-compute comparison. Patient-cluster intervals condition on the fitted models. One seed per fold is not three independent repetitions; the final E's three-seed ensemble is also not three full-system replications. Teacher confirmation is recorded from the user's statement. These statistical and compute limitations remain part of the scientific interpretation.

## Supplementary preference result

E11 88.0224625%, formal A 90.4531349%, formal E 90.5209738% aligned Dice_mean. E minus E11 is +2.4985112 pp, paired patient interval [+1.282858,+4.433427] pp. E uses deterministic CT guidance and a three-seed SAM2.1 combination. It does not achieve the historical 91% target. It is a repeated-cohort result with additional pretrained models and compute.

The final report separates this result from the original P/A claim and the earlier consensus endpoints. It does not assign v9's old GED to E.
