#import "shared.typ": *
#import "../templates/UM_Typst_Template/UM_Slide_Typst_Template/tokens.typ": *
#let pilots = json("../presentation/pilot_evidence.json")
#let preference90 = json("../results/course_project/preference90.json")
#let pref-a = json("../results/course_project/formal_a.json")
#let final-e = json("../results/course_project/formal_e.json")
#let fixed(value, digits: 4) = {
  let parts = str(calc.round(value, digits: digits)).split(".")
  let decimals = parts.at(1, default: "")
  parts.first() + "." + decimals + "0" * (digits - decimals.len())
}
#let signed(value, digits: 3) = (if value > 0 { "+" } else { "" }) + fixed(value, digits: digits)
#let consensus-rows = tables.consensus.map(row => if row.first() == "E11" {
  ("Reproduced DiffOSeg", ..row.slice(1))
} else { row })
#let preference-stages = (
  ("Reproduced DiffOSeg", final-e.means.e11, "Frozen four-fold baseline"),
  ("v9", preference90.means.v9, "Combined consensus / preference branches"),
  ("Localization + registration", preference90.means.full, "CT alignment, localization and native-grid refinement"),
  ("Pref-A", pref-a.means.candidate, "Patient cross-fitting and SAM adaptation"),
  ("Final E", final-e.means.candidate, "Deterministic guidance and three-seed SAM2.1"),
)
#let preference-path = preference-stages.enumerate().map(((i, row)) => (
  row.at(0), fixed(100 * row.at(1)),
  if i == 0 { "-" } else { signed(100 * (row.at(1) - preference-stages.at(i - 1).at(1)), digits: 4) },
  row.at(2),
))
#set document(title: "DiffOSeg: Reproduction of a Consensus Claim and Controlled Extensions", author: ("Keru ZHENG", "Yanchen MU", "Huanrong LIU"))
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: .65em)
#set heading(numbering: "1.")
#show heading: set text(fill: um-blue)
#set page(paper: "a4", margin: (x: 19mm, top: 20mm, bottom: 19mm),
  header: [
    #set text(size: 8pt, fill: um-blue)
    CISC8006 · Course project evidence report #h(1fr) 11 November 2026
  ],
  footer: context [
    #set text(size: 8pt, fill: um-blue)
    Submission candidate #h(1fr) #counter(page).display()
  ],
)
#align(right)[#image("../templates/UM_Typst_Template/UM_Slide_Typst_Template/assets/um-logo-horizontal-blue.png", width: 42mm)]
#v(4mm)
#text(size: 22pt, weight: "bold", fill: um-blue)[DiffOSeg: Reproduction of a Consensus Claim and Controlled Extensions]
#v(3mm)
#text(size: 12pt)[Keru ZHENG · Yanchen MU · Huanrong LIU]
#linebreak()
#text(size: 10pt)[University of Macau · CISC8006 Advanced Machine Learning]
#v(3mm)
*Abstract.* We evaluate the submitted claim that DiffOSeg's probabilistic consensus strategy (P) improves over average annotation aggregation (A) on LIDC-IDRI [1]. A locally frozen comparison uses one patient fold, three paired training seeds per condition and 30 generated masks per image. Six runs completed. Mean GED is 0.104276 for P and 0.104358 for A. Mean threshold-aware Dice is 90.3139% and 90.1986%, respectively. Both paired 95% intervals include zero, yielding an *inconclusive* claim under the registered rule. Separately, existing four-fold studies investigate boundary correction, polar residual sampling, cumulative-vote supervision and energy calibration against a frozen Reproduced DiffOSeg baseline. The full v4 extension reduces GED50 relative to v3 by 0.003426, while its small Dice increments remain uncertain. A supplementary preference system reaches 90.5210% aligned Dice. We preserve negative results, distinguish the protocols and document limits in prior cohort exposure, training variability and compute matching.

= Target claim and scope

The submitted unit of work is Table 3's P/A comparison, rather than the paper's full experimental programme. The stated tier is *scaled-faithful*: retain the Stage I task, four-rater labels, 128×128 images, 250-step cosine categorical diffusion and 30-sample evaluation, while reducing four-fold evaluation to one fixed patient fold.

The protocol defines the paired difference as P minus A. Support requires a negative GED confidence interval and a positive DiceSoft confidence interval. A significantly reversed metric yields “not supported.” Otherwise the result is “inconclusive.” The outcome and fidelity tier are separate fields.

This dossier preserves that original claim. The four-fold extensions have a separate baseline and endpoint contract. Teacher confirmation is recorded from the team's communication on 8 November 2026. Responsibility allocation and verification execution are recorded separately; no unreported deadline or specific exemption is inferred.

= Data and frozen execution

The preprocessed LIDC file contains 1,609 CT patches from 214 patients and four binary masks per patch. A patient-disjoint fold uses 160 training patients and 54 held-out patients, with 450 held-out patches. The split seed is 1337 and the selected fold is 0. The data SHA-256 begins #raw("6c401c96b0e1737d4")#text(";") the complete value and evidence hashes are in the accompanying manifest.

P and A each use seeds 20261, 20262 and 20263, batch size 12 and 120,000 updates. Initialization hashes match within each pair. The architecture, optimization settings, data split, augmentation and final EMA checkpoint rule are shared. Recorded execution uses one RTX 5090 and 12.157 GPU-hours, including preflight and evaluation, within the frozen 240 GPU-hour ceiling.

#pagebreak()
= Method and evaluation

For P, the implementation samples a subset size uniformly from 1 to 4, selects that many annotators uniformly, and averages their masks. A averages all four masks. The A path consumes unused subset draws so paired random streams remain aligned. Both use a per-pixel categorical draw from the consensus probabilities before diffusion. Shared intensity multiplication uses Uniform[0.9,1.1] and clips to [0,1]. These operational interpretations are explicitly disclosed.

For masks $a,b$, the distance is $d(a,b)=1-"IoU"(a,b)$. The empirical GED uses all cross-pairs and both within-bank pair sets, including their diagonals:
$ "GED" = 2 "mean"(d(Y,X)) - "mean"(d(Y,Y)) - "mean"(d(X,X)). $
The primary implementation retains the released empty/empty IoU convention; an alternative is reported separately. DiceSoft thresholds the mean ground-truth and prediction masks at 0.1, 0.3, 0.5, 0.7 and 0.9, computes foreground Dice at each threshold, then averages them.

Within a seed, patch scores have equal weight. The final estimate averages three seeds equally. The paired bootstrap resamples both the three training seeds and the 54 patients, retaining all patches of each sampled patient. We use 10,000 percentile replicates with seed 271828. This preserves within-patient dependence and P/A pairing.

= Original claim result

#course-table(("Seed", "P GED30", "A GED30", "P DS30 (%)", "A DS30 (%)"), tables.pa)
#v(2mm)
The mean GED difference is *−0.0000817264*, with 95% CI *[−0.0053970626, +0.0054403074]*. The mean DiceSoft difference is *+0.1153060 percentage points*, with CI *[−0.2291579, +0.4960257]* pp. Both intervals cross zero. The outcome is *inconclusive*. Patient-only intervals conditional on the fitted seeds and the alternative empty-mask metric retain the same outcome.

The paper reports P/A GED30 of 0.0773/0.1029 and DiceSoft30 of 92.20/90.43% [1]. The local P/A gap is substantially smaller. P's GED improves in one paired seed and worsens in two. Its DiceSoft ordering is positive in all three, but the combined seed-and-patient uncertainty is unresolved. The single-fold result does not establish the paper's four-fold absolute scores.

== Evidence validation

The audit verified the frozen source and protocol identities, six checkpoint hashes and identical paired initialization. All 2,700 case records reproduce the reported aggregates. Repeating the bootstrap reproduces the CI endpoints to floating-point precision. The saved-mask replay on 8 November 2026 scored all 2,700 saved case groups, comprising 81,000 predictions, using integer set intersections. With the frozen float32 accumulation, every primary case score matches exactly. A separate float64 diagnostic differs by at most $2.013 times 10^(-7)$. These checks establish the integrity of this comparison, while leaving fidelity to unspecified author settings unresolved.

#pagebreak()
= Discrepancy diagnosis

*Observed facts.* A's absolute metrics are comparatively close to the paper values, while P's reported advantage is not recovered. No metric has a significantly reversed paired interval. The evidence therefore supports the declared inconclusive result, rather than either superiority or equivalence.

*Known implementation choices.* The paper does not uniquely specify the subset-weight law or the handling of fractional consensus as a categorical state. The local experiment freezes a uniform subset-size sampler and categorical target draw. Its intensity-scaling range, patient ordering and final-checkpoint policy are also explicit choices. Prior corrected P-only runs had already evaluated the same folds. The local freeze precedes this P/A comparison, not all exposure to these patients.

*Analytic mechanism.* Conditional on subset size $k$, an annotator enters P with probability $k/4$ and receives weight $1/k$. Its expected weight is therefore $1/4$. P and A share the expected per-pixel consensus, although a shared random subset can change the joint spatial distribution. This fact shows why a gain is not automatic; it does not prove the reason for the observed discrepancy.

*Unresolved explanations.* Split composition, finite seed variability, consensus encoding and augmentation could affect the result. No factorial study isolates their effects. The source corrections were applied together, so individual performance gains cannot be assigned to them. Further testing would need its own hypothesis, scope and budget.

= Improvement hypotheses and pilots

The extensions use *Reproduced DiffOSeg*, a separately frozen four-fold baseline called E11 in the archive [3]. Its training preserves soft consensus and omits the unspecified intensity scaling used in the P/A archive. Stage II uses stable foreground-area ranks as simulated preferences, rather than persistent identities of four radiologists. The extension endpoints are GED50, DiceSoft50 and aligned preference Dice; K=30 remains secondary.

#course-table(("Hypothesis / limitation", "Inner pilot evidence", "Rule and subsequent outcome"), (
  ("H1 Boundary: localization error",
   "Dice " + fixed(100 * pilots.boundary.dice_min, digits: 2) + "–" + fixed(100 * pilots.boundary.dice_max, digits: 2) + "%",
   "Finite outputs; Dice ≥80% in every fold. Passed; full v1 evaluated."),
  ("H2 Polar: contour variation",
   "Dice " + fixed(100 * pilots.polar.dice_min, digits: 2) + "–" + fixed(100 * pilots.polar.dice_max, digits: 2) + "%",
   "Finite outputs; Dice ≥78% in every fold. Passed; full v2 was negative."),
  ("H3 Votes: pixel disagreement",
   "ΔGED50 " + signed(pilots.votes.mean_delta.at("GED@50"), digits: 6) + "; ΔDiceSoft50 " + signed(100 * pilots.votes.mean_delta.at("Dice_soft@50")) + " pp",
   "Select among 12 settings per fold; maximize the smaller relative GED / DiceSoft gain. Full v3 evaluated."),
  ("H4 Energy: sample-distribution mismatch", "Decoder-only failed; energy revision passed",
   "All three fold-equal inner means must improve over the locked v3 surrogate. See the next section."),
), size: 9pt, inset: 5pt)
#v(2mm)

H1/H2 report the range of standalone-model Dice across four inner folds. Their fit sanity gates permit further training and evaluation; they do not establish outer performance gains. H3 reports the fold-equal changes of the selected settings against its inner surrogate proposal. It uses engineering checks and parameter selection, not a fixed numerical superiority threshold. Selection patients are excluded from the fitted inner models and residual library, but have been reused during development.

The pilot numbers are extracted from 20 archived fold records, with source and protocol hashes in #raw("presentation/pilot_evidence.json") [6]. These are existing results. No new training, inference or preregistration was performed for this report revision.

#pagebreak()
= V4 method and promotion decision

== Joint decoding and distribution calibration

The v4 method starts from frozen image models and saved diffusion predictions [3]. Four rank heads predict area-ordered preference masks; four cumulative-vote heads predict regions supported by at least one, two, three or four annotations. A joint decoder enumerates the 16 possible four-bit label states at each pixel. It combines the rank probabilities with the vote-count law and normalizes within each count group, so the four rank marginals and cumulative supports agree on the expected count. A shared location correction moves the continuous maps toward a reference centroid estimated only from inner training patients.

The decoder and location module use 12 learned scalars. A separate readout adds 10 scalars to calibrate the foreground frequency of generated masks. Given frequency $f$ and sample count $K$, it allocates $floor(K f + 0.5)$ foreground labels at each pixel and assigns them using the fixed sample-order rule. Each K=10/30/50 bank uses only its own prefix. The preference output keeps the existing boundary blend.

The modules are fitted separately for 320 Adam updates each on the inner calibration patients. The readout minimizes
$ L = "GED"_50 + 3 (1 - "DiceSoft"_50). $
The forward objective scores discrete masks; a smooth surrogate supplies gradients. The inner study uses a surrogate sample bank because the original full-fold diffusion checkpoints had already seen the inner validation patients. Outer evaluation instead uses the frozen diffusion-derived bank. Calibration and check groups are separate within each outer training fold, while the check group has been reused across development rounds.

== Failed pilot, revision and outer evaluation

The registered promotion rule requires *all three fold-equal inner point estimates* to improve against the locked v3 surrogate: ΔGED50 < 0, ΔDiceSoft50 > 0 and Δpreference Dice > 0. It is not a confidence-interval rule and differs from the original P/A decision criterion. Dice changes below are percentage points; GED changes are in its original units.

#course-table(("Inner pilot", "ΔGED50", "ΔDiceSoft50", "ΔPref. Dice", "Decision"), (
  ("Decoder only", signed(pilots.v4.decoder_only.mean_delta.at("GED@50"), digits: 6),
   signed(100 * pilots.v4.decoder_only.mean_delta.at("Dice_soft@50")),
   signed(100 * pilots.v4.decoder_only.mean_delta.at("Dice_mean")), "Fail; outer test not run"),
  ("With energy readout", signed(pilots.v4.energy.mean_delta.at("GED@50"), digits: 6),
   signed(100 * pilots.v4.energy.mean_delta.at("Dice_soft@50")),
   signed(100 * pilots.v4.energy.mean_delta.at("Dice_mean")), "Pass; full v4 evaluated"),
), size: 9.5pt, inset: 5pt)
#v(2mm)

The first decoder improved Dice but worsened GED, so it was not promoted. The energy revision kept the preference decoder fixed and trained the final sample distribution directly. It passed the mean-based rule and also improved all three scores in each inner fold. The four parameter records were then locked, all outer predictions were sealed, and only then were outer scores read. The failed pilot and its gate log remain archived [6].

We focus on v4 because its failed pilot led to a targeted revision and full component controls. This editorial choice does not make it the best preference system or replace the original endpoints. Reused checks and the inner-to-outer sample-bank change limit what the pilot can establish; the outer results provide the subsequent test.

#pagebreak()
= Controlled extension results

The following table uses identical original patient folds and equal weighting of four fold means. DS denotes DiceSoft. These scores are distinct from the P/A single-fold experiment.

#course-table(("Method", "GED30", "DS30 (%)", "GED50", "DS50 (%)"), consensus-rows)
#v(2mm)

The polar study fails to improve all three declared directions against v1. Its DiceSoft50 falls by 0.938 pp, with interval [−1.165,−0.752] pp. This negative result remains part of the portfolio. Vote supervision and energy calibration subsequently improve the distribution metrics in their respective comparisons.

For v4 versus v3, GED50 changes by *−0.003426*, CI *[−0.005314,−0.001551]*. DiceSoft50 changes by *+0.093 pp*, CI *[−0.005,+0.189]* pp. Aligned preference Dice changes by *+0.009 pp*, CI *[−0.092,+0.107]* pp. The strongest evidence concerns GED. Small positive Dice estimates do not establish a reliable gain.

== Component controls

#course-table(("Variant", "GED50", "DS50 (%)", "Preference Dice (%)"), tables.ablation, size: 9.5pt)
#v(2mm)
Removing the energy readout worsens GED50 to 0.115012. Reusing the original sample order instead improves GED50 relative to full v4. Thus the controls support the energy readout's contribution within this fitted system and do not support a benefit from the new sample ordering. The controls disable components at inference using the same fitted parameters; they are not independently retrained systems. In particular, the no-energy control is not the failed decoder-only pilot: it removes the readout from the fitted full system. The energy-only control has lower GED but worse DiceSoft, illustrating a trade-off rather than uniform superiority.

== Fairness and limitations

Matched data, patient partitions, metrics and reference predictions make these within-study comparisons interpretable. Additional networks, supervision and inference make total compute unequal. Patient bootstrap intervals condition on the fitted models, and one seed per fold is not three independent repetitions. The check cohort has been reused across development rounds. These constraints limit causal and generalization claims and require explicit acceptance under the course's uncertainty and compute rules.

#pagebreak()
= Supplementary preference study

A later programme targets aligned four-rank preference Dice [4]. The following table follows the locked primary stages. It reports equal means across the same four patient folds; the baseline-to-v9 increment combines earlier extensions, and all listed increments compare systems rather than isolate individual components.

#course-table(("System", "Dice (%)", "Δ previous (pp)", "Main change"), preference-path, size: 9pt, inset: 5pt)
#v(2mm)

The localization/registration stage adds *+#fixed(100 * (preference90.means.full - preference90.means.v9))* pp over v9, with paired patient CI [+#fixed(100 * preference90.delta_ci95.at(0)), +#fixed(100 * preference90.delta_ci95.at(1))] pp. It combines training-time CT alignment, image-based localization and refinement on the native grid. Pref-A adds patient cross-fitted guidance and SAM adaptation, improving by *+#fixed(pref-a.delta_vs_reference_pp)* pp, with paired patient CI [+#fixed(pref-a.paired_reference_ci95_pp.at(0)), +#fixed(pref-a.paired_reference_ci95_pp.at(1))] pp. Final E adds deterministic continuous guidance and a three-seed SAM2.1 Hiera-Tiny combination; its increment over Pref-A is *+#fixed(final-e.delta_vs_reference_pp)* pp, CI [+#fixed(final-e.paired_reference_ci95_pp.at(0)), +#fixed(final-e.paired_reference_ci95_pp.at(1))] pp.

Final E includes earlier fitted components and public generic visual pretraining. All four outer-fold predictions were sealed before scoring the selected candidate. Its fold-level results are:

#course-table(("Fold", "Patients", "Patches", "Reproduced DiffOSeg (%)", "Pref-A (%)", "Final E (%)"), tables.preference, size: 9pt, inset: 5pt)
#v(2mm)
Final E achieves *90.5209738%* Dice_mean, *+2.4985112 pp* over Reproduced DiffOSeg, with paired patient CI *[+1.282858,+4.433427]* pp. Fold 3 slightly worsens relative to Pref-A. The historical 91% point-estimate target remains unmet.

An independent calculation recomputed 19,308 Dice values for Reproduced DiffOSeg, Pref-A and Final E from the original masks and sealed predictions, with maximum discrepancy zero. It checked mutually exclusive coverage of all 214 patients and 1,609 patches. This establishes numeric integrity within the repeatedly used cohort; it is not independent institutional validation.

Final E's three seeds are ensemble members, not three full training-and-selection repetitions. The additional models and compute prevent an equal-budget superiority claim. Its GED and DiceSoft were not newly evaluated, so v9's consensus scores are not assigned to Final E. These preference results do not answer the original P/A claim. Pref-A is the later system's name, distinct from A = average aggregation in that claim.

#pagebreak()
= Reproducibility and compute

The portable package contains frozen P/A code, six-run metadata, 2,700 case scores, machine summaries, generated tables and 20 aggregate pilot records with source hashes [6]. Images, weights and prediction banks remain external; anonymized patient clusters preserve resampling. The separate four-fold source does not replace the frozen P/A implementation.

The CPU check validates hashes, coverage, pairing and the claim decision; NumPy enables bootstrap replay. The 8 November saved-mask replay used no training or fresh inference. Full model replay requires the original data, weights, environment and configurations.

P/A used 12.157 GPU-hours within its 240-hour ceiling. The inventory of 145 archive roots is not a complete compute total: extension timings and model-search costs remain incompletely reconciled. Whole-programme budget compliance and equal-compute superiority are not established.

= Agent use, responsibility and delivery

Codex assisted the audit, evidence packaging and analysis. A consequential review recovered the P/A workflow from archived runs when it was missing from the README. Verified source identities and recomputed statistics established a completed experiment with an inconclusive result. Tests cover missing/duplicate cases, changed pairing, invalid metrics and small mask fixtures. The technical checks were executed by Codex at the team's request; no independent classmate audit is claimed.

#course-table(("Member", "Responsibility weight", "Assigned responsibility"), (
  ("Keru ZHENG", "30%", "Claim, P/A method, data and protocol"),
  ("Yanchen MU", "30%", "Evaluation, statistics and diagnosis"),
  ("Huanrong LIU", "40%", "Improvements, integration, report and slides"),
), size: 9pt, inset: 4pt)
#v(1mm)
These responsibility weights do not measure speaking time or individual authorship of every artifact. The presentation assigns five slides and about five minutes to each member: Zheng 1–5, Mu 6–10 and Liu 11–15. All 15 PPTX pages contain English speaker notes, timing and sources; their editable text is in #raw("presentation/speaker_notes.json"). The report, slides and pilot extractor share the same archived numerical evidence.

= Conclusions

The original single-fold, three-seed P/A comparison is complete and inconclusive. The separate four-fold study shows a supported GED gain from energy calibration, while small Dice increments remain uncertain. The final preference system reaches 90.5210% with added models and compute. Negative pilots, fixed-weight controls, repeated cohort exposure and conditional uncertainty constrain these conclusions. External validation and complete historical compute accounting remain outstanding.

= References and artifact sources

#set par(justify: false)
#set text(size: 9pt)

[1] H. Zhang, X. Luo, Y. Chen and K. Li. *DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model*. 2025. arXiv:2507.13087v2. #link("https://arxiv.org/abs/2507.13087v2")[Primary paper].

[2] Frozen P/A comparison, 20261025_table3_pa_claim_v1. Portable protocol, results and source identities: #raw("results/course_project/") and #raw("manifest.json").

[3] Reproduction and extension protocols: #raw("docs/e11_lidc.md"), #raw("docs/lidc_boundary/"), #raw("docs/lidc_polar/"), #raw("docs/lidc_vote/") and #raw("docs/lidc_joint/").

[4] Preference progression and Final E audit: #raw("docs/lidc_reproduction_to_best.md") and #raw("docs/lidc_91/outer_stable_refinement_results.md")#text(";") machine reports: #raw("preference90.json"), #raw("formal_a.json") and #raw("formal_e.json") in the course evidence package.

[5] CISC8006 specification and later announcements: #raw("docs/course_project_requirement/"). Operational draft status and later announcements remain distinct.

[6] Pilot evidence and extraction: #raw("presentation/pilot_evidence.json") and #raw("presentation/prepare_pilot_evidence.py"). See #raw("docs/course_project/evidence_map.md") and #raw("docs/course_project/defense.md").

#v(2mm)
#text(size: 8pt)[Design attribution: UM Typst Template, © 2026 HUANRONG LIU, CC BY 4.0. This A4 report uses its palette and logo, not an official course-report layout. University assets retain their own terms; see LICENSES.md.]
