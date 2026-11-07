#import "shared.typ": *
#import "../templates/UM_Typst_Template/UM_Slide_Typst_Template/tokens.typ": *
#set document(title: "DiffOSeg: Reproduction of a Consensus Claim and Controlled Extensions", author: ("Keru ZHENG", "Yanchen MU", "Huanrong LIU"))
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: .65em)
#set heading(numbering: "1.")
#show heading: set text(fill: um-blue)
#set page(paper: "a4", margin: (x: 19mm, top: 20mm, bottom: 19mm),
  header: [
    #set text(size: 8pt, fill: um-blue)
    CISC8006 · Course project evidence report #h(1fr) 8 November 2026
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
*Abstract.* We evaluate the submitted claim that DiffOSeg's probabilistic consensus strategy (P) improves over average annotation aggregation (A) on LIDC-IDRI [1]. A locally frozen comparison uses one patient fold, three paired training seeds per condition and 30 generated masks per image. Six runs completed. Mean GED is 0.104276 for P and 0.104358 for A. Mean threshold-aware Dice is 90.3139% and 90.1986%, respectively. Both paired 95% intervals include zero, yielding an *inconclusive* claim under the registered rule. Separately, existing four-fold studies investigate boundary correction, polar residual sampling, cumulative-vote supervision and energy calibration against a frozen E11 baseline. The full v4 extension reduces GED50 relative to v3 by 0.003426, while its small Dice increments remain uncertain. A supplementary preference system reaches 90.5210% aligned Dice. We preserve negative results, distinguish the protocols and document limits in prior cohort exposure, training variability and compute matching.

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

The audit verified the frozen source and protocol identities, six checkpoint hashes and identical paired initialization. All 2,700 case records reproduce the reported aggregates. Repeating the bootstrap reproduces the CI endpoints to floating-point precision. A fresh independent replay scores all 2,700 saved case groups, comprising 81,000 predictions, using integer set intersections. With the frozen float32 accumulation, every primary case score matches exactly. A separate float64 diagnostic differs by at most $2.013 times 10^(-7)$. These checks establish the integrity of this comparison, while leaving fidelity to unspecified author settings unresolved.

#pagebreak()
= Discrepancy diagnosis

*Observed facts.* A's absolute metrics are comparatively close to the paper values, while P's reported advantage is not recovered. No metric has a significantly reversed paired interval. The evidence therefore supports the declared inconclusive result, rather than either superiority or equivalence.

*Known implementation choices.* The paper does not uniquely specify the subset-weight law or the handling of fractional consensus as a categorical state. The local experiment freezes a uniform subset-size sampler and categorical target draw. Its intensity-scaling range, patient ordering and final-checkpoint policy are also explicit choices. Prior corrected P-only runs had already evaluated the same folds. The local freeze precedes this P/A comparison, not all exposure to these patients.

*Analytic mechanism.* Conditional on subset size $k$, an annotator enters P with probability $k/4$ and receives weight $1/k$. Its expected weight is therefore $1/4$. P and A share the expected per-pixel consensus, although a shared random subset can change the joint spatial distribution. This fact shows why a gain is not automatic; it does not prove the reason for the observed discrepancy.

*Unresolved explanations.* Split composition, finite seed variability, consensus encoding and augmentation could affect the result. No factorial study isolates their effects. The source corrections were applied together, so individual performance gains cannot be assigned to them. Further testing would need its own hypothesis, scope and budget.

= Improvement hypotheses and pilots

The extension programme uses another frozen, four-fold E11 baseline. It retains soft consensus during training and omits the unspecified intensity scaling. It also trains Stage II using stable foreground-area ranks as simulated preferences. These are not persistent identities of four real radiologists.

#course-table(("Hypothesis", "Mechanism and pilot", "Disposition"), (
  ("H1 Boundary", "Four-rank image model corrects localization; inner Dice gate 0.80", "Full v1 evaluation"),
  ("H2 Polar", "Angular residual sampling models correlated boundaries; inner gate 0.78", "Full v2 negative result"),
  ("H3 Votes", "Cumulative vote supervision models per-pixel disagreement", "Full v3 evaluation"),
  ("H4 Energy", "Calibrate the final discrete sample distribution", "Failed initial pilot; full v4"),
), size: 9pt)

Each study preserves its historical protocol and selection rule. Inner training, calibration and check patients remain within the respective outer training fold. Previously seen results on the same cohort are disclosed. The hypothesis count and completed pilots exceed the course minimum, and the team has reported teacher confirmation.

The report selects v4 to explain a complete improvement investigation. This editorial choice does not replace the historical endpoint: GED50, DiceSoft50 and aligned preference Dice were the extension's declared targets. K=30 was preserved as a secondary endpoint throughout.

#pagebreak()
= Controlled extension results

The following table uses identical original patient folds and equal weighting of four fold means. DS denotes DiceSoft. These scores are distinct from the P/A single-fold experiment.

#course-table(("Method", "GED30", "DS30 (%)", "GED50", "DS50 (%)"), tables.consensus)
#v(2mm)

The polar study fails to improve all three declared directions against v1. Its DiceSoft50 falls by 0.938 pp, with interval [−1.165,−0.752] pp. This negative result remains part of the portfolio. Vote supervision and energy calibration subsequently improve the distribution metrics in their respective comparisons.

For v4 versus v3, GED50 changes by *−0.003426*, CI *[−0.005314,−0.001551]*. DiceSoft50 changes by *+0.093 pp*, CI *[−0.005,+0.189]* pp. Aligned preference Dice changes by *+0.009 pp*, CI *[−0.092,+0.107]* pp. The strongest evidence concerns GED. Small positive Dice estimates do not establish a reliable gain.

== Component controls

#course-table(("Variant", "GED50", "DS50 (%)", "Preference Dice (%)"), tables.ablation, size: 9.5pt)
#v(2mm)
Removing the energy readout worsens GED50 to 0.115012. Reusing the original sample order instead improves GED50 relative to full v4. Thus the controls support the energy readout's contribution within this fitted system and do not support a benefit from the new sample ordering. The controls disable components at inference using the same fitted parameters; they are not independently retrained systems.

== Fairness and limitations

Matched data, patient partitions, metrics and reference predictions make these within-study comparisons interpretable. Additional networks, supervision and inference make total compute unequal. Patient bootstrap intervals condition on the fitted models, and one seed per fold is not three independent repetitions. The check cohort has been reused across development rounds. These constraints limit causal and generalization claims and require explicit acceptance under the course's uncertainty and compute rules.

#pagebreak()
= Supplementary preference study

A later programme targets aligned four-rank preference Dice. Its final selected system E uses deterministic CT guidance and three SAM2.1 Hiera-Tiny seeds, along with earlier fitted components and public generic visual pretraining. All original four-fold predictions were sealed before scoring this candidate.

#course-table(("Fold", "Patients", "Patches", "E11 (%)", "A (%)", "E (%)"), tables.preference, size: 9.5pt)
#v(2mm)
E achieves *90.5209738%* Dice_mean. Its improvement over E11 is *+2.4985112 pp*, with paired patient interval *[+1.282858,+4.433427]* pp. Relative to formal A, the improvement is +0.067839 pp. Fold 3 slightly worsens relative to A. The result does not meet the historical 91% point-estimate target.

An independent audit recomputed 19,308 Dice values for E11, A and E from the original masks and sealed predictions, with maximum discrepancy zero. It checked mutually exclusive coverage of all 214 patients and 1,609 patches. This is strong numeric integrity evidence for a repeated-cohort study, not independent institutional validation.

E's ensemble uses three seeds as model members. It is not three independent repetitions of the entire training-and-selection pipeline. Its historical GED and DiceSoft endpoints were not newly evaluated, so v9's consensus values are not attached to E. E's preference Dice also does not answer the original P/A claim.

= Reproducibility and compute

The portable package includes the P/A source archive, separately hashed metric dependency, six-run metadata, 2,700 scores, machine summaries and generated tables. Integer patient clusters preserve resampling while removing original identifiers. The current E11 training modules are not substituted for the P/A frozen implementation.

A standard-library CPU smoke validates hashes, source identity, paired case coverage, initialization and the outcome. NumPy enables exact bootstrap replay. Full model replay additionally requires external data, weights, runtime receipts and original configuration identities. The requested replay checks all saved P/A masks and recomputes the bootstrap. It does not include retraining, fresh model inference or a clean installation on another person's machine.

P/A consumed 12.157 GPU-hours under its 240-hour ceiling. The package inventories 145 historical archive roots but does not claim a complete aggregate compute total. Distributed extension receipts and model-search costs have not been fully reconciled. No claim of whole-programme budget compliance or equal-compute superiority follows from that inventory.

#pagebreak()
= Agent use, responsibility and delivery

Codex materially assisted the evidence audit, packaging implementation and analysis. A consequential critical-review episode occurred when the current README failed to expose the P/A workflow. Searching the archived runs, inspecting the exact source and reproducing the statistics changed the assessment to “completed experiment, inconclusive claim.” The evidence, rather than an agent assertion, supports that decision.

The checking tests reject missing/duplicate cases, changed pairing and invalid metrics; additional fixtures validate disjoint masks, equal distributions and the empty-mask convention. At the team's request, Codex executes the technical checks. The record identifies the agent as executor and does not represent an independent member or classmate audit.

#course-table(("Member", "Weight", "Assigned responsibility"), (
  ("Keru ZHENG", "30%", "Claim, P/A method, data and protocol"),
  ("Yanchen MU", "30%", "Evaluation, statistics and diagnosis"),
  ("Huanrong LIU", "40%", "Improvements, integration, report and slides"),
), size: 9pt)
#v(2mm)
These are team-assigned responsibility weights. They do not measure time or retrospectively attribute every shared artifact to a single member.

The requested repeat verification independently scores the full frozen P/A prediction banks and recovers the same inconclusive decision. The initial float32/float64 discrepancy and its resolution are retained. Teacher confirmation is recorded as reported by the team; it does not change the scientific limitations or constitute a separately retrieved approval document.

= Conclusions

The frozen single-fold, three-seed comparison completed and returned an inconclusive P/A result. The extension portfolio provides multiple falsifiable hypotheses, pilots, negative findings and a full controlled study. Energy calibration yields a supported GED improvement within the later four-fold setting, while small Dice increments remain uncertain. The supplementary E system improves aligned preference Dice on the repeatedly used cohort, with added compute and model dependencies.

These findings support a defensible research account when the original claim, extension scope and evidence limits remain separate. Responsibilities, delegated verification and teacher confirmation are now documented with their actual sources. Full historical compute accounting and independent-cohort validation remain limitations.

= References and artifact sources

#set par(justify: false)

[1] H. Zhang, X. Luo, Y. Chen and K. Li. *DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model*. 2025. arXiv:2507.13087v2. #link("https://arxiv.org/abs/2507.13087v2")[Primary paper].

[2] Frozen P/A protocol and results, 20261025_table3_pa_claim_v1. Portable copies: results/course_project/pa_protocol.json and pa_results.json. The complete source fingerprint is recorded in manifest.json.

[3] E11 adaptation and extension protocols: docs/e11_lidc.md, docs/lidc_boundary/, docs/lidc_polar/, docs/lidc_vote/ and docs/lidc_joint/. Numerical source hashes are preserved in the course manifest.

[4] Formal E report and independent audit, 20261106_preference91_formal_e. See docs/lidc_91/outer_stable_refinement_results.md and the original frozen predictions.

[5] CISC8006 course specification and later announcements, locally preserved in docs/course_project_requirement/. Operational draft status and subsequent announcements must be distinguished.

#v(3mm)
#text(size: 8pt)[Design attribution: UM Typst Template, © 2026 HUANRONG LIU, CC BY 4.0. This A4 report adapts its UM palette and logo assets; the source repository provides poster and slide templates, not an official course-report layout. University identity assets retain their own terms. Content and source attribution are detailed in LICENSES.md.]
