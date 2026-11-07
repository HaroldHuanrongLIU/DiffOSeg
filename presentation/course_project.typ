#import "@preview/touying:0.7.4": *
#import themes.university: *
#import "@preview/numbly:0.1.0": numbly
#import "../templates/UM_Typst_Template/UM_Slide_Typst_Template/tokens.typ": *
#import "../templates/UM_Typst_Template/UM_Slide_Typst_Template/components.typ": *

#let template-copyright-holder = [HUANRONG LIU]
#let template-copyright-year = "2026"
#let template-license-url = "https://creativecommons.org/licenses/by/4.0/"

#let um-corner-logo = place(
  top + right,
  dx: 0pt - logo-layout.right,
  dy: logo-layout.top,
)[
  #image("../templates/UM_Typst_Template/UM_Slide_Typst_Template/assets/um-logo-horizontal-blue.png", width: logo-layout.width)
]

#let title-layout = (
  author-gap: 1.5em,
  institution-gap: .2em,
  institution-size: .9em,
)

#show: university-theme.with(
  aspect-ratio: "16-9",
  align: horizon,
  progress-bar: true,
  header-right: self => (
    text(size: 1.1em, utils.display-current-heading(level: 1))
      + h(logo-layout.width + logo-layout.right + space-sm)
  ),
  footer-a: [
    © #template-copyright-year #template-copyright-holder · #link(template-license-url)[CC BY 4.0]
  ],
  footer-c: self => {
    set text(fill: um-blue)
    h(1fr)
    utils.display-info-date(self)
    h(1fr)
    context utils.slide-counter.display() + " / " + utils.last-slide-number
    h(1fr)
  },
  config-page(fill: white, foreground: um-corner-logo),
  config-common(new-section-slide-fn: none),
  config-colors(
    primary: um-blue,
    secondary: um-blue-light,
    tertiary: um-yellow,
    neutral-lightest: white,
    neutral-darkest: black,
  ),
  config-info(
    title: [DiffOSeg: Consensus and Preference],
    short-title: [DiffOSeg course project],
    subtitle: [CISC8006 · Claim reproduction and controlled extensions],
    author: [Keru ZHENG, Yanchen MU, Huanrong LIU],
    authors: grid(
      columns: (auto, auto, auto),
      column-gutter: title-layout.author-gap,
      ..(
        ([Keru ZHENG], [University of Macau]),
        ([Yanchen MU], [University of Macau]),
        ([Huanrong LIU], [University of Macau]),
      ).map(((author, institution)) => [
        #set align(center)
        #author
        #v(title-layout.institution-gap)
        #text(size: title-layout.institution-size, institution)
      ]),
    ),
    date: datetime(year: 2026, month: 11, day: 8),
    institution: [CISC8006 Advanced Machine Learning],
    contact: none,
    logo: none,
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))


#import "../report/shared.typ": course-table, tables

#title-slide()

= Original claim

== Claim and decision

#text(size: 1.3em, weight: "bold", fill: um-blue)[Does probabilistic consensus outperform averaging?]
#v(.8em)
Stage I · LIDC-IDRI · one patient fold · three paired seeds
#v(.7em)
- GED30 should decrease
- DiceSoft30 should increase
- Both paired 95% intervals must support the expected directions

#v(.7em)
*Observed outcome: inconclusive*

== Consensus strategies

#two-column(left-width: 1fr, right-width: 1fr, gutter: 1.5em)[
  #text(fill: um-blue, weight: "bold")[P: probabilistic consensus]
  #v(.6em)
  Sample 1–4 annotators, choose a subset uniformly and average its masks.
][
  #text(fill: um-blue-light, weight: "bold")[A: average aggregation]
  #v(.6em)
  Average all four annotation masks.
]
#v(1em)
Both conditions share architecture, training budget, augmentation and paired initialization.
#v(.6em)
The frozen implementation samples a categorical target from the consensus probabilities.

== Frozen comparison

#course-table(("Item", "Setting"), (
 ("Held-out data", "54 patients, 450 patches"),
 ("Training data", "160 patients"),
 ("Paired seeds", "20261, 20262, 20263"),
 ("Training", "120000 updates each, batch 12"),
 ("Diffusion / evaluation", "250 cosine steps / 30 samples"),
 ("Recorded compute", "12.157 GPU-hours on one RTX 5090"),
), size: 17pt, inset: 6pt)
#v(.6em)
Local protocol freeze precedes this P/A comparison. Earlier P-only runs had examined these patients.

== Three paired seeds

#course-table(("Seed", "P GED30", "A GED30", "P DS30 (%)", "A DS30 (%)"), tables.pa, size: 16pt, inset: 6pt)
#v(.8em)
P's GED improves in one seed and worsens in two.
#v(.4em)
DiceSoft has the expected ordering in all three seeds.

== Paired uncertainty

#course-table(("P minus A", "Mean change", "Paired 95% interval"), (
 ("GED30", "−0.000082", "[−0.005397, +0.005440]"),
 ("DiceSoft30", "+0.1153 pp", "[−0.2292, +0.4960] pp"),
), size: 18pt, inset: 8pt)
#v(.9em)
Both intervals include zero.
#v(.6em)
10,000 paired seed-by-patient bootstrap draws retain every patch of each sampled patient.
#v(.6em)
Patient-only and alternative empty-mask analyses reach the same outcome.

== Discrepancy diagnosis

- Pairing, checkpoint identity and metric recomputation passed
- The local P/A gap is smaller than the paper's gap
- Subset sampling, categorical encoding and augmentation include disclosed assumptions
- Split composition and three-seed variability remain possible explanations

#v(.7em)
P and A have the same expected per-pixel consensus under the frozen sampler. Shared subsets can still change spatial dependence.
#v(.5em)
No controlled study isolates the cause of the discrepancy.

= Extensions

== Improvement portfolio

#course-table(("Hypothesis", "Mechanism", "Evidence"), (
 ("H1 Boundary", "Correct localization", "Full v1 + image-only control"),
 ("H2 Polar", "Model boundary variation", "Full v2 negative result"),
 ("H3 Votes", "Supervise local disagreement", "Full v3 + control"),
 ("H4 Energy", "Calibrate sample distribution", "Failed pilot + full v4"),
), size: 16pt, inset: 6pt)
#v(.7em)
These studies use the separately frozen E11 four-fold baseline.
#v(.4em)
Historical primary endpoints: GED50, DiceSoft50 and aligned preference Dice.

== Four-fold consensus results

#course-table(("Method", "GED30", "DS30 (%)", "GED50", "DS50 (%)"), tables.consensus, size: 16pt, inset: 6pt)
#v(.6em)
v4 versus v3: GED50 change −0.003426, CI [−0.005314, −0.001551].
#v(.4em)
Small positive Dice changes have intervals that cross zero.

== Energy calibration controls

#course-table(("Variant", "GED50", "DS50 (%)", "Preference (%)"), tables.ablation, size: 15pt, inset: 5pt)
#v(.6em)
Removing the energy readout worsens GED.
#v(.3em)
The original sample order beats the new ordering in this fitted system.
#v(.3em)
Controls remove components at inference without retraining.

== Supplementary preference result

#course-table(("Fold", "Patients", "Patches", "E11 (%)", "A (%)", "E (%)"), tables.preference, size: 15pt, inset: 5pt)
#v(.6em)
E: *90.5210%*, +2.4985 pp versus E11.
#v(.4em)
Deterministic CT guidance and three SAM2.1 seeds add models and compute. The historical 91% target remains unmet.

= Study limits

== Interpretation limits

- Original claim and extensions use different frozen baselines
- Repeated use of the cohort limits generalization claims
- Patient intervals condition on the fitted extension models
- Three ensemble members are not three full-system repetitions
- Complete historical compute accounting remains unavailable

#v(.7em)
Final E's preference Dice does not replace the P/A GED30 and DiceSoft30 test.

== Evidence and team roles

#text(fill: um-blue, weight: "bold")[Portable CPU review]
#v(.5em)
#raw("python3 -B scripts/check_course_project.py")
#v(.7em)
Full replay: 2,700 case groups and 81,000 predictions. Frozen-precision metric error: zero.
#v(.7em)
*Codex technical review:* recovered the P/A archive and verified the metrics and bootstrap.
#v(.5em)
Responsibility weights: Zheng 30%, Mu 30%, Liu 40%.

== Conclusions

#text(size: 1.15em, weight: "bold", fill: um-blue)[The submitted P/A experiment is complete and inconclusive.]
#v(.9em)
Energy calibration improves GED in the separate four-fold extension.
#v(.7em)
Negative studies and component controls constrain the explanation.
#v(.7em)
The evidence package makes the results inspectable while preserving their limits.
