#import "@preview/touying:0.7.4": *
#import themes.university: *
#import "@preview/numbly:0.1.0": numbly
#import "../templates/UM_Typst_Template/UM_Slide_Typst_Template/tokens.typ": *
#import "../templates/UM_Typst_Template/UM_Slide_Typst_Template/components.typ": *

#let um-corner-logo = place(
  top + right,
  dx: 0pt - logo-layout.right,
  dy: logo-layout.top,
)[
  #image("../templates/UM_Typst_Template/UM_Slide_Typst_Template/assets/um-logo-horizontal-blue.png", width: logo-layout.width)
]

#let title-layout = (
  author-gap: 1.5em,
)

#show: university-theme.with(
  aspect-ratio: "16-9",
  align: horizon,
  progress-bar: true,
  header-right: self => (
    text(size: 1.1em, utils.display-current-heading(level: 1))
      + h(logo-layout.width + logo-layout.right + space-sm)
  ),
  footer-a: [E11-1049],
  footer-c: self => {
    set text(fill: um-blue)
    h(1fr)
    utils.display-info-date(self)
    h(1fr)
    context str(counter(page).get().first() + 1) + " / " + str(counter(page).final().first() + 1)
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
    subtitle: none,
    author: [Keru ZHENG, Yanchen MU, Huanrong LIU],
    authors: grid(
      columns: (auto, auto, auto),
      column-gutter: title-layout.author-gap,
      [Keru ZHENG],
      [Yanchen MU],
      [Huanrong LIU],
    ),
    date: datetime(year: 2026, month: 11, day: 11),
    institution: [CISC8006 Advanced Machine Learning],
    contact: none,
    logo: none,
  ),
)

#set heading(numbering: numbly("{1}.", default: "1.1"))


#import "../report/shared.typ": course-table

#show table: it => align(center, it)
#set table(align: left)

// Values remain linked to the frozen evidence; no experiment is rerun here.
#let pa = json("../results/course_project/pa_results.json").primary
#let v4 = json("../results/course_project/v4.json").metrics
#let pilots = json("pilot_evidence.json")
#let fmt(value, digits: 4) = {
  let parts = str(calc.round(value, digits: digits)).split(".")
  let decimals = parts.at(1, default: "")
  parts.first() + "." + decimals + "0" * (digits - decimals.len())
}

#title-slide()

= Paper and problem

== Multi-rater segmentation

#let example-layout = (width: 720pt, height: 155pt, image-width: 1600pt, dx: -60pt, dy: -5pt)
#align(center)[
  #box(width: example-layout.width, height: example-layout.height, clip: true)[
    #align(top + left)[
      #move(dx: example-layout.dx, dy: example-layout.dy)[
        #image("assets/diffoseg-paper-consensus.png", width: example-layout.image-width, height: example-layout.image-width * 9 / 16, fit: "stretch")
      ]
    ]
  ]
]
#v(.35em)
#align(center)[#text(size: 18pt)[One CT patch · four expert annotations]]
#v(.7em)
#two-column(gutter: 1.8em, alignment: top)[
  #text(fill: um-blue, weight: "bold")[Consensus]
  #v(.3em)
  Which segmentation patterns are shared, and where do annotations disagree?
][
  #text(fill: um-blue-light, weight: "bold")[Preference]
  #v(.3em)
  How can a prediction reflect a particular annotation style?
]

== DiffOSeg in two stages

#let framework-layout = (
  left-width: 510pt,
  right-width: 1fr,
  gutter: 32pt,
  width: 510pt,
  height: 294pt,
  image-width: 742.6pt,
)
#two-column(left-width: framework-layout.left-width, right-width: framework-layout.right-width, gutter: framework-layout.gutter)[
  #box(width: framework-layout.width, height: framework-layout.height, clip: true)[
    #align(top + left)[
      #image("assets/diffoseg-paper-framework.png", width: framework-layout.image-width, height: framework-layout.image-width * 9 / 16, fit: "stretch")
    ]
  ]
][
  #set text(size: 19pt)
  #text(fill: um-blue, weight: "bold")[Stage I]
  #v(.2em)
  Image-conditioned categorical diffusion learns a distribution of masks.
  #v(.7em)
  #text(fill: um-blue-light, weight: "bold")[Stage II]
  #v(.2em)
  Initialize from Stage I; add adaptive preference prompts.
  #v(.7em)
  *Our target:* the Stage I consensus strategy.
]

== P versus A

#two-column(gutter: 1.8em)[
  #text(fill: um-blue, weight: "bold")[P: probabilistic consensus]
  #v(.5em)
  Choose 1–4 annotations, then average the selected subset.
  #v(.6em)
  $z_P = frac(1, abs(S)) sum_(j in S) z_j$
  #v(.4em)
  The training target changes with the subset.
][
  #text(fill: um-blue-light, weight: "bold")[A: average aggregation]
  #v(.5em)
  Average all four annotations on every visit.
  #v(.6em)
  $z_A = 1 / 4 sum_(j=1)^4 z_j$
  #v(.4em)
  The consensus probabilities stay fixed.
]
#v(.8em)
#text(size: 18pt)[Frozen local choices: uniform subset size and subset; per-pixel categorical target draws in both arms. These operational details are disclosed assumptions.]

= Claim reproduction

== Claim and metrics

#text(size: 24pt, weight: "bold", fill: um-blue)[Does P outperform A on LIDC-IDRI?]
#v(.4em)
Table 3 · Stage I · scaled-faithful · one patient fold · three paired seeds
#v(.6em)
#course-table(("Endpoint", "What it measures", "Expected change"), (
  ("GED30", "Mismatch between mask distributions", "P − A < 0"),
  ("DiceSoft30", "Threshold-aware overlap of mean masks", "P − A > 0"),
), size: 19pt, inset: 7pt)
#v(.45em)
#text(size: 18pt)[30 = generated masks per image. DiceSoft uses five thresholds.]
#v(.5em)
#text(size: 19pt)[
  *Supported:* both paired 95% intervals favor P. \
  *Not supported:* either metric is significantly reversed. \
  *Inconclusive:* all other outcomes.
]

== Paired experiment

#course-table(("Design choice", "Frozen local comparison"), (
  ("Patient separation", "160 train / 54 test patients; 450 test patches"),
  ("Repeated training", "P and A × 3 paired seeds; matched initialization"),
  ("Shared budget", "120,000 updates; batch 12; final EMA"),
  ("Preserved settings", "128 × 128; 250 cosine steps; K = 30"),
  ("Compute", "12.157 GPU-hours used / 240-hour ceiling"),
), size: 18pt, inset: 6pt)
#v(.6em)
#text(size: 19pt)[Paper: four folds. Local test: one fold, with frozen consensus encoding and augmentation assumptions.]
#v(.35em)
#text(size: 18pt)[Earlier P-only runs had already examined this cohort. The freeze predates this P/A comparison, not all prior exposure.]

== P/A result: inconclusive

#course-table(("Metric", "Paper P / A", "Local P / A"), (
  ("GED30 ↓", "0.0773 / 0.1029", fmt(pa.P_mean.at(0), digits: 6) + " / " + fmt(pa.A_mean.at(0), digits: 6)),
  ("DiceSoft30 ↑", "92.20 / 90.43%", fmt(100 * pa.P_mean.at(1)) + " / " + fmt(100 * pa.A_mean.at(1)) + "%"),
), size: 18pt, inset: 6pt)
#v(.2em)
#image("assets/pa-paired-intervals.svg", width: 100%, height: 127pt, fit: "contain")
#v(.15em)
#text(size: 20pt, weight: "bold", fill: um-blue)[Both paired intervals cross zero: the declared outcome is inconclusive.]
#v(.25em)
#text(size: 17pt)[95% intervals use 10,000 paired seed-by-patient draws; patients retain all patches. Paper: four-fold mean. Local: three-seed mean on one fold.]

== Discrepancy diagnosis

#two-column(gutter: 1.6em)[
  #set text(size: 21pt)
  #text(fill: um-blue, weight: "bold")[Checks completed]
  #v(.4em)
  - Matching split, budget and paired initialization
  - Source and checkpoint identities checked
  - Saved mask metrics and bootstrap reproduced
][
  #set text(size: 21pt)
  #text(fill: um-blue-light, weight: "bold")[Explanations still open]
  #v(.4em)
  - Consensus encoding and augmentation choices
  - Split composition and training variability
  - One-fold scale and prior cohort exposure
]
#v(.65em)
#text(size: 21pt)[Under our sampler, $E[z_P] = z_A$ at each pixel; shared subsets can still change spatial dependence.]
#v(.45em)
#text(size: 19pt)[This mechanism does not identify the cause of the paper–local gap. An inconclusive result does not prove equivalence.]

= Controlled extensions

== Hypotheses and pilot gates

#text(size: 18pt)[Separate four-fold extensions of Reproduced DiffOSeg; pilots use outer-training patients.]
#v(.45em)
#course-table(("Hypothesis / limitation", "Quantitative inner evidence", "Rule / next step"), (
  ([*H1 · Boundary* \ Localization error],
   "Dice " + fmt(100 * pilots.boundary.dice_min, digits: 2) + "–" + fmt(100 * pilots.boundary.dice_max, digits: 2) + "%",
   "≥80% in every fold → v1"),
  ([*H2 · Polar* \ Contour variation],
   "Dice " + fmt(100 * pilots.polar.dice_min, digits: 2) + "–" + fmt(100 * pilots.polar.dice_max, digits: 2) + "%",
   "≥78% in every fold → v2"),
  ([*H3 · Votes* \ Pixel disagreement],
   [ΔGED50 #fmt(pilots.votes.mean_delta.at("GED@50"), digits: 6) \ ΔDiceSoft50 +#fmt(100 * pilots.votes.mean_delta.at("Dice_soft@50"), digits: 3) pp],
   [12 options/fold; maximize \ minimum relative gain → v3]),
  ([*H4 · Energy* \ Distribution mismatch],
   [Decoder-only fails; \ energy readout passes],
   [Three-metric gate → v4 \ Details on the next slide]),
), size: 17pt, inset: 6pt)
#v(.5em)
#text(size: 17pt)[H1/H2: standalone Dice ranges across inner folds; finite-output checks also required. H3: mean changes vs its inner surrogate; parameter selection, not a fixed cutoff.]
#v(.35em)
#text(size: 18pt)[Passing a pilot permits evaluation. v2 later failed all three outer improvement directions.]

== Why v4 was promoted

#text(size: 21pt, weight: "bold", fill: um-blue)[Gate: all three four-fold mean changes must improve]
#v(.3em)
#text(size: 17pt)[Candidate − locked v3 surrogate, K=50. Required: ΔGED < 0; both ΔDice > 0. Dice changes are percentage points.]
#v(.5em)
#course-table(("Inner pilot", "ΔGED50", "ΔDiceSoft50", "ΔPref. Dice", "Decision"), (
  ("Decoder only", "+" + fmt(pilots.v4.decoder_only.mean_delta.at("GED@50"), digits: 6),
   "+" + fmt(100 * pilots.v4.decoder_only.mean_delta.at("Dice_soft@50"), digits: 3),
   "+" + fmt(100 * pilots.v4.decoder_only.mean_delta.at("Dice_mean"), digits: 3), "FAIL; no outer test"),
  ("+ Energy readout", fmt(pilots.v4.energy.mean_delta.at("GED@50"), digits: 6),
   "+" + fmt(100 * pilots.v4.energy.mean_delta.at("Dice_soft@50"), digits: 3),
   "+" + fmt(100 * pilots.v4.energy.mean_delta.at("Dice_mean"), digits: 3), "PASS → full v4"),
), size: 17pt, inset: 7pt)
#v(.6em)
#text(size: 18pt)[*Revision:* keep the 16-state rank/vote decoder; fit a frequency readout with *GED + 3 × DiceSoft loss* to calibrate the sampled masks.]
#v(.45em)
#text(size: 19pt, weight: "bold", fill: um-blue)[Pass inner gate → lock parameters → seal outer predictions → score]
#v(.4em)
#text(size: 17pt)[This is a point-estimate gate, not a significance test. Inner checks were reused during development. We focus on v4 because the failed pilot led to a targeted revision and full controls.]

== Results and controls

#course-table(("Variant", "GED50 ↓", "DiceSoft50 (%) ↑"), (
  ("Votes v3", fmt(v4.at("GED@50").v3, digits: 6), fmt(100 * v4.at("Dice_soft@50").v3, digits: 3)),
  ("Full v4", fmt(v4.at("GED@50").full, digits: 6), fmt(100 * v4.at("Dice_soft@50").full, digits: 3)),
  ("No energy readout", fmt(v4.at("GED@50").no_energy, digits: 6), fmt(100 * v4.at("Dice_soft@50").no_energy, digits: 3)),
  ("Original sample order", fmt(v4.at("GED@50").no_order, digits: 6), fmt(100 * v4.at("Dice_soft@50").no_order, digits: 3)),
  ("Energy-only on v3", fmt(v4.at("GED@50").energy_only, digits: 6), fmt(100 * v4.at("Dice_soft@50").energy_only, digits: 3)),
), size: 18pt, inset: 6pt)
#v(.55em)
#text(size: 20pt, weight: "bold", fill: um-blue)[v4 − v3 GED50: −0.003426; 95% CI [−0.005314, −0.001551]]
#v(.35em)
#text(size: 18pt)[The old order improves GED further; energy-only trades DiceSoft for GED. The small v4 Dice gains have intervals crossing zero.]
#v(.3em)
#text(size: 17pt)[No-energy is a fixed-weight inference control, not the failed pilot. Patient CIs condition on fitted models; total compute is not matched.]

== Preference extension

#two-column(left-width: 1.25fr, right-width: 1fr, gutter: 1.2em)[
  #image("assets/preference-results.svg", width: 100%, height: 210pt, fit: "contain")
][
  #set text(size: 19pt)
  #text(size: 26pt, weight: "bold", fill: um-blue)[90.5210%]
  #v(.3em)
  *+2.4985 pp* vs Reproduced DiffOSeg
  #v(.5em)
  Localization / registration, followed by SAM adaptation and a three-seed SAM2.1 combination.
  #v(.5em)
  Extra models and compute; the historical 91% target remains unmet.
]
#v(.4em)
#text(size: 18pt)[Four-fold preference Dice averages four area-ranked masks. Ranks are size-based styles, not persistent clinician identities.]
#v(.3em)
#text(size: 17pt)[Same repeatedly used cohort. Pref-A is an extension system, unrelated to A = average aggregation in the P/A test.]

= Evidence and limits

== Agent use and evidence

#course-table(("Phase", "Consequential action", "Verification record"), (
  ("Understand / audit", "Recovered the omitted P/A archive", "Frozen source, six runs and pairing"),
  ("Implement / check", "Built CPU evidence checks", "Metric fixtures and package checks"),
  ("Analyze / challenge", "Retained the inconclusive decision", "Full saved-mask and bootstrap replay"),
), size: 17pt, inset: 7pt)
#v(.8em)
#text(size: 21pt, fill: um-blue, weight: "bold")[Reported results are traceable; evidence checks passed.]

== Limits and next steps

#course-table(("Current limit", "Next verification"), (
  ("P/A: one fold and only three seeds", "More folds / seeds; isolate encoding assumptions"),
  ("Repeated use of the LIDC cohort", "Freeze the method and test a new cohort"),
  ("Added models and incomplete compute totals", "Account for costs; compare matched budgets"),
  ("Extension CIs condition on fitted models", "Repeat the complete training and selection process"),
), size: 19pt, inset: 8pt)
#v(.75em)
#text(size: 20pt)[A three-seed ensemble is not three independent system repetitions.]
#v(.45em)
#text(size: 19pt)[The P/A claim, four-fold consensus study and preference extension retain separate conclusions.]

== Conclusions

#text(size: 25pt, weight: "bold", fill: um-blue)[P/A reproduction: complete, but inconclusive]
#v(.45em)
The local advantages are small and both paired intervals cross zero.
#v(.85em)
#text(size: 25pt, weight: "bold", fill: um-blue)[Energy calibration: a supported local GED gain]
#v(.45em)
Negative results and component controls limit the explanation.
#v(.85em)
#text(size: 25pt, weight: "bold", fill: um-blue)[Preference extension: 90.5210% with added resources]
#v(.45em)
Promising on this cohort; generalization and compute fairness remain open.
