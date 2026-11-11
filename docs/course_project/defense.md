# Presentation and individual-defense preparation

This is preparation material, not answers supplied during an assessed defense. All three members must speak and understand the full claim. The user assigned Zheng/Mu/Liu responsibility weights of 30/30/40. The speaking allocation below is a plan, not a completed presentation record.

## Fifteen-minute talk

The English deck was reorganized on 2026-11-11 into exactly 15 pages, including the cover. It retains the pinned UM template. Scientific results and the submitted claim are unchanged.

| Speaker | Slides / topic | Target time |
|---|---|---:|
| Keru ZHENG | 1–5: problem, paper framework, P/A mechanism, claim and metrics | 4.5 min |
| Yanchen MU | 6–8: protocol, paper/local comparison, uncertainty and diagnosis | 4.5 min |
| Huanrong LIU | 9–15: hypotheses, v4, preference extension, evidence, limits and conclusions | 6 min |

Five-minute Q&A follows under the later course announcement. Slide 13 gives a brief verbal account of agent decisions and verified evidence. Its visible conclusion is: "Reported results are traceable; evidence checks passed." Commands and replay counts are retained below as speaker reference, rather than displayed on the slide or scheduled as a live demonstration.

### Optional artifact walkthrough for Q&A

If a demonstration is requested, open `results/course_project/pa_results.json`, identify the primary decision and intervals, then run `python3 -B scripts/check_course_project.py` from the repository root. This CPU command checks the portable evidence package; it does not retrain, infer masks or repeat the historical full mask replay. The separate historical replay covered 2,700 case groups and 81,000 saved predictions. Codex executed the technical checks, and team members remain accountable for explaining them. No live agent is needed during the assessed defense.

## Questions each member should answer

- What is the exact submitted claim? One Stage I P/A comparison on a frozen patient fold, three seeds per condition, GED30 and DiceSoft30, with a joint CI rule.
- Why is the result inconclusive? Both primary paired intervals include zero, despite the expected mean ordering.
- Why resample patients? Patches within a patient are dependent. Preserve all patches from each sampled patient and use the same weights for P and A.
- What does seed resampling add? It represents variability across the three fitted training seeds, with limited precision because there are only three.
- What changes between P and A? The annotator subset before consensus averaging; architecture, budget and paired RNG controls remain shared.
- Is the test cohort untouched? No. The archive discloses prior P-only evaluation. The local freeze predates this P/A comparison.
- Why does E not answer the claim? E reports aligned preference Dice on another four-fold baseline, with different implementation choices and added models.
- Which ablation is most informative? Removing v4's energy readout worsens GED; original sample ordering outperforms the new ordering, so not every component helps.
- Is the final comparison compute matched? No. Explicitly discuss added training, teachers, pretrained models and inference. Global accounting is incomplete.
- Which uncertainty is missing for extensions? Most intervals condition on fitted models; a three-seed ensemble is not three full-pipeline repetitions.
- What is the actual independent contribution? Point to code, tests, data/metric validation and method components, then give only the individual's verifiable work.
- What did the agent get checked on? The missed P/A archive, protocol identities, pairing, mask metrics and bootstrap, plus source tests. The user delegated technical checking to Codex; the executor is recorded accordingly. Full mask replay now covers all 2700 case groups.

## Slide sources and notes

The timing below totals 15 minutes. Sources and figure credits remain in these accompanying notes, following the UM template convention. Local results refer to the frozen JSON/CSV evidence, not a new experiment.

| Page | Time | Speaking point and evidence |
|---|---:|---|
| 1 · Title | 0:15 | Introduce the paper and the course task: one claim, its reproduction, and controlled extensions. Author names and UM branding are user supplied. |
| 2 · Multi-rater segmentation | 0:45 | Point to the CT and four masks. A single hard consensus hides differences; consensus and preference answer distinct questions. The displayed example is the first LIDC row of Zhang et al. (2025), Figure 2, image and annotation columns only. It is published paper evidence, not a local prediction. |
| 3 · DiffOSeg in two stages | 1:15 | Follow Stage I annotations → noisy mask → image-conditioned denoising. Stage II initializes from Stage I and adds preference prompts. Highlight Stage I as our reproduction target. Zhang et al., Figure 1 and Sections 2.1–2.2. The figure viewport removes unused outer whitespace only. |
| 4 · P versus A | 1:00 | Use the same four annotations to explain subset averaging versus all-mask averaging. Uniform subset-size sampling and per-pixel categorical draws describe our frozen interpretation, not a fully specified author implementation. Sources: submitted claim, `claim_and_protocol.md`, `pa_protocol.json`, archived P/A source. |
| 5 · Claim and metrics | 1:15 | GED compares mask distributions using IoU distance. DiceSoft compares mean prediction/annotation masks across five thresholds. K=30 is the prediction count. State the joint CI decision rule before results. Sources: paper Table 3; `claim_and_protocol.md`; `pa_metrics_set.py`. |
| 6 · Paired experiment | 1:15 | Explain patient separation, paired seeds/initialization, the shared 120,000-update budget and final EMA. Contrast four paper folds with one local fold. Disclose prior P-only exposure and encoding/augmentation assumptions. Sources: `pa_runs.json`, `pa_protocol.json`, `pa_results.json`. |
| 7 · P/A result: inconclusive | 2:00 | Paper P/A GED30: 0.0773/0.1029; DiceSoft30: 92.20/90.43%. Local means: 0.104276/0.104358 and 90.3139/90.1986%. Local differences are −0.0000817264, CI [−0.0053970626,+0.0054403074], and +0.1153060 pp, CI [−0.2291579,+0.4960257] pp. The intervals resample paired seeds and patients, retaining all patches within each sampled patient. GED favors P in only one seed; DiceSoft favors P in all three. Both primary intervals include zero. Sources: paper Table 3, `pa_results.json`, `pa_cases.csv`. |
| 8 · Discrepancy diagnosis | 1:15 | Verification reduces concern about reporting, pairing and metric mistakes; it does not establish exact fidelity to unpublished author details. The expected subset weight for each of four annotators is 1/4, so per-pixel means agree while spatial dependence can differ. No causal experiment identifies the discrepancy. Source: `diagnosis.md`; full replay record. |
| 9 · Hypotheses and pilot gates | 1:15 | Explicitly change scope to the separate four-fold Reproduced DiffOSeg extension. H1 inner Dice across folds is 89.5818%, 89.5340%, 89.8010%, 89.0577%; its fit gate requires finite outputs and at least 80% in every fold. H2 polar-only Dice is 86.3267%, 86.6294%, 86.8356%, 87.1807%, above its 78% fit gate in every fold. These are fit sanity gates, not proof of outer improvement: v2 subsequently failed all three outer directions. H3 selects among 12 consensus settings per fold by maximizing the minimum relative GED reduction / DiceSoft gain against its inner surrogate; the selected mean changes are GED50 −0.0044118193 and DiceSoft50 +1.6109572 pp. This is parameter selection with engineering checks, not a fixed superiority cutoff. H4 uses the three-metric gate on the next slide. Sources: presentation/pilot_evidence.json; v1–v4 archived result/lock records and protocols. |
| 10 · Why v4 was promoted | 1:30 | Define delta as candidate minus the locked v3 surrogate. The original gate requires all three fold-equal inner point estimates to improve: GED50 < 0, DiceSoft50 > 0 and preference Dice > 0. Decoder-only mean changes are +0.0032700911 GED, +0.1310775 pp DiceSoft and +0.1505663 pp preference Dice: fail, with no outer test. The energy revision gives −0.0069438320 GED, +0.1939209 pp DiceSoft and +0.1505663 pp preference Dice: pass. It also improves all three metrics in each inner fold, although the registered gate is on the four-fold means. Mechanism: four rank/vote heads → 16 joint binary states → a calibrated frequency → sampled masks. Keep the decoder fixed and train a frequency readout using empirical IoU GED plus three times DiceSoft loss. After passing, lock four parameter records, seal all outer predictions, then score. This gate is distinct from the original P/A confidence-interval decision rule. The check set was reused during method development. Promotion preceded the v4 outer scores; highlighting v4 in this talk is an editorial choice because it has a documented failure, targeted revision and full controls, not a claim that it is the best final system. Sources: presentation/pilot_evidence.json; docs/lidc_joint/protocol.md, energy_protocol.md and results.md. |
| 11 · Results and controls | 1:00 | v4−v3 GED50: −0.003426, CI [−0.005314,−0.001551]. No-energy worsens GED, but old sample order improves it. Energy-only has the lowest GED among these rows while sacrificing DiceSoft. These are fixed-weight inference controls, not independently retrained ablations. In particular, the no-energy outer control is not the failed decoder-only pilot: it uses the fitted energy-system parameters and was predeclared for component analysis. Small full-v4 Dice gains have CIs crossing zero. Source: `v4.json`, `improvements.md`. |
| 12 · Preference extension | 0:45 | Reproduced DiffOSeg 88.0225% → Pref-A 90.4531% → final E 90.5210%. The gain over Reproduced DiffOSeg is +2.4985 pp, paired patient CI [+1.2829,+4.4334] pp, conditional on fitted models. E adds deterministic CT guidance and a three-seed SAM2.1 combination. Mention the localization/registration work preceding these final systems. Pref-A is not average aggregation A. Area ranks represent size styles, not stable clinician IDs. The 91% point-estimate target remains unmet. Sources: `formal_e.json`, `docs/lidc_reproduction_to_best.md`. |
| 13 · Agent use and evidence | 0:30 | Explain one consequential decision: recovering the P/A archive changed the assessment to completed/inconclusive. State that reported results are traceable and evidence checks passed; keep the command walkthrough as optional Q&A reference. The historical full replay scored 2,700 case groups / 81,000 predictions; it did not retrain. Codex executed the technical checks, with student accountability; do not claim an independent classmate audit. Sources: `AI_USAGE.md`, `peer_audit.md`, replay summary. |
| 14 · Limits and next steps | 0:30 | Pair each limitation with a future test. A seed ensemble is not repeated full-system evaluation. Added compute, repeated cohort exposure and conditional patient intervals constrain extension claims. Sources: `improvements.md`, `compute.md`, `claim_and_protocol.md`. |
| 15 · Conclusions | 0:30 | End with three bounded findings: inconclusive original P/A claim; supported local GED improvement with v4; preference gains with added resources. Do not infer equivalence, clinical generalization, or superiority under matched total compute. |

Paper: Han Zhang, Xiangde Luo, Yong Chen and Kang Li. *DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model* (2025), [arXiv:2507.13087v2](https://arxiv.org/html/2507.13087v2). Figure files are unchanged downloads; presentation-only viewports are described above. Paper figures remain under their original terms and are excluded from the UM template's CC BY 4.0 license. Asset identities and regeneration instructions are in `presentation/assets/README.md`.

The requested deck has no extra appendix pages. Detailed seed/fold tables, controls and experiment history remain in `results/course_project/tables.md`, the report, and the linked experiment documents for Q&A preparation.

## Pilot-number provenance and regeneration

`presentation/pilot_evidence.json` contains only aggregate metrics, fold numbers, decisions and source identities from 20 historical fold records. It omits patient identifiers, case indices, images, weights and predictions. Its source list gives archive-relative paths and SHA-256 hashes; the five protocol hashes bind the displayed rules. The numbers distinguish fit sanity gates, inner parameter selection, the v4 promotion gate and outer evaluation. No new training, prediction, significance test or preregistration was performed for this slide update.

Rebuild the portable pilot evidence from the existing local archive:

```bash
python3 -B presentation/prepare_pilot_evidence.py --archive-root /mnt/hdd1/phd_course/cisc8006/DiffOSeg_runs
```

The extractor reads each original `result.json` or `lock-F.json`, preserves the locked controls and computes equal means across four folds. It checks the H1/H2 recorded gate flags, the H3 selected option, the failed/passed H4 decisions, and agreement with the already archived v4 inner summary. Compile the slides after regeneration using the existing paired PDF/PPTX workflow.
