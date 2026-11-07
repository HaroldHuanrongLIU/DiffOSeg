# Presentation and individual-defense preparation

This is preparation material, not answers supplied during an assessed defense. All three members must speak and understand the full claim. The user assigned Zheng/Mu/Liu responsibility weights of 30/30/40. The speaking allocation below is a plan, not a completed presentation record.

## Fifteen-minute talk

| Speaker | Slides / topic | Target time |
|---|---|---:|
| Keru ZHENG | 1–4: claim, P/A mechanism and frozen protocol | 4.5 min |
| Yanchen MU | 5–7: results, uncertainty and discrepancy diagnosis | 4.5 min |
| Huanrong LIU | 8–14: improvements, controls, limits and reproducibility | 6 min |

Five-minute Q&A follows under the later course announcement. A bounded demo can replace part of the final three minutes: show the manifest, run the CPU smoke, and explain the claim-to-artifact link. No training or live agent is needed.

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

Slide 1: supplied author names and user-selected UM template.
Slides 2–7: submitted claim, pa_protocol.json, pa_results.json, pa_runs.json, pa_cases.csv and diagnosis.md.
Slides 8–10: v1/v2/v3/v4 protocols and machine-readable summaries; K=50 remains the historical primary endpoint.
Slide 11: formal_e.json and independent mask audit.
Slides 12–14: disclosed limitations, reproducibility commands and AI_USAGE.md.

Paper facts and Table 3 references use Zhang et al., arXiv:2507.13087v2. Local numbers come from the exact artifacts above. Sources are recorded here instead of adding long credit lines to slide bodies, following the selected template's guidance.
