# Consequential coding-agent use

Updated 2026-11-11. The user delegated technical checking to Codex, assigned team responsibilities at 30:30:40 and reported teacher confirmation. This ledger names the actual executor rather than presenting an agent run as a member's manual action.

Tool: Codex in the desktop app. Exact model snapshot/version and sampling settings were not exposed in the visible task record. No token/cost estimate is inferred.

| Date / phase | Delegated task | Decision and evidence | Agent verification | Human instruction / verification distinction |
|---|---|---|---|---|
| 2026-11-08 / understanding and audit | Read requirements and claim; audit repository and archives | Original claim retained; completed P/A archive recovered | Source/protocol/checkpoint identities, six-run results, paired statistics | User authorized closeout; agent performed technical inspection |
| 2026-11-08 / implementation and validation | Build evidence package, CPU checks, report and slides | Scripts and deliverables prepared; original experiments preserved | validation.json: smoke, tests, source checks, PDF/PPTX review | User selected template, supplied authors and delegated checking |
| 2026-11-08 / analysis and critical review | Separate P/A, consensus extensions and preference result | Inconclusive P/A outcome retained; scope limits disclosed | Numeric and source evidence, including original 19308-score preference audit | Teacher confirmation recorded from the user's statement |
| 2026-11-08 / requested results replay | Recompute all saved P/A masks and bootstrap | Full metric/statistical replay completed | replay_20261108_verified/summary.json: 2700 case groups, 81000 predictions, exact frozen-precision agreement | Executed by Codex at the user's request; no manual member or independent-classmate execution is claimed |
| 2026-11-11 / presentation revision | Restructure the English slides into the approved 15-page narrative, retaining the UM template | Paper/problem context added; P/A, v4 and preference evidence remain separate; no experiment or scientific result changed | 15-page PDF visual review; 15 PPTX slide images match same-source renders; evidence smoke passed; presentation_validation.json | User explicitly requested the 15-page rebuild and PDF/PPTX; Codex performed the artifact checks |

## Critical-review episodes

1. The README omitted the P/A workflow. Archive discovery and verification changed the assessment to “experiment complete, claim inconclusive.” This conclusion rests on the stored source, runs and recomputed statistics.
2. Full replay initially compared independent float64 arithmetic with historical float32 accumulation using a 1e-7 tolerance. Case 116 of P/20261 differed by 1.0374e-7, while the reference implementation reproduced the stored value. The initial failure is retained. The checker then independently calculated integer intersections and used the frozen float32 division/means for the primary comparison, keeping a separate float64 diagnostic. Agreement was tested at the stricter 1e-12 tolerance. All primary scores match exactly; the largest float64 diagnostic difference is 2.0123e-7. No historical score or decision was changed.

## Relevant user instructions

- Read-only audit: “结合仓库现有代码、实验结果和文档，对照课程要求及已提交的 claim，检查我们完成了什么、还缺什么。先只读审计，不修改文件。”
- Closeout: “按你建议的顺序收尾。”
- Allocation: “你帮我分工一下，3：3：4的比重；zheng：mu：liu”
- Delegated checking: “你替代成员检查”
- Teacher confirmation: “老师确认”
- Verification request: “重新跑一次复现结果就行。”

The last request was implemented as replay of frozen predictions and statistics. No model retraining or fresh model inference occurred. The user-reported teacher confirmation is not an independently retrieved written approval or an unreported exception.

## Historical evidence and responsibility

Historical test logs and receipts demonstrate software/evidence activities, not the identity of whoever originally prompted or reviewed an agent. The member weights in CONTRIBUTIONS.md allocate responsibilities and do not retrospectively identify personal authorship of every artifact.

## Data handling

Medical arrays were processed locally. Portable CSVs contain local integer patient clusters rather than original identifiers. No image or mask was uploaded by these scripts. Codex received code/document excerpts and aggregate numerical results; prior public browsing queried paper/repository pages. Historical sessions' complete data-egress history is not asserted.

Each member remains responsible for understanding the work. A live agent must not answer assessed defense questions unless the instructor permits it.
