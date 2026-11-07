# CISC8006 Course Project：要求原文汇编

整理日期：2026-11-08。以下保留课程页面和公告的原文，按来源分节；格式转换为 Markdown。展示排期表为公告附件，链接见对应公告，附件内容未转录。

**状态说明：**完整规范页面自称 publication draft，部分操作细则须待教师正式公布；不同日期的原文可能存在差异，请核对后续公告。

---

## Course Project: Full Specification and Grading Criteria

来源：[https://ummoodle.um.edu.mo/mod/page/view.php?id=1202975](https://ummoodle.um.edu.mo/mod/page/view.php?id=1202975)

## CISC8006 Course Project: Specification and Grading Criteria

> Draft for publication · 2026-08-22 Course: CISC8006 Advanced Machine Learning: Theory & Methods Assessment: 100% course project; no examination Publication status: not final until the operational decisions in Section 11 are announced in UMMoodle

For a shorter version suitable for students, see the [Student Guide](https://ummoodle.um.edu.mo/mod/page/view.php?id=1202972). This detailed specification remains the reference for edge cases and full operational rules.

### 1. Assessment summary

There is no midterm or final examination. The entire course grade is determined by one team-based course project with individual technical assessment.

The project has three scientific stages:

1. **Identify a paper and define one testable claim.** Select a machine-learning or deep-learning paper and convert one central result into a precise, feasible reproduction target.
2. **Reproduce the claim.** Reconstruct the relevant method, evaluation protocol, and evidence. Compare the result with the paper and diagnose agreement or discrepancy.
3. **Improve upon the paper.** Form diagnosis-driven hypotheses, pilot alternatives, and fully evaluate at least one improvement with fair controls and ablations.

Coding agents such as Claude Code, Codex, or an approved equivalent must be used materially. Agents may assist with understanding, implementation, testing, debugging, experimental design, analysis, and review. Students remain responsible for every submitted line of code, command, citation, result, and claim.

Each team must maintain a reproducible repository and give a final presentation. Every student must also complete an individual technical defense.

### 2. What the project is—and is not

#### 2.1 The unit of work is one defensible claim

The project is not a request to reproduce an entire paper. Each team must freeze one target such as:

- one table row or figure trend;
- one comparison between two methods;
- one ablation result;
- one component, loss, or algorithmic claim; or
- one bounded capability claim under a specified evaluation protocol.

A strong target claim identifies:

- the system or method being tested;
- the dataset and split;
- the baseline or comparison;
- the metric and direction of comparison;
- the reproduction tier;
- the compute ceiling;
- the expected paper result or ordering; and
- the rule used to classify the target claim as supported, not supported, or inconclusive.

Example:

> Under the approved scaled-faithful setting, Method B will recover the paper's reported ordering over Method A on validation negative log-likelihood, using the frozen split, metric implementation, and compute-matched protocol.

#### 2.2 Reproduction tiers

The instructor will approve one of three tiers.

| Tier | Definition |
| --- | --- |
| **Exact reproduction** | The original data, model, and evaluation can realistically be rerun. |
| **Scaled-faithful reproduction** | Scale is reduced, but the central hypothesis, comparison, and evaluation remain faithful. |
| **Component reproduction** | One central algorithmic claim, loss, module, or ablation is independently reconstructed and tested. |

The tiers carry no automatic grade advantage. Scope and workload are normalized at approval. A rigorous component reproduction can receive a higher mark than a superficial exact rerun.

Running an official repository unchanged is not sufficient. If official code is used, the team must independently reconstruct, audit, test, or materially validate at least one important component or claim.

#### 2.3 Paper eligibility

A proposed paper must have a course-feasible empirical claim and an auditable path to data, evaluation, and implementation. The approval form must disclose:

- paper citation and primary-source link;
- target claim, table, figure, or equation;
- official and third-party code;
- data and model access;
- licences and permitted use;
- expected hardware, runtime, storage, and API cost;
- evaluation metrics and baselines;
- major feasibility or contamination risks; and
- the fallback component claim if the preferred scope fails.

Survey papers, theory-only papers, or projects dependent on inaccessible private systems require explicit approval and a separately testable empirical claim. No project may depend on confidential employer data, leaked material, hidden benchmark labels, or unapproved private APIs.

### 3. Required project workflow

#### 3.1 Reproduction requirements

The reproduction must:

1. state the exact target claim and approved tier;
2. document all intentional deviations from the paper;
3. reproduce the relevant data split, preprocessing, metric, and comparison as faithfully as feasible;
4. validate the evaluation implementation on a fixture, known example, or synthetic test;
5. report repeated runs and uncertainty for stochastic comparisons;
6. compare the result directly with the paper using the same units and definitions; and
7. diagnose discrepancies rather than hiding them.

At least three seeds are required by default for stochastic comparisons. An exemption must be approved in the Week 5 protocol against the team's fixed compute ceiling and must include an alternative uncertainty or sensitivity analysis.

Claim status and reproduction scope are separate fields. Report the claim as **supported**, **not supported**, or **inconclusive** under the frozen protocol; separately identify the reproduction tier and any fidelity limitations.

#### 3.2 Improvement requirements

After the reproduction result is understood, each team must:

1. propose **at least three** improvement hypotheses;
2. connect each hypothesis to a diagnosed limitation, failure mode, or theoretical mechanism;
3. pilot **at least two** approved hypotheses;
4. preserve all pilot results, including negative results;
5. select **at least one** hypothesis for full evaluation;
6. compare it fairly against the frozen reproduction baseline;
7. include appropriate controls and ablations; and
8. report uncertainty, limitations, and failure cases.

An improvement does not need to produce a positive gain. A well-designed, honestly reported null or negative result can receive full credit. Marks reward scientific reasoning and evidence—not the size or sign of the gain.

### 4. Required use of coding agents

#### 4.1 Minimum requirement

Each team must document at least one **consequential** coding-agent interaction in each of these three phases:

1. paper understanding, codebase audit, or reproduction planning;
2. implementation, testing, debugging, or experiment execution; and
3. improvement design, result analysis, or adversarial review.

An interaction is consequential when it changes—or could reasonably change—the claim, code, configuration, data processing, metric, result, or project decision. Routine autocomplete, spelling correction, and low-stakes formatting need not be logged.

The choice of tool, subscription level, number of prompts, and token expenditure is not graded. The quality of delegation, verification, and scientific judgment is graded. The instructor will identify at least one approved, reasonably accessible workflow. A team that cannot access an approved tool, or experiences a material service outage, must report it promptly and may complete an equivalent instructor-approved agent task; no advantage is awarded for a paid tier.

#### 4.2 Student accountability

Students must be able to explain and defend all agent-assisted work. In particular:

- review every consequential code change;
- run tests or other checks appropriate to the change;
- verify generated citations against primary sources;
- inspect commands before execution, especially destructive or expensive commands;
- never expose credentials, personal data, confidential data, or restricted material;
- distinguish imported code, agent-assisted code, and team-authored modifications; and
- record why consequential suggestions were accepted, modified, rejected, or escalated.

“The agent generated it” is neither evidence nor an excuse.

#### 4.3 Consequential-interaction ledger

The repository must contain an agent-use ledger, normally `AI_USAGE.md`. Supporting evidence may be stored as sanitized excerpts in `logs/agents/` or submitted to the instructor as a private annex when public release would expose personal information or non-redistributable material. Raw transcripts need not be public. Record consequential interactions rather than every minor chat message.

| Field | Required information |
| --- | --- |
| Date and team member | Who delegated or reviewed the task |
| Tool/model/version | Record the visible product, model, version, date, and controllable settings when available; state when a field is not exposed |
| Delegated task | The concrete objective and relevant context |
| Decision | Accepted, modified, rejected, or escalated |
| Human verification | Tests, inspection, derivation, comparison, or review performed |
| Evidence link | Prompt/output excerpt, diff, commit, test, experiment, or saved artifact |

Additional requirements:

- preserve representative prompt/output excerpts or an exportable transcript when available;
- never record hidden chain-of-thought; a concise task/output summary is sufficient;
- redact secrets or restricted information and leave a redaction marker with the reason;
- if a model output becomes experimental input, record the model snapshot/date, decoding settings, saved output, and hash; and
- include at least one critical-review episode in which a consequential proposal was verified and then accepted, modified, rejected, or escalated. The record must explain why the proposal was not accepted merely on the agent's authority.

#### 4.4 Prohibited uses

The following are prohibited:

- fabricated or altered citations, results, logs, commits, or agent records;
- hidden test-set tuning or selecting only favorable runs;
- unattributed copying of code, text, figures, or data;
- concealing material agent assistance;
- submitting code or analysis that the team cannot explain;
- bypassing access controls or licence restrictions;
- uploading personal data or unauthorized third-party confidential, unpublished, embargoed, or restricted material to an agent; and
- using a live agent to answer assessed defense questions unless the instructor explicitly authorizes a demonstration.

Suspected fabrication, plagiarism, or deliberate concealment is handled under UM academic-integrity procedures.

### 5. Repository and evidence package

#### 5.1 Repository requirements

Each team must maintain one version-controlled repository with meaningful development history. Large datasets and model weights should normally remain outside Git; provide retrieval instructions, versions, and checksums instead.

Recommended structure:

```text
.gitignore
.env.example
README.md
AI_USAGE.md
CONTRIBUTIONS.md
LICENSES.md
SECURITY.md
environment/           # lockfile, container, or environment specification
configs/               # versioned experiment configurations
data/README.md          # provenance, licence, preparation, checksums
src/                    # implementation
tests/                  # unit, integration, and evaluation fixtures
scripts/                # setup, smoke test, reproduction, and improvement runs
experiments/registry.csv
logs/agents/
results/                # machine-readable summaries and generated figures
report/
presentation/
```

Equivalent structures are acceptable if all required evidence is easy to locate.

For every external dataset, model, and code component, record its source, version, licence, modifications, required notices, and whether redistribution is permitted. Restricted assets must be referenced and retrieved through authorized instructions rather than committed to the repository.

The repository must exclude credentials and sensitive local files, provide placeholder variables through `.env.example`, and pass a secret scan before each assessed release. Use least-privilege credentials, rotate any exposed secret immediately, and document which data were sent to each external API or agent. New scraping, human-subject data collection, or use of biometric, medical, or sensitive personal data requires prior instructor and applicable UM approval.

#### 5.2 Minimum final package

The final tagged release must include:

- a meaningful commit history and a tagged final commit;
- setup instructions and a one-command smoke test;
- a locked environment or container specification;
- data preparation, provenance, licences, and checksums;
- the exact configuration and command for every reported table or figure;
- tests for critical implementation and evaluation logic;
- an experiment registry containing commit, configuration, seed, hardware, runtime, cost, and result;
- a results-to-artifacts table linking every major claim to logs and saved outputs;
- a registry and concise evidence summary for failed and negative experiments; large redundant checkpoints need not be retained;
- attribution for paper code, third-party code, data, models, and AI assistance;
- a final report using the published template and proposed eight-page main-text limit, excluding references and appendices; and
- the presentation deck and any demonstration instructions.

The report must separate facts, results, interpretations, and limitations. It should contain: target claim; paper and method summary; deviations; protocol; reproduction result; discrepancy diagnosis; improvement hypotheses; controlled evaluation; ablations; limitations; agent-use reflection; and references.

#### 5.3 Experiment registry

A minimum registry row should contain:

```text
run_id, timestamp, git_commit, config, data_version, seed,
hardware, runtime, cost, status, metrics, artifact_path, notes
```

For services that do not expose per-run cost, record a reasonable estimate or `not separately metered` and explain the basis in the repository README.

Do not overwrite an earlier run to make a later result look cleaner. Corrections should create a new run or an auditable amendment.

### 6. Fifteen-week project timeline

Deadlines are relative to the weekly class and will be posted in UMMoodle. Week 1 was delivered on 2026-08-19 and Week 2 is scheduled for 2026-08-26. Later calendar dates remain subject to official holiday and makeup-class arrangements; the week-number requirements below remain the grading contract unless the instructor announces a revision.

| Week | Required milestone | Acceptance evidence | Rubric connection |
| --- | --- | --- | --- |
| **1** | Project launch; interests survey; identify possible teammates and a first one-sentence target claim | Team interests and provisional claim | Formative |
| **2** | Submit a three-paper shortlist by the start of class; register the team when registration opens; initialize the repository, agent ledger, and security/data-egress rules | Paper-feasibility table covering code, data, licence, model/API access, compute, and metrics; repository link; initial `AI_USAGE.md`, `.gitignore`, and `.env.example` | Preparation for Scope |
| **3** | Freeze paper, reproduction tier, one target claim/table/figure, inherited dependencies, fallback scope, and team compute ceiling | Instructor-approved scope contract | **Scope: 5 points** |
| **4** | Submit the claim map | Assumptions, algorithm components, baselines, split, metric, expected result, confounders, and evidence rule | Protocol |
| **5** | Submit the reproduction protocol and agent-use plan | Evaluation harness, populated ledger examples, security/data boundaries, run-registration rule, and one-command smoke test | **Protocol: 10 points** |
| **6** | Lock the environment and data path; produce the first baseline | Dependency lock, data provenance/checksums, first registered runs, synthetic test, and small baseline or documented failure | **Pilot: 10 points** |
| **7** | Pass/rescope gate and live audit | Code/configuration walkthrough, runnable fixture, early result, agent-ledger audit, and contribution check | Scope, Protocol, Pilot |
| **8** | Reproduction v1 and improvement design | Quantitative result with uncertainty, direct paper comparison, discrepancy list, and three diagnosis-driven hypotheses; approved pilots may begin | Reproduction |
| **9** | Final reproduction dossier, improvement approval, and pilot launch | Claim status, evidence table, discrepancy diagnosis, approved hypotheses, falsifiable predictions, confounders, compute-matched plan, and first registered pilot runs | **Reproduction: 20 points** |
| **10** | Complete pilots of at least two approved improvements | Registered pilot runs, negative results, comparison summary, and justified selection of one full evaluation | Improvement |
| **11** | Fully evaluate at least one improvement; complete peer artifact audit | Controls, core ablations, uncertainty, failure cases, sanitized peer setup/smoke test, expected output/hash, claim-to-artifact link, and documented repairs | **Improvement: 20 points** |
| **12** | Complete final ablations and enter the common artifact freeze | Immutable tagged commit/hash; final registry, plots/tables, report, deck, agent ledger, contribution record, and signed freeze checklist | **Artifact/report/provenance: 15 points** |
| **13** | Project Symposium A | Team presentation, demonstration, individual defense; answers addendum within 72 hours | Presentation and individual criteria |
| **14** | Project Symposium B | Same evidence standard and frozen artifact | Presentation and individual criteria |
| **15** | Project Symposium C and course synthesis | Same frozen evidence standard; archive and reflection referring to the Week 12 tag | Presentation and individual criteria |

The Week 3, Week 5, and Week 6 checkpoints each permit one corrective revision within seven calendar days after instructor feedback. The original submission remains in the record. Week 9, Week 11, and Week 12 are final evidence deadlines for the corresponding rubric components; any general late-submission rule will be announced in UMMoodle before it takes effect.

Peer auditors receive only a sanitized setup packet: no credentials, private agent annexes, or restricted data. They perform the published smoke test and checklist rather than expensive training. The audited team receives a short repair window, records every repair, and preserves the original audit result.

#### 6.1 Rescoping

One instructor-approved scope revision is allowed through the end of Week 7 without penalty. The revised claim must preserve a defensible connection to the paper and must update the scope contract, protocol, repository, and agent ledger.

Later rescoping requires written approval. An approved rescope has no separate automatic deduction. Evidence invalidated or never produced because of the change cannot earn the directly affected rubric points, but the same defect is not penalized twice. Deliberate nondisclosure or a knowingly invalid claim is an academic-integrity matter; unavoidable paper, data, or infrastructure failures are judged from the team's documented evidence and timely escalation.

#### 6.2 Common artifact freeze

All teams freeze the same evidence package at the end of Week 12: code, environment, data instructions, configurations, registry, logs, results, report, presentation deck, agent ledger, and contribution record. The frozen package must be identified by an immutable tag and commit hash. This prevents later presentation groups from receiving extra experimental or writing time.

After the freeze, teams may:

- correct presentation wording or formatting;
- make accessibility corrections; and
- complete an instructor-approved packaging repair that does not change scientific outputs.

Every post-freeze change must be listed in an amendment log and retain the original frozen file for comparison. Unapproved post-freeze experiments, changed results, or replacement evidence are not graded.

### 7. Grading rubric

#### 7.1 Grade composition

| Component | Points | Score type |
| --- | --- | --- |
| Scope contract and understanding of the target claim | 5 | Team |
| Reproduction protocol, feasibility, and evaluation design | 10 | Team |
| Pilot implementation and live-audit readiness | 10 | Team |
| Final reproduction evidence and discrepancy diagnosis | 20 | Team |
| Improvement hypotheses, controlled experiments, and ablations | 20 | Team |
| Reproducible artifact, final report, and agent-use provenance | 15 | Team |
| Team presentation and demonstration | 5 | Team |
| Individual technical defense | 10 | Individual |
| Individual contribution evidence | 5 | Individual |
| **Total** | **100** | **85 team + 15 individual** |

For student (i):

[ \text{Course grade}_i = \text{team components out of 85} + \text{individual defense out of 10} + \text{individual contribution out of 5}. ]

#### 7.2 Detailed point allocation

##### A. Scope contract and understanding — 5 points

| Subcriterion | Points |
| --- | --- |
| Precise, falsifiable target claim tied to a specific paper result | 2 |
| Correct understanding of the method, assumptions, and intended comparison | 2 |
| Feasible approved tier, fallback scope, and compute/data/licence screen | 1 |

##### B. Reproduction protocol, feasibility, and evaluation design — 10 points

| Subcriterion | Points |
| --- | --- |
| Faithful data, split, preprocessing, baseline, and metric contract | 3 |
| Controls, seeds/uncertainty, contamination checks, and discrepancy rule | 3 |
| Compute, cost, runtime, licence, security, and fallback plan | 2 |
| Claim-to-code/config/run/artifact traceability designed before results | 2 |

##### C. Pilot implementation and live-audit readiness — 10 points

| Subcriterion | Points |
| --- | --- |
| Locked environment, data pipeline, and one-command smoke test | 3 |
| Correct core implementation with meaningful tests or fixtures | 3 |
| Registered small baseline result or well-diagnosed failure | 2 |
| Live code/configuration audit and active agent/contribution records | 2 |

##### D. Final reproduction evidence and discrepancy diagnosis — 20 points

| Subcriterion | Points |
| --- | --- |
| Fidelity to the approved paper claim and transparent deviations | 5 |
| Correct quantitative evidence, repeated runs, and uncertainty | 6 |
| Direct paper comparison using consistent metrics and units | 3 |
| Technical diagnosis of agreement, discrepancy, or failure | 4 |
| Honest scope, limitations, and supported/not-supported/inconclusive claim status | 2 |

##### E. Improvement hypotheses, experiments, and ablations — 20 points

| Subcriterion | Points |
| --- | --- |
| Three mechanism- or diagnosis-driven hypotheses with falsifiable predictions | 4 |
| At least two auditable pilots and justified selection of the full experiment | 3 |
| Fair compute-matched comparison with appropriate controls | 5 |
| Ablations, robustness/uncertainty analysis, and failure cases | 5 |
| Evidence-calibrated interpretation, including negative results and limitations | 3 |

##### F. Reproducible artifact, report, and agent-use provenance — 15 points

| Subcriterion | Points |
| --- | --- |
| Runnable repository, README, locked environment, and meaningful history | 4 |
| Data/code/model provenance, licences, required notices, and security hygiene | 1 |
| Experiment registry and results-to-artifacts traceability | 2 |
| Clear final report with correct citations and evidence-calibrated claims | 5 |
| Complete consequential-agent ledger with substantive human verification | 3 |

##### G. Team presentation and demonstration — 5 points

| Subcriterion | Points |
| --- | --- |
| Clear claim-to-evidence story and readable technical material | 2 |
| Effective demonstration or artifact walkthrough | 1 |
| Timing, coordination, and coherent coverage of the required story | 2 |

##### H. Individual technical defense — 10 points

| Subcriterion | Points |
| --- | --- |
| Explains the paper claim, method, and assumptions | 3 |
| Explains relevant code, configurations, evaluation, and evidence | 3 |
| Defends improvement choices, agent decisions, failures, and limitations | 2 |
| Answers follow-up questions accurately and acknowledges uncertainty | 2 |

##### I. Individual contribution evidence — 5 points

| Subcriterion | Points |
| --- | --- |
| Sustained, identifiable contribution supported by commits, roles, and logs | 3 |
| Professional collaboration supported by confidential peer feedback and defense evidence | 2 |

Peer feedback is evidence, not an automatic vote. The instructor triangulates contribution statements, repository history, experiment and agent logs, confidential peer feedback, and technical questioning.

#### 7.3 Performance-level descriptors

| Component | Excellent | Proficient | Developing | Insufficient |
| --- | --- | --- | --- | --- |
| **Scope, 5** | **5:** precise, feasible, and technically understood | **4:** sound with minor ambiguity | **2–3:** broad or partly unsupported | **0–1:** unsuitable, unapproved, or not testable |
| **Protocol, 10** | **9–10:** faithful, controlled, auditable, and feasible | **7–8:** sound with limited gaps | **4–6:** important design or feasibility gaps | **0–3:** invalid or largely absent protocol |
| **Pilot, 10** | **9–10:** runnable, tested, registered, and audit-ready | **7–8:** substantially ready with minor gaps | **4–6:** brittle or incomplete pipeline/evidence | **0–3:** no credible executable pilot |
| **Reproduction, 20** | **18–20:** faithful evidence with uncertainty and strong diagnosis | **14–17:** substantially correct with limited omissions | **8–13:** partial or weakly comparable evidence | **0–7:** missing, invalid, or unsupported reproduction |
| **Improvement, 20** | **18–20:** well-motivated, controlled, ablated, and honestly interpreted | **14–17:** sound investigation with limited gaps | **8–13:** weak controls, motivation, or evidence | **0–7:** no meaningful or valid investigation |
| **Artifact/report/agents, 15** | **14–15:** independently navigable, reproducible, and traceable | **11–13:** mostly reproducible with minor manual steps | **7–10:** brittle or materially incomplete records | **0–6:** inaccessible, non-runnable, or untraceable |
| **Presentation, 5** | **5:** clear, balanced, technical, and well timed | **4:** strong with minor issues | **2–3:** unclear, uneven, or weakly evidenced | **0–1:** missing or seriously inadequate |
| **Defense, 10** | **9–10:** accurate command of the full project | **7–8:** sound understanding with minor gaps | **4–6:** narrow or shallow understanding | **0–3:** cannot explain submitted work |
| **Contribution, 5** | **5:** sustained, identifiable, well-supported contribution | **4:** substantive contribution with minor gaps | **2–3:** uneven or weakly documented contribution | **0–1:** negligible or unverifiable contribution |

### 8. Evidence and scoring rules

These rules prevent polished presentation from substituting for scientific evidence while avoiding multiple penalties for the same defect.

1. **Evidence gate.** A claimed result must link to the code, configuration, registered run, and saved output that produced it. Unsupported claims earn no credit in the subcriterion that depends on them; unrelated evidence is scored normally.
2. **Comparable-baseline rule.** A valid comparable baseline follows the frozen split, preprocessing, metric, and evaluation protocol; uses the relevant paper or approved reproduction method; respects the compute contract or discloses a justified normalization; includes the required repetitions or approved alternative; and is traceable to artifacts. Without such a baseline, improvement comparisons that depend on it are not credited. Sound hypotheses, diagnostics, negative findings, and other independently valid evidence remain eligible for credit.
3. **Execution and repair rule.** The assessor performs one documented setup/smoke-test attempt on the declared supported environment using the submitted instructions, with one clarification request if needed. If packaging fails, the team may submit a packaging-only repair within 72 hours of notice; the original is retained, and scientific outputs may not change. Only rubric evidence that cannot be verified is affected. A timely documented infrastructure failure outside the team's control is not treated as a packaging failure.
4. **Official-code rule.** A team using official code must still provide independent reconstruction, testing, audit, or material validation of an important component or claim. Missing independent work receives no credit under the directly relevant testing, audit, or fidelity subcriteria; the remaining evidence is scored on its merits.
5. **Agent-use rule.** If a team does not demonstrate the required consequential use in all three phases—and has no approved equivalent—it receives **0/3** for the agent-ledger subcriterion. Incomplete but genuine provenance may receive partial credit. Deliberate concealment or fabricated records are integrity matters, not ordinary missing documentation.
6. **Individual defense rule.** A student who misses the defense without approved accommodation receives **0/10** for Individual technical defense. An approved absence requires a separate defense of equivalent scope.
7. **Contribution rule.** Low or unverifiable participation affects the individual Defense and Contribution components, not otherwise valid team evidence. Peer ratings alone do not determine the mark; they are triangulated with commits, logs, declared roles, and questioning.
8. **No double counting.** The same underlying defect is not deducted repeatedly unless it independently invalidates evidence in more than one rubric component. Any affected points and the evidence supporting the decision will be identified in feedback.
9. **Integrity rule.** Fabricated evidence, plagiarism, deliberate concealment, or prohibited data use is handled through UM academic-integrity procedures rather than by an improvised rubric cap.

### 9. Presentation and defense

Each team must present the frozen project artifact during Weeks 13–15.

The proposed quality-preserving format per team is:

- **15 minutes:** target claim, reproduction, agent workflow, improvement, limitations, and demonstration;
- **10 minutes:** technical questions and individual defense; and
- **2 minutes:** transition.

All teams, including later symposium cohorts, submit the final report and presentation deck at the Week 12 common freeze. Exact cohort allocation will be published after team registration and before the freeze.

All members must speak and receive at least two minutes of individual questioning. Teammates may not answer a question directed to another student. If team size or class capacity makes equivalent questioning impossible inside the team slot, the instructor will schedule a separate or parallel defense using the same difficulty-banded question bank.

The demonstration is normally a bounded smoke test, recorded result, or code/configuration/log walkthrough—not a costly live training run. The instructor may select one claim, configuration, experiment, code path, or agent decision for examination. A recorded fallback is acceptable when live infrastructure is unreliable, provided it links to the frozen evidence.

The response-to-questions addendum is due within 72 hours. It may clarify answers and correct presentation errors, but it may not silently replace the Week 12 frozen evidence.

### 10. Submission checklist

Before the Week 12 freeze, confirm that:

- [ ] The target paper, claim, reproduction tier, and deviations are explicit.
- [ ] The data, split, metrics, baselines, and compute contract are frozen.
- [ ] The repository installs in a clean environment and the smoke test runs.
- [ ] Every reported result links to a commit, config, registered run, and artifact.
- [ ] Repeated runs and uncertainty are reported, or an approved exemption is documented.
- [ ] The claim status is stated as supported, not supported, or inconclusive, separately from tier and fidelity.
- [ ] Three improvement hypotheses, two pilots, and one full evaluation are present.
- [ ] Controls, ablations, negative results, and limitations are preserved.
- [ ] `AI_USAGE.md`, representative agent evidence, and a documented critical-review episode are complete.
- [ ] Licences, citations, provenance, the data-egress inventory, and secret/security checks are complete.
- [ ] `CONTRIBUTIONS.md` accurately records individual roles and evidence.
- [ ] The final report, deck, and demonstration instructions match the immutable Week 12 tag and hash.

### 11. Items to confirm before final publication

The rubric weights and core scientific principles match the Week 1 course contract. The detailed repository, security, repair, and presentation rules in this draft are post-Week-1 operational clarifications and do not take effect until the instructor publishes the final policy in UMMoodle. Before publication, confirm:

1. final team-size rule, symposium allocation, and any separate-defense schedule for the enrolled class;
2. numerical per-team GPU, storage, runtime, and paid-API ceiling;
3. repository visibility, private-annex route, and submission mechanism;
4. exact deadlines after holiday and makeup-class arrangements are confirmed;
5. paper-eligibility details such as publication window or public-code expectations;
6. the final report template and whether the proposed eight-page main-text limit applies;
7. the approved accessible agent options, outage procedure, and accommodation route;
8. grade conversion, fractional-point rounding, and general late-submission policy; and
9. any other UM-mandated attendance, accommodation, data-governance, or academic-integrity wording.

Last modified: Wednesday, 2 September 2026, 6:10 PM

---

## Course Project and Grading: Student Guide

来源：[https://ummoodle.um.edu.mo/mod/page/view.php?id=1202972](https://ummoodle.um.edu.mo/mod/page/view.php?id=1202972)

## CISC8006 Course Project and Grading: Student Guide

> Short student version · 2026-09-02\
> **No midterm or final examination: the course project is 100% of the grade.**

This guide summarizes the main grading rules. Exact dates, team size, resource limits, report length, submission method, late policy, and presentation schedule will be announced in UMMoodle. Follow the final UMMoodle announcement if it differs from this guide.

### At a glance

- The project is completed in a team, but **15 points are individual**.
- Your goal is to test **one clear claim**, not to reproduce an entire paper.
- A larger or more expensive project does not automatically receive a higher grade.
- Coding agents such as Claude Code or Codex must be used and documented.

**Project path:** choose 1 claim → reproduce it → propose 3 improvements → pilot 2 → fully test 1 → freeze the evidence → present and defend it.

### What your team must do

#### 1. Choose one paper and one claim

Choose a machine-learning or deep-learning paper with a result that your team can test. The target may be one table row, figure trend, method comparison, or important component.

The instructor will approve one reproduction level:

- **Exact:** rerun the original setup.
- **Scaled-faithful:** use less data or compute but keep the same research question and fair comparison.
- **Component:** rebuild and test one important part of the method.

All three levels can receive full marks. Running the official repository without checking, rebuilding, or testing an important part is not enough.

#### 2. Reproduce the claim

Freeze the claim, data split, data preparation, baseline, metric, units, compute limit, and decision rule before the final runs. Record all important differences from the paper.

Test the evaluation code on a small known or synthetic example. Use at least **three random seeds** for comparisons affected by randomness, unless the instructor approves another uncertainty check. Compare your result directly with the paper and report the claim as:

- **supported**;
- **not supported**; or
- **inconclusive**—the evidence is not clear enough.

#### 3. Improve upon the paper

Use the reproduction result to find a real weakness or failure. Your team must:

- propose at least **three** improvement ideas;
- pilot at least **two** approved ideas; and
- fully evaluate at least **one** idea against the frozen reproduction baseline.

Use fair controls and ablations—tests that remove or change one part to show what it does. Report uncertainty, limitations, failed runs, and negative results. An improvement does **not** need to win: a careful and honestly explained null or negative result can receive full credit.

### How to use coding agents

Record at least one important coding-agent interaction in each phase:

1. paper understanding, code review, or reproduction planning;
2. implementation, testing, debugging, or experiment runs; and
3. improvement design, result analysis, or critical review.

Keep the record in `AI_USAGE.md`. For each important interaction, note the task, tool, decision, human checks, and an evidence link. Include at least one case where your team checks an agent suggestion and then accepts, changes, or rejects it for a clear reason.

You are responsible for every submitted line of code, command, citation, result, and claim. You must be able to explain the work during the defense. Never send passwords, personal data, confidential material, or restricted data to an agent.

### What the repository must contain

Keep one version-controlled repository with meaningful commit history. It must include:

- clear setup instructions, a locked environment, and one command that quickly checks whether the system works;
- data/code/model sources, versions, licences, preparation steps, and checksums;
- versioned code, configurations, exact run commands, and tests;
- an experiment record linking every reported result to its commit, configuration, seed, runtime, cost, and saved output;
- failed and negative experiment records;
- `AI_USAGE.md` and `CONTRIBUTIONS.md`; and
- the final report, slides, demonstration instructions, and fixed Week 12 Git tag and commit hash.

Large data and model files should normally stay outside Git. Do not commit credentials or overwrite old runs to make the record look cleaner.

### Grading: 100 points

| Component | Points | Type |
| --- | --- | --- |
| Scope and understanding of the claim | 5 | Team |
| Reproduction protocol and evaluation design | 10 | Team |
| Pilot implementation and audit readiness | 10 | Team |
| Final reproduction and difference analysis | 20 | Team |
| Improvement experiments and ablations | 20 | Team |
| Repository, report, and agent-use record | 15 | Team |
| Team presentation and demonstration | 5 | Team |
| Individual technical defense | 10 | Individual |
| Individual contribution | 5 | Individual |
| **Total** | **100** | **85 team + 15 individual** |

### Key milestones

| When | Required progress |
| --- | --- |
| **Weeks 1–2** | Form the team, shortlist three papers, check feasibility, create the repository, and start `AI_USAGE.md`. |
| **Week 3** | Freeze the paper, reproduction level, one claim, fallback, inherited dependencies, and compute limit. |
| **Weeks 4–5** | Submit the claim map, reproduction protocol, agent plan, evaluation check, and one-command system check. |
| **Weeks 6–7** | Lock the environment and data path, produce a small baseline, and pass the live code/configuration review. One approved rescope is allowed through the end of Week 7 without penalty. |
| **Weeks 8–9** | Submit the final reproduction, paper comparison, difference analysis, uncertainty, and three improvement ideas. |
| **Weeks 10–11** | Pilot two improvements, fully test one, run controls and ablations, and complete the peer evidence check. |
| **Week 12** | Freeze the complete repository, experiment record, results, report, slides, agent record, and contribution record. All teams share this deadline. |
| **Weeks 13–15** | Give the team presentation and demonstration. Every member must speak and complete an individual defense. |

Weeks 3, 5, and 6 allow one corrective revision within seven days of instructor feedback. Weeks 9, 11, and 12 are final evidence deadlines for their graded components.

### Important grading rules

- Every result must link to the code, configuration, registered run, and saved output that produced it.
- Improvement marks require a fair and traceable reproduction baseline.
- Preserve failed and negative runs; do not report only favourable results.
- Missing the required three-phase agent use without an approved alternative gives **0/3** for the agent-record item.
- Missing the individual defense without an approved accommodation gives **0/10** for the defense.
- Do not use a live agent to answer assessed defense questions unless the instructor explicitly allows a demonstration.
- After the common Week 12 freeze, do not add experiments or change scientific results. A short answers addendum is due within 72 hours after the presentation, but it cannot replace frozen evidence.
- Fabricated evidence, plagiarism, hidden test-set tuning, concealed agent use, and prohibited data use are handled under UM academic-integrity procedures.

For unusual cases and full operational rules, consult the detailed course-project specification and the final UMMoodle announcements.

Last modified: Wednesday, 2 September 2026, 6:10 PM

---

## Project Team Registration (2–3 students)

来源：[https://ummoodle.um.edu.mo/mod/feedback/view.php?id=1183499](https://ummoodle.um.edu.mo/mod/feedback/view.php?id=1183499)

**Closed:** Sunday, 30 August 2026, 11:59 PM

Submit one response per team. Each team must have 2–3 students. Enter names exactly as shown in UMMoodle. Presentation dates and order will be assigned later by random draw.

The feedback is not open

---

## Final Project Paper and Claim

来源：[https://ummoodle.um.edu.mo/mod/feedback/view.php?id=1202783](https://ummoodle.um.edu.mo/mod/feedback/view.php?id=1202783)

**Closed:** Tuesday, 8 September 2026, 11:59 PM

Submit one response per team.

The feedback is not open

---

## Course Schedule

来源：[https://ummoodle.um.edu.mo/mod/page/view.php?id=1173077](https://ummoodle.um.edu.mo/mod/page/view.php?id=1173077)

Course information

Class time: Wednesdays, 19:00–22:00\
Venue: E11-1021(L)\
Semester: 17 August–30 November 2026

The schedule below reflects the current course plan. Any approved timetable or university-calendar change will be announced on UMMoodle.

Weekly schedule

Week 1 — 19 August\
Course introduction; from language models to LLMs and agents.\
Project milestone: launch, prerequisite diagnostic, team formation, interests survey, and a first one-sentence target claim.

Week 2 — 26 August\
Inside an LLM at inference time: tokenization, embeddings, attention, causal masks, Transformer blocks, decoding, and KV cache.\
Project milestone: shortlist three papers and check code, data, licences, compute, and evaluation access.

Week 3 — 2 September\
The model lifecycle: data, pretraining, scaling, optimization, supervised fine-tuning, post-training, inference, and model choice.\
Project milestone: freeze the paper, reproduction tier, target claim, and compute ceiling.

Week 4 — 9 September\
Using and evaluating LLM systems: prompt contracts, structured outputs, context engineering, retrieval, RAG, sampling, and evaluation design.\
Project milestone: submit a claim map covering assumptions, baselines, splits, metrics, and expected results.

Week 5 — 16 September\
Agents and reliable agent-assisted research: tools, workflows, traces, cost, latency, prompt injection, privacy, and scientific verification.\
Project milestone: reproduction protocol and agent-use policy due; evaluation harness and one-command smoke test must run.

Week 6 — 23 September\
Language modeling as maximum likelihood: autoregressive factorization, teacher forcing, masking, NLL, cross-entropy, perplexity, and SFT.\
Project milestone: locked environment, data provenance, first logs, and a small baseline result or documented failure.

Week 7 — 30 September\
Entropy, KL divergence, and regularized control: KL direction, chain rule, reference policies, constrained objectives, and the effect of beta.\
Project milestone: pass/rescope gate with baseline logs and a live code/configuration audit.

Week 8 — 7 October\
Policy gradients and credit assignment: trajectories, returns, REINFORCE, baselines, advantages, and sparse rewards.\
Project milestone: reproduction v1 with uncertainty and three diagnosis-driven improvement hypotheses.

Week 9 — 14 October\
Preference data, reward models, and the RLHF map: Bradley–Terry models, rankings, noise, calibration, and reward hacking.\
Project milestone: final reproduction dossier and an approved improvement portfolio.

Week 10 — 21 October\
Actor–critic, PPO, GRPO, and the RLHF optimization loop: advantages, importance ratios, clipping, rollouts, and group baselines.\
Project milestone: run minimal pilots of at least two approved hypotheses and select one for full evaluation.

Week 11 — 28 October\
DPO and direct preference optimization: derivation, implicit rewards, reference-relative margins, support, beta, and policy mismatch.\
Project milestone: fully evaluate at least one improvement with controls and ablations; peer artifact audit.

Week 12 — 4 November\
Off-policy learning, distribution shift, and trustworthy evaluation: behavior-policy support, importance sampling, effective sample size, and fixed-budget evaluation.\
Project milestone: final ablations and common artifact freeze for every team.

Week 13 — 11 November\
Project Symposium A.\
Project milestone: presentation and individual technical defense; response-to-questions addendum due within 72 hours.

Week 14 — 18 November\
Project Symposium B.\
Project milestone: presentation and individual technical defense; response-to-questions addendum due within 72 hours.

Week 15 — 25 November\
Project Symposium C and course synthesis.\
Project milestone: presentation and individual technical defense; response-to-questions addendum due within 72 hours.

Last modified: Wednesday, 19 August 2026, 3:05 PM

---

## CISC8006 Final Project Presentation Schedule

来源：[https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=528620](https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=528620)

原文发布时间：Tuesday, 3 November 2026, 11:30 AM

Dear students,

Update (November 4): The SmolVLA team (Hu Manke, Chen Yuyang, Zheng Ruisheng) has moved to the final slot on November 25, 21:10–21:30. All other teams’ published times remain unchanged. Please use the revised attachment.

I’m sorry I didn’t post the presentation schedule before last weekend as promised. Thank you for your patience.

The attached schedule lists each team’s presentation date and time. We’ll meet on Wednesday, November 11, 18, and 25, in E11-1021(L). Each team will have 15 minutes to present, followed by 5 minutes for Q&A. All times are Macau time.

Please check your team’s slot. If you’d like to request a change, please email me at mengqu@um.edu.mo as soon as possible. Include your team members’ names, your assigned slot, and any alternative times that work for your team. I’ll do my best to accommodate your request.

Best,\
Meng Qu

附件：[CISC8006-presentation-schedule.xlsx](https://ummoodle.um.edu.mo/pluginfile.php/1865376/mod_forum/attachment/659996/CISC8006-presentation-schedule.xlsx?forcedownload=1)

---

## Final Project Presentations: November 11, 18, and 25

来源：[https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=503834](https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=503834)

原文发布时间：Wednesday, 28 October 2026, 6:20 PM

Dear all,

Our final project presentations will take place over three consecutive weeks, starting on November 11: **November 11, 18, and 25**.

Each group will have **15 minutes for the presentation**, followed by **5 minutes for Q&A**.

The presentation order will be the reverse of the order in which groups discussed their project topics with me: groups that discussed their topics with me later will present earlier.

I will post the presentation schedule on UM Moodle later this week. If your group has any special scheduling needs, please speak with me during today’s class.

Best regards,\
Meng Qu

---

## Office Hour Change This Week and Course Project Paper Confirmation

来源：[https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=318140](https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=318140)

原文发布时间：Wednesday, 16 September 2026, 12:15 PM

Dear students,

This week's office hour is moved to Thursday, September 17, 1:00–2:00 pm, at my office, N21-2001F. There will be no office hour today, Wednesday, September 16. This change applies to this week only; the usual Wednesday 2:00–3:00 pm schedule will resume next week.

If your team has not yet confirmed its selected course project paper with me, you must meet me either in today's class (Wednesday, September 16) or during tomorrow's office hour (Thursday, September 17, 1:00–2:00 pm) to confirm your paper choice.

Best regards,\
Meng Qu

---

## Action Required: Decide and Submit Your Project Paper and Claim by 6 September

来源：[https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=272846](https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=272846)

原文发布时间：Wednesday, 2 September 2026, 6:18 PM

Dear students,

By the end of this week - **Sunday, 6 September 2026, 11:59 PM** - each team needs to:

1. decide on the paper for the course project;
2. define one clear and testable central claim; and
3. submit the paper and claim using the [**Final Project Paper and Claim**](https://ummoodle.um.edu.mo/mod/feedback/view.php?id=1202783) form introduced in Siyu's announcement.

Each team should submit one response.

During next week, **7-13 September**, I will review all submitted papers and claims. If I find feasibility problems, an unclear or overly broad claim, data or compute limitations, evaluation risks, or other concerns, I will contact the team so that we can discuss and refine the paper and claim together.

Our goal is for every team to have a finalized paper and claim by the end of next week - **Sunday, 13 September 2026**.

Please submit a complete response by this Sunday so that there is enough time for review and discussion.

Best,\
Qu Meng

---

## Final Project Paper and Claim Submission

来源：[https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=272744](https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=272744)

原文发布时间：Wednesday, 2 September 2026, 5:15 PM

Dear students,

Please submit your final project paper and claim using the “Final Project Paper and Claim” form in the General section by Tuesday, 8 September 2026, 11:59 PM.

Each team should submit one response. “Paper” refers to the paper on which your final project is based, and “Claim” refers to the central claim of your final project.

Submission form: [https://ummoodle.um.edu.mo/mod/feedback/view.php?id=](https://ummoodle.um.edu.mo/mod/feedback/view.php?id=)[1202783](tel:1202783)

Thank you.

---

## Project Team Registration (2–3 Students)

来源：[https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=260465](https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=260465)

原文发布时间：Monday, 24 August 2026, 2:16 PM

Dear students,

Please register your team using the “Project Team Registration (2–3 students)” form in the General section by Sunday, 30 August 2026, 11:59 PM.

Each team must have 2–3 students. Please submit one response per team and enter members’ names exactly as shown in UMMoodle.

Registration form: [https://ummoodle.um.edu.mo/mod/feedback/view.php?id=1183499](https://ummoodle.um.edu.mo/mod/feedback/view.php?id=1183499)

Presentation dates and order will be assigned later by random draw.

Thank you.

---

## Team Formation Deadline Extended to Sunday, 30 August

来源：[https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=258518](https://ummoodle.um.edu.mo/mod/forum/discuss.php?d=258518)

原文发布时间：Saturday, 22 August 2026, 2:35 PM

Dear all,

The deadline for forming project teams has been extended to Sunday, 30 August 2026. UM's course add/drop period ends on Wednesday, 26 August at 10:00, so the class roster may still change before then. Moving the team formation deadline until after the add/drop period will help avoid major team changes caused by enrolment updates.

Please finalize your team by the end of Sunday, 30 August.

Best,\
Meng
