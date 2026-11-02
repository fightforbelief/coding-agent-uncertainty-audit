# Early Test-Command Signals in Coding-Agent Traces: An Exploratory Audit of Measurement and Predictive Value

Yunjie Guo and Hanqing Yang | 3 November 2026

## Abstract

Test-related tool use is an appealing early indicator of a coding agent's eventual outcome, but a detected command need not imply that tests executed, and predictive associations can reflect task difficulty or trajectory duration. We examine these distinctions in a pinned Open-SWE-Traces snapshot containing 207,489 trajectories from four scaffold/model configurations. A stratified audit covers 1,196 trajectories at two prefix windows. Its AI-assisted returns were reported as individually human-reviewed, and disagreements were resolved through explicitly authorized AI adjudication rather than an independent human adjudicator. These labels therefore provide a descriptive measurement audit, not an independent gold standard. In the R15 detector-positive audit frame, the weighted shares labeled as a definite test command and confirmed test execution were 97.22% and 74.05%, respectively. Excluding audit trajectories from every model partition leaves 159,867 known-outcome trajectories. The preselected primary exploratory comparison adds four frozen test-command proxies to structural logistic regression at 15 completed rounds. On 20,012 repository-held-out test trajectories, average precision increases from 0.641246 to 0.642153: Delta AP = 0.000907, with a paired repository-cluster bootstrap 95% interval [-0.002150, 0.004284]. The interval includes zero. A prespecified secondary nonlinear baseline shows a larger positive increment, while the main within-issue test-command odds-ratio intervals span one. Across 7,000 within-issue bootstrap refits, 860 fail convergence or rank checks. The findings do not establish a stable incremental benefit for the primary model, a causal effect of testing, or independently validated measurement of test execution. They support treating command observability, predictive value and behavioral explanation as separate empirical questions.

**Keywords:** coding agents; software engineering traces; measurement validity; early outcome prediction; repository-held-out evaluation; conditional logistic regression

## 1. Introduction

Software engineering agents interact with repositories through sequences of searches, edits and program executions. Benchmarks such as SWE-bench evaluate whether a proposed modification resolves an issue [2], while systems such as SWE-agent and OpenHands expose interfaces through which agents can inspect code and execute commands [3,4]. Such trajectories contain information beyond the final outcome. A visible test command, an explicit failing test, or a failure followed by a passing summary could plausibly help identify trajectories that need additional attention before they terminate.

That possibility involves several distinct claims. First, a log-derived rule must measure the behavior its name suggests. A command string may be visible even when the runner is unavailable, dependencies fail to import, collection stops, or output is unrecoverable. Second, an observable event may add little predictive information beyond the surrounding trajectory's structure. Third, even a useful predictor need not describe an intervention that improves outcomes. Agents may test because an issue is difficult, because a plausible patch already exists, or because their scaffold prompts them to do so. These distinctions matter when a feature called `ran_tests` is interpreted as evidence of successful verification or as a reason to change an agent's policy.

We study frozen command and output proxies on a historical Open-SWE-Traces snapshot [1]. Our design combines a full feature census, a stratified measurement audit, repository-held-out prediction at five fixed checkpoints, and issue-conditional comparisons. We retain the original dataset split, exclude every audit trajectory from all modeling partitions, and freeze the new computational protocol before scoring. An earlier analysis of this same snapshot helped motivate the extension; consequently, the present work remains exploratory despite these controls.

The measurement route also differs from the original plan. Two returned annotation packages originated as AI-assisted drafts. The project owner stated that they had subsequently received separate individual human review and that reviewers had not viewed each other's judgments. The owner then selected AI candidate adjudication in place of the planned third human adjudicator. We preserve this provenance rather than presenting the resulting labels as an independent human gold standard.

Our contribution is an auditable empirical characterization of this particular snapshot and analysis route. We ask: **RQ1**, what distinctions and uncertainties remain between detected commands and evidence of test execution? **RQ2**, do fixed early test-command proxies improve held-out failure ranking beyond structural features? **RQ3**, how do those associations change when comparisons are made within issues and within issue/configuration cells? We report the preselected primary contrast alongside every planned sensitivity analysis, including unstable bootstrap fits and positive secondary findings.

## 2. Background and scope

SWE-bench frames repository-level issue resolution as an executable evaluation problem [2]. SWE-agent investigates the interface between a language model and its computing environment [3], and OpenHands provides a platform for agents that interact with code, command lines and other tools [4]. These works motivate attention to trajectories and scaffolds, but they do not establish the predictive or causal interpretation of the specific proxies studied here. Our data are Open-SWE-Traces, not a new evaluation run on the original SWE-bench task set.

The distinction between an underlying construct and its operationalization is central to measurement research. Jacobs and Wallach emphasize making assumptions about constructs and their measurements explicit [5]. We apply that distinction to test-related behavior: command presence, process initiation, test execution and test outcome are separate targets. Agreement between submitted labels, particularly labels with shared AI assistance, cannot establish all of these targets' validity.

Trajectories are also dependent observations. Repeated attempts share issues and repositories; treating them as independent would understate the structure of the data. We therefore keep repositories separated across prediction partitions and resample whole repositories for interval estimation, following the general rationale for cluster-aware inference [6]. Within-issue conditional logistic models compare outcome-discordant groups through a likelihood that removes group intercepts [8]. These controls address particular sources of dependence and heterogeneity; they do not create random treatment assignment or an independent dataset replication. This paper is a focused audit, not a systematic review or a claim of priority over all trace-prediction methods.

## 3. Data and protocol

### 3.1 Pinned snapshot, outcomes and exclusions

The source is `nvidia/Open-SWE-Traces` at revision `c144d9d160dcd9f5ebcd9520d2d54111dda8b681` [1]. All 84 source shards were checked against the pinned manifest before extraction. The snapshot has 207,489 unique trajectory identifiers and 22,320 issue identifiers. Its four configuration labels are OpenHands/`minimax_m25` (49,948 trajectories), OpenHands/`qwen35_122b` (55,488), SWE-agent/`minimax_m25` (57,268), and SWE-agent/`qwen35_122b` (44,785). These are historical dataset labels; the analysis does not evaluate current versions of those systems.

There are 160,731 trajectories with `resolved` equal to zero or one and 46,758 with other or missing outcomes. The prediction target is failure, defined as Y = 1 when `resolved == 0`. Unknown outcomes are excluded rather than treated as failures. We exclude all 1,196 audit trajectory identifiers from train, validation, test and explanatory analyses; 864 of those identifiers have known outcomes. The resulting modeling pool contains 159,867 trajectories, 17,086 issues and 2,349 repositories. Table 1 gives the retained historical repository-connected-component split, whose seed is 20261024. Repository sets are disjoint across its three partitions.

**Table 1. Known-label modeling pool after audit exclusions, before checkpoint-reaching filters. Positive outcome is task failure.**

| Partition | Trajectories | Repositories | Issues | Failure share |
| --- | --- | --- | --- | --- |
| Train | 113,350 | 1,650 | 12,177 | 0.588258 |
| Validation | 26,504 | 352 | 2,771 | 0.625113 |
| Test | 20,013 | 347 | 2,138 | 0.586519 |

### 3.2 Study history and inferential status

The extension was partly motivated by inspecting seven positive trajectories in the earlier study's test set. Its protocol was committed locally before new scoring (initial freeze `e104202`), and implementation checks were frozen before the first new prediction score. Freezing prevents subsequent endpoint or parameter switching within this extension; it cannot erase earlier exposure to the snapshot. Every new analysis is therefore exploratory. The earlier study's separately frozen result is described in Appendix A and is not used as an independent replication of the present findings.

The planned checkpoints are K = 5, 10, 15, 20 and 30 completed interaction rounds. The primary K is the largest member of {15, 10, 5} reached by at least 70% of known-label, non-audit train/validation trajectories. This selects K = 15 with coverage 99.9693%, without using test-set coverage or scores. The primary predictive population requires reaching its checkpoint. A separate fixed-population sensitivity analysis retains only trajectories reaching 30 rounds, then evaluates every K on that same population.

### 3.3 Prefix construction and frozen features

A completed round follows the frozen parser's assistant/tool interaction boundaries, rather than a wall-clock interval or token budget. Structural features use messages up to the completed-round cut. We reuse 18 original fields: completed rounds and checkpoint availability; assistant-message, tool-call, tool-observation and distinct-tool counts; user, assistant and observation character counts; observation-line count; and counts for eight tool families. No outcome or audit label enters extraction. Before any new score, all 207,489 K5 structural rows were required to match the historical feature matrix exactly; they did.

The command detector operates on recovered tool-call commands. Its fixed rules recognize supported runner invocations and common wrappers, including environment assignments, working-directory commands and selected package-manager or interpreter prefixes. Only high-confidence detections enter the predictive variables. Ambiguous commands and missing command text are retained as diagnostics. The four behavior proxies are **command presence**, an indicator of any high-confidence detection; **command count**, log(1 + number of detected calls); **explicit failure**, an indicator that a detected call has a parsed test-failure marker; and **failure-then-pass**, an indicator of an earlier explicit failure followed by a later explicit passing summary from the same runner with no failure in that later call.

These definitions describe visible evidence, not verified execution or code correctness. The internal name `ran_tests` denotes command presence. For short trajectories, the full available prefix is retained for explanatory analyses. Command evidence is selected by the frozen round <= K rule; an available unfinished terminal call does not create an additional completed round. The same rule is used throughout extraction, without examining messages after the selected prefix to relabel an earlier event.

## 4. Measurement audit

### 4.1 Sampling frames and labels

The frozen audit contains 1,196 unique trajectories, 2,392 R5/R15 windows and 1,197 stratum memberships. Sampling is stratified within scaffold/model configurations. Primary strata target R5 detector positives; newly positive R15 trajectories with an R5 known-negative command history; R15 known negatives; and R15 missing-command cases. Two additional diagnostic frames target ambiguous commands. Appendix B specifies the restricted eligibility rules. These are enriched audit frames, not a simple random sample of all trajectories. One cross-frame overlap is retained with its own membership weights.

For a particular frame F and label category c, the descriptive weighted share is the sum of w_i times the category indicator divided by the sum of w_i, where w_i = N_h/n_h for the corresponding stratum/configuration cell. Memberships are analyzed in their declared frames; their inclusion probabilities are not summed across frames. R5-positive memberships contribute to both R5 and R15 summaries, so the six reported summary sample sizes must not be added to infer the unique audit size. We retain ambiguity, unrecoverable commands and insufficient evidence as uncertainty rather than automatically counting them as negative.

### 4.2 Review provenance and reconciliation

The two submitted packages are retained byte-for-byte. The owner identified Yunjie Guo and Hanqing Yang as the respective reviewers, reported individual itemwise review, accepted the review-timeline explanation, and stated that there had been no mutual viewing before submission. The packages' AI-draft origins and approximate timing records remain preserved. These are owner attestations; the software did not authenticate identities, independently establish chronology, or reconstruct exact review timestamps. We make no claim of unaided independent human annotation.

Across submitted first-round command-status judgments, raw agreement is 91.56% at R5 (Cohen's kappa 0.7473) and 67.81% at R15 (kappa 0.5566). These are sample agreement statistics on the enriched audit, not accuracy estimates. There are 2,321 differing fields across 770 windows, including 737 windows with core-label differences and 33 with runner-name-only differences. The owner explicitly selected the previously generated AI candidate values for those fields. A further 506 definitional consistency changes in 225 windows set execution/output-related fields to `not_applicable` when the final command decision was no-test-command. Both original rounds and every reconciliation source remain available.

### 4.3 Descriptive measurement findings

Table 2 reports the resulting AI-reconciled label distributions. The R15 positive audit frame has 97.22% weighted definite-command labels, but only 74.05% weighted execution-confirmed labels; the weighted process-started share is 93.93%. Those categories answer different questions. A visible invocation can coexist with missing output or a pre-test failure. The command-positive share among determinate R15 labels is 99.40%, but that conditional ratio excludes uncertainty and is not independently validated precision.

**Table 2. Stratum-weighted AI-reconciled command labels within the declared audit frames. Positive/determinate excludes uncertain labels; it is not gold-standard precision. Sample n counts memberships in the indicated summary.**

| Audit frame | Sample n | Positive / all | Uncertain / all | Positive / determinate |
| --- | --- | --- | --- | --- |
| R5 positive (A) | 124 | 98.26% | 1.74% | 100.00% |
| R15 positive (A+B) | 324 | 97.22% | 2.20% | 99.40% |
| R15 known negative (C) | 200 | 0.40% | 24.42% | 0.53% |
| R15 missing command (D) | 495 | 0.00% | 83.43% | 0.00% |
| R5 ambiguous (AMB5) | 78 | 1.58% | 7.30% | 1.70% |
| R15 ambiguous (AMB15) | 100 | 4.92% | 16.75% | 5.91% |

The R15 known-negative frame retains 24.42% uncertainty, and the missing-command diagnostic retains 83.43% uncertainty. Thus, a negative detector output cannot be interpreted uniformly as evidence that testing did not occur. Neither the known-negative positive share nor its determinate-only version is a false-negative rate or recall estimate: their denominator is a detector-defined audit frame. The diagnostic frames remain separate from the main descriptive estimates. No formal measurement confidence interval was retrospectively invented after observing the labels; classification ambiguity ranges are not sampling confidence intervals.

## 5. Statistical methods

### 5.1 Held-out prediction

At each checkpoint, M0 is L2-regularized logistic regression using structural features. M1 adds the explicit-failure proxy. M2 adds all four behavior proxies to M0. We select inverse regularization C from {0.01, 0.1, 1, 10} by validation average precision, breaking ties toward smaller C. The standardizer is fit only on training data during selection; it and the final model are then refit on train plus validation. Logistic optimization uses `lbfgs` with a 5,000-iteration limit. No repository or issue identifier is supplied as a predictor.

The primary exploratory estimand is Delta AP = AP(M2) - AP(M0) at K = 15 in the checkpoint-reaching population. AP uses the non-interpolated precision-recall summary implemented by scikit-learn [7]. We also report AUROC, Brier score and failure prevalence as the random-ranking AP reference. AP differences compare identical scored trajectories within each cell. Raw AP across different checkpoint populations is not itself a paired longitudinal comparison.

Intervals resample the test repositories with replacement, carrying all their trajectories and using the same draw for both models. There are 1,000 paired draws with seed 20261092; we report the 2.5th and 97.5th percentiles of the AP differences. These intervals condition on the fitted models and saved test predictions; they do not incorporate retraining, model-selection or audit-label uncertainty. Prediction vectors are saved once for each prespecified scoring cell. The five main logistic cells, five fixed-population logistic cells and two nonlinear cells form 12 cells in total. There is no multiplicity-adjusted confirmatory testing, and secondary endpoints do not replace the primary contrast.

### 5.2 Nonlinear sensitivity baseline

A prespecified secondary histogram-gradient-boosting baseline compares M0 and M2 feature sets at K = 5 and 15. Validation AP selects from maximum leaf counts {15, 31} and learning rates {0.05, 0.1}. The remaining settings are 100 boosting iterations, early stopping enabled and random seed 20261103. The selected model is refit on train plus validation and evaluated once on test; paired intervals follow the same repository resampling. This tests a different model class, not an alternative primary endpoint selected after seeing scores.

### 5.3 Within-issue comparisons

At K = 10 and 15, explanatory analyses use all 159,867 known-label, non-audit trajectories, including shorter available prefixes. Model A is ordinary logistic regression with the four behavior proxies, log(1 + tool-call count), distinct-tool count, log(1 + observation characters), and configuration indicators. Model B uses issue-conditional logistic regression with the same predictors where identified. Model C conditions on issue/configuration cells and omits configuration indicators; the protocol requires at least 500 outcome-discordant cells to run C. We also fit B separately within each of the four configurations, yielding 14 analyses.

Only groups with both outcomes identify the conditional effects. Predictors with no within-group variation are removed and logged. The exact conditional likelihood is evaluated with a vectorized dynamic program; its likelihood and gradient are checked against statsmodels ConditionalLogit at each point estimate [8]. Covariates are scaled for numerical optimization and reported back in their original units. Failure is the positive outcome, so an odds ratio above one denotes association with failure, not success.

Each analysis requests 500 repository-bootstrap refits with seed 20261092. Repeated repositories contribute repeated whole-group likelihood terms, equivalent to assigning distinct replicate identifiers to duplicated groups; duplicates are not merged into enlarged conditional groups. Rank-deficient and nonconvergent fits are recorded as failed rather than given zero coefficients. Percentile intervals use successful refits only. High failure rates can alter their sampling distribution, so nominal 95% coverage is not established for unstable analyses. These models estimate observational associations and do not identify the effect of forcing an agent to run tests.

## 6. Results

### 6.1 Coverage and event support

All planned feature rows and prediction cells passed identity and provenance checks. K15 reaches 113,316 training, 26,495 validation and 20,012 test trajectories. The test failure prevalence at K15 is 0.586548. K30 reaches 19,614 test trajectories, or 98.01% of the known-label non-audit test pool. Command-related evidence becomes more common later in the trajectory: among checkpoint-reaching eligible trajectories across all partitions, high-confidence command presence rises from 1,020/159,867 at K5 to 35,816/159,823 at K15 and 86,639/156,326 at K30. Failure-then-pass is particularly sparse early: one trajectory at K5, 715 at K15 and 8,982 at K30. These counts describe detector outputs and do not replace the audit's execution labels.

### 6.2 Primary predictive comparison

At the preselected K15 checkpoint, M0 has AP 0.641246, AUROC 0.567562 and Brier score 0.239811. M2 has AP 0.642153, AUROC 0.569493 and Brier score 0.239553 (Table 3). The primary Delta AP is 0.000906580, with paired repository-bootstrap interval [-0.002149890, 0.004284314]. Its point estimate is small and positive, but its interval includes zero; incremental AP is not established under this comparison. This is not an equivalence test or evidence that every potentially useful test-related representation has zero value.

**Table 3. Primary-checkpoint logistic results (K15; test n = 20,012). Higher AP/AUROC and lower Brier are better; no operational utility threshold was tested.**

| Model | Average precision | AUROC | Brier |
| --- | --- | --- | --- |
| M0 | 0.641246 | 0.567562 | 0.239811 |
| M1 | 0.641611 | 0.568702 | 0.239705 |
| M2 | 0.642153 | 0.569493 | 0.239553 |

M1 at K15 has AP 0.641611 and Delta AP 0.000365150, with interval [-0.000606030, 0.001310412]. Thus, the narrower failure feature does not establish incremental AP at the primary checkpoint either. All validation choices and per-cell coefficients are retained in the supplementary files rather than chosen to optimize the test-set narrative.

### 6.3 Checkpoints, fixed population and nonlinear models

Table 4 and Figure 1 show the planned checkpoint contrasts. Every main logistic M2-versus-M0 interval includes zero. The K20 point increment, 0.002018722, is larger than the primary increment but remains secondary and has an interval spanning zero. At K5, the narrower M1 contrast has a very small positive interval excluding zero; this does not establish the broader primary K15 M2 contrast. Exact values and additional metrics appear in Appendix C.

**Table 4. All planned main-population logistic contrasts. Intervals are paired 95% repository-bootstrap percentiles for M2 minus M0. K15 is the preselected primary exploratory contrast.**

| K | Test n | M0 AP | M1 Delta AP | M2 Delta AP | M2 95% interval |
| --- | --- | --- | --- | --- | --- |
| 5 | 20,013 | 0.640799 | 0.000069 | 0.000272 | [-0.000019, 0.000607] |
| 10 | 20,013 | 0.639777 | 0.000261 | 0.001361 | [-0.001125, 0.003995] |
| 15 | 20,012 | 0.641246 | 0.000365 | 0.000907 | [-0.002150, 0.004284] |
| 20 | 19,986 | 0.646857 | 0.000726 | 0.002019 | [-0.000681, 0.004885] |
| 30 | 19,614 | 0.661830 | 0.000252 | 0.001008 | [-0.001968, 0.004082] |

![Figure 1. Prespecified exploratory checkpoint contrasts and known-label non-audit coverage. Solid lines denote the main checkpoint-reaching population and dashed lines the fixed reached-30 cohort; bands show unadjusted paired repository-bootstrap percentile intervals. Population and information content change with K.](figures/fig_checkpoints.png)

Restricting to the same reached-30 cohort yields K15 Delta AP 0.000582591 with interval [-0.001289762, 0.002582330]. None of the fixed-population logistic M2 intervals excludes zero. This sensitivity analysis reduces changes in cohort membership across K, but does not equalize the information accumulated over time or remove unmeasured difficulty.

The nonlinear baseline gives a different secondary result (Table 5). At K15 its M0 AP is 0.660063 and M2 AP is 0.663951, with Delta AP 0.003888397 and interval [0.000538769, 0.007222448]. At K5 the nonlinear contrast is negative and its interval includes zero. The K15 nonlinear finding is compatible with model-class-dependent predictive information in these proxies. It does not convert the primary logistic result into a positive finding, establish practical usefulness, or validate the proxy as true execution.

**Table 5. Prespecified secondary nonlinear baseline. The K15 M2 contrast is positive, but does not replace the primary logistic endpoint.**

| K | Model | AP | Delta AP vs M0 | 95% paired interval |
| --- | --- | --- | --- | --- |
| 5 | M0 | 0.651683 | -- | -- |
| 5 | M2 | 0.651181 | -0.000502 | [-0.001289, 0.000237] |
| 15 | M0 | 0.660063 | -- | -- |
| 15 | M2 | 0.663951 | 0.003888 | [0.000539, 0.007222] |

### 6.4 Within-issue support, associations and stability

The eligible pool contains 17,086 issues, of which 7,018 have outcome variation and contribute 69,453 rows to the issue-conditional analysis. There are 61,216 issue/configuration cells; 8,714 are outcome-discordant and contribute 25,028 rows, exceeding the support threshold for Model C. Groups contain at most 12 trajectories per issue and three per issue/configuration cell.

Table 6 and Figure 2 summarize the command-presence coefficient. At K10 its failure odds ratio is 0.8654 in ordinary logistic regression, 1.0804 conditional on issue, and 1.2510 conditional on issue/configuration. At K15 the corresponding values are 0.9545, 0.8921 and 0.8255. Every displayed interval spans one. The point estimates vary with adjustment and checkpoint; they do not support a single stable direction for a testing effect. The conditional populations also differ from the ordinary-regression population, so changes in odds ratios are not a pure decomposition of confounding.

**Table 6. Main command-presence associations with failure. A: ordinary logistic; B: issue conditional; C: issue/configuration conditional. Intervals use successful refits only; coverage is not established when refit failure is substantial.**

| K | Model | Rows | Groups | Failure OR | 95% interval | Valid refits |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | A | 159,867 | -- | 0.8654 | [0.7178, 1.0197] | 472/500 |
| 10 | B | 69,453 | 7,018 | 1.0804 | [0.8449, 1.3730] | 500/500 |
| 10 | C | 25,028 | 8,714 | 1.2510 | [0.8882, 1.8370] | 500/500 |
| 15 | A | 159,867 | -- | 0.9545 | [0.8590, 1.0519] | 500/500 |
| 15 | B | 69,453 | 7,018 | 0.8921 | [0.7750, 1.0532] | 497/500 |
| 15 | C | 25,028 | 8,714 | 0.8255 | [0.6672, 1.0702] | 325/500 |

![Figure 2. Main command-presence odds ratios for failure, with successful-refit percentile intervals. Every displayed interval spans one. A: ordinary logistic; B: issue conditional; C: issue/configuration conditional. Logarithmic horizontal scale; vertical line denotes OR = 1. K15 Model C retains only 325/500 refits.](figures/fig_within_issue.png)

All 14 point-estimate optimizations met the frozen acceptance criterion, but this does not certify a finite, well-identified maximum-likelihood estimate. Of the requested 7,000 refits, 860 failed convergence or rank checks. Main-model valid counts at K10 are 472/500, 500/500 and 500/500 for A/B/C; at K15 they are 500/500, 497/500 and 325/500. Configuration-specific valid counts range from 265/500 to 500/500. Sparse behavior and group support are important context, but the failure log does not distinguish every failure's causal source. We therefore report the counts rather than attributing all failures to one mechanism. The K10 SWE-agent/Qwen subgroup also has numerically extreme coefficients despite passing the gradient criterion; its magnitude and percentile intervals are not reliable effect estimates. Appendix D gives these diagnostics, all stability counts and configuration-specific command-presence outputs.

## 7. Discussion

The most direct result is the separation between a command signal and a verified behavior. The audit's high definite-command share can coexist with a substantially smaller execution-confirmed share, even before considering that the reconciled labels are AI-assisted and not an independent reference standard. Replacing unknown command evidence with a negative label would obscure precisely the cases where the trace cannot answer the measurement question. A useful measurement pipeline should preserve these distinctions rather than interpreting every detected runner invocation as a completed test run.

The primary prediction result is also more limited than a positive point estimate suggests. The broad proxy set improves AP by about nine ten-thousandths in the selected logistic comparison, with a cluster interval spanning both deterioration and improvement. This does not establish a practically useful early-warning feature. The positive K15 nonlinear secondary result deserves to be reported rather than suppressed: a richer functional form may extract information that the specified logistic model does not. However, model dependence, multiple exploratory contrasts and reuse of the historical snapshot make an independent frozen-dataset evaluation necessary before drawing a general claim. No decision threshold, human-review workload, runtime saving or downstream intervention benefit was tested here.

Within-issue adjustment answers a different question from held-out ranking. It compares trajectories sharing an issue, or an issue/configuration cell, rather than asking whether a score ranks unseen repositories well. The command-presence intervals span one in the main models and point directions change with the comparison. Even a more precise observational association would not settle causality. Testing can follow successful localization or respond to an already failing attempt, and the trace does not randomize these decisions.

Finally, computation completing is not the same as every interval being reliable. Repository resampling exposes weaknesses that can be hidden by reporting only point estimates. Some subgroup analyses retain little more than half of their requested bootstrap refits. Percentiles from those selected successful fits may not represent the intended resampling distribution. The failure counts are part of the result, not implementation details to discard. A future study should independently validate annotations, predefine how to handle weakly identified features, and evaluate any resulting rule on a genuinely new snapshot. Those are follow-up studies, not changes made to improve the present results after observing them.

## 8. Limitations and research integrity

**Measurement and shared assistance.** The review packages began as AI drafts and AI candidates resolved disagreements. Shared rule or model errors may survive review and create apparently high agreement. Owner attestations establish the declared provenance used by the project, not independent identity or chronology verification. The final labels must not be presented as independently human-adjudicated ground truth. No retrospective formal measurement interval corrects this limitation.

**Snapshot and outcome selection.** Only four historical configuration groups are represented, and 46,758 trajectories lack the modeled binary outcome. The analysis cannot establish behavior or predictive performance for those unknown-outcome trajectories, new models, new scaffolds, later dataset revisions or production settings. Repository separation reduces one form of leakage but does not prove that related projects, duplicated text or shared upstream code are absent across partitions.

**Prefix and task-selection effects.** A completed interaction round is not a uniform quantity of work. Conditioning on reaching K changes the sampled population, while the fixed reached-30 sensitivity still conditions on eventual duration. Short prefixes in the explanatory analyses differ from the checkpoint-reaching predictive populations. These choices limit temporal and causal interpretation even though they were fixed in advance.

**Inference and model dependence.** Test-bootstrap intervals condition on saved fitted predictors. Within-issue intervals exclude failed refits, sometimes at substantial rates. A small gradient can also coexist with numerically saturated coefficients: the implementation does not provide a formal separation diagnosis or establish a finite maximum-likelihood estimate for every subgroup. Multiple endpoints are exploratory and lack familywise correction; there is no equivalence margin, prospective power guarantee, or intervention experiment. A positive secondary interval cannot compensate for the primary interval spanning zero. Conversely, that primary result does not rule out every nonlinear or richer representation.

**AI assistance and contributor roles.** AI assisted the annotation drafts, adjudication, computational implementation and manuscript preparation. Yunjie Guo and Hanqing Yang are listed as authors in the supplied order and were identified by the project owner as the two individual reviewers. This reported review role does not imply independent unaided annotation. The local artifact package preserves the evidence needed to inspect the review and adjudication provenance. No public artifact release or external submission is asserted here.

## 9. Conclusion

On this pinned coding-agent snapshot, detected test-command evidence and confirmed execution are distinct measurement targets. Adding four frozen command-related proxies does not establish incremental average precision in the preselected K15 logistic comparison. A positive nonlinear secondary contrast shows that the predictive conclusion depends on model class, while the main within-issue command-presence intervals span one and several bootstrap analyses are unstable. The evidence supports a qualified exploratory characterization of observable traces, not an independently validated behavioral gold standard, a causal claim about testing, or a demonstrated operational benefit.

## Reproducibility and availability

The local research package retains the pinned shard manifest, metadata, audit exclusion identifiers, frozen feature matrix, first-round submissions, AI reconciliation provenance, code bindings, prediction vectors, validation choices, prediction-bootstrap draws, and within-issue percentile summaries and failed-refit indices. The source data remain available from the pinned provider revision [1]; raw trajectory shards are not duplicated in the delivery archive. The primary feature matrix SHA-256 is `6276f1d7c1d932d7c9e9e27fe2e2570304797d7cc954298909402d0fd2075489`. The exclusion-list SHA-256 is `069d6421a4f57b63df47c7de4f68810f53d230725b43b1037ca6cf836c09e7ff`.

The numerical environment uses Python 3.12.10, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, scikit-learn 1.9.1 and statsmodels 0.14.6. The recorded full suite has 333 passing tests and no skips; subsequent scheduling checks also pass. Checks include exact historical K5 feature equality and conditional-likelihood verification. These checks establish implementation properties, not scientific validity of the annotations or nominal coverage after failed refits. The manuscript tables are generated from locked result files, and the accompanying manifest records their hashes. Public release and venue submission remain separate author decisions.

## References

[1] NVIDIA. Open-SWE-Traces. Dataset, pinned revision c144d9d160dcd9f5ebcd9520d2d54111dda8b681. [Pinned dataset card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/blob/c144d9d160dcd9f5ebcd9520d2d54111dda8b681/README.md). Accessed 3 November 2026.

[2] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR. [Paper](https://arxiv.org/abs/2310.06770).

[3] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS. [Paper](https://arxiv.org/abs/2405.15793).

[4] Xingyao Wang, Boxuan Li, Yufan Song, et al. 2025. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. ICLR. [Paper](https://arxiv.org/abs/2407.16741).

[5] Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and Fairness. Proceedings of the ACM Conference on Fairness, Accountability, and Transparency, 375-385. [DOI: 10.1145/3442188.3445901](https://doi.org/10.1145/3442188.3445901).

[6] A. Colin Cameron and Douglas L. Miller. 2015. A Practitioner's Guide to Cluster-Robust Inference. Journal of Human Resources 50(2), 317-372. [DOI: 10.3368/jhr.50.2.317](https://doi.org/10.3368/jhr.50.2.317).

[7] Scikit-learn developers. average_precision_score: API documentation and metric definition. [Documentation](https://sklearn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Accessed 3 November 2026.

[8] Statsmodels developers. ConditionalLogit, version 0.14.6: API documentation. [Documentation](https://www.statsmodels.org/v0.14.6/generated/statsmodels.discrete.conditional_models.ConditionalLogit.html). Accessed 3 November 2026.

## Appendix A. Relationship to the historical analysis

The original frozen K5 analysis scored 20,118 test trajectories and compared a historical structural model with that model plus the original strict explicit-test-failure signal. Structural AP was 0.640960128243 and augmented AP was 0.641026019969, giving Delta AP 0.000065891727 and a paired repository-bootstrap interval [0.000005607830, 0.000132822866]. The OpenHands-only historical replication also matched its stored predictions and metrics. The original PDF and decision record remain unchanged.

The current extension excludes all audit identifiers, expands the proxy set, uses a separate command detector, adds checkpoints and selects K15 as its primary exploratory endpoint. Even its K5 M1 cell is therefore not identical to the historical specification. The old result and the new results are not two independent studies; they reuse the snapshot. We do not interpret their numerical differences as an intervention effect, a correction of the historical result, or a before/after improvement.

## Appendix B. Audit frames and proxy support

The frozen primary strata use parser-eligible trajectories and the following rule outputs: A requires high-confidence R5 command presence; B requires high-confidence R15 presence, R5 detector absence, and no missing R5 execution command; C requires R15 detector absence and no missing R15 execution command; D requires R15 detector absence with a missing R15 execution command. The additional AMB5/AMB15 frames require the corresponding ambiguous-command flag. The reported R15 positive summary combines A and B; it is not a blanket estimate for all possible R15-positive histories, including those outside these eligibility rules. Each configuration's N_h/n_h is retained in the sampling design.

The command-status categories are definite test command, definite no-test command, ambiguous, command unrecoverable and insufficient evidence. Separate fields retain execution confirmation, process status, pre-test failure detail, output status and mixed pass/failure composition. An explicit failure and a process-started status are not equivalent to successful test execution. Neither same-runner failure-then-pass nor an explicit passing summary establishes that the final patch resolves its issue.

**Table A1. Support for frozen detector proxies among checkpoint-reaching eligible trajectories across all prediction partitions.**

| K | Eligible n | Command present | Explicit failure | Failure-then-pass |
| --- | --- | --- | --- | --- |
| 5 | 159,867 | 1,020 | 64 | 1 |
| 10 | 159,864 | 14,142 | 1,303 | 124 |
| 15 | 159,823 | 35,816 | 4,659 | 715 |
| 20 | 159,543 | 56,764 | 10,117 | 2,271 |
| 30 | 156,326 | 86,639 | 23,547 | 8,982 |

Table A1 counts detector outputs among checkpoint-reaching known-label, non-audit trajectories pooled across the three prediction partitions. These are descriptive support counts, not additional predictive evaluations. The explanatory models at K10/K15 retain shorter trajectories as described in Section 5.3, so their denominators differ. Per-configuration covariate distributions and all retained coefficients are supplied in the accompanying CSV/JSON artifacts.

## Appendix C. Additional predictive results

**Table A2. Complete main-population logistic metrics. All AP contrasts use the same K-specific M0. Intervals are unadjusted exploratory paired-bootstrap summaries.**

| K | Model | AP | AUROC | Brier | Delta AP | 95% paired interval |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | M0 | 0.640799 | 0.557202 | 0.240135 | -- | -- |
| 5 | M1 | 0.640868 | 0.557345 | 0.240117 | 0.000069 | [0.000003, 0.000169] |
| 5 | M2 | 0.641071 | 0.557598 | 0.240090 | 0.000272 | [-0.000019, 0.000607] |
| 10 | M0 | 0.639777 | 0.559388 | 0.240105 | -- | -- |
| 10 | M1 | 0.640038 | 0.559556 | 0.240072 | 0.000261 | [-0.000304, 0.000849] |
| 10 | M2 | 0.641138 | 0.560599 | 0.239970 | 0.001361 | [-0.001125, 0.003995] |
| 15 | M0 | 0.641246 | 0.567562 | 0.239811 | -- | -- |
| 15 | M1 | 0.641611 | 0.568702 | 0.239705 | 0.000365 | [-0.000606, 0.001310] |
| 15 | M2 | 0.642153 | 0.569493 | 0.239553 | 0.000907 | [-0.002150, 0.004284] |
| 20 | M0 | 0.646857 | 0.573203 | 0.239113 | -- | -- |
| 20 | M1 | 0.647583 | 0.574814 | 0.238977 | 0.000726 | [-0.000423, 0.001895] |
| 20 | M2 | 0.648875 | 0.576545 | 0.238673 | 0.002019 | [-0.000681, 0.004885] |
| 30 | M0 | 0.661830 | 0.586637 | 0.236871 | -- | -- |
| 30 | M1 | 0.662082 | 0.587018 | 0.236837 | 0.000252 | [-0.000072, 0.000600] |
| 30 | M2 | 0.662838 | 0.588424 | 0.236574 | 0.001008 | [-0.001968, 0.004082] |

Table A2 contains all main-population logistic metrics. M1 and M2 intervals compare each model with the same K-specific M0. All prediction contrasts have 1,000 valid paired bootstrap draws. Model-selection grids, selected C values and M2 standardized coefficients are preserved in the cell-level result files. No failed prediction-bootstrap draws occurred.

**Table A3. Fixed reached-30 population, M2 versus M0 at each checkpoint.**

| K | Test n | M2 AP | Delta AP | 95% paired interval |
| --- | --- | --- | --- | --- |
| 5 | 19,614 | 0.644254 | 0.000139 | [-0.000088, 0.000402] |
| 10 | 19,614 | 0.644045 | 0.000934 | [-0.001059, 0.003027] |
| 15 | 19,614 | 0.645746 | 0.000583 | [-0.001290, 0.002582] |
| 20 | 19,614 | 0.651458 | 0.001079 | [-0.000533, 0.002885] |
| 30 | 19,614 | 0.662838 | 0.001008 | [-0.001968, 0.004082] |

Table A3 reports the M2 sensitivity contrast on the fixed reached-30 population. Its test cohort has 19,614 trajectories at every checkpoint. The fixed training and validation cohorts have 110,791 and 25,921 trajectories, respectively. The complete fixed-population metrics and nonlinear parameter grids accompany the manuscript as supplementary machine-readable files.

## Appendix D. Within-issue stability and configuration comparisons

**Table A4. Complete bootstrap refit accounting (500 requested per analysis). A completed point estimate does not imply all resamples were estimable.**

| Analysis | K | Point-estimate rows | Valid refits | Failed refits |
| --- | --- | --- | --- | --- |
| A | 10 | 159,867 | 472 | 28 |
| B | 10 | 69,453 | 500 | 0 |
| B: OH/M | 10 | 4,769 | 307 | 193 |
| B: OH/Q | 10 | 7,348 | 500 | 0 |
| B: SWE/M | 10 | 7,041 | 500 | 0 |
| B: SWE/Q | 10 | 5,870 | 265 | 235 |
| C | 10 | 25,028 | 500 | 0 |
| A | 15 | 159,867 | 500 | 0 |
| B | 15 | 69,453 | 497 | 3 |
| B: OH/M | 15 | 4,769 | 330 | 170 |
| B: OH/Q | 15 | 7,348 | 500 | 0 |
| B: SWE/M | 15 | 7,041 | 500 | 0 |
| B: SWE/Q | 15 | 5,870 | 444 | 56 |
| C | 15 | 25,028 | 325 | 175 |

Table A4 counts every requested within-issue bootstrap refit. Failures include lack of sufficient design rank and optimization failures under the frozen acceptance criteria. Successful-refit percentiles are reported rather than imputing a coefficient for a failed fit. Where many fits fail, those percentiles should be treated as unstable exploratory summaries rather than intervals with demonstrated nominal coverage.

**Table A5. Configuration-specific issue-conditional command-presence odds ratios for task failure. Intervals use successful refits only.**

| Configuration | K | Failure OR | 95% interval | Valid refits |
| --- | --- | --- | --- | --- |
| B: OH/M | 10 | 1.6313 | [0.5906, 3.5788] | 307/500 |
| B: OH/Q | 10 | 2.0337 | [0.2445, 36.8278] | 500/500 |
| B: SWE/M | 10 | 1.1663 | [0.7137, 1.7511] | 500/500 |
| B: SWE/Q | 10 | 2.83e-15 | [1.55e-15, 4.87e-15] | 265/500 |
| B: OH/M | 15 | 0.8380 | [0.5146, 1.3270] | 330/500 |
| B: OH/Q | 15 | 1.1968 | [0.6710, 2.3868] | 500/500 |
| B: SWE/M | 15 | 0.7920 | [0.5615, 1.1225] | 500/500 |
| B: SWE/Q | 15 | 0.6887 | [0.2048, 2.8978] | 444/500 |

Table A5 reports the configuration-specific issue-conditional command-presence outputs. OH denotes OpenHands, SWE denotes SWE-agent, M denotes `minimax_m25`, and Q denotes `qwen35_122b`. These subgroup models use outcome-discordant issues within each configuration. All behavior-covariate estimates, dropped variables, likelihood-check discrepancies and failed-refit indices are retained in the underlying analysis JSON files. Point-estimate completion does not resolve sparse support or selection among successful bootstrap fits.

The K10 SWE/Q fit requires particular caution. Its reported command-presence OR is 2.83e-15, accompanied by a command-count OR of 1.33e21 and an explicit-failure OR of 4.75e55. The explicit-failure coefficient's bootstrap percentiles collapse to its point value. These numerically extreme and saturated outputs are preserved for transparency, but neither their coefficient magnitudes nor their percentile intervals should be interpreted as reliable evidence of a behavioral effect. The frozen gradient-based acceptance criterion and agreement with the reference likelihood do not establish existence of a finite maximum-likelihood estimate. We did not modify or refit the model after observing these diagnostics.
