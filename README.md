# p8-coding-agent-uncertainty-audit

Independent validation of trajectory-level reliability signals on the pinned historical Open-SWE-Traces v1.0 snapshot.

The completed manuscript is [`paper.pdf`](paper.pdf): **Yunjie Guo and Hanqing Yang**, *Early Test-Command Signals in Coding-Agent Traces: Measurement, Predictive Value, and Within-Issue Associations* (3 November 2026, 13 pages). The accompanying 14-page PDF, LaTeX source, tables, and figures are in [`output/final_paper_20261103/`](output/final_paper_20261103/README.md). The 27 October 2026 frozen-protocol paper is [`paper_20261027.pdf`](paper_20261027.pdf).

Completed exploratory extension (2026-11-03): owner-attested individual review of the two AI-assisted returns and owner-authorized AI adjudication are complete. The full pinned 207,489-trajectory feature census, 12 prediction cells and 14 within-issue analyses have completed, with all 1,196 audit trajectories excluded from every model partition. The preselected K=15 primary exploratory delta AP is 0.000906580475, with paired repository-cluster bootstrap interval [-0.00214988951, 0.00428431441]; incremental AP is not established. All new findings remain exploratory. AI sources, deviations and failed bootstrap fits are disclosed; no third independent human adjudication or independently verified gold standard is claimed.

The October 27 decision remains: `RESEARCH CLOSED — SMALL POSITIVE INCREMENTAL AP; PRACTICAL VALUE NOT ESTABLISHED`.

Pinned dataset revision: `c144d9d160dcd9f5ebcd9520d2d54111dda8b681` (commit time 2026-07-15T06:29:42Z).

The frozen primary endpoint is repository-held-out Delta AP: average precision of L2-regularized logistic regression on structural features plus a strict explicit test-failure indicator, minus average precision of the structural model alone. On the scored test set the structural AP is 0.6409601282425617 and the augmented AP is 0.6410260199693942. Delta AP is 0.00006589172683257871. The paired repository-cluster bootstrap 95% interval is [0.00000560783009556365, 0.00013282286584345433]. The interval excludes zero, and the absolute increment is very small. The primary feature is sparse, so this result does not establish practical usefulness, novelty, or an exact reproduction of a published method. L3 was not run.

## Run on hanqing-win

From the project root, with the project virtual environment:

```text
set PYTHONPATH=src
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\00_remote_environment_audit.py
.venv\Scripts\python.exe scripts\02_dataset_schema_audit.py --mode census
.venv\Scripts\python.exe scripts\02_dataset_schema_audit.py --mode parse-sample
.venv\Scripts\python.exe scripts\03_parser_validation.py --mode draw
.venv\Scripts\python.exe scripts\gate05_marker_review_queue.py
.venv\Scripts\python.exe scripts\gate05_l1_anchor_audit.py
.venv\Scripts\python.exe scripts\gate05_literature_overlap.py
.venv\Scripts\python.exe scripts\gate075_issue_distribution.py
.venv\Scripts\python.exe scripts\gate075_marker_queue_v2.py
.venv\Scripts\python.exe scripts\gate08_prepare_review_package.py
.venv\Scripts\python.exe scripts\gate09_fresh_validation.py
.venv\Scripts\python.exe scripts\l2_fit_confirmatory.py
```

Raw trajectories, fitted pipelines, and run artifacts are not in this repository.
