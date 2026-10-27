# p8-coding-agent-uncertainty-audit

Independent validation study of trajectory-level reliability signals on the pinned historical Open-SWE-Traces v1.0 snapshot.

This checkout completed Gate 0 with decision `REDESIGN`. Gate 0.5 remains `REDESIGN AMENDMENT PREPARED — AWAITING HUMAN REVIEW`. Gate 0.75 remains `PRE-FREEZE AUDIT PREPARED — MANUAL VALIDATION AND PROTOCOL DECISION PENDING`. Gate 0.8 prepares the human-review package. None of these stages fits confirmatory models, downloads the remaining trajectory bodies, or upgrades the state to `GO`.

Current state: `HUMAN REVIEW PACKAGE PREPARED — STUDENT LABELS AND PROJECT-OWNER DECISIONS PENDING`. See `docs/DECISION_LOG.md`.

Pinned revision: `c144d9d160dcd9f5ebcd9520d2d54111dda8b681` (commit time 2026-07-15T06:29:42Z).

Remote project path: `C:\Users\hanqi\Desktop\projects\p8-coding-agent-uncertainty-audit`.

## Run on hanqing-win

From the project root, with the project virtual environment:

```text
set PYTHONPATH=src
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\00_remote_environment_audit.py
.venv\Scripts\python.exe scripts\02_dataset_schema_audit.py --mode census
.venv\Scripts\python.exe scripts\02_dataset_schema_audit.py --mode parse-sample
.venv\Scripts\python.exe scripts\03_parser_validation.py --mode draw
```

Gate 0 reports live in `artifacts/reports/` and are left in place. Gate 0.5 adds the amendment under `docs/` and the review queue under `artifacts/manual_review/`. Gate 0.75 adds versioned reports and `marker_review_queue_v2.csv` without replacing the 548-row queue. Raw trajectories are not committed.

```text
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\gate05_marker_review_queue.py
.venv\Scripts\python.exe scripts\gate05_l1_anchor_audit.py
.venv\Scripts\python.exe scripts\gate05_literature_overlap.py
.venv\Scripts\python.exe scripts\gate075_issue_distribution.py
.venv\Scripts\python.exe scripts\gate075_marker_queue_v2.py
.venv\Scripts\python.exe scripts\gate08_prepare_review_package.py
```
