"""Verify the Gate 0.9 adjudication archive and write the frozen L2 protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\hanqi\Desktop\projects\p8-coding-agent-uncertainty-audit")
ZIP_PATH = ROOT / "artifacts" / "l2_inputs" / "P8_GATE09_AI_ADJUDICATION_RESULTS.zip"
DEST = ROOT / "artifacts" / "l2_inputs" / "gate09_ai_adjudication_results"
EXPECTED_ZIP = "55833daab1f4364f39f8cf5ee01e38cd014988970835f4f38f5c5f8075395181"
EXPECTED_ROWS = {
    "01_explicit_exception_adjudicated.csv": (50, 47, 3),
    "02_explicit_test_failure_adjudicated.csv": (50, 50, 0),
    "03_openhands_exit_status_adjudicated.csv": (50, 50, 0),
    "04_diagnostic_adjudicated.csv": (84, None, None),
    "05_random_observation_adjudicated.csv": (24, None, None),
}

PROTOCOL = r"""# L2 protocol, frozen before the confirmatory fit

**State:** protocol frozen. Results are not in this file. They are written only after the run finishes.

This file does not replace `docs/protocol.md` or `configs/feature_spec.yaml`. Those remain the earlier provisional records.

Research question, unchanged:

> How well do published trajectory-reliability signals transfer to repository-level coding-agent traces, and do observable execution-feedback features add incremental predictive value beyond simple structural baselines under repository-held-out evaluation?

Primary endpoint:

`Delta_AUPRC = AUPRC(structural + execution-feedback) - AUPRC(structural-only)`

Operational AUPRC is `sklearn.metrics.average_precision_score`. The positive class is `resolved == 0`. `resolved == 1` is negative. `resolved == -1` is excluded from supervised fitting and evaluation and is counted separately. The unit is the trajectory. Labels are not majority-voted.

## Adjudication evidence

The input is the AI-assisted adjudication of the eight fresh shards. It is project evidence. It is not human-verified ground truth. `reviewer_label` stays blank. `student_verified` stays false. These labels are not dataset outcomes and not predictors.

| Signal | AI labels | Conditional one-sided 95% finite-population lower bound | L2 decision |
|---|---|---:|---|
| `explicit_exception` | 47 true positives, 3 false positives | 0.8522570532915361 | Excluded from the primary model. Below 0.90 |
| strict `explicit_test_failure` | 50 true positives | 0.9419345961564922 | Sole primary cross-scaffold execution-feedback feature |
| `tool_exit_nonzero` | 50 true positives, OpenHands only | 0.941889378774265 | Secondary OpenHands analysis only |

The bounds are copied from the adjudication manifest. They assume the AI labels are correct and apply only to predicted positives on the eight selected shards. They do not establish precision on all 84 shards and do not include AI labeling error. The three exception false-positive sample ids are not used to edit the parser.

## Other signals

`explicit_test_pass` is exploratory. Build, setup or collection, and unresolved package-level `FAIL` stay separate and are not strict test failures. `interrupt_or_timeout` and `execution_status_unknown` are excluded from the primary model. Exit-status and test-result discordance is diagnostic. There is no `execution_failure` feature. A missing SWE-agent exit code stays missing.

## L1

`L1 ANCHOR UNRESOLVED` is accepted as non-blocking for this incremental-prediction experiment. This is not an exact HTC reproduction and novelty is not established.

## Design that is already fixed

- Dataset revision `c144d9d160dcd9f5ebcd9520d2d54111dda8b681`.
- Repository connected components, seed `20261024`.
- Checkpoint 5 when at least 70% of known-label trajectories reach five completed rounds, otherwise 3. Coverage is measured on the full snapshot before fitting.
- Structural predictors are the minimal set in `configs/l2_feature_spec_frozen_v1.yaml`. The augmented primary model adds only `prefix_explicit_test_failure_any`.
- Logistic regression, L2 penalty, no class weighting, `C` in `{0.01, 0.1, 1.0, 10.0}`, maximum 5000 iterations. `C` is chosen by validation average precision. Ties take the smaller `C`. The selected model is refit on train plus validation. The test set is scored once.
- Paired repository-cluster bootstrap, at least 1000 replicates, seed `20261092`.
- Issue weights `1/n_j` are a sensitivity metric inside one split. They do not replace the primary estimate.
"""

FEATURE_SPEC = r"""# Frozen L2 feature specification v1.
# Written after the adjudication archive hash was verified.
# This file does not replace configs/feature_spec.yaml or configs/feature_spec_v2.yaml.

specification_version: l2_feature_spec_frozen_v1
parser_version: v2
feature_dictionary_version: v2
research_question: >-
  How well do published trajectory-reliability signals transfer to
  repository-level coding-agent traces, and do observable execution-feedback
  features add incremental predictive value beyond simple structural baselines
  under repository-held-out evaluation?
primary_endpoint: >-
  Delta_AUPRC = AUPRC(structural + strict explicit_test_failure)
  - AUPRC(structural-only)
auprc_implementation: sklearn.metrics.average_precision_score
positive_class: "resolved == 0"
checkpoint_rule:
  prefer_cycles: 5
  minimum_known_label_coverage: 0.70
  fallback_cycles: 3
structural_features:
  - prefix_completed_rounds
  - prefix_checkpoint_reached
  - prefix_assistant_message_count
  - prefix_tool_call_count
  - prefix_tool_observation_count
  - prefix_distinct_tool_count
  - prefix_tool_family_execute
  - prefix_tool_family_edit
  - prefix_tool_family_read
  - prefix_tool_family_search
  - prefix_tool_family_think
  - prefix_tool_family_submit
  - prefix_tool_family_other
  - prefix_tool_family_unknown
  - prefix_user_content_chars
  - prefix_assistant_content_chars
  - prefix_tool_observation_chars
  - prefix_tool_observation_lines
primary_added_feature: prefix_explicit_test_failure_any
secondary_openhands_added_feature: prefix_tool_exit_nonzero_any
excluded_from_primary:
  - explicit_exception
  - explicit_test_pass
  - build_or_compile_failure
  - test_setup_or_collection_failure
  - package_level_failure_unresolved
  - interrupt_or_timeout
  - execution_status_unknown
  - discordant_exit_status_and_test_result
  - tool_exit_nonzero
forbidden_combination:
  - execution_failure
tool_family_rule: >-
  Fixed name mapping in prefix_features.tool_family. It is not refit after
  seeing test average precision.
audit_only_not_predictors:
  - n_complete_cycles
  - parse_status
  - repo
  - instance_id
  - trajectory_id
model:
  family: sklearn.linear_model.LogisticRegression
  penalty: l2
  class_weight: null
  C_grid: [0.01, 0.1, 1.0, 10.0]
  tie_break: smaller C
  max_iter: 5000
  selection_metric: validation average precision
  refit: train plus validation
split_seed: 20261024
bootstrap_seed: 20261092
bootstrap_replicates: 1000
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not ZIP_PATH.is_file():
        raise SystemExit(f"missing archive: {ZIP_PATH}")
    actual = sha256(ZIP_PATH)
    if actual != EXPECTED_ZIP:
        raise SystemExit(f"zip hash mismatch: {actual}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for item in manifest["outputs"]:
            payload = archive.read(item["name"])
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise SystemExit(f"member hash mismatch: {item['name']}")
        if DEST.exists():
            raise SystemExit(f"refusing to overwrite {DEST}")
        DEST.mkdir(parents=True)
        for info in archive.infolist():
            target = DEST / info.filename
            if not target.resolve().is_relative_to(DEST.resolve()):
                raise SystemExit(f"unsafe member: {info.filename}")
            target.write_bytes(archive.read(info))
    for name, (rows, positives, negatives) in EXPECTED_ROWS.items():
        with (DEST / name).open(encoding="utf-8-sig", newline="") as handle:
            table = list(csv.DictReader(handle))
        if len(table) != rows:
            raise SystemExit(f"{name} has {len(table)} rows")
        if "resolved" in table[0] or any(row["reviewer_label"] != "" for row in table):
            raise SystemExit(f"{name} failed label isolation")
        if any(row["student_verified"].strip().lower() != "false" for row in table):
            raise SystemExit(f"{name} student_verified is not false")
        if positives is not None:
            labels = [row["ai_adjudication_label"] for row in table]
            if labels.count("true_positive") != positives or labels.count("false_positive") != negatives:
                raise SystemExit(f"{name} label counts do not match the archive")
    protocol = ROOT / "artifacts" / "reports" / "L2_PROTOCOL_FROZEN.md"
    spec = ROOT / "configs" / "l2_feature_spec_frozen_v1.yaml"
    for path in (protocol, spec):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    protocol.write_text(PROTOCOL, encoding="utf-8")
    spec.write_text(FEATURE_SPEC, encoding="utf-8")
    print(json.dumps({"zip_sha256": actual, "extracted": str(DEST), "protocol": str(protocol)}))


if __name__ == "__main__":
    main()
