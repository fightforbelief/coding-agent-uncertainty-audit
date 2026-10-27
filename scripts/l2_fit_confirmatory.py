"""Fit the frozen L2 models after the feature table exists.

The feature parquet is read first. ``resolved`` is joined only after that
schema is confirmed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from p8audit.l2_models import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    fit_selected,
    issue_weighted_average_precision,
    paired_bootstrap,
    select_c,
)
from p8audit.prefix_features import choose_checkpoint, predictor_fields

ROOT = Path(r"C:\Users\hanqi\Desktop\projects\p8-coding-agent-uncertainty-audit")
RUN = ROOT / "artifacts" / "l2_run_20261027_01"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(y: np.ndarray, proba: np.ndarray) -> float:
    return float(average_precision_score(y, proba))


def main() -> None:
    feature_path = RUN / "prefix_features.parquet"
    label_path = RUN / "labels_for_join.parquet"
    schema = json.loads((RUN / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("resolved_present") is not False or "resolved" in schema["columns"]:
        raise SystemExit("feature schema is not frozen without resolved")
    feature_columns = pq.ParquetFile(feature_path).schema_arrow.names
    if "resolved" in feature_columns:
        raise SystemExit("feature parquet contains resolved")
    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)
    merged = features.merge(labels, on=["shard", "row_index"], how="inner", suffixes=("", "_label"), validate="one_to_one")
    if len(merged) != len(features) or not (merged["trajectory_id"] == merged["trajectory_id_label"]).all():
        raise SystemExit("feature and label rows do not match")
    if not (merged["repo"] == merged["repo_label"]).all() or not (merged["config"] == merged["config_label"]).all():
        raise SystemExit("repository or config changed between passes")
    n_known = int((merged["resolved"] != -1).sum())
    reaching = (merged["resolved"] != -1) & (merged["n_complete_cycles"] >= 5)
    n_reaching = int(reaching.sum())
    checkpoint = choose_checkpoint(n_known, n_reaching)
    coverage = pd.DataFrame(
        [
            {
                "known_label_trajectories": n_known,
                "known_label_reaching_five_complete_rounds": n_reaching,
                "coverage": n_reaching / n_known,
                "selected_checkpoint": checkpoint,
                "parse_failures": int((~merged["parse_eligible"].astype(bool)).sum()),
            }
        ]
    )
    coverage.to_csv(RUN / "L2_PREFIX_COVERAGE.csv", index=False)

    fields = predictor_fields(checkpoint, include_exit=False)
    eligible = merged["parse_eligible"].astype(bool) & merged["resolved"].isin([0, 1]) & merged[fields].notna().all(axis=1)
    work = merged.loc[eligible].copy()
    work["y"] = (work["resolved"] == 0).astype(int)
    structural_fields = [name for name in fields if not name.endswith("prefix_explicit_test_failure_any")]
    augmented_fields = fields
    if structural_fields == augmented_fields:
        raise SystemExit("augmented model did not add the test-failure feature")

    def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
        return frame[columns].to_numpy(dtype=float)

    train = work[work["split"] == "train"]
    valid = work[work["split"] == "validation"]
    test = work[work["split"] == "test"]
    if train["y"].nunique() < 2 or valid["y"].nunique() < 2 or test["y"].nunique() < 2:
        raise SystemExit("a split is missing an outcome class")
    structural_c, structural_grid = select_c(matrix(train, structural_fields), train["y"].to_numpy(), matrix(valid, structural_fields), valid["y"].to_numpy())
    augmented_c, augmented_grid = select_c(matrix(train, augmented_fields), train["y"].to_numpy(), matrix(valid, augmented_fields), valid["y"].to_numpy())
    fit_rows = work[work["split"].isin(["train", "validation"])]
    structural_model = fit_selected(matrix(fit_rows, structural_fields), fit_rows["y"].to_numpy(), structural_c)
    augmented_model = fit_selected(matrix(fit_rows, augmented_fields), fit_rows["y"].to_numpy(), augmented_c)
    structural_proba = structural_model.predict_proba(matrix(test, structural_fields))[:, 1]
    augmented_proba = augmented_model.predict_proba(matrix(test, augmented_fields))[:, 1]
    y_test = test["y"].to_numpy()
    structural_ap = average_precision(y_test, structural_proba)
    augmented_ap = average_precision(y_test, augmented_proba)
    delta = augmented_ap - structural_ap
    bootstrap = paired_bootstrap(
        y_test,
        structural_proba,
        augmented_proba,
        test["repo"].astype(str).tolist(),
        seed=BOOTSTRAP_SEED,
        n_replicates=BOOTSTRAP_REPLICATES,
    )
    low, high = bootstrap.percentile_interval()
    includes_zero = low <= 0.0 <= high
    weighted_structural = issue_weighted_average_precision(y_test, structural_proba, test["instance_id"].astype(str).tolist())
    weighted_augmented = issue_weighted_average_precision(y_test, augmented_proba, test["instance_id"].astype(str).tolist())

    exit_field = f"k{checkpoint}_prefix_tool_exit_nonzero_any"
    openhands = work[(work["config"] == "openhands") & work[exit_field].notna()].copy()
    oh_structural = structural_fields
    oh_fields = structural_fields + [exit_field]
    oh_train = openhands[openhands["split"] == "train"]
    oh_valid = openhands[openhands["split"] == "validation"]
    oh_test = openhands[openhands["split"] == "test"]
    oh_structural_c, _ = select_c(matrix(oh_train, oh_structural), oh_train["y"].to_numpy(), matrix(oh_valid, oh_structural), oh_valid["y"].to_numpy())
    oh_augmented_c, _ = select_c(matrix(oh_train, oh_fields), oh_train["y"].to_numpy(), matrix(oh_valid, oh_fields), oh_valid["y"].to_numpy())
    oh_fit = openhands[openhands["split"].isin(["train", "validation"])]
    oh_structural_model = fit_selected(matrix(oh_fit, oh_structural), oh_fit["y"].to_numpy(), oh_structural_c)
    oh_augmented_model = fit_selected(matrix(oh_fit, oh_fields), oh_fit["y"].to_numpy(), oh_augmented_c)
    oh_y = oh_test["y"].to_numpy()
    oh_structural_ap = average_precision(oh_y, oh_structural_model.predict_proba(matrix(oh_test, oh_structural))[:, 1])
    oh_augmented_ap = average_precision(oh_y, oh_augmented_model.predict_proba(matrix(oh_test, oh_fields))[:, 1])

    event = f"k{checkpoint}_prefix_explicit_test_failure_any"
    parsed = merged[merged["parse_eligible"].astype(bool)]
    audit_rows = []
    for shard, group in parsed.groupby("shard", sort=True):
        audit_rows.append(
            {
                "shard": shard,
                "parsed_rows": int(len(group)),
                "event_positives": int((group[event] == 1).sum()),
            }
        )
    pd.DataFrame(audit_rows).to_csv(RUN / "L2_FEATURE_AUDIT_BY_SHARD.csv", index=False)
    config_rows = []
    for (config, dataset_split), group in parsed.groupby(["config", "dataset_split"], sort=True):
        config_rows.append(
            {
                "config": config,
                "dataset_split": dataset_split,
                "parsed_rows": int(len(group)),
                "strict_test_failure_positives": int((group[event] == 1).sum()),
            }
        )
    pd.DataFrame(config_rows).to_csv(RUN / "L2_FEATURE_COUNTS_BY_CONFIG.csv", index=False)

    subgroup_rows = []
    test_frame = test.copy()
    test_frame["structural_proba"] = structural_proba
    test_frame["augmented_proba"] = augmented_proba
    for (config, dataset_split), group in test_frame.groupby(["config", "dataset_split"], sort=True):
        if group["y"].nunique() < 2:
            structural_group = None
            augmented_group = None
        else:
            structural_group = average_precision(group["y"].to_numpy(), group["structural_proba"].to_numpy())
            augmented_group = average_precision(group["y"].to_numpy(), group["augmented_proba"].to_numpy())
        subgroup_rows.append(
            {
                "config": config,
                "dataset_split": dataset_split,
                "rows": int(len(group)),
                "structural_ap": structural_group,
                "augmented_ap": augmented_group,
            }
        )

    metrics = pd.DataFrame(
        [
            {"model": "structural", "split": "test", "average_precision": structural_ap, "C": structural_c},
            {"model": "augmented", "split": "test", "average_precision": augmented_ap, "C": augmented_c},
            {"model": "delta", "split": "test", "average_precision": delta, "C": None},
            {"model": "structural_issue_weighted", "split": "test", "average_precision": weighted_structural, "C": structural_c},
            {"model": "augmented_issue_weighted", "split": "test", "average_precision": weighted_augmented, "C": augmented_c},
            {"model": "delta_issue_weighted", "split": "test", "average_precision": weighted_augmented - weighted_structural, "C": None},
            {"model": "openhands_structural", "split": "test", "average_precision": oh_structural_ap, "C": oh_structural_c},
            {"model": "openhands_augmented_exit", "split": "test", "average_precision": oh_augmented_ap, "C": oh_augmented_c},
            {"model": "openhands_delta", "split": "test", "average_precision": oh_augmented_ap - oh_structural_ap, "C": None},
        ]
    )
    metrics.to_csv(RUN / "L2_TEST_METRICS.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(RUN / "L2_SUBGROUP_TEST_AP.csv", index=False)
    pd.DataFrame(
        {
            "structural_ap": bootstrap.structural_ap,
            "augmented_ap": bootstrap.augmented_ap,
            "delta_auprc": bootstrap.deltas,
        }
    ).to_csv(RUN / "L2_BOOTSTRAP_DELTA_AUPRC.csv", index=False)
    pd.DataFrame({"replicate": bootstrap.skipped_replicates, "reason": "missing_outcome_class_or_empty"}).to_csv(
        RUN / "L2_BOOTSTRAP_SKIPPED.csv", index=False
    )
    predictions = test[["trajectory_id", "instance_id", "repo", "split", "y"]].copy()
    predictions["structural_proba"] = structural_proba
    predictions["augmented_proba"] = augmented_proba
    predictions.to_csv(RUN / "L2_TEST_PREDICTIONS.csv", index=False)
    joblib.dump(structural_model, RUN / "structural_pipeline.joblib")
    joblib.dump(augmented_model, RUN / "augmented_pipeline.joblib")
    (RUN / "hyperparameters.json").write_text(
        json.dumps(
            {
                "structural_C": structural_c,
                "structural_validation_ap": structural_grid,
                "augmented_C": augmented_c,
                "augmented_validation_ap": augmented_grid,
                "openhands_structural_C": oh_structural_c,
                "openhands_augmented_C": oh_augmented_c,
                "checkpoint": checkpoint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    direction = "positive" if delta > 0 else "negative" if delta < 0 else "zero"
    if includes_zero:
        claim = "This experiment did not establish a clear positive incremental AUPRC effect under this protocol."
    elif delta > 0:
        claim = "The paired interval is entirely above zero, so the measured incremental test AUPRC is positive under this protocol and these caveats."
    else:
        claim = "The paired interval is entirely below zero. The experiment did not establish a positive incremental AUPRC effect."
    results = f"""# L2 confirmatory results

Research question, unchanged:

> How well do published trajectory-reliability signals transfer to repository-level coding-agent traces, and do observable execution-feedback features add incremental predictive value beyond simple structural baselines under repository-held-out evaluation?

Checkpoint: {checkpoint} completed rounds. Known-label trajectories reaching five rounds: {n_reaching} / {n_known} ({n_reaching / n_known:.6f}).

Eligible supervised rows: {len(work)}. Test rows: {len(test)}. Parse failures in the snapshot: {int((~merged['parse_eligible'].astype(bool)).sum())}.

Primary test average precision, `sklearn.metrics.average_precision_score`:

- Structural only: {structural_ap:.6f}. Selected C: {structural_c}.
- Structural plus strict `explicit_test_failure`: {augmented_ap:.6f}. Selected C: {augmented_c}.
- Delta AUPRC: {delta:.6f} ({direction}).

Paired repository-cluster 95% percentile interval, seed {BOOTSTRAP_SEED}, requested replicates {BOOTSTRAP_REPLICATES}, valid replicates {bootstrap.n_valid}, skipped {len(bootstrap.skipped_replicates)}: [{low:.6f}, {high:.6f}]. The interval {"includes" if includes_zero else "does not include"} zero.

{claim}

Issue-weighted test sensitivity, weights `1/n_j` inside the eligible test rows only: structural {weighted_structural:.6f}, augmented {weighted_augmented:.6f}, difference {weighted_augmented - weighted_structural:.6f}. This does not replace the unweighted primary estimate.

OpenHands-only secondary test average precision: structural {oh_structural_ap:.6f} (C {oh_structural_c}), plus `prefix_tool_exit_nonzero_any` {oh_augmented_ap:.6f} (C {oh_augmented_c}), difference {oh_augmented_ap - oh_structural_ap:.6f}. Missing SWE-agent exit status was not coded as zero. This is not the primary endpoint.

Subgroup test scores are descriptive and were not used to choose features. See `L2_SUBGROUP_TEST_AP.csv`.

The precision decision for strict test failure comes from AI adjudication of 50 predicted positives on eight fresh shards, not from all 84 shards. Those labels are not human-verified ground truth. The bootstrap does not include uncertainty in the AI labels. L1 remains `L1 ANCHOR UNRESOLVED`. HTC was not exactly reproduced. Novelty has not been established.
"""
    (RUN / "L2_RESULTS.md").write_text(results, encoding="utf-8")
    (RUN / "L2_CLAIM_LEDGER.md").write_text(
        "\n".join(
            [
                "# L2 claim ledger",
                "",
                f"- Primary Delta AUPRC is {delta:.6f}.",
                f"- 95% paired interval is [{low:.6f}, {high:.6f}] and {'includes' if includes_zero else 'does not include'} zero.",
                f"- {claim}",
                "- AI adjudication is not independent human ground truth.",
                "- Precision evidence is limited to the eight fresh shards.",
                "- L1 remains unresolved.",
                "- HTC was not exactly reproduced.",
                "- Novelty has not been established.",
                "- No `execution_failure` feature was created.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "dataset_revision": "c144d9d160dcd9f5ebcd9520d2d54111dda8b681",
        "split_seed": 20261024,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "checkpoint": checkpoint,
        "feature_sha256": sha256(feature_path),
        "label_sha256": sha256(label_path),
        "adjudication_zip_sha256": sha256(ROOT / "artifacts" / "l2_inputs" / "P8_GATE09_AI_ADJUDICATION_RESULTS.zip"),
        "structural_C": structural_c,
        "augmented_C": augmented_c,
        "structural_ap": structural_ap,
        "augmented_ap": augmented_ap,
        "delta_auprc": delta,
        "interval": [low, high],
        "interval_includes_zero": includes_zero,
        "valid_bootstrap_replicates": bootstrap.n_valid,
        "skipped_bootstrap_replicates": len(bootstrap.skipped_replicates),
        "eligible_rows": int(len(work)),
        "test_rows": int(len(test)),
    }
    (RUN / "L2_RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
