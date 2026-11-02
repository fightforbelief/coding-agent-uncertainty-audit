"""Rerun the frozen L2 fit from saved tables. Do not rewrite that run."""

from __future__ import annotations

import json
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from p8audit.l2_models import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    fit_selected,
    paired_bootstrap,
    select_c,
)
from p8audit.prefix_features import predictor_fields

OUT = ROOT / "analysis_v2"
PRIMARY_STRUCTURAL = 0.6409601282425617
PRIMARY_AUGMENTED = 0.6410260199693942
OPENHANDS_STRUCTURAL = 0.6800000121048011
OPENHANDS_AUGMENTED = 0.6789736160889954
TOLERANCE = 1e-9


def _run_dir() -> Path:
    candidates = []
    if os.environ.get("P8_L2_RUN"):
        candidates.append(Path(os.environ["P8_L2_RUN"]))
    candidates.append(ROOT / "artifacts" / "l2_run_20261027_01")
    for path in candidates:
        if (path / "prefix_features.parquet").is_file() and (path / "labels_for_join.parquet").is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise SystemExit(
        "缺少冻结运行的 prefix_features.parquet 和 labels_for_join.parquet。"
        f"已查找：{searched}。这两份表在 hanqing-win 的 "
        "artifacts/l2_run_20261027_01/，当前机器连不上，因此没有重跑主分析。"
    )


def _matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].to_numpy(dtype=float)


def _close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= TOLERANCE


def main(output_dir: Path | None = None) -> None:
    output = output_dir or OUT
    names = ("preds_primary.parquet", "preds_openhands.parquet", "meta.parquet", "00_reproduce_report.json")
    if any((output / name).exists() for name in names):
        raise SystemExit("输出已存在。请使用 --output-dir 指定新的目录，避免覆盖既有研究结果。")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = _run_dir()
    features = pd.read_parquet(run_dir / "prefix_features.parquet")
    labels = pd.read_parquet(run_dir / "labels_for_join.parquet")
    merged = features.merge(
        labels,
        on=["shard", "row_index"],
        how="inner",
        suffixes=("", "_label"),
        validate="one_to_one",
    )
    if not (merged["trajectory_id"] == merged["trajectory_id_label"]).all():
        raise SystemExit("特征表和标签表的 trajectory_id 对不上，已停止。")
    fields = predictor_fields(5, include_exit=False)
    structural_fields = [name for name in fields if not name.endswith("prefix_explicit_test_failure_any")]
    event = "k5_prefix_explicit_test_failure_any"
    eligible = merged["parse_eligible"].astype(bool) & merged["resolved"].isin([0, 1]) & merged[fields].notna().all(axis=1)
    work = merged.loc[eligible].copy()
    work["y"] = (work["resolved"] == 0).astype(int)
    train = work[work["split"] == "train"]
    valid = work[work["split"] == "validation"]
    test = work[work["split"] == "test"]
    structural_c, _ = select_c(_matrix(train, structural_fields), train["y"].to_numpy(), _matrix(valid, structural_fields), valid["y"].to_numpy())
    augmented_c, _ = select_c(_matrix(train, fields), train["y"].to_numpy(), _matrix(valid, fields), valid["y"].to_numpy())
    fit_rows = work[work["split"].isin(["train", "validation"])]
    structural_model = fit_selected(_matrix(fit_rows, structural_fields), fit_rows["y"].to_numpy(), structural_c)
    augmented_model = fit_selected(_matrix(fit_rows, fields), fit_rows["y"].to_numpy(), augmented_c)
    structural_proba = structural_model.predict_proba(_matrix(test, structural_fields))[:, 1]
    augmented_proba = augmented_model.predict_proba(_matrix(test, fields))[:, 1]
    y_test = test["y"].to_numpy()
    structural_ap = float(average_precision_score(y_test, structural_proba))
    augmented_ap = float(average_precision_score(y_test, augmented_proba))
    report = {
        "sklearn": sklearn.__version__,
        "structural_ap": structural_ap,
        "augmented_ap": augmented_ap,
        "structural_c": structural_c,
        "augmented_c": augmented_c,
        "test_rows": int(len(test)),
        "source_run": str(run_dir),
    }
    if not _close(structural_ap, PRIMARY_STRUCTURAL) or not _close(augmented_ap, PRIMARY_AUGMENTED):
        report["primary_match"] = False
        (output / "00_reproduce_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit(
            "主结果没有在容差 1e-9 内复现，已停止，没有写入预测文件。"
            f" structural={structural_ap} augmented={augmented_ap} sklearn={sklearn.__version__}"
        )
    if len(test) != 20118:
        raise SystemExit(f"测试集行数是 {len(test)}，不是 20,118。已停止。")

    exit_field = "k5_prefix_tool_exit_nonzero_any"
    openhands = work[(work["config"] == "openhands") & work[exit_field].notna()].copy()
    oh_fields = structural_fields + [exit_field]
    oh_train = openhands[openhands["split"] == "train"]
    oh_valid = openhands[openhands["split"] == "validation"]
    oh_test = openhands[openhands["split"] == "test"]
    oh_fit = openhands[openhands["split"].isin(["train", "validation"])]
    oh_structural_c, _ = select_c(_matrix(oh_train, structural_fields), oh_train["y"].to_numpy(), _matrix(oh_valid, structural_fields), oh_valid["y"].to_numpy())
    oh_augmented_c, _ = select_c(_matrix(oh_train, oh_fields), oh_train["y"].to_numpy(), _matrix(oh_valid, oh_fields), oh_valid["y"].to_numpy())
    oh_structural_model = fit_selected(_matrix(oh_fit, structural_fields), oh_fit["y"].to_numpy(), oh_structural_c)
    oh_augmented_model = fit_selected(_matrix(oh_fit, oh_fields), oh_fit["y"].to_numpy(), oh_augmented_c)
    oh_structural_p = oh_structural_model.predict_proba(_matrix(oh_test, structural_fields))[:, 1]
    oh_augmented_p = oh_augmented_model.predict_proba(_matrix(oh_test, oh_fields))[:, 1]
    oh_structural_ap = float(average_precision_score(oh_test["y"].to_numpy(), oh_structural_p))
    oh_augmented_ap = float(average_precision_score(oh_test["y"].to_numpy(), oh_augmented_p))
    report.update(
        {
            "primary_match": True,
            "openhands_structural_ap": oh_structural_ap,
            "openhands_augmented_ap": oh_augmented_ap,
            "openhands_structural_match": _close(oh_structural_ap, OPENHANDS_STRUCTURAL),
            "openhands_augmented_match": _close(oh_augmented_ap, OPENHANDS_AUGMENTED),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
    )
    bootstrap = paired_bootstrap(
        y_test,
        structural_proba,
        augmented_proba,
        test["repo"].astype(str).tolist(),
        seed=BOOTSTRAP_SEED,
        n_replicates=BOOTSTRAP_REPLICATES,
    )
    low, high = bootstrap.percentile_interval()
    report["bootstrap_interval"] = [low, high]
    report["bootstrap_valid"] = bootstrap.n_valid

    primary = pd.DataFrame(
        {
            "trajectory_id": test["trajectory_id"].astype(str),
            "issue_id": test["instance_id"].astype(str),
            "repo": test["repo"].astype(str),
            "scaffold": test["config"].astype(str),
            "model": test["dataset_split"].astype(str),
            "label": test["y"].astype(int),
            "feature_value": test[event].astype(int),
            "p_structural": structural_proba,
            "p_augmented": augmented_proba,
        }
    )
    openhands_preds = pd.DataFrame(
        {
            "trajectory_id": oh_test["trajectory_id"].astype(str),
            "issue_id": oh_test["instance_id"].astype(str),
            "repo": oh_test["repo"].astype(str),
            "scaffold": oh_test["config"].astype(str),
            "model": oh_test["dataset_split"].astype(str),
            "label": oh_test["y"].astype(int),
            "feature_value": oh_test[exit_field].astype(int),
            "p_structural": oh_structural_p,
            "p_augmented": oh_augmented_p,
        }
    )
    meta = pd.DataFrame(
        {
            "trajectory_id": merged["trajectory_id"].astype(str),
            "issue_id": merged["instance_id"].astype(str),
            "repo": merged["repo"].astype(str),
            "scaffold": merged["config"].astype(str),
            "model": merged["dataset_split"].astype(str),
            "resolved": merged["resolved"].astype(int),
            "split": merged["split"].astype(str),
            "n_rounds_total": merged["n_complete_cycles"],
            "parser_v2_strict_fail_r5": merged[event],
        }
    )
    if len(meta) != 207489:
        raise SystemExit(f"meta 行数是 {len(meta)}，不是 207,489。已停止，没有覆盖已有输出。")
    primary.to_parquet(output / "preds_primary.parquet", index=False)
    openhands_preds.to_parquet(output / "preds_openhands.parquet", index=False)
    meta.to_parquet(output / "meta.parquet", index=False)
    (output / "00_reproduce_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    main(parser.parse_args().output_dir)
