"""Workbook generation, reconciliation, and unreviewed summaries."""

import csv
from pathlib import Path

import pytest

from p8audit.review_workbook import (
    apply_human_labels,
    summarize_review,
    write_review_package,
)


def _row(sample_id: str, family: str, purpose: str, key: str, excerpt: str = "traceback") -> dict:
    return {
        "sample_id": sample_id,
        "observation_key": key,
        "signal_family": family,
        "review_purpose": purpose,
        "config": "openhands",
        "split": "minimax_m25",
        "parquet_shard": "data/shard.parquet",
        "language": "python",
        "matched_pattern": "traceback",
        "parser_predicted_label": "positive",
        "sampling_probability": "2/10 (0.2000000000)",
        "context_excerpt": excerpt,
        "excerpt_truncated": "false",
        "reviewer_label": "",
        "reviewer_notes": "",
        "student_verified": "false",
        "resolved": "0",
    }


def _fixture() -> list[dict]:
    shared = "data/shard.parquet|traj-1|4"
    return [
        _row("g075-00001", "explicit_exception", "precision_validation", "data/shard.parquet|traj-1|1"),
        _row("g075-00002", "explicit_test_failure", "precision_validation", "data/shard.parquet|traj-1|2"),
        _row("g075-00003", "tool_exit_nonzero", "precision_validation", "data/shard.parquet|traj-1|3"),
        _row("g075-00004", "execution_status_unknown", "diagnostic_coverage", shared, "full excerpt stays"),
        _row("g075-00005", "missed_test_format", "diagnostic_coverage", shared, "full excerpt stays"),
    ]


def test_package_drops_outcome_fields_and_keeps_the_excerpt(tmp_path: Path) -> None:
    manifest = write_review_package(_fixture(), tmp_path / "package")
    exception_path = tmp_path / "package" / "01_precision_explicit_exception.csv"
    with exception_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert "resolved" not in rows[0]
    assert rows[0]["context_excerpt"] == "traceback"
    assert rows[0]["student_verified"] == "false"
    assert rows[0]["reviewer_label"] == ""
    diagnostic = tmp_path / "package" / "04_diagnostic_review.csv"
    with diagnostic.open(encoding="utf-8-sig", newline="") as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    assert {row["repeated_observation_slot"] for row in diagnostic_rows} == {"yes"}
    assert manifest["summary"]["unreviewed_rows"] == 5
    assert manifest["summary"]["student_verified_rows"] == 0
    assert manifest["summary"]["true_positive"] == 0
    assert manifest["xlsx_written"] is False


def test_summary_does_not_count_a_blank_label_as_verified() -> None:
    rows = _fixture()
    rows[0]["student_verified"] = "true"
    summary = summarize_review(rows)
    assert summary["student_verified_rows"] == 0
    assert summary["unreviewed_rows"] == 5
    rows[0]["reviewer_label"] = "true_positive"
    summary = summarize_review(rows)
    assert summary["reviewed_rows"] == 1
    assert summary["true_positive"] == 1
    assert summary["student_verified_rows"] == 1
    assert summary["unreviewed_rows"] == 4


def test_reconcile_matches_stable_ids_and_does_not_invent_verification() -> None:
    template = _fixture()
    labels = [
        {
            "sample_id": "g075-00001",
            "observation_key": template[0]["observation_key"],
            "signal_family": "explicit_exception",
            "review_purpose": "precision_validation",
            "reviewer_label": "ambiguous",
            "reviewer_notes": "two readings",
        }
    ]
    merged = apply_human_labels(template, labels)
    assert merged[0]["reviewer_label"] == "ambiguous"
    assert merged[0]["student_verified"] == "false"
    assert merged[1]["reviewer_label"] == ""
    summary = summarize_review(merged)
    assert summary["ambiguous"] == 1
    assert summary["student_verified_rows"] == 0


def test_reconcile_rejects_a_mismatched_observation_or_unknown_label() -> None:
    template = _fixture()
    bad_key = {
        "sample_id": "g075-00001",
        "observation_key": "other",
        "signal_family": "explicit_exception",
        "review_purpose": "precision_validation",
        "reviewer_label": "true_positive",
    }
    with pytest.raises(ValueError, match="observation_key"):
        apply_human_labels(template, [bad_key])
    bad_label = {
        "sample_id": "g075-00001",
        "observation_key": template[0]["observation_key"],
        "signal_family": "explicit_exception",
        "review_purpose": "precision_validation",
        "reviewer_label": "proxy_positive",
    }
    with pytest.raises(ValueError, match="unknown reviewer_label"):
        apply_human_labels(template, [bad_label])


def test_package_refuses_the_source_queue_filename(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refusing"):
        write_review_package(_fixture(), tmp_path / "marker_review_queue_v2.csv")
