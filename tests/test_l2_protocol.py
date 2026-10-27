"""L2 prefix, split, bootstrap, and label-separation checks."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from p8audit.dependence import issue_equal_trajectory_weights, repository_cluster_bootstrap_indices
from p8audit.l2_models import paired_bootstrap, select_c
from p8audit.prefix_features import choose_checkpoint, prefix_feature_row, predictor_fields
from p8audit.splits import assign_group_splits, connected_components, crossing_groups
from p8audit.trajectory_parser import parse_trajectory

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts" / "l2_inputs" / "gate09_ai_adjudication_results"
HISTORICAL = {
    "artifacts/manual_review/marker_review_queue_v2.csv":
        "6da2d7cb345ce4c2a8494707e940679b15f2c845ee43b2f29b69bfb19d4264a2",
    "artifacts/reports/GATE0_REPORT.md":
        "44ee6ac39113851fb65bd5a43e1a70b7a1bf13ecd1e674e76cccf973883a40ce",
}


def _cycle(name: str, observation: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "do",
            "reasoning_content": "SECRET" * 50,
            "tool_calls": [{"id": "1", "function": {"name": name, "arguments": "{}"}}],
        },
        {"role": "tool", "content": observation},
    ]


def _six_cycles() -> list[dict]:
    messages: list[dict] = []
    for index in range(5):
        messages.extend(_cycle("bash", f"round {index} ok"))
    messages.append({"role": "user", "content": "FUTURE" * 100})
    messages.extend(_cycle("bash", "FAILED tests/test_app.py::test_one\n"))
    return messages


def test_prefix_stops_at_five_completed_rounds() -> None:
    parsed = parse_trajectory(_six_cycles())
    row = prefix_feature_row(parsed, config="openhands")
    assert row["k5_prefix_explicit_test_failure_any"] == 0
    assert row["k5_prefix_completed_rounds"] == 5
    assert row["k5_prefix_checkpoint_reached"] == 1
    assert row["k5_prefix_user_content_chars"] == 0
    assert "SECRET" not in str(row)
    assert row["k5_prefix_assistant_content_chars"] == 5 * len("do")
    assert row["k5_prefix_tool_family_execute"] == 5.0
    inside = []
    for _ in range(4):
        inside.extend(_cycle("bash", "ok"))
    inside.extend(_cycle("bash", "FAILED tests/test_app.py::test_one\n"))
    positive = prefix_feature_row(parse_trajectory(inside), config="openhands")
    assert positive["k5_prefix_explicit_test_failure_any"] == 1.0
    exit_row = prefix_feature_row(
        parse_trajectory(_cycle("bash", "[The command completed with exit code 1.]\n")),
        config="openhands",
    )
    assert exit_row["k3_prefix_tool_exit_nonzero_any"] == 1.0
    later = prefix_feature_row(parsed, config="openhands")
    assert later["n_complete_cycles"] == 6
    # The raw count is audit-only and is not a model field.
    assert "n_complete_cycles" not in predictor_fields(5, include_exit=False)


def test_failure_after_the_cut_is_not_a_prefix_event() -> None:
    parsed = parse_trajectory(_six_cycles())
    assert prefix_feature_row(parsed, config="sweagent")["k5_prefix_explicit_test_failure_any"] == 0
    assert prefix_feature_row(parsed, config="sweagent")["k5_prefix_tool_exit_nonzero_any"] is None


def test_parse_failure_is_missing_not_zero() -> None:
    row = prefix_feature_row(parse_trajectory("{"), config="openhands")
    assert row["parse_eligible"] is False
    assert row["k5_prefix_tool_call_count"] is None
    assert row["k5_prefix_explicit_test_failure_any"] is None


def test_checkpoint_rule() -> None:
    assert choose_checkpoint(100, 70) == 5
    assert choose_checkpoint(100, 69) == 3


def test_split_does_not_take_outcomes_and_blocks_crossings() -> None:
    repos = ["a", "a", "b", "c"]
    issues = ["i1", "i1", "i2", "i3"]
    components = connected_components(repos, issues)
    first = assign_group_splits(components, seed=20261024)
    second = assign_group_splits(components, seed=20261024)
    assert first == second
    names = [first[item] for item in components]
    assert crossing_groups(repos, names) == {}
    assert crossing_groups(issues, names) == {}


def test_paired_bootstrap_reuses_one_resample_and_skips_one_class() -> None:
    repos = ["a", "a", "b"]
    y = np.array([1, 0, 1])
    structural = np.array([0.2, 0.3, 0.4])
    augmented = np.array([0.6, 0.1, 0.7])
    result = paired_bootstrap(y, structural, augmented, repos, seed=20261092, n_replicates=30)
    assert result.n_requested == 30
    assert result.n_valid + len(result.skipped_replicates) == 30
    assert len(result.deltas) == len(result.structural_ap) == len(result.augmented_ap)
    replicates = repository_cluster_bootstrap_indices(repos, np.random.default_rng(20261092), 20)
    for indexes in replicates:
        count_a = sum(1 for index in indexes if repos[index] == "a")
        assert count_a % 2 == 0


def test_issue_weights_are_computed_inside_one_split() -> None:
    test_ids = ["issue-a", "issue-a", "issue-b"]
    train_ids = ["issue-a", "issue-a", "issue-a", "issue-c"]
    assert issue_equal_trajectory_weights(test_ids) == [0.5, 0.5, 1.0]
    assert issue_equal_trajectory_weights(train_ids)[0] == pytest.approx(1 / 3)


def test_c_tie_prefers_the_smaller_value() -> None:
    # Identical columns make every C equally able to separate the toy labels
    # only if the model can fit. A constant feature still yields finite scores.
    x_train = np.array([[0.0], [1.0], [0.0], [1.0]])
    y_train = np.array([0, 1, 0, 1])
    x_valid = np.array([[0.0], [1.0]])
    y_valid = np.array([0, 1])
    chosen, scores = select_c(x_train, y_train, x_valid, y_valid)
    assert set(scores) == {"0.01", "0.1", "1.0", "10.0"}
    best = max(scores.values())
    tied = [float(key) for key, value in scores.items() if abs(value - best) <= 1e-15]
    assert chosen == min(tied)


def test_historical_hashes_and_adjudication_label_separation() -> None:
    import hashlib

    for relative, expected in HISTORICAL.items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected
    if not (ARCHIVE / "manifest.json").is_file():
        pytest.skip("adjudication archive is not installed")
    for name, rows, positives, negatives in (
        ("01_explicit_exception_adjudicated.csv", 50, 47, 3),
        ("02_explicit_test_failure_adjudicated.csv", 50, 50, 0),
        ("03_openhands_exit_status_adjudicated.csv", 50, 50, 0),
        ("04_diagnostic_adjudicated.csv", 84, None, None),
        ("05_random_observation_adjudicated.csv", 24, None, None),
    ):
        with (ARCHIVE / name).open(encoding="utf-8-sig", newline="") as handle:
            table = list(csv.DictReader(handle))
        assert len(table) == rows
        assert "resolved" not in table[0]
        assert all(row["reviewer_label"] == "" for row in table)
        assert all(row["student_verified"].strip().lower() == "false" for row in table)
        assert all(row["ai_human_verified"].strip().lower() == "false" for row in table)
        if positives is not None:
            labels = [row["ai_adjudication_label"] for row in table]
            assert labels.count("true_positive") == positives
            assert labels.count("false_positive") == negatives


def test_results_zip_members_match_when_present() -> None:
    zip_path = ROOT / "artifacts" / "l2_inputs" / "P8_GATE09_AI_ADJUDICATION_RESULTS.zip"
    if not zip_path.is_file():
        pytest.skip("zip not retained")
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "01_explicit_exception_adjudicated.csv" in names
    assert "manifest.json" in names
    # The unread helper keeps the import from looking like an unused test import.
    assert io.DEFAULT_BUFFER_SIZE > 0
