"""V3.4.1 common-support rules. These tests do not fit a model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from v3_4_1_support import (
    PUBLISHED_V34,
    STATUS_BLOCKED,
    STATUS_COMPLETE,
    STATUS_PARTIAL,
    characterize,
    decide_status,
    nesting_passes,
    source_problems,
    v34_variation_sets,
)
from v3_4_lib import issue_support

ROOT = Path(__file__).resolve().parents[2]
CENSUS = Path(r"C:\Users\hanqi\Desktop\Projects\Research_Portfolio_2026\_p8-v3_4-census-staging\runs\v3_4_census_20261102T153940Z\census.parquet")
SUMMARY = Path(r"C:\Users\hanqi\Desktop\Projects\Research_Portfolio_2026\_p8-v3_4-census-staging\runs\v3_4_census_20261102T153940Z\issue_support_summary.csv")


def _row(**overrides) -> dict:
    base = {
        "trajectory_id": "t",
        "issue_id": "issue",
        "instance_id": "issue",
        "repo": "repo",
        "scaffold": "openhands",
        "model": "minimax_m25",
        "resolved": 0,
        "split": "train",
        "n_rounds_total": 20,
        "n_rounds_observed": 20,
        "parser_v2_strict_fail_r5": 0,
        "census_parse_ok": True,
        "parser_status": "ok",
        "prefix5_status": "match",
        "prefix15_status": "match",
        "shard": "data/x",
        "v3_ran_tests_r5": 0,
        "v3_ran_tests_r15": 0,
        "exec_missing_r5": 0,
        "exec_missing_r15": 0,
        "exec_calls_r5": 1,
        "exec_calls_r15": 1,
        "v3_n_test_runs_r5": 0,
        "v3_n_test_runs_r15": 0,
        "v3_test_fail_r5": 0,
        "v3_test_fail_r15": 0,
    }
    base.update(overrides)
    return base


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_issue_level_variation_and_single_trajectory() -> None:
    frame = _frame(
        [
            _row(trajectory_id="a1", issue_id="varied", v3_ran_tests_r5=1, v3_ran_tests_r15=1, v3_n_test_runs_r5=1, v3_n_test_runs_r15=1, resolved=0),
            _row(trajectory_id="a2", issue_id="varied", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=0),
            _row(trajectory_id="b1", issue_id="alone", resolved=1),
        ]
    )
    result = characterize(frame)
    issues = result["issues"].set_index("issue_id")
    assert int(issues.loc["varied", "test_behavior_variation_r5"]) == 1
    assert int(issues.loc["varied", "outcome_variation"]) == 0
    assert int(issues.loc["alone", "test_behavior_variation_r5"]) == 0
    assert int(issues.loc["alone", "n_trajectories"]) == 1
    assert result["headline"]["behavior_only_r5"] == 1
    assert result["headline"]["neither_r5"] == 1


def test_unknown_label_is_excluded_from_outcome_and_four_cell() -> None:
    frame = _frame(
        [
            _row(trajectory_id="u1", issue_id="labels", v3_ran_tests_r5=1, v3_ran_tests_r15=1, resolved=-1, v3_test_fail_r5=1),
            _row(trajectory_id="u2", issue_id="labels", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=0),
        ]
    )
    result = characterize(frame)
    issue = result["issues"].iloc[0]
    assert int(issue["n_unknown_labels"]) == 1
    assert int(issue["n_known_labels"]) == 1
    assert int(issue["outcome_variation"]) == 0
    assert int(issue["test_behavior_variation_r5"]) == 1
    cell = result["four_cell"].loc[result["four_cell"]["checkpoint"] == 5].iloc[0]
    assert int(cell["n_exposed_failure"]) == 0
    assert int(cell["n_exposed_success"]) == 0
    assert int(cell["n_unexposed_failure"]) == 1
    assert int(cell["n_unexposed_success"]) == 0
    pair = result["discordant"].loc[result["discordant"]["checkpoint"] == 5].iloc[0]
    assert int(pair["total_discordant_pairs"]) == 0


def test_missing_command_is_not_a_known_negative() -> None:
    frame = _frame(
        [
            _row(trajectory_id="m1", issue_id="missing", v3_ran_tests_r5=1, v3_ran_tests_r15=1, exec_missing_r5=1, resolved=0, v3_test_fail_r5=1),
            _row(trajectory_id="m2", issue_id="missing", v3_ran_tests_r5=0, v3_ran_tests_r15=0, exec_missing_r5=2, exec_missing_r15=2, resolved=1),
        ]
    )
    result = characterize(frame)
    issue = result["issues"].iloc[0]
    assert int(issue["n_test_run_r5"]) == 1
    assert int(issue["n_known_nonrun_r5"]) == 0
    assert int(issue["n_missing_command_r5"]) == 1
    assert int(issue["test_behavior_variation_r5"]) == 0
    assert int(issue["v34_loose_flag_variation_r5"]) == 1
    cell = result["four_cell"].loc[result["four_cell"]["checkpoint"] == 5].iloc[0]
    assert int(cell["n_exposed_failure"]) == 1
    assert int(cell["n_unexposed_success"]) == 0
    sets = v34_variation_sets(frame)
    stats = issue_support(frame)
    assert "missing" in sets["loose_issues"]
    assert "missing" not in sets["strict_issues"]
    assert stats["issues_with_r5_test_flag_variation"] == 1
    assert stats["issues_with_strict_run_versus_known_nonrun"] == 0


def test_four_cell_and_discordant_products() -> None:
    frame = _frame(
        [
            _row(trajectory_id="p1", issue_id="pairs", v3_ran_tests_r5=1, v3_ran_tests_r15=1, v3_n_test_runs_r5=1, v3_n_test_runs_r15=1, resolved=0),
            _row(trajectory_id="p2", issue_id="pairs", v3_ran_tests_r5=1, v3_ran_tests_r15=1, v3_n_test_runs_r5=1, v3_n_test_runs_r15=1, resolved=0),
            _row(trajectory_id="p3", issue_id="pairs", v3_ran_tests_r5=1, v3_ran_tests_r15=1, resolved=1),
            _row(trajectory_id="p4", issue_id="pairs", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=0),
            _row(trajectory_id="p5", issue_id="pairs", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=1),
            _row(trajectory_id="p6", issue_id="pairs", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=1),
            _row(trajectory_id="p7", issue_id="pairs", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=1),
        ]
    )
    result = characterize(frame)
    cell = result["four_cell"].loc[result["four_cell"]["checkpoint"] == 5].iloc[0]
    assert (int(cell["n_exposed_failure"]), int(cell["n_exposed_success"]), int(cell["n_unexposed_failure"]), int(cell["n_unexposed_success"])) == (2, 1, 1, 3)
    pair = result["discordant"].loc[result["discordant"]["checkpoint"] == 5].iloc[0]
    assert int(pair["positive_discordant_pairs"]) == 6
    assert int(pair["negative_discordant_pairs"]) == 1
    assert int(pair["total_discordant_pairs"]) == 7
    assert pair["direction"] == "both"
    assert result["pairs_r5"]["cells_both"] == 1
    assert result["pairs_r5"]["positive_pair_products"] == 6
    assert result["pairs_r5"]["negative_pair_products"] == 1


def test_same_issue_across_groups_is_not_within_group_support() -> None:
    frame = _frame(
        [
            _row(trajectory_id="g1", issue_id="cross", scaffold="openhands", model="minimax_m25", v3_ran_tests_r5=1, v3_ran_tests_r15=1, resolved=0),
            _row(trajectory_id="g2", issue_id="cross", scaffold="sweagent", model="qwen35_122b", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=1),
            _row(trajectory_id="g3", issue_id="inside", scaffold="sweagent", model="minimax_m25", v3_ran_tests_r5=1, v3_ran_tests_r15=1, resolved=0),
            _row(trajectory_id="g4", issue_id="inside", scaffold="sweagent", model="minimax_m25", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=1),
            _row(trajectory_id="g5", issue_id="inside", scaffold="openhands", model="qwen35_122b", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=1),
        ]
    )
    result = characterize(frame)
    cells = result["cells"].set_index(["issue_id", "scaffold", "model"])
    assert int(cells.loc[("cross", "openhands", "minimax_m25"), "joint_variation_r5"]) == 0
    assert int(cells.loc[("cross", "sweagent", "qwen35_122b"), "joint_variation_r5"]) == 0
    assert int(result["issues"].set_index("issue_id").loc["cross", "joint_variation_r5"]) == 1
    assert int(cells.loc[("inside", "sweagent", "minimax_m25"), "joint_variation_r5"]) == 1
    assert result["cross"]["both_with_within_group_joint_issues"] == 1
    assert result["cross"]["both_with_within_group_joint_cells"] == 1
    assert result["headline"]["cells"] == 4
    pair = result["discordant"]
    inside = pair.loc[
        (pair["issue_id"] == "inside")
        & (pair["scaffold"] == "sweagent")
        & (pair["model"] == "minimax_m25")
        & (pair["checkpoint"] == 5)
    ].iloc[0]
    assert int(inside["positive_discordant_pairs"]) == 1
    assert int(inside["negative_discordant_pairs"]) == 0
    assert inside["direction"] == "positive_only"


def test_multiple_trajectories_and_positive_only_direction() -> None:
    frame = _frame(
        [
            _row(trajectory_id="n1", issue_id="only-neg", v3_ran_tests_r5=1, v3_ran_tests_r15=1, resolved=1),
            _row(trajectory_id="n2", issue_id="only-neg", v3_ran_tests_r5=0, v3_ran_tests_r15=0, resolved=0),
        ]
    )
    result = characterize(frame)
    pair = result["discordant"].loc[result["discordant"]["checkpoint"] == 5].iloc[0]
    assert int(pair["positive_discordant_pairs"]) == 0
    assert int(pair["negative_discordant_pairs"]) == 1
    assert pair["direction"] == "negative_only"
    assert result["pairs_r5"]["issues_any"] == 1
    assert result["pairs_r5"]["cells_any"] == 1


def test_short_prefix_keeps_available_rounds_and_nested_window() -> None:
    frame = _frame(
        [
            _row(
                trajectory_id="short",
                issue_id="short-issue",
                n_rounds_total=4,
                n_rounds_observed=4,
                prefix15_status="no_prefix_boundary",
                v3_ran_tests_r5=1,
                v3_ran_tests_r15=1,
                v3_n_test_runs_r5=1,
                v3_n_test_runs_r15=1,
                exec_calls_r5=2,
                exec_calls_r15=2,
            ),
            _row(
                trajectory_id="later",
                issue_id="later-issue",
                n_rounds_total=8,
                n_rounds_observed=8,
                prefix15_status="no_prefix_boundary",
                v3_ran_tests_r5=0,
                v3_ran_tests_r15=1,
                v3_n_test_runs_r5=0,
                v3_n_test_runs_r15=1,
                exec_calls_r5=1,
                exec_calls_r15=3,
            ),
        ]
    )
    result = characterize(frame)
    assert set(result["short_rows"]["trajectory_id"]) == {"short", "later"}
    assert result["nesting"]["n_short_r15_missing"] == 0
    assert result["nesting"]["n_ran_not_nested"] == 0
    assert result["nesting"]["n_short_rounds_le_5_window_differs"] == 0
    assert result["nesting"]["n_short_r15_added_after_r5"] == 1
    broken = frame.copy()
    broken.loc[broken["trajectory_id"] == "short", "v3_ran_tests_r15"] = 0
    broken_result = characterize(broken)
    assert broken_result["nesting"]["n_ran_not_nested"] == 1
    assert broken_result["nesting"]["n_short_r5_positive_r15_negative"] == 1


def test_variation_sets_match_v34_issue_support() -> None:
    frame = _frame(
        [
            _row(trajectory_id="s1", issue_id="both", v3_ran_tests_r5=1, resolved=0),
            _row(trajectory_id="s2", issue_id="both", v3_ran_tests_r5=0, resolved=1),
            _row(trajectory_id="s3", issue_id="cross", scaffold="openhands", model="minimax_m25", v3_ran_tests_r5=1, resolved=0),
            _row(trajectory_id="s4", issue_id="cross", scaffold="sweagent", model="qwen35_122b", v3_ran_tests_r5=0, resolved=1),
            _row(trajectory_id="s5", issue_id="loose-only", v3_ran_tests_r5=1, resolved=0),
            _row(trajectory_id="s6", issue_id="loose-only", v3_ran_tests_r5=0, exec_missing_r5=1, resolved=1),
        ]
    )
    sets = v34_variation_sets(frame)
    stats = issue_support(frame)
    assert len(sets["loose_issues"]) == stats["issues_with_r5_test_flag_variation"] == 3
    assert len(sets["strict_issues"]) == stats["issues_with_strict_run_versus_known_nonrun"] == 2
    assert len(sets["outcome_issues"]) == stats["issues_with_outcome_variation"] == 3
    assert len(sets["both_issues"]) == stats["issues_behavior_and_outcome"] == 3
    assert len(sets["strict_both_issues"]) == stats["issues_strict_behavior_and_outcome"] == 2
    assert len(sets["within_loose_issues"]) == stats["issues_with_within_group_test_variation"] == 2
    result = characterize(frame)
    assert result["alignment"]["internal_match"] is True
    assert result["alignment"]["published_match"] is False


def test_behavior_definitions_keep_test_failure_separate() -> None:
    frame = _frame(
        [
            _row(trajectory_id="f1", issue_id="fail-only", v3_ran_tests_r5=1, v3_ran_tests_r15=1, v3_test_fail_r5=1, v3_test_fail_r15=1, resolved=0),
            _row(trajectory_id="f2", issue_id="fail-only", v3_ran_tests_r5=1, v3_ran_tests_r15=1, v3_test_fail_r5=0, v3_test_fail_r15=0, resolved=1),
            _row(trajectory_id="f3", issue_id="parser", parser_v2_strict_fail_r5=1, resolved=0),
            _row(trajectory_id="f4", issue_id="parser", parser_v2_strict_fail_r5=0, resolved=1),
            _row(trajectory_id="f5", issue_id="parser-missing", parser_v2_strict_fail_r5=None, resolved=0),
        ]
    )
    result = characterize(frame)
    behavior = result["by_behavior"].set_index("variable")
    assert int(behavior.loc["v3_ran_tests_r5", "n_issues_with_variation"]) == 0
    assert int(behavior.loc["v3_test_fail_r5", "n_issues_with_variation"]) == 1
    assert int(behavior.loc["v3_test_fail_r5", "n_positive_trajectories"]) == 1
    assert int(behavior.loc["v3_test_fail_r5", "n_valid_negative_trajectories"]) == 4
    assert int(behavior.loc["v3_test_fail_r5", "n_issues_with_same_cell_joint_variation"]) == 1
    assert int(behavior.loc["parser_v2_strict_fail_r5", "n_missing_trajectories"]) == 1
    assert int(behavior.loc["parser_v2_strict_fail_r5", "n_issues_with_variation"]) == 1
    assert int(behavior.loc["parser_v2_strict_fail_r5", "n_cells_with_joint_variation"]) == 1


def test_parse_failure_stays_unusable() -> None:
    frame = _frame(
        [
            _row(trajectory_id="e1", issue_id="error", census_parse_ok=False, v3_ran_tests_r5=None, v3_ran_tests_r15=None, exec_missing_r5=None, exec_missing_r15=None, v3_test_fail_r5=None, v3_test_fail_r15=None, resolved=0),
            _row(trajectory_id="e2", issue_id="error", v3_ran_tests_r5=1, v3_ran_tests_r15=1, resolved=1),
        ]
    )
    result = characterize(frame)
    issue = result["issues"].iloc[0]
    assert int(issue["n_test_run_r5"]) == 1
    assert int(issue["n_known_nonrun_r5"]) == 0
    assert int(issue["test_behavior_variation_r5"]) == 0
    assert int(result["by_behavior"].set_index("variable").loc["v3_ran_tests_r5", "n_missing_trajectories"]) == 1


def test_status_gates_and_runner_has_no_model() -> None:
    assert decide_status(source_ok=False, counts_complete=True, alignment_ok=True, nesting_ok=True, tests_ok=True) == STATUS_BLOCKED
    assert decide_status(source_ok=True, counts_complete=True, alignment_ok=False, nesting_ok=True, tests_ok=True) == STATUS_PARTIAL
    assert decide_status(source_ok=True, counts_complete=True, alignment_ok=True, nesting_ok=False, tests_ok=True) == STATUS_PARTIAL
    assert decide_status(source_ok=True, counts_complete=True, alignment_ok=True, nesting_ok=True, tests_ok=False) == STATUS_PARTIAL
    assert decide_status(source_ok=True, counts_complete=True, alignment_ok=True, nesting_ok=True, tests_ok=True) == STATUS_COMPLETE
    text = (ROOT / "analysis_v2" / "v3_4_1_support.py").read_text(encoding="utf-8")
    text += (ROOT / "analysis_v2" / "v3_4_1_run.py").read_text(encoding="utf-8")
    for token in ("LogisticRegression", "mixedlm", "average_precision", "roc_auc", "fillna(0)", "00_reproduce"):
        assert token not in text


def test_saved_v34_summary_targets_remain_published() -> None:
    saved = pd.read_csv(SUMMARY).set_index("metric")["value"]
    for key, expected in PUBLISHED_V34.items():
        assert int(saved.loc[key]) == expected


def test_full_census_reproduces_v34_counts() -> None:
    columns = [
        "trajectory_id",
        "issue_id",
        "scaffold",
        "model",
        "resolved",
        "census_parse_ok",
        "v3_ran_tests_r5",
        "v3_ran_tests_r15",
        "exec_missing_r5",
        "exec_missing_r15",
        "exec_calls_r5",
        "exec_calls_r15",
        "v3_n_test_runs_r5",
        "v3_n_test_runs_r15",
        "v3_test_fail_r5",
        "v3_test_fail_r15",
        "parser_v2_strict_fail_r5",
        "prefix5_status",
        "prefix15_status",
        "n_rounds_observed",
        "n_rounds_total",
    ]
    frame = pd.read_parquet(CENSUS, columns=columns)
    stats = issue_support(frame)
    for key, expected in PUBLISHED_V34.items():
        assert stats[key] == expected
    result = characterize(frame)
    assert result["alignment"]["internal_match"] is True
    assert result["alignment"]["published_match"] is True
    assert result["nesting"]["n_no_prefix_boundary"] == 50
    assert result["nesting"]["n_ran_not_nested"] == 0
    assert result["nesting"]["n_short_r15_missing"] == 0
    assert result["cross"]["joint_issues_outside_published_both"] == 0
    assert nesting_passes(result["nesting"]) is True
    assert result["headline"]["issues"] == 22320
    assert result["headline"]["trajectories"] == 207489


def test_source_problem_detects_a_bad_join() -> None:
    left = _frame([_row(trajectory_id="t1", issue_id="i", instance_id="i")])
    right = left.copy()
    right["resolved"] = -1
    problems = source_problems(left, right)
    assert any("resolved" in item for item in problems)
