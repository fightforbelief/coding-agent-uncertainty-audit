"""V3.4.1 common-support counts from a saved census frame.

This module does not read trajectories, fit a model, or change a detector.
Missing commands stay out of the known-negative class. ``resolved == -1``
stays out of outcome variation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from v3_4_lib import is_true_flag, issue_support

STATUS_COMPLETE = "V3.4.1 COMPLETE — COMMON SUPPORT CHARACTERIZED"
STATUS_PARTIAL = "V3.4.1 PARTIAL — SUPPORT COUNTS INCOMPLETE"
STATUS_BLOCKED = "V3.4.1 BLOCKED — SOURCE OR JOIN FAILURE"

PUBLISHED_V34 = {
    "issues_with_r5_test_flag_variation": 856,
    "issues_with_strict_run_versus_known_nonrun": 856,
    "issues_with_outcome_variation": 7029,
    "issues_behavior_only": 526,
    "issues_outcome_only": 6699,
    "issues_behavior_and_outcome": 330,
    "issues_strict_behavior_and_outcome": 330,
    "issues_with_within_group_test_variation": 734,
}
PUBLISHED_BUCKETS = {
    5: {"run": 1304, "known_nonrun": 206174, "missing_command": 11, "unusable": 0},
    15: {"run": 44480, "known_nonrun": 162514, "missing_command": 495, "unusable": 0},
}
GROUP_ORDER = (
    ("openhands", "minimax_m25", "OpenHands / Minimax"),
    ("openhands", "qwen35_122b", "OpenHands / Qwen"),
    ("sweagent", "minimax_m25", "SWE-agent / Minimax"),
    ("sweagent", "qwen35_122b", "SWE-agent / Qwen"),
)
BEHAVIOR_VARIABLES = (
    "v3_ran_tests_r5",
    "v3_ran_tests_r15",
    "v3_test_fail_r5",
    "v3_test_fail_r15",
    "parser_v2_strict_fail_r5",
)


def nesting_passes(nesting: dict[str, Any]) -> bool:
    """The saved r15 prefix is usable when these window checks all hold."""
    expected = {
        "n_no_prefix_boundary": 50,
        "n_prefix15_mismatch": 0,
        "n_other_prefix15_status": 0,
        "n_rounds_below_15": 50,
        "n_short_with_rounds_ge_15": 0,
        "n_below_15_without_boundary_flag": 0,
        "n_round_source_mismatch": 0,
        "n_ran_not_nested": 0,
        "n_test_calls_not_nested": 0,
        "n_exec_calls_not_nested": 0,
        "n_short_r15_missing": 0,
        "n_short_r5_positive_r15_negative": 0,
        "n_short_rounds_le_5_window_differs": 0,
    }
    return all(nesting.get(key) == value for key, value in expected.items())


def decide_status(
    *,
    source_ok: bool,
    counts_complete: bool,
    alignment_ok: bool,
    nesting_ok: bool,
    tests_ok: bool,
) -> str:
    if not source_ok:
        return STATUS_BLOCKED
    if counts_complete and alignment_ok and nesting_ok and tests_ok:
        return STATUS_COMPLETE
    return STATUS_PARTIAL


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _parse_ok(series: pd.Series) -> np.ndarray:
    return series.map(is_true_flag).to_numpy(dtype=bool)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Add exposure and outcome indicators. Stored measurements are not filled."""
    out = frame.copy()
    resolved = _numeric(out["resolved"])
    out["is_success"] = resolved == 1
    out["is_failure"] = resolved == 0
    out["is_unknown_label"] = resolved == -1
    out["is_known_label"] = out["is_success"] | out["is_failure"]
    out["is_invalid_label"] = ~(out["is_known_label"] | out["is_unknown_label"])
    ok = _parse_ok(out["census_parse_ok"])
    for checkpoint in (5, 15):
        ran = _numeric(out[f"v3_ran_tests_r{checkpoint}"])
        missing = _numeric(out[f"exec_missing_r{checkpoint}"])
        run = ok & (ran == 1)
        known = ok & (ran == 0) & (missing == 0)
        absent = ok & (ran == 0) & (missing > 0)
        finite = np.isfinite(ran) & np.isfinite(missing)
        out[f"run_r{checkpoint}"] = run
        out[f"known_nonrun_r{checkpoint}"] = known
        out[f"missing_command_r{checkpoint}"] = absent
        out[f"unusable_r{checkpoint}"] = ~(run | known | absent)
        out[f"loose_one_r{checkpoint}"] = ok & finite & (ran == 1)
        out[f"loose_zero_r{checkpoint}"] = ok & finite & (ran == 0)
        fail = _numeric(out[f"v3_test_fail_r{checkpoint}"])
        fail = np.where(ok, fail, np.nan)
        out[f"fail_pos_r{checkpoint}"] = fail == 1
        out[f"fail_neg_r{checkpoint}"] = fail == 0
        out[f"fail_missing_r{checkpoint}"] = ~((fail == 1) | (fail == 0))
    parser = _numeric(out["parser_v2_strict_fail_r5"])
    out["parser_pos"] = parser == 1
    out["parser_neg"] = parser == 0
    out["parser_missing"] = ~((parser == 1) | (parser == 0))
    rounds = _numeric(out["n_rounds_observed"])
    out["rounds_observed_num"] = rounds
    return out


def v34_variation_sets(frame: pd.DataFrame) -> dict[str, set[Any]]:
    """Same r5 sets as ``issue_support``. Missing-command zeros stay in the loose set."""
    valid = frame.loc[frame["census_parse_ok"].map(is_true_flag)].copy()
    valid = valid.loc[valid["v3_ran_tests_r5"].notna() & valid["exec_missing_r5"].notna()]
    behavior = valid.groupby("issue_id", dropna=False)["v3_ran_tests_r5"].agg(["min", "max"])
    loose = set(behavior.index[(behavior["min"] == 0) & (behavior["max"] == 1)])
    strict_rows = valid.loc[(valid["v3_ran_tests_r5"] == 1) | ((valid["v3_ran_tests_r5"] == 0) & (valid["exec_missing_r5"] == 0))]
    strict_span = strict_rows.groupby("issue_id", dropna=False)["v3_ran_tests_r5"].agg(["min", "max"])
    strict = set(strict_span.index[(strict_span["min"] == 0) & (strict_span["max"] == 1)])
    resolved = pd.to_numeric(frame["resolved"], errors="coerce")
    known = frame.loc[resolved.isin([0, 1])]
    outcomes = known.groupby("issue_id", dropna=False)["resolved"].agg(["min", "max"])
    outcome = set(outcomes.index[(outcomes["min"] == 0) & (outcomes["max"] == 1)])
    grouped = valid.groupby(["issue_id", "scaffold", "model"], dropna=False)["v3_ran_tests_r5"].agg(["min", "max"])
    within_rows = grouped[(grouped["min"] == 0) & (grouped["max"] == 1)].reset_index()
    within_issues = set(within_rows["issue_id"]) if len(within_rows) else set()
    within_cells = set(zip(within_rows["issue_id"], within_rows["scaffold"], within_rows["model"], strict=True)) if len(within_rows) else set()
    return {
        "loose_issues": loose,
        "strict_issues": strict,
        "outcome_issues": outcome,
        "both_issues": loose & outcome,
        "strict_both_issues": strict & outcome,
        "within_loose_issues": within_issues,
        "within_loose_cells": within_cells,
    }


def aggregate(prepared: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    work = prepared.copy()
    named = {
        "n_trajectories": ("trajectory_id", "size"),
        "n_known_labels": ("is_known_label", "sum"),
        "n_unknown_labels": ("is_unknown_label", "sum"),
        "n_invalid_labels": ("is_invalid_label", "sum"),
        "n_success": ("is_success", "sum"),
        "n_failure": ("is_failure", "sum"),
    }
    for checkpoint in (5, 15):
        named[f"n_test_run_r{checkpoint}"] = (f"run_r{checkpoint}", "sum")
        named[f"n_known_nonrun_r{checkpoint}"] = (f"known_nonrun_r{checkpoint}", "sum")
        named[f"n_missing_command_r{checkpoint}"] = (f"missing_command_r{checkpoint}", "sum")
        named[f"n_unusable_r{checkpoint}"] = (f"unusable_r{checkpoint}", "sum")
        named[f"n_loose_zero_r{checkpoint}"] = (f"loose_zero_r{checkpoint}", "sum")
        named[f"n_loose_one_r{checkpoint}"] = (f"loose_one_r{checkpoint}", "sum")
        named[f"n_fail_pos_r{checkpoint}"] = (f"fail_pos_r{checkpoint}", "sum")
        named[f"n_fail_neg_r{checkpoint}"] = (f"fail_neg_r{checkpoint}", "sum")
        named[f"a_r{checkpoint}"] = (f"a_r{checkpoint}", "sum")
        named[f"b_r{checkpoint}"] = (f"b_r{checkpoint}", "sum")
        named[f"c_r{checkpoint}"] = (f"c_r{checkpoint}", "sum")
        named[f"d_r{checkpoint}"] = (f"d_r{checkpoint}", "sum")
    for checkpoint in (5, 15):
        work[f"a_r{checkpoint}"] = work[f"run_r{checkpoint}"] & work["is_failure"]
        work[f"b_r{checkpoint}"] = work[f"run_r{checkpoint}"] & work["is_success"]
        work[f"c_r{checkpoint}"] = work[f"known_nonrun_r{checkpoint}"] & work["is_failure"]
        work[f"d_r{checkpoint}"] = work[f"known_nonrun_r{checkpoint}"] & work["is_success"]
    named["n_parser_pos"] = ("parser_pos", "sum")
    named["n_parser_neg"] = ("parser_neg", "sum")
    named["n_parser_missing"] = ("parser_missing", "sum")
    grouped = work.groupby(keys, dropna=False).agg(**named).reset_index()
    for column in grouped.columns:
        if column in keys or not pd.api.types.is_numeric_dtype(grouped[column]):
            continue
        grouped[column] = grouped[column].astype(int)
    grouped["outcome_variation"] = ((grouped["n_success"] > 0) & (grouped["n_failure"] > 0)).astype(int)
    for checkpoint in (5, 15):
        behavior = (grouped[f"n_test_run_r{checkpoint}"] > 0) & (grouped[f"n_known_nonrun_r{checkpoint}"] > 0)
        loose = (grouped[f"n_loose_zero_r{checkpoint}"] > 0) & (grouped[f"n_loose_one_r{checkpoint}"] > 0)
        grouped[f"test_behavior_variation_r{checkpoint}"] = behavior.astype(int)
        grouped[f"v34_loose_flag_variation_r{checkpoint}"] = loose.astype(int)
        grouped[f"joint_variation_r{checkpoint}"] = (behavior & grouped["outcome_variation"].eq(1)).astype(int)
        grouped[f"fail_variation_r{checkpoint}"] = (
            (grouped[f"n_fail_pos_r{checkpoint}"] > 0) & (grouped[f"n_fail_neg_r{checkpoint}"] > 0)
        ).astype(int)
        grouped[f"fail_joint_r{checkpoint}"] = (
            grouped[f"fail_variation_r{checkpoint}"].eq(1) & grouped["outcome_variation"].eq(1)
        ).astype(int)
    grouped["parser_variation"] = ((grouped["n_parser_pos"] > 0) & (grouped["n_parser_neg"] > 0)).astype(int)
    grouped["parser_joint"] = (grouped["parser_variation"].eq(1) & grouped["outcome_variation"].eq(1)).astype(int)
    return grouped


def _direction(positive: int, negative: int) -> str:
    if positive > 0 and negative > 0:
        return "both"
    if positive > 0:
        return "positive_only"
    if negative > 0:
        return "negative_only"
    return "none"


def four_cell_tables(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_rows: list[pd.DataFrame] = []
    pair_rows: list[pd.DataFrame] = []
    for checkpoint in (5, 15):
        part = cells[["issue_id", "scaffold", "model"]].copy()
        part["checkpoint"] = checkpoint
        part["n_exposed_failure"] = cells[f"a_r{checkpoint}"].astype(int)
        part["n_exposed_success"] = cells[f"b_r{checkpoint}"].astype(int)
        part["n_unexposed_failure"] = cells[f"c_r{checkpoint}"].astype(int)
        part["n_unexposed_success"] = cells[f"d_r{checkpoint}"].astype(int)
        positive = part["n_exposed_failure"] * part["n_unexposed_success"]
        negative = part["n_exposed_success"] * part["n_unexposed_failure"]
        pairs = part.copy()
        pairs["positive_discordant_pairs"] = positive.astype(int)
        pairs["negative_discordant_pairs"] = negative.astype(int)
        pairs["total_discordant_pairs"] = (positive + negative).astype(int)
        pairs["direction"] = [
            _direction(int(left), int(right))
            for left, right in zip(pairs["positive_discordant_pairs"], pairs["negative_discordant_pairs"], strict=True)
        ]
        count_rows.append(part)
        pair_rows.append(pairs)
    counts = pd.concat(count_rows, ignore_index=True)
    discordant = pd.concat(pair_rows, ignore_index=True)
    sort_keys = ["checkpoint", "issue_id", "scaffold", "model"]
    return counts.sort_values(sort_keys).reset_index(drop=True), discordant.sort_values(sort_keys).reset_index(drop=True)


def _pair_summary(discordant: pd.DataFrame, checkpoint: int) -> dict[str, Any]:
    part = discordant.loc[discordant["checkpoint"] == checkpoint]
    any_rows = part.loc[part["total_discordant_pairs"] > 0]
    summary = {
        "cells_any": int(len(any_rows)),
        "issues_any": int(any_rows["issue_id"].nunique()) if len(any_rows) else 0,
        "positive_pair_products": int(part["positive_discordant_pairs"].sum()),
        "negative_pair_products": int(part["negative_discordant_pairs"].sum()),
    }
    for name in ("positive_only", "negative_only", "both", "none"):
        chosen = part.loc[part["direction"] == name]
        summary[f"cells_{name}"] = int(len(chosen))
        summary[f"issues_{name}"] = int(chosen["issue_id"].nunique()) if len(chosen) else 0
    by_group = []
    for scaffold, model, label in GROUP_ORDER:
        group = part.loc[(part["scaffold"] == scaffold) & (part["model"] == model)]
        informative = group.loc[group["total_discordant_pairs"] > 0]
        by_group.append(
            {
                "scaffold": scaffold,
                "model": model,
                "label": label,
                "cells_any": int(len(informative)),
                "issues_any": int(informative["issue_id"].nunique()) if len(informative) else 0,
                "cells_positive_only": int((group["direction"] == "positive_only").sum()),
                "cells_negative_only": int((group["direction"] == "negative_only").sum()),
                "cells_both": int((group["direction"] == "both").sum()),
                "positive_pair_products": int(group["positive_discordant_pairs"].sum()),
                "negative_pair_products": int(group["negative_discordant_pairs"].sum()),
            }
        )
    summary["by_group"] = by_group
    return summary


def _behavior_row(
    name: str,
    *,
    positive: int,
    negative: int,
    missing: int,
    issues: int,
    cells: int,
    joint_issues: int,
    joint_cells: int,
    same_cell_issues: int,
) -> dict[str, Any]:
    return {
        "variable": name,
        "n_positive_trajectories": positive,
        "n_valid_negative_trajectories": negative,
        "n_missing_trajectories": missing,
        "n_issues_with_variation": issues,
        "n_cells_with_variation": cells,
        "n_issues_with_variation_and_outcome_variation": joint_issues,
        "n_issues_with_same_cell_joint_variation": same_cell_issues,
        "n_cells_with_joint_variation": joint_cells,
    }


def _same_cell_issues(cells: pd.DataFrame, column: str) -> int:
    return int(cells.loc[cells[column] == 1, "issue_id"].nunique()) if len(cells) else 0


def behavior_table(prepared: pd.DataFrame, issues: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for checkpoint in (5, 15):
        rows.append(
            _behavior_row(
                f"v3_ran_tests_r{checkpoint}",
                positive=int(prepared[f"run_r{checkpoint}"].sum()),
                negative=int(prepared[f"known_nonrun_r{checkpoint}"].sum()),
                missing=int(prepared[f"missing_command_r{checkpoint}"].sum() + prepared[f"unusable_r{checkpoint}"].sum()),
                issues=int(issues[f"test_behavior_variation_r{checkpoint}"].sum()),
                cells=int(cells[f"test_behavior_variation_r{checkpoint}"].sum()),
                joint_issues=int(issues[f"joint_variation_r{checkpoint}"].sum()),
                joint_cells=int(cells[f"joint_variation_r{checkpoint}"].sum()),
                same_cell_issues=_same_cell_issues(cells, f"joint_variation_r{checkpoint}"),
            )
        )
        rows.append(
            _behavior_row(
                f"v3_test_fail_r{checkpoint}",
                positive=int(prepared[f"fail_pos_r{checkpoint}"].sum()),
                negative=int(prepared[f"fail_neg_r{checkpoint}"].sum()),
                missing=int(prepared[f"fail_missing_r{checkpoint}"].sum()),
                issues=int(issues[f"fail_variation_r{checkpoint}"].sum()),
                cells=int(cells[f"fail_variation_r{checkpoint}"].sum()),
                joint_issues=int(issues[f"fail_joint_r{checkpoint}"].sum()),
                joint_cells=int(cells[f"fail_joint_r{checkpoint}"].sum()),
                same_cell_issues=_same_cell_issues(cells, f"fail_joint_r{checkpoint}"),
            )
        )
    rows.append(
        _behavior_row(
            "parser_v2_strict_fail_r5",
            positive=int(prepared["parser_pos"].sum()),
            negative=int(prepared["parser_neg"].sum()),
            missing=int(prepared["parser_missing"].sum()),
            issues=int(issues["parser_variation"].sum()),
            cells=int(cells["parser_variation"].sum()),
            joint_issues=int(issues["parser_joint"].sum()),
            joint_cells=int(cells["parser_joint"].sum()),
            same_cell_issues=_same_cell_issues(cells, "parser_joint"),
        )
    )
    return pd.DataFrame(rows)


def group_table(prepared: pd.DataFrame, cells: pd.DataFrame, discordant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scaffold, model, label in GROUP_ORDER:
        trajectories = prepared.loc[(prepared["scaffold"] == scaffold) & (prepared["model"] == model)]
        group_cells = cells.loc[(cells["scaffold"] == scaffold) & (cells["model"] == model)]
        row: dict[str, Any] = {
            "scaffold": scaffold,
            "model": model,
            "group": label,
            "n_trajectories": int(len(trajectories)),
            "n_known_labels": int(trajectories["is_known_label"].sum()) if len(trajectories) else 0,
            "n_unknown_labels": int(trajectories["is_unknown_label"].sum()) if len(trajectories) else 0,
            "n_issues": int(group_cells["issue_id"].nunique()) if len(group_cells) else 0,
            "n_cells": int(len(group_cells)),
            "n_cells_with_two_or_more_trajectories": int((group_cells["n_trajectories"] >= 2).sum()) if len(group_cells) else 0,
        }
        for checkpoint in (5, 15):
            runs = int(trajectories[f"run_r{checkpoint}"].sum()) if len(trajectories) else 0
            missing = int(trajectories[f"missing_command_r{checkpoint}"].sum()) if len(trajectories) else 0
            row[f"n_test_run_r{checkpoint}"] = runs
            row[f"n_missing_command_r{checkpoint}"] = missing
            row[f"test_run_rate_r{checkpoint}"] = None if len(trajectories) == 0 else runs / len(trajectories)
            row[f"missing_command_rate_r{checkpoint}"] = None if len(trajectories) == 0 else missing / len(trajectories)
            row[f"n_behavior_cells_r{checkpoint}"] = int(group_cells[f"test_behavior_variation_r{checkpoint}"].sum()) if len(group_cells) else 0
            row[f"n_behavior_issues_r{checkpoint}"] = int(group_cells.loc[group_cells[f"test_behavior_variation_r{checkpoint}"] == 1, "issue_id"].nunique()) if len(group_cells) else 0
            row[f"n_joint_cells_r{checkpoint}"] = int(group_cells[f"joint_variation_r{checkpoint}"].sum()) if len(group_cells) else 0
            row[f"n_joint_issues_r{checkpoint}"] = int(group_cells.loc[group_cells[f"joint_variation_r{checkpoint}"] == 1, "issue_id"].nunique()) if len(group_cells) else 0
            pairs = discordant.loc[(discordant["checkpoint"] == checkpoint) & (discordant["scaffold"] == scaffold) & (discordant["model"] == model)]
            informative = pairs.loc[pairs["total_discordant_pairs"] > 0]
            row[f"n_discordant_cells_r{checkpoint}"] = int(len(informative))
            row[f"n_discordant_issues_r{checkpoint}"] = int(informative["issue_id"].nunique()) if len(informative) else 0
            row[f"positive_discordant_pair_products_r{checkpoint}"] = int(pairs["positive_discordant_pairs"].sum()) if len(pairs) else 0
            row[f"negative_discordant_pair_products_r{checkpoint}"] = int(pairs["negative_discordant_pairs"].sum()) if len(pairs) else 0
        row["n_outcome_cells"] = int(group_cells["outcome_variation"].sum()) if len(group_cells) else 0
        row["n_outcome_issues"] = int(group_cells.loc[group_cells["outcome_variation"] == 1, "issue_id"].nunique()) if len(group_cells) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def short_trajectory_frame(prepared: pd.DataFrame) -> pd.DataFrame:
    status = prepared["prefix15_status"].astype(str)
    short = status.eq("no_prefix_boundary") | (prepared["rounds_observed_num"] < 15) | status.eq("mismatch")
    columns = [
        "trajectory_id",
        "issue_id",
        "scaffold",
        "model",
        "resolved",
        "n_rounds_observed",
        "n_rounds_total",
        "prefix5_status",
        "prefix15_status",
        "v3_ran_tests_r5",
        "v3_ran_tests_r15",
        "v3_n_test_runs_r5",
        "v3_n_test_runs_r15",
        "exec_calls_r5",
        "exec_calls_r15",
        "exec_missing_r5",
        "exec_missing_r15",
    ]
    audit = prepared.loc[short, columns].copy()
    audit["r15_exposure"] = np.select(
        [
            prepared.loc[short, "run_r15"].to_numpy(),
            prepared.loc[short, "known_nonrun_r15"].to_numpy(),
            prepared.loc[short, "missing_command_r15"].to_numpy(),
        ],
        ["run", "known_nonrun", "missing_command"],
        default="unusable",
    )
    return audit.sort_values(["n_rounds_observed", "trajectory_id"]).reset_index(drop=True)


def nesting_audit(prepared: pd.DataFrame) -> dict[str, Any]:
    ran5 = _numeric(prepared["v3_ran_tests_r5"])
    ran15 = _numeric(prepared["v3_ran_tests_r15"])
    calls5 = _numeric(prepared["v3_n_test_runs_r5"])
    calls15 = _numeric(prepared["v3_n_test_runs_r15"])
    exec5 = _numeric(prepared["exec_calls_r5"])
    exec15 = _numeric(prepared["exec_calls_r15"])
    comparable = np.isfinite(ran5) & np.isfinite(ran15)
    status = prepared["prefix15_status"].astype(str)
    short = status.eq("no_prefix_boundary").to_numpy()
    rounds = prepared["rounds_observed_num"].to_numpy(dtype=float)
    total = _numeric(prepared["n_rounds_total"])
    both_rounds = np.isfinite(rounds) & np.isfinite(total)
    le5 = short & np.isfinite(rounds) & (rounds <= 5)
    window_equal = (
        np.isfinite(ran5)
        & np.isfinite(ran15)
        & (ran5 == ran15)
        & np.isfinite(calls5)
        & np.isfinite(calls15)
        & (calls5 == calls15)
        & np.isfinite(exec5)
        & np.isfinite(exec15)
        & (exec5 == exec15)
        & (_numeric(prepared["exec_missing_r5"]) == _numeric(prepared["exec_missing_r15"]))
    )
    round_counts = (
        pd.Series(rounds[short & np.isfinite(rounds)]).value_counts().sort_index().astype(int).to_dict()
        if short.any()
        else {}
    )
    return {
        "n_no_prefix_boundary": int(short.sum()),
        "n_prefix15_match": int(status.eq("match").sum()),
        "n_prefix15_mismatch": int(status.eq("mismatch").sum()),
        "n_other_prefix15_status": int((~status.isin(["match", "no_prefix_boundary", "mismatch"])).sum()),
        "n_rounds_below_15": int(np.nansum(rounds < 15)),
        "n_short_with_rounds_ge_15": int((short & np.isfinite(rounds) & (rounds >= 15)).sum()),
        "n_below_15_without_boundary_flag": int((~short & np.isfinite(rounds) & (rounds < 15)).sum()),
        "n_round_source_mismatch": int((both_rounds & (rounds != total)).sum()),
        "n_ran_not_nested": int((comparable & (ran15 < ran5)).sum()),
        "n_test_calls_not_nested": int((np.isfinite(calls5) & np.isfinite(calls15) & (calls15 < calls5)).sum()),
        "n_exec_calls_not_nested": int((np.isfinite(exec5) & np.isfinite(exec15) & (exec15 < exec5)).sum()),
        "n_short_r15_missing": int((short & ~np.isfinite(ran15)).sum()),
        "n_short_r5_positive_r15_negative": int((short & comparable & (ran5 == 1) & (ran15 == 0)).sum()),
        "n_short_rounds_le_5": int(le5.sum()),
        "n_short_rounds_le_5_window_differs": int((le5 & ~window_equal).sum()),
        "n_short_r15_run": int((short & prepared["run_r15"].to_numpy()).sum()),
        "n_short_r15_known_nonrun": int((short & prepared["known_nonrun_r15"].to_numpy()).sum()),
        "n_short_r15_missing_command": int((short & prepared["missing_command_r15"].to_numpy()).sum()),
        "n_short_r15_added_after_r5": int((short & comparable & (ran5 == 0) & (ran15 == 1)).sum()),
        "round_counts": {str(int(key)): int(value) for key, value in round_counts.items()},
    }


def _alignment(frame: pd.DataFrame, issues: pd.DataFrame, cells: pd.DataFrame, sets: dict[str, set[Any]]) -> dict[str, Any]:
    stats = issue_support(frame)
    comparisons = {
        "loose_issues": (len(sets["loose_issues"]), int(issues["v34_loose_flag_variation_r5"].sum()), stats["issues_with_r5_test_flag_variation"]),
        "strict_issues": (len(sets["strict_issues"]), int(issues["test_behavior_variation_r5"].sum()), stats["issues_with_strict_run_versus_known_nonrun"]),
        "outcome_issues": (len(sets["outcome_issues"]), int(issues["outcome_variation"].sum()), stats["issues_with_outcome_variation"]),
        "both_issues": (len(sets["both_issues"]), int(((issues["v34_loose_flag_variation_r5"] == 1) & (issues["outcome_variation"] == 1)).sum()), stats["issues_behavior_and_outcome"]),
        "strict_both_issues": (len(sets["strict_both_issues"]), int(issues["joint_variation_r5"].sum()), stats["issues_strict_behavior_and_outcome"]),
        "within_loose_issues": (len(sets["within_loose_issues"]), int(cells.loc[cells["v34_loose_flag_variation_r5"] == 1, "issue_id"].nunique()), stats["issues_with_within_group_test_variation"]),
        "within_loose_cells": (len(sets["within_loose_cells"]), int(cells["v34_loose_flag_variation_r5"].sum()), int(cells["v34_loose_flag_variation_r5"].sum())),
    }
    internal_ok = all(len(set(values)) == 1 for values in comparisons.values())
    published = {}
    published_ok = True
    for key, expected in PUBLISHED_V34.items():
        observed = stats[key]
        published[key] = {"expected": expected, "observed": observed, "match": observed == expected}
        published_ok = published_ok and observed == expected
    for checkpoint, expected in PUBLISHED_BUCKETS.items():
        observed_bucket = {
            "run": int(issues[f"n_test_run_r{checkpoint}"].sum()) if "n_test_run_r5" in issues else int(frame[f"run_r{checkpoint}"].sum()) if f"run_r{checkpoint}" in frame else None,
        }
        # Bucket totals are trajectory sums. Use the prepared indicators when present.
        if f"run_r{checkpoint}" in frame.columns:
            observed_bucket = {
                "run": int(frame[f"run_r{checkpoint}"].sum()),
                "known_nonrun": int(frame[f"known_nonrun_r{checkpoint}"].sum()),
                "missing_command": int(frame[f"missing_command_r{checkpoint}"].sum()),
                "unusable": int(frame[f"unusable_r{checkpoint}"].sum()),
            }
            match = observed_bucket == expected
            published[f"bucket_r{checkpoint}"] = {"expected": expected, "observed": observed_bucket, "match": match}
            published_ok = published_ok and match
    return {
        "internal_match": internal_ok,
        "published_match": published_ok,
        "comparisons": {key: {"set": values[0], "table": values[1], "issue_support": values[2]} for key, values in comparisons.items()},
        "published": published,
        "issue_support": stats,
    }


def _cross(sets: dict[str, set[Any]], cells: pd.DataFrame) -> dict[str, Any]:
    loose = sets["loose_issues"]
    both = sets["both_issues"]
    within = sets["within_loose_issues"]
    strict_behavior_issues = set(cells.loc[cells["test_behavior_variation_r5"] == 1, "issue_id"])
    outcome_issues = set(cells.loc[cells["outcome_variation"] == 1, "issue_id"])
    joint_issues = set(cells.loc[cells["joint_variation_r5"] == 1, "issue_id"])
    loose_joint_issues = set(cells.loc[(cells["v34_loose_flag_variation_r5"] == 1) & (cells["outcome_variation"] == 1), "issue_id"])
    joint_cells = cells.loc[cells["joint_variation_r5"] == 1]
    loose_joint_cells = cells.loc[(cells["v34_loose_flag_variation_r5"] == 1) & (cells["outcome_variation"] == 1)]
    return {
        "issue_behavior_with_within_group_behavior_issues": len(loose & within),
        "issue_behavior_without_within_group_behavior_issues": len(loose - within),
        "strict_issue_behavior_with_within_group_strict_issues": len(sets["strict_issues"] & strict_behavior_issues),
        "both_with_within_group_joint_issues": len(both & joint_issues),
        "both_with_within_group_joint_cells": int(joint_cells["issue_id"].isin(both).sum()),
        "both_with_loose_within_group_joint_issues": len(both & loose_joint_issues),
        "both_with_loose_within_group_joint_cells": int(loose_joint_cells["issue_id"].isin(both).sum()),
        "joint_issues_outside_published_both": len(joint_issues - both),
        "within_734_with_within_group_outcome_issues": len(within & outcome_issues),
        "within_734_without_within_group_outcome_issues": len(within - outcome_issues),
        "within_734_with_same_cell_joint_issues": len(within & joint_issues),
        "within_group_outcome_without_within_group_behavior_issues": len(outcome_issues - within),
        "within_group_behavior_cells": int(cells["v34_loose_flag_variation_r5"].sum()),
        "within_group_strict_behavior_issues": len(strict_behavior_issues),
        "within_group_strict_behavior_cells": int(cells["test_behavior_variation_r5"].sum()),
        "within_group_outcome_issues": len(outcome_issues),
        "within_group_outcome_cells": int(cells["outcome_variation"].sum()),
        "within_group_joint_issues": len(joint_issues),
        "within_group_joint_cells": int(len(joint_cells)),
        "within_group_outcome_cells_among_734": int(cells.loc[cells["issue_id"].isin(within) & cells["outcome_variation"].eq(1)].shape[0]),
        "strict_behavior_without_within_group_outcome_issues": len(strict_behavior_issues - outcome_issues),
        "within_group_outcome_without_strict_behavior_issues": len(outcome_issues - strict_behavior_issues),
    }


def characterize(frame: pd.DataFrame) -> dict[str, Any]:
    """Build every V3.4.1 table from one joined trajectory frame."""
    required = {
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
        "prefix15_status",
        "n_rounds_observed",
        "n_rounds_total",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"required columns missing: {missing}")
    prepared = prepare(frame)
    sets = v34_variation_sets(frame)
    issues = aggregate(prepared, ["issue_id"]).sort_values("issue_id").reset_index(drop=True)
    cells = aggregate(prepared, ["issue_id", "scaffold", "model"]).sort_values(["issue_id", "scaffold", "model"]).reset_index(drop=True)
    counts, discordant = four_cell_tables(cells)
    nesting = nesting_audit(prepared)
    alignment = _alignment(prepared, issues, cells, sets)
    cross = _cross(sets, cells)
    behavior = behavior_table(prepared, issues, cells)
    groups = group_table(prepared, cells, discordant)
    unexpected = sorted(set(zip(prepared["scaffold"], prepared["model"], strict=True)).difference((item[0], item[1]) for item in GROUP_ORDER))
    neither = int(((issues["v34_loose_flag_variation_r5"] == 0) & (issues["outcome_variation"] == 0)).sum())
    headline = {
        "trajectories": int(len(prepared)),
        "issues": int(issues.shape[0]),
        "cells": int(cells.shape[0]),
        "cells_with_two_or_more": int((cells["n_trajectories"] >= 2).sum()),
        "behavior_issues_r5": int(issues["test_behavior_variation_r5"].sum()),
        "behavior_issues_r15": int(issues["test_behavior_variation_r15"].sum()),
        "outcome_issues": int(issues["outcome_variation"].sum()),
        "both_issues_r5": int(issues["joint_variation_r5"].sum()),
        "both_issues_r15": int(issues["joint_variation_r15"].sum()),
        "behavior_only_r5": int(((issues["test_behavior_variation_r5"] == 1) & (issues["outcome_variation"] == 0)).sum()),
        "outcome_only": int(((issues["test_behavior_variation_r5"] == 0) & (issues["outcome_variation"] == 1)).sum()),
        "neither_r5": int(((issues["test_behavior_variation_r5"] == 0) & (issues["outcome_variation"] == 0)).sum()),
        "loose_neither": neither,
        "known_labels": int(prepared["is_known_label"].sum()),
        "unknown_labels": int(prepared["is_unknown_label"].sum()),
        "invalid_labels": int(prepared["is_invalid_label"].sum()),
        "unexpected_groups": [{"scaffold": scaffold, "model": model} for scaffold, model in unexpected],
    }
    issue_columns = [
        "issue_id",
        "n_trajectories",
        "n_known_labels",
        "n_unknown_labels",
        "n_success",
        "n_failure",
        "outcome_variation",
        "n_test_run_r5",
        "n_known_nonrun_r5",
        "n_missing_command_r5",
        "test_behavior_variation_r5",
        "n_test_run_r15",
        "n_known_nonrun_r15",
        "n_missing_command_r15",
        "test_behavior_variation_r15",
        "joint_variation_r5",
        "joint_variation_r15",
        "v34_loose_flag_variation_r5",
    ]
    cell_columns = [
        "issue_id",
        "scaffold",
        "model",
        "n_trajectories",
        "n_known_labels",
        "n_unknown_labels",
        "n_success",
        "n_failure",
        "n_test_run_r5",
        "n_known_nonrun_r5",
        "n_missing_command_r5",
        "test_behavior_variation_r5",
        "outcome_variation",
        "joint_variation_r5",
        "n_test_run_r15",
        "n_known_nonrun_r15",
        "n_missing_command_r15",
        "test_behavior_variation_r15",
        "joint_variation_r15",
        "v34_loose_flag_variation_r5",
    ]
    return {
        "issues": issues[issue_columns],
        "cells": cells[cell_columns],
        "four_cell": counts,
        "discordant": discordant,
        "by_group": groups,
        "by_behavior": behavior,
        "short_rows": short_trajectory_frame(prepared),
        "headline": headline,
        "cross": cross,
        "alignment": alignment,
        "nesting": nesting,
        "pairs_r5": _pair_summary(discordant, 5),
        "pairs_r15": _pair_summary(discordant, 15),
        "sets": sets,
    }


def source_problems(census: pd.DataFrame, meta: pd.DataFrame) -> list[str]:
    """Join and corpus gates. A failure here blocks the audit."""
    problems: list[str] = []
    if len(census) != 207489 or len(meta) != 207489:
        problems.append(f"row count census={len(census)} meta={len(meta)}")
    if census["trajectory_id"].duplicated().any() or meta["trajectory_id"].duplicated().any():
        problems.append("duplicate trajectory ids")
        return problems
    if set(census["trajectory_id"]) != set(meta["trajectory_id"]):
        problems.append("census and metadata trajectory ids differ")
        return problems
    shared = [
        "issue_id",
        "instance_id",
        "repo",
        "scaffold",
        "model",
        "resolved",
        "split",
        "n_rounds_total",
        "parser_v2_strict_fail_r5",
        "shard",
    ]
    left = census[["trajectory_id", *shared]].copy()
    right = meta[["trajectory_id", *shared]].copy()
    merged = left.merge(right, on="trajectory_id", suffixes=("_census", "_meta"), validate="one_to_one")
    for column in shared:
        mismatch = int((merged[f"{column}_census"].astype(str) != merged[f"{column}_meta"].astype(str)).sum())
        if mismatch:
            problems.append(f"{column} disagrees on {mismatch} rows")
    if int((census["issue_id"].astype(str) != census["instance_id"].astype(str)).sum()):
        problems.append("issue_id is not the stored instance_id")
    resolved = pd.to_numeric(census["resolved"], errors="coerce")
    expected = {1: 65244, 0: 95487, -1: 46758}
    for value, count in expected.items():
        observed = int((resolved == value).sum())
        if observed != count:
            problems.append(f"resolved {value} count is {observed}")
    if int(census["issue_id"].nunique()) != 22320:
        problems.append(f"issue count is {int(census['issue_id'].nunique())}")
    if int(census["repo"].nunique()) != 2654:
        problems.append(f"repo count is {int(census['repo'].nunique())}")
    return problems
