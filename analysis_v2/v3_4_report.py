"""Descriptive V3.4 tables. This module does not fit a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from census_integrity import parser_v2_failure_trigger, render_trigger, stratified_sample, v3_failure_trigger
from p8v3.testcmd import parse_test_output
from traj_adapter import iter_calls
from v3_4_lib import (
    CHECKPOINTS,
    compare_v33,
    high_confidence_buckets,
    is_true_flag,
    issue_support,
    metric_summary,
    outcome_kind,
    sha256_file,
)

BINARY_METRICS = (
    "v3_ran_tests_r5",
    "v3_ran_tests_r15",
    "v3_ran_tests_amb_r5",
    "v3_ran_tests_amb_r15",
    "v3_test_fail_r5",
    "v3_test_fail_r15",
    "v3_collect_build_fail_r5",
    "v3_collect_build_fail_r15",
    "v3_anyobs_fail_r5",
    "v3_anyobs_fail_r15",
    "loose_r5",
    "loose_r15",
)
COUNT_METRICS = (
    "v3_n_test_runs_r5",
    "v3_n_test_runs_r15",
    "v3_test_fail_obs_r5",
    "v3_test_fail_obs_r15",
    "v2_explicit_test_failure_r5",
    "v2_explicit_test_failure_r15",
    "v2_explicit_test_pass_r5",
    "v2_explicit_test_pass_r15",
    "v2_build_or_compile_failure_r5",
    "v2_build_or_compile_failure_r15",
    "v2_test_setup_or_collection_failure_r5",
    "v2_test_setup_or_collection_failure_r15",
    "v2_package_level_failure_unresolved_r5",
    "v2_package_level_failure_unresolved_r15",
    "v2_execution_status_unknown_r5",
    "v2_execution_status_unknown_r15",
    "high_unknown_outputs_r5",
    "high_unknown_outputs_r15",
)


def _ok_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["census_parse_ok"].map(is_true_flag).astype(bool)


def _subset(frame: pd.DataFrame, spec: tuple) -> pd.DataFrame:
    kind = spec[0]
    if kind == "all":
        return frame
    if kind == "scaffold":
        return frame.loc[frame["scaffold"] == spec[1]]
    if kind == "model":
        return frame.loc[frame["model"] == spec[1]]
    if kind == "pair":
        return frame.loc[(frame["scaffold"] == spec[1]) & (frame["model"] == spec[2])]
    if kind == "outcome":
        kinds = frame["resolved"].map(outcome_kind)
        return frame.loc[kinds == spec[1]]
    raise KeyError(kind)


GROUPS = (
    ("all", ("all",)),
    ("OpenHands", ("scaffold", "openhands")),
    ("SWE-agent", ("scaffold", "sweagent")),
    ("Minimax", ("model", "minimax_m25")),
    ("Qwen", ("model", "qwen35_122b")),
    ("OpenHands / Minimax", ("pair", "openhands", "minimax_m25")),
    ("OpenHands / Qwen", ("pair", "openhands", "qwen35_122b")),
    ("SWE-agent / Minimax", ("pair", "sweagent", "minimax_m25")),
    ("SWE-agent / Qwen", ("pair", "sweagent", "qwen35_122b")),
    ("known-label", ("outcome", "known")),
    ("unknown-label", ("outcome", "unknown")),
)


def summary_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for label, spec in GROUPS:
        group = _subset(frame, spec)
        for metric in (*BINARY_METRICS, *COUNT_METRICS):
            if metric not in group.columns:
                continue
            item = metric_summary(group[metric], count=metric in COUNT_METRICS)
            item["group"] = label
            item["metric"] = metric
            item["denominator"] = "n_valid rows; missing stays out of the rate" if metric not in COUNT_METRICS else "sum over n_valid; not a trajectory rate"
            rows.append(item)
        for checkpoint in CHECKPOINTS:
            buckets = high_confidence_buckets(group, checkpoint)
            for name, value in buckets.items():
                rows.append(
                    {
                        "group": label,
                        "metric": f"bucket_r{checkpoint}_{name}",
                        "n_total": int(len(group)),
                        "n_valid": int(_ok_mask(group).sum()),
                        "n_error": int((~_ok_mask(group)).sum()),
                        "positive_count": int(value),
                        "rate": None,
                        "denominator": "partition of trajectories; missing command is separate from a known non-hit",
                    }
                )
    return rows


def runner_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    ok = _ok_mask(frame)
    for checkpoint in CHECKPOINTS:
        column = f"high_runners_r{checkpoint}"
        unknown = f"high_unknown_outputs_r{checkpoint}"
        calls = f"v3_n_test_runs_r{checkpoint}"
        for (scaffold, model), group in frame.groupby(["scaffold", "model"], dropna=False, sort=True):
            valid = group.loc[ok.loc[group.index], column].dropna()
            counts: dict[str, int] = {}
            for text in valid.astype(str):
                for runner in text.split("|"):
                    if runner:
                        counts[runner] = counts.get(runner, 0) + 1
            n_calls = int(group.loc[ok.loc[group.index], calls].dropna().sum())
            n_unknown = int(group.loc[ok.loc[group.index], unknown].dropna().sum())
            for runner, count in sorted(counts.items()):
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "scaffold": scaffold,
                        "model": model,
                        "runner": runner,
                        "hits": count,
                        "high_confidence_calls": n_calls,
                        "unknown_outputs": n_unknown,
                        "unknown_output_rate": None if n_calls == 0 else n_unknown / n_calls,
                    }
                )
    return rows


def crosstab_rows(frame: pd.DataFrame, left: str, right: str) -> list[dict[str, Any]]:
    def level(value: Any) -> str:
        if pd.isna(value):
            return "<missing>"
        return str(int(value))

    table = pd.crosstab(frame[left].map(level), frame[right].map(level), dropna=False)
    rows = []
    for left_value in table.index:
        for right_value in table.columns:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "left_value": left_value,
                    "right_value": right_value,
                    "n": int(table.loc[left_value, right_value]),
                }
            )
    return rows


def _rate_text(frame: pd.DataFrame, metric: str, *, count: bool = False) -> str:
    item = metric_summary(frame[metric], count=count)
    if count:
        return f"合计 {item['positive_count']}，有效轨迹 {item['n_valid']}，缺失 {item['n_error']}"
    if item["rate"] is None:
        return f"0 个有效分母（缺失 {item['n_error']}）"
    return f"{item['positive_count']} / {item['n_valid']} = {item['rate']:.6f}，缺失 {item['n_error']}"


def _lookup(rows: list[dict[str, Any]], group: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if row["group"] == group and row["metric"] == metric:
            return row
    raise KeyError((group, metric))


def assessment(frame: pd.DataFrame, *, tests_passed: bool, round_mismatches: int) -> list[str]:
    problems = []
    ok = _ok_mask(frame)
    n_error = int((~ok).sum())
    if n_error:
        problems.append(f"parse errors: {n_error}")
    mismatches = int((frame["prefix5_status"] == "mismatch").sum()) + int((frame["prefix15_status"] == "mismatch").sum())
    if mismatches:
        problems.append(f"prefix mismatches: {mismatches}")
    if round_mismatches:
        problems.append(f"n_rounds_total mismatches against recomputed completed rounds: {round_mismatches}")
    v33 = compare_v33(frame)
    if not v33["pass"]:
        problems.append(f"V3.3 recheck differences: {len(v33['differences'])}")
    if not tests_passed:
        problems.append("pytest did not pass")
    for checkpoint in CHECKPOINTS:
        buckets = high_confidence_buckets(frame, checkpoint)
        if sum(buckets.values()) != len(frame):
            problems.append(f"r{checkpoint} execution buckets do not partition the corpus")
    invalid = int(frame["resolved"].map(outcome_kind).eq("invalid").sum())
    if invalid:
        problems.append(f"invalid outcomes: {invalid}")
    return problems


def _answers(frame: pd.DataFrame, support: dict[str, Any], cross_rows: list[dict[str, Any]]) -> list[str]:
    def cell(left_value: str, right_value: str, right: str) -> int:
        return sum(
            row["n"]
            for row in cross_rows
            if row["right"] == right and row["left_value"] == left_value and row["right_value"] == right_value
        )

    lines = [
        f"1. 前 5 个完成轮次里，至少出现一次高置信度测试命令的轨迹是 {_rate_text(frame, 'v3_ran_tests_r5')}。高置信度调用次数是 {_rate_text(frame, 'v3_n_test_runs_r5', count=True)}。调用次数不等于完整测试套件次数。",
        f"2. 前 15 个完成轮次的对应轨迹比例是 {_rate_text(frame, 'v3_ran_tests_r15')}。调用次数是 {_rate_text(frame, 'v3_n_test_runs_r15', count=True)}。",
        f"3. OpenHands 前 5 轮高置信度测试轨迹比例是 {_rate_text(_subset(frame, ('scaffold', 'openhands')), 'v3_ran_tests_r5')}。SWE-agent 是 {_rate_text(_subset(frame, ('scaffold', 'sweagent')), 'v3_ran_tests_r5')}。",
        f"4. Minimax 前 5 轮是 {_rate_text(_subset(frame, ('model', 'minimax_m25')), 'v3_ran_tests_r5')}。Qwen 是 {_rate_text(_subset(frame, ('model', 'qwen35_122b')), 'v3_ran_tests_r5')}。",
        "5. parser v2 strict fail 与 V3 test fail 的交叉计数："
        f" 0×0 = {cell('0', '0', 'v3_test_fail_r5')}，0×1 = {cell('0', '1', 'v3_test_fail_r5')}，"
        f"1×0 = {cell('1', '0', 'v3_test_fail_r5')}，1×1 = {cell('1', '1', 'v3_test_fail_r5')}。"
        "与 v3_anyobs_fail_r5："
        f" 0×0 = {cell('0', '0', 'v3_anyobs_fail_r5')}，0×1 = {cell('0', '1', 'v3_anyobs_fail_r5')}，"
        f"1×0 = {cell('1', '0', 'v3_anyobs_fail_r5')}，1×1 = {cell('1', '1', 'v3_anyobs_fail_r5')}。"
        "这些数字描述两个检测器是否给出相同标记。",
        "6. 分歧样本保留触发该检测器的 observation，并写明工具、命令和检测原因。样本不能换成检测器的正确率。",
        f"7. 有 {support['issues_with_r5_test_flag_variation']} 个 issue 的前 5 轮高置信度测试标记同时出现 0 和 1。其中严格的“检测到运行”对“命令已抽出但没有高置信度测试命令”有 {support['issues_with_strict_run_versus_known_nonrun']} 个 issue。",
        f"8. 测试标记变异且已知结局同时有 0 和 1 的 issue 有 {support['issues_behavior_and_outcome']} 个。严格测试对比且结局有差异的 issue 有 {support['issues_strict_behavior_and_outcome']} 个。",
        f"9. 同一 scaffold/model 组内就出现测试标记差异的 issue 有 {support['issues_with_within_group_test_variation']} 个。这个数量只说明后续独立人工审阅或 issue 内比较是否有可观察的支持度。",
    ]
    return lines


def render_report(
    *,
    status: str,
    frame: pd.DataFrame,
    support: dict[str, Any],
    v33: dict[str, Any],
    problems: list[str],
    provenance: dict[str, Any],
    metadata: dict[str, Any],
    categories: dict[str, int],
    round_mismatches: int,
) -> str:
    rows = summary_rows(frame)
    lines = [
        "# V3.4 全库描述性普查",
        "",
        f"状态：`{status}`",
        "",
        "本报告只描述测试命令、测试输出和 issue 内标记是否不同。它不估计预测增量、因果效应或检测器正确率。",
        "",
        "## 数据与 metadata",
        "",
        f"- 固定 revision：`{provenance.get('revision')}`",
        f"- Manifest SHA-256：`{provenance.get('manifest_sha256')}`",
        f"- 分片核验：{provenance.get('verified')} / {provenance.get('expected_shards')}",
        f"- Metadata 行数：{metadata.get('rows')}",
        f"- parser v2 阳性（历史冻结特征）：{metadata.get('parser_flag_positives')}",
        f"- parser v2 缺失：{metadata.get('parser_flag_nulls')}",
        f"- 重算完成轮次与历史 `n_complete_cycles` 不一致：{round_mismatches}",
        "",
        "## 直接回答",
        "",
    ]
    cross_rows = crosstab_rows(frame, "parser_v2_strict_fail_r5", "v3_test_fail_r5")
    cross_rows.extend(crosstab_rows(frame, "parser_v2_strict_fail_r5", "v3_anyobs_fail_r5"))
    lines.extend(f"- {line}" for line in _answers(frame, support, cross_rows))
    lines.extend(["", "## 分组比例", ""])
    lines.append("| group | metric | n_total | n_valid | n_error | positive_count | rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        if row["metric"] not in {"v3_ran_tests_r5", "v3_ran_tests_r15", "v3_test_fail_r5", "v3_test_fail_r15", "v3_n_test_runs_r5", "v3_n_test_runs_r15"}:
            continue
        rate = "" if row["rate"] is None else f"{row['rate']:.6f}"
        lines.append(
            f"| {row['group']} | {row['metric']} | {row['n_total']} | {row['n_valid']} | {row['n_error']} | {row['positive_count']} | {rate} |"
        )
    lines.extend(["", "## 高置信度命令、已知未命中、命令缺失", ""])
    for checkpoint in CHECKPOINTS:
        buckets = high_confidence_buckets(frame, checkpoint)
        lines.append(f"前 {checkpoint} 轮：`{json.dumps(buckets, ensure_ascii=False)}`")
    lines.extend(["", "## Issue 支持度", "", "```json", json.dumps(support, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## V3.3 八片复核", ""])
    lines.append(f"复核 {'通过' if v33['pass'] else '不一致'}。")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"observed_totals": v33["observed_totals"], "differences": v33["differences"]}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.extend(["", "## 命令抽取", ""])
    for key, value in sorted(categories.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 未解决事项", ""])
    if problems:
        lines.extend(f"- {item}" for item in problems)
    else:
        lines.append("- 无。")
    lines.extend(
        [
            "",
            "## 没有做的事",
            "",
            "- 没有运行 `00_reproduce.py`。",
            "- 没有训练模型，没有计算 AUPRC、ROC-AUC 或 Delta AP。",
            "- 没有修改 parser v2、`p8v3.testcmd`、L2 五轮 checkpoint 或冻结特征。",
            "- 没有开始人工标注、issue 固定效应回归或 L3。",
            "",
        ]
    )
    return "\n".join(lines)


def _trigger_block(kind: str, trajectory: Any, scaffold: str) -> list[str]:
    calls = iter_calls({"scaffold": scaffold, "trajectory": trajectory})
    if kind == "v3":
        trigger = v3_failure_trigger(calls)
        reason = "unresolved"
        if trigger.get("status") == "identified":
            outcome = parse_test_output(trigger.get("output"))
            frameworks = ",".join(sorted(outcome.frameworks))
            reason = f"high-confidence test command; parse_test_output.test_failed; frameworks={frameworks}"
    else:
        trigger = parser_v2_failure_trigger(trajectory, scaffold)
        reason = "unresolved" if trigger.get("status") != "identified" else ",".join(trigger.get("reasons") or [])
    lines = [
        f"message_index: {trigger.get('observation_index')}",
        f"detection_reason: {reason}",
    ]
    lines.extend(render_trigger(trigger))
    return lines


def disagreement_markdown(frame: pd.DataFrame, shard_paths: dict[str, Path]) -> str:
    sections = [
        "# Disagreement sample",
        "",
        "诊断样本使用种子 `20261102`。样本只定位触发检测器的 observation。",
        "它不估计 recall 或 precision。找不到触发证据时记为 unresolved，不用另一条 observation 顶替。",
        "",
    ]
    specs = (
        ("v3_test_fail_r5=1 and parser_v2=0", (frame["v3_test_fail_r5"] == 1) & (frame["parser_v2_strict_fail_r5"] == 0), 30, "v3"),
        ("parser_v2=1 and v3_test_fail_r5=0", (frame["parser_v2_strict_fail_r5"] == 1) & (frame["v3_test_fail_r5"] == 0), 10, "parser"),
    )
    for title, mask, limit, kind in specs:
        pool = frame.loc[mask.fillna(False)]
        chosen, strata = stratified_sample(pool, limit, 20261102)
        sections.extend([f"## {title}", "", f"可用行 {len(pool)}。抽出 {len(chosen)}。", ""])
        sections.append("| scaffold | model | population | sampled |")
        sections.append("|---|---|---:|---:|")
        for item in strata:
            sections.append(f"| {item['scaffold']} | {item['model']} | {item['population']} | {item['sampled']} |")
        sections.append("")
        wanted: dict[str, set[str]] = {}
        for row in chosen.itertuples(index=False):
            wanted.setdefault(str(row.shard), set()).add(str(row.trajectory_id))
        loaded: dict[str, Any] = {}
        for relative, ids in wanted.items():
            path = shard_paths.get(relative)
            if path is None or not Path(path).is_file():
                continue
            reader = pq.ParquetFile(path)
            remaining = set(ids)
            for batch in reader.iter_batches(batch_size=50, columns=["trajectory_id", "trajectory"]):
                for record in batch.to_pylist():
                    trajectory_id = str(record["trajectory_id"])
                    if trajectory_id in remaining:
                        loaded[trajectory_id] = record["trajectory"]
                        remaining.remove(trajectory_id)
                if not remaining:
                    break
        for row in chosen.itertuples(index=False):
            trajectory_id = str(row.trajectory_id)
            sections.extend([f"### {trajectory_id}", "", f"shard: {row.shard}", f"scaffold: {row.scaffold}", f"model: {row.model}"])
            trajectory = loaded.get(trajectory_id)
            if trajectory is None:
                sections.extend(["trigger: unresolved (trajectory bytes were not reloaded)", ""])
                continue
            sections.extend(_trigger_block(kind, trajectory, str(row.scaffold)))
    return "\n".join(sections)


def write_tables(run_dir: Path, frame: pd.DataFrame, categories: dict[str, int]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(frame)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "census_summary.csv", index=False)
    model_labels = {
        "OpenHands",
        "SWE-agent",
        "Minimax",
        "Qwen",
        "OpenHands / Minimax",
        "OpenHands / Qwen",
        "SWE-agent / Minimax",
        "SWE-agent / Qwen",
    }
    summary.loc[summary["group"].isin(model_labels)].to_csv(run_dir / "scaffold_model_summary.csv", index=False)
    support = issue_support(frame)
    pd.DataFrame([{"metric": key, "value": json.dumps(value) if isinstance(value, dict) else value} for key, value in support.items()]).to_csv(
        run_dir / "issue_support_summary.csv",
        index=False,
    )
    cross = crosstab_rows(frame, "parser_v2_strict_fail_r5", "v3_test_fail_r5")
    cross.extend(crosstab_rows(frame, "parser_v2_strict_fail_r5", "v3_anyobs_fail_r5"))
    pd.DataFrame(cross).to_csv(run_dir / "parser_v2_vs_v3_crosstab.csv", index=False)
    extraction = [{"record_type": "command", "category": key, "n_calls": value} for key, value in sorted(categories.items())]
    for item in runner_rows(frame):
        item["record_type"] = "runner"
        extraction.append(item)
    pd.DataFrame(extraction).to_csv(run_dir / "command_extraction_summary.csv", index=False)
    return {"summary_rows": rows, "support": support, "cross_rows": cross, "v33": compare_v33(frame)}


def hash_outputs(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.name == "V3_4_SHA256_MANIFEST.csv":
            continue
        rows.append({"path": str(path.relative_to(run_dir)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    frame = pd.DataFrame(rows)
    frame.to_csv(run_dir / "V3_4_SHA256_MANIFEST.csv", index=False)
    return frame


def short_review(status: str, run_dir: Path, report: str) -> str:
    return "\n".join(
        [
            "# V3.4 full-corpus census review",
            "",
            f"状态：`{status}`",
            "",
            f"运行目录：`{run_dir}`",
            "",
            "完整表格、交叉表、issue 支持度和分歧样本都在该目录。下面保留同一份描述性结论。",
            "",
            report,
            "",
        ]
    )
