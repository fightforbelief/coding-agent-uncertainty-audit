"""Reporting, provenance, and disagreement selection for the v3 census.

These helpers do not read the pinned dataset and do not fit a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from p8audit.execution_signals_v2 import classify_execution_v2
from p8audit.schema import PINNED_REVISION, SHARD_GROUPS
from p8audit.trajectory_parser import parse_trajectory
from p8v3.testcmd import detect_test_command, parse_test_output
from traj_adapter import iter_calls

BLOCKED = "DATASET_REVISION_VERIFICATION_BLOCKED"
KNOWN_RESOLVED = {-1, 0, 1}
BINARY_METRICS = (
    "v3_ran_tests_r5",
    "v3_ran_tests_amb_r5",
    "v3_test_fail_r5",
    "v3_collect_build_fail_r5",
    "v3_anyobs_fail_r5",
    "loose_r5",
    "v3_ran_tests_r15",
    "v3_ran_tests_amb_r15",
    "v3_test_fail_r15",
    "v3_collect_build_fail_r15",
    "v3_anyobs_fail_r15",
    "loose_r15",
)
COUNT_METRICS = ("v3_n_test_runs_r5", "v3_n_test_runs_r15")


def require_data_root(raw: str | None, *, repository_root: Path) -> Path:
    """Reject a missing value before Path('') can become the working directory."""
    if raw is None or str(raw).strip() == "":
        raise SystemExit("P8_DATA_ROOT is required and must not be empty.")
    path = Path(raw)
    if not path.is_dir():
        raise SystemExit(f"P8_DATA_ROOT is not a directory: {raw}")
    resolved = path.resolve()
    if resolved == repository_root.resolve():
        raise SystemExit("P8_DATA_ROOT must not be the repository root.")
    return resolved


def verify_dataset_revision(manifest_path: Path, data_root: Path) -> list[dict]:
    """Match a local manifest that itself records the pinned revision.

    Byte counts and a shard total are not accepted as a substitute. Absence of
    that manifest is ``DATASET_REVISION_VERIFICATION_BLOCKED``.
    """
    if not manifest_path.is_file():
        raise SystemExit(BLOCKED)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(BLOCKED) from exc
    if not isinstance(payload, dict) or payload.get("revision") != PINNED_REVISION:
        raise SystemExit(BLOCKED)
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise SystemExit(BLOCKED)
    inventory: list[dict] = []
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or "path" not in item or "bytes" not in item:
            raise SystemExit(BLOCKED)
        relative = str(item["path"]).replace("\\", "/")
        if relative in seen:
            raise SystemExit(BLOCKED)
        if not any(relative.startswith(prefix + "/") or relative.startswith(prefix) for prefix in SHARD_GROUPS):
            raise SystemExit(BLOCKED)
        local = data_root / relative
        try:
            size = int(item["bytes"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(BLOCKED) from exc
        if not local.is_file() or local.stat().st_size != size:
            raise SystemExit(f"shard inventory mismatch: {relative}")
        seen.add(relative)
        inventory.append({"path": relative, "bytes": size, "local": local})
    if len(inventory) != 84:
        raise SystemExit(BLOCKED)
    return inventory


def compare_trajectory_ids(meta: pd.DataFrame, census: pd.DataFrame) -> dict:
    meta_ids = meta["trajectory_id"].astype(str)
    census_ids = census["trajectory_id"].astype(str)
    meta_set = set(meta_ids)
    census_set = set(census_ids)
    return {
        "meta_rows": int(len(meta)),
        "census_rows": int(len(census)),
        "meta_duplicate_ids": int(meta_ids.duplicated().sum()),
        "census_duplicate_ids": int(census_ids.duplicated().sum()),
        "missing_from_census": sorted(meta_set - census_set),
        "extra_in_census": sorted(census_set - meta_set),
    }


def ids_match(comparison: dict) -> bool:
    return (
        comparison["meta_duplicate_ids"] == 0
        and comparison["census_duplicate_ids"] == 0
        and not comparison["missing_from_census"]
        and not comparison["extra_in_census"]
    )


def _valid_mask(group: pd.DataFrame, metric: str) -> pd.Series:
    return group[metric].notna()


def metric_line(group: pd.DataFrame, metric: str, *, count: bool) -> dict:
    total = int(len(group))
    valid = _valid_mask(group, metric)
    n_valid = int(valid.sum())
    n_error = total - n_valid
    if count:
        observed = group.loc[valid, metric]
        return {
            "n_total": total,
            "n_valid": n_valid,
            "n_error": n_error,
            "positive_or_sum": float(observed.sum()) if n_valid else 0.0,
            "rate": None,
        }
    positives = int((group.loc[valid, metric] == 1).sum())
    rate = positives / n_valid if n_valid else None
    return {
        "n_total": total,
        "n_valid": n_valid,
        "n_error": n_error,
        "positive_or_sum": positives,
        "rate": rate,
    }


def outcome_audit(frame: pd.DataFrame) -> dict:
    numeric = pd.to_numeric(frame["resolved"], errors="coerce")
    invalid = ~numeric.isin([-1, 0, 1])
    return {
        "n_invalid_or_missing_outcome": int(invalid.sum()),
        "known_label": frame.loc[numeric.isin([0, 1])].copy(),
        "invalid_rows": int(invalid.sum()),
    }


def render_report(frame: pd.DataFrame, error_count: int) -> tuple[str, str]:
    """Return markdown and ``COMPLETE`` or ``PARTIAL``.

    Missing census metrics stay missing. Rates use valid rows only.
    """
    outcomes = outcome_audit(frame)
    status = "PARTIAL" if error_count or outcomes["n_invalid_or_missing_outcome"] else "COMPLETE"
    lines = [
        "# v3 census",
        "",
        f"Status: `{status}`",
        "",
        "A parsing exception is missing data, not a negative observation. "
        "`v3_n_test_runs_rK` sums high-confidence calls over valid trajectories. "
        "It is not a trajectory count and not a count of completed test suites. "
        "Rates divide by `n_valid` only.",
        "",
        f"Rows in the joined table: {len(frame)}. Census parse errors: {error_count}. "
        f"Outcomes outside {{-1, 0, 1}}: {outcomes['n_invalid_or_missing_outcome']}.",
        "",
    ]
    if status == "PARTIAL":
        lines.append("These rates are not full-corpus measurements.")
        lines.append("")
    subsets = [("all-rows", frame), ("known-label resolved in {0, 1}", outcomes["known_label"])]
    for title, subset in subsets:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| metric | n_total | n_valid | n_error | positive_or_sum | rate |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for metric in (*BINARY_METRICS, *COUNT_METRICS):
            summary = metric_line(subset, metric, count=metric in COUNT_METRICS)
            rate = "" if summary["rate"] is None else f"{summary['rate']:.6g}"
            lines.append(
                f"| {metric} | {summary['n_total']} | {summary['n_valid']} | {summary['n_error']} | "
                f"{summary['positive_or_sum']:.0f} | {rate} |"
            )
        lines.append("")
        if len(subset):
            for (scaffold, model), group in subset.groupby(["scaffold", "model"], dropna=False, sort=True):
                lines.append(f"### {scaffold} / {model}")
                lines.append("")
                lines.append("| metric | n_total | n_valid | n_error | positive_or_sum | rate |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for metric in (*BINARY_METRICS, *COUNT_METRICS):
                    summary = metric_line(group, metric, count=metric in COUNT_METRICS)
                    rate = "" if summary["rate"] is None else f"{summary['rate']:.6g}"
                    lines.append(
                        f"| {metric} | {summary['n_total']} | {summary['n_valid']} | {summary['n_error']} | "
                        f"{summary['positive_or_sum']:.0f} | {rate} |"
                    )
                lines.append("")
    lines.extend(["## 交叉表", "", "单元格 `<missing>` 是缺失，不是 0。", ""])
    for left, right in (
        ("parser_v2_strict_fail_r5", "v3_test_fail_r5"),
        ("parser_v2_strict_fail_r5", "v3_anyobs_fail_r5"),
    ):
        table = pd.crosstab(
            frame[left].map(lambda value: "<missing>" if pd.isna(value) else str(int(value))),
            frame[right].map(lambda value: "<missing>" if pd.isna(value) else str(int(value))),
            dropna=False,
        )
        lines.append(f"### {left} × {right}")
        lines.append("")
        lines.append("```")
        lines.append(table.to_string())
        lines.append("```")
        lines.append("")
    lines.extend(["## runner 分布与 unknown 输出", ""])
    lines.append("`unknown_outputs` 只累加有效行。解析失败的行记在 `n_error`，不加进命中数。")
    lines.append("")
    for k in (5, 15):
        lines.append(f"### r{k}")
        lines.append("")
        lines.append("| config | model | runner | hits | n_error |")
        lines.append("|---|---|---|---:|---:|")
        runner_column = f"high_runners_r{k}"
        unknown_column = f"high_unknown_outputs_r{k}"
        for (scaffold, model), group in frame.groupby(["scaffold", "model"], dropna=False, sort=True):
            valid = group[runner_column].notna()
            n_error = int((~valid).sum())
            counts: dict[str, int] = {}
            hits = 0
            for text in group.loc[valid, runner_column].astype(str):
                for runner in text.split("|"):
                    if not runner:
                        continue
                    counts[runner] = counts.get(runner, 0) + 1
                    hits += 1
            unknown = int(group.loc[valid, unknown_column].sum()) if int(valid.sum()) else 0
            for runner, count in sorted(counts.items()):
                lines.append(f"| {scaffold} | {model} | {runner} | {count} | {n_error} |")
            rate = unknown / hits if hits else None
            shown = "" if rate is None else f"{rate:.6g}"
            lines.append(f"| {scaffold} | {model} | unknown_output_rate | {shown} | {n_error} |")
        lines.append("")
    if status == "PARTIAL":
        lines.append("Status remains `PARTIAL`. Do not quote these rates as full-corpus measurements.")
        lines.append("")
    return "\n".join(lines), status


def stratified_sample(frame: pd.DataFrame, limit: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    """Deterministic sample stratified by scaffold and model."""
    rng = np.random.default_rng(seed)
    grouped = list(frame.groupby(["scaffold", "model"], dropna=False, sort=True))
    keys = [key for key, _group in grouped]
    remaining = {key: len(group) for key, group in grouped}
    allocation = {key: 0 for key in keys}
    left = min(limit, int(len(frame)))
    while left > 0 and any(value > 0 for value in remaining.values()):
        progressed = False
        for key in keys:
            if left == 0:
                break
            if remaining[key] <= 0:
                continue
            allocation[key] += 1
            remaining[key] -= 1
            left -= 1
            progressed = True
        if not progressed:
            break
    parts = []
    report = []
    lookup = {key: group for key, group in grouped}
    for key in keys:
        group = lookup[key]
        population = int(len(group))
        take = allocation[key]
        report.append(
            {
                "scaffold": key[0],
                "model": key[1],
                "population": population,
                "sampled": take,
            }
        )
        if take == 0:
            continue
        chosen = rng.choice(group.index.to_numpy(), size=take, replace=False)
        parts.append(group.loc[list(chosen)])
    if not parts:
        return frame.iloc[0:0].copy(), report
    return pd.concat(parts), report


def v3_failure_trigger(calls: list[dict]) -> dict:
    """First high-confidence failed test call inside round <= 5."""
    for call in calls:
        if int(call["round"]) > 5:
            continue
        hit = detect_test_command(call.get("command"))
        if hit is None or hit.confidence != "high":
            continue
        if parse_test_output(call.get("output")).test_failed:
            return {
                "status": "identified",
                "round": call["round"],
                "tool": call.get("tool"),
                "command": call.get("command"),
                "output": call.get("output"),
                "observation_index": call.get("observation_index"),
            }
    return {"status": "unresolved", "round": None, "tool": None, "command": None, "output": None}


def parser_v2_failure_trigger(trajectory, config: str) -> dict:
    """First parser-v2 test-failure observation inside the frozen five-round prefix."""
    parsed = parse_trajectory(trajectory)
    used = min(parsed.n_complete_cycles, 5)
    if used < 1:
        return {"status": "unresolved", "reason": "prefix has no completed round"}
    end = parsed.prefix_message_count(used)
    if end is None:
        return {"status": "unresolved", "reason": "prefix end is missing"}
    calls = iter_calls({"config": config, "trajectory": trajectory})
    by_observation = {
        call["observation_index"]: call for call in calls if call.get("observation_index") is not None
    }
    for message in parsed.messages:
        if message.role != "tool" or message.index >= end:
            continue
        signals = classify_execution_v2(message.content)
        if not signals.explicit_test_failure:
            continue
        call = by_observation.get(message.index)
        return {
            "status": "identified",
            "round": None if call is None else call["round"],
            "tool": None if call is None else call.get("tool"),
            "command": None if call is None else call.get("command"),
            "output": message.content,
            "observation_index": message.index,
            "call_matched": call is not None,
            "reasons": list(signals.reasons),
        }
    return {"status": "unresolved", "reason": "no parser-v2 test failure inside the five-round prefix"}


def excerpt(text: str | None) -> str:
    if not text:
        return ""
    if len(text) <= 3000:
        return text
    omitted = len(text) - 3000
    return text[:1500] + f"\n[...TRUNCATED {omitted} chars...]\n" + text[-1500:]


def render_trigger(trigger: dict) -> list[str]:
    if trigger.get("status") != "identified":
        reason = trigger.get("reason", "triggering observation was not identified")
        return [f"trigger: unresolved ({reason})", ""]
    lines = [
        "trigger: identified",
        f"round: {trigger.get('round')}",
        f"tool: {trigger.get('tool')}",
        f"command: {trigger.get('command')}",
        "```",
        excerpt(trigger.get("output")),
        "```",
        "",
    ]
    return lines
