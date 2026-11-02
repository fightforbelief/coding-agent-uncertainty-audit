"""Write the V3.4.1 common-support audit from the saved V3.4 census.

The saved parquet files are read and not modified. No model is fit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from p8audit.schema import PINNED_REVISION
from v3_4_1_support import (
    characterize,
    decide_status,
    nesting_passes,
    source_problems,
)
from v3_4_lib import sha256_file

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT.parent / "_p8-v3_4-census-staging"
SOURCE_RUN = STAGING / "runs" / "v3_4_census_20261102T153940Z"
CENSUS_PATH = SOURCE_RUN / "census.parquet"
META_PATH = STAGING / "metadata" / "meta.parquet"
UPLOADS = ROOT / "uploads"
REVIEW_PATH = ROOT / "analysis_v2" / "V3_4_1_ISSUE_SUPPORT_REVIEW.md"
INPUTS = {
    "census": CENSUS_PATH,
    "metadata": META_PATH,
    "issue_support_summary": SOURCE_RUN / "issue_support_summary.csv",
    "census_summary": SOURCE_RUN / "census_summary.csv",
    "scaffold_model_summary": SOURCE_RUN / "scaffold_model_summary.csv",
    "run_manifest": SOURCE_RUN / "V3_4_RUN_MANIFEST.json",
    "final_report": SOURCE_RUN / "V3_4_FINAL_REPORT.md",
}
CODE_FILES = (
    ROOT / "analysis_v2" / "v3_4_1_support.py",
    ROOT / "analysis_v2" / "v3_4_1_run.py",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    census = pd.read_parquet(CENSUS_PATH)
    meta = pd.read_parquet(META_PATH)
    return census, meta


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _pytest(run_dir: Path) -> tuple[str, bool]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    plain = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    text = "$ python -m pytest -q\n" + (plain.stdout or "") + (plain.stderr or "")
    if plain.returncode == 0:
        return text, True
    tmp = run_dir / "_pytest_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)
    rerun = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(tmp / "base")],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    text += "\n\n$ python -m pytest -q -p no:cacheprovider --basetemp <run>/_pytest_tmp/base\n"
    text += (rerun.stdout or "") + (rerun.stderr or "")
    shutil.rmtree(tmp, ignore_errors=True)
    return text, rerun.returncode == 0


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator} / {denominator}"
    return f"{numerator:,} / {denominator:,} ({numerator / denominator:.2%})"


def _group_lines(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Group | Trajectories | Known labels | Test run r5 | Test run r15 | Outcome cells | Joint cells r5 | Discordant cells r5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| {group} | {n:,} | {known:,} | {r5} | {r15} | {outcome:,} | {joint:,} | {discord:,} |".format(
                group=row.group,
                n=int(row.n_trajectories),
                known=int(row.n_known_labels),
                r5=_rate(int(row.n_test_run_r5), int(row.n_trajectories)),
                r15=_rate(int(row.n_test_run_r15), int(row.n_trajectories)),
                outcome=int(row.n_outcome_cells),
                joint=int(row.n_joint_cells_r5),
                discord=int(row.n_discordant_cells_r5),
            )
        )
    return lines


def _pair_lines(title: str, summary: dict[str, Any]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        f"- 有 discordant pair-products 的 cells：{summary['cells_any']:,}；对应 issue：{summary['issues_any']:,}。",
        f"- 只有 positive 方向的 cells：{summary['cells_positive_only']:,}（issue {summary['issues_positive_only']:,}）。",
        f"- 只有 negative 方向的 cells：{summary['cells_negative_only']:,}（issue {summary['issues_negative_only']:,}）。",
        f"- 两个方向都有的 cells：{summary['cells_both']:,}（issue {summary['issues_both']:,}）。",
        f"- positive pair-products 合计：{summary['positive_pair_products']:,}。",
        f"- negative pair-products 合计：{summary['negative_pair_products']:,}。",
        "",
        "方向按 cell 计数。一个 issue 可以在不同组里落入不同方向，所以方向上的 issue 数可以重叠。pair-product 是格子内 `a×d` 或 `b×c`，不是独立样本量。",
        "",
        "| Group | Cells | Issues | Positive only | Negative only | Both | Positive products | Negative products |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_group"]:
        lines.append(
            "| {label} | {cells:,} | {issues:,} | {positive:,} | {negative:,} | {both:,} | {pp:,} | {np:,} |".format(
                label=row["label"],
                cells=row["cells_any"],
                issues=row["issues_any"],
                positive=row["cells_positive_only"],
                negative=row["cells_negative_only"],
                both=row["cells_both"],
                pp=row["positive_pair_products"],
                np=row["negative_pair_products"],
            )
        )
    return lines


def _behavior_lines(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Variable | Positive trajectories | Valid negative trajectories | Missing trajectories | Variation issues | Variation cells | Issue-level joint issues | Same-cell joint issues | Same-cell joint cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| {name} | {pos:,} | {neg:,} | {missing:,} | {issues:,} | {cells:,} | {joint:,} | {same:,} | {same_cells:,} |".format(
                name=row.variable,
                pos=int(row.n_positive_trajectories),
                neg=int(row.n_valid_negative_trajectories),
                missing=int(row.n_missing_trajectories),
                issues=int(row.n_issues_with_variation),
                cells=int(row.n_cells_with_variation),
                joint=int(row.n_issues_with_variation_and_outcome_variation),
                same=int(row.n_issues_with_same_cell_joint_variation),
                same_cells=int(row.n_cells_with_joint_variation),
            )
        )
    return lines


def render_summary(result: dict[str, Any], *, status: str, run_dir: Path, tests_ok: bool) -> str:
    head = result["headline"]
    cross = result["cross"]
    nesting = result["nesting"]
    published = result["alignment"]["published"]
    stats = result["alignment"]["issue_support"]
    groups = result["by_group"]
    joint_cells = int(groups["n_joint_cells_r5"].sum())
    if joint_cells:
        top = groups.loc[groups["n_joint_cells_r5"].idxmax()]
        dominance = f"{top['group']} 占 r5 joint cells 的 {int(top['n_joint_cells_r5']):,} / {joint_cells:,} ({int(top['n_joint_cells_r5']) / joint_cells:.1%})。"
    else:
        dominance = "r5 joint cells 为 0。"
    zero_joint = [str(name) for name in groups.loc[groups["n_joint_cells_r5"] == 0, "group"]]
    zero_discord = [str(name) for name in groups.loc[groups["n_discordant_cells_r5"] == 0, "group"]]
    round_text = "、".join(f"{rounds} 轮 {count} 条" for rounds, count in nesting["round_counts"].items()) or "无"
    published_text = "、".join(
        f"{key} 期望 {item['expected']}、复算 {item['observed']}"
        for key, item in published.items()
        if isinstance(item, dict) and "observed" in item and not str(key).startswith("bucket_")
    )
    lines = [
        "# V3.4.1 Issue × Scaffold/Model Common-Support Audit",
        "",
        f"状态：`{status}`",
        "",
        "这是描述性支持度审计。它统计同一个 issue、同一个 scaffold/model 里是否同时看得到测试行为差异和已知结局差异。它不估计关联，也不表示预测价值。",
        "",
        "## 数据",
        "",
        f"- Census：`{CENSUS_PATH}`",
        f"- Metadata：`{META_PATH}`",
        f"- 来源运行：`{SOURCE_RUN}`",
        f"- 本次运行：`{run_dir}`",
        f"- 数据 revision：`{PINNED_REVISION}`",
        f"- 轨迹 {head['trajectories']:,}；issue {head['issues']:,}；已知标签 {head['known_labels']:,}；unknown 标签 {head['unknown_labels']:,}；非法标签 {head['invalid_labels']:,}。",
        f"- 观察到的 issue × scaffold × model cells：{head['cells']:,}。其中至少两条轨迹的 cells：{head['cells_with_two_or_more']:,}。",
        "",
        "暴露的定义是高置信度 `v3_ran_tests`。明确的未运行是该标记为 0，并且窗口内没有缺失命令。缺失命令单独计数，不进入四格表的未运行格。`resolved == -1` 不进入结局差异，也不进入四格表。",
        "",
        "## 与 V3.4 计数的核对",
        "",
        f"在同一张 census 上重新调用 V3.4 的 `issue_support`：{published_text}。",
        "",
        f"内部三套实现是否一致：`{result['alignment']['internal_match']}`。与已保存的 V3.4 汇总是否一致：`{result['alignment']['published_match']}`。测试是否通过：`{tests_ok}`。",
        "",
        f"V3.4 的 734 是 loose within-group flag variation，也就是同组里标记同时出现 0 和 1；缺失命令上的 0 仍留在这套计数里。严格的同组行为差异要求同组里既有高置信度运行，也有明确未运行。严格的 issue 数是 {cross['within_group_strict_behavior_issues']:,}，cells 是 {cross['within_group_strict_behavior_cells']:,}。loose cells 是 {cross['within_group_behavior_cells']:,}。",
        "",
        "## Issue 层",
        "",
        f"- 严格测试行为差异，前 5 轮：{head['behavior_issues_r5']:,}。前 15 轮：{head['behavior_issues_r15']:,}。",
        f"- 已知结局差异：{head['outcome_issues']:,}。V3.4 loose 口径下的结局差异是 {stats['issues_with_outcome_variation']:,}。",
        f"- 前 5 轮严格行为差异且有结局差异：{head['both_issues_r5']:,}。前 15 轮：{head['both_issues_r15']:,}。",
        f"- 前 5 轮只有严格行为差异：{head['behavior_only_r5']:,}。只有结局差异：{head['outcome_only']:,}。两者都没有：{head['neither_r5']:,}。",
        f"- V3.4 loose 行为差异的 issue 是 {stats['issues_with_r5_test_flag_variation']:,}。其中至少有一个同组 loose 行为差异的 issue 是 {cross['issue_behavior_with_within_group_behavior_issues']:,}。只在不同 scaffold/model 之间出现行为差异、没有任何同组行为差异的 issue 是 {cross['issue_behavior_without_within_group_behavior_issues']:,}。",
        "",
        "## 同一个 scaffold/model 内",
        "",
        f"- 历史 330 个同时有 loose 行为差异和结局差异的 issue 中，至少有一个严格同组 joint cell 的 issue：{cross['both_with_within_group_joint_issues']:,}。这些 issue 上的严格 joint cells：{cross['both_with_within_group_joint_cells']:,}。",
        f"- 同一批 330 个 issue 中，至少有一个 loose 同组 joint cell 的 issue：{cross['both_with_loose_within_group_joint_issues']:,}；对应 cells：{cross['both_with_loose_within_group_joint_cells']:,}。",
        f"- 落在这 330 个 issue 之外的严格 joint issue：{cross['joint_issues_outside_published_both']:,}。",
        f"- 734 个 loose 同组行为差异 issue 中，至少有一个同组结局差异 cell 的 issue：{cross['within_734_with_within_group_outcome_issues']:,}。没有同组结局差异的 issue：{cross['within_734_without_within_group_outcome_issues']:,}。",
        f"- 这 734 个 issue 中，行为差异和结局差异出现在同一个 cell 的 issue：{cross['within_734_with_same_cell_joint_issues']:,}。",
        f"- 有同组结局差异、但没有 loose 同组行为差异的 issue：{cross['within_group_outcome_without_within_group_behavior_issues']:,}。",
        f"- 同组结局差异 cells：{cross['within_group_outcome_cells']:,}，涉及 issue {cross['within_group_outcome_issues']:,}。734 个 issue 内部的同组结局差异 cells：{cross['within_group_outcome_cells_among_734']:,}。",
        f"- 严格同组行为差异且没有同组结局差异的 issue：{cross['strict_behavior_without_within_group_outcome_issues']:,}。有同组结局差异但没有严格同组行为差异的 issue：{cross['within_group_outcome_without_strict_behavior_issues']:,}。",
        "",
        "## 四组支持度",
        "",
        *_group_lines(groups),
        "",
        dominance,
        "",
        "r5 同组 joint cells 为 0 的组：" + ("、".join(zero_joint) if zero_joint else "没有。"),
        "",
        "r5 discordant cells 为 0 的组：" + ("、".join(zero_discord) if zero_discord else "没有。"),
        "",
        "各组 missing-command 轨迹占该组轨迹的比例：",
        "",
    ]
    for row in groups.itertuples(index=False):
        lines.append(
            f"- {row.group}：r5 {_rate(int(row.n_missing_command_r5), int(row.n_trajectories))}；r15 {_rate(int(row.n_missing_command_r15), int(row.n_trajectories))}。"
        )
    lines.extend(["", "## Discordant pair-products", ""])
    lines.extend(_pair_lines("前 5 轮", result["pairs_r5"]))
    lines.extend(["", *_pair_lines("前 15 轮", result["pairs_r15"]), ""])
    lines.extend(
        [
            "## 不同行为变量",
            "",
            "测试执行的有效阴性是明确未检测到高置信度测试命令。测试失败和 parser v2 的有效阴性是标记本身为 0。`v3_test_fail = 0` 包含没有高置信度测试命令的轨迹，所以它和 `v3_ran_tests` 是两个变量。Issue-level joint 表示这个 issue 同时有该变量的差异和已知结局差异。Same-cell joint 表示这两件事出现在同一个 scaffold/model cell。",
            "",
            *_behavior_lines(result["by_behavior"]),
            "",
            "## 短于 15 个完成轮次的轨迹",
            "",
            f"- `prefix15_status = no_prefix_boundary`：{nesting['n_no_prefix_boundary']:,}。",
            f"- 完成轮次少于 15：{nesting['n_rounds_below_15']:,}。两者不一致的短轨迹：边界标记但轮次不少于 15 的有 {nesting['n_short_with_rounds_ge_15']:,}，轮次少于 15 但没有边界标记的有 {nesting['n_below_15_without_boundary_flag']:,}。",
            f"- 前 15 轮前缀 mismatch：{nesting['n_prefix15_mismatch']:,}。其他前缀状态：{nesting['n_other_prefix15_status']:,}。",
            f"- 历史完成轮次与本次观察完成轮次不一致：{nesting['n_round_source_mismatch']:,}。",
            f"- 短轨迹的完成轮次分布：{round_text}。",
            f"- 短轨迹的 r15 暴露：运行 {nesting['n_short_r15_run']:,}，明确未运行 {nesting['n_short_r15_known_nonrun']:,}，缺失命令 {nesting['n_short_r15_missing_command']:,}，r15 标记缺失 {nesting['n_short_r15_missing']:,}。",
            f"- 短轨迹里 r5 为 0 且 r15 为 1：{nesting['n_short_r15_added_after_r5']:,}。这表示第 6 轮及以后、仍在该轨迹已完成轮次内的命令进入了 r15。",
            f"- r15 阳性少于 r5 阳性的轨迹：{nesting['n_ran_not_nested']:,}。测试调用次数或执行调用次数违反嵌套的轨迹：{nesting['n_test_calls_not_nested']:,}、{nesting['n_exec_calls_not_nested']:,}。",
            f"- 完成轮次不超过 5 的短轨迹有 {nesting['n_short_rounds_le_5']:,}。其中 r5 与 r15 的运行标记、测试调用次数、执行调用数或缺失命令数不一致的有 {nesting['n_short_rounds_le_5_window_differs']:,}。",
            "",
            "窗口规则是 `round <= 15`。轨迹的完成轮次少于 15 时，`prefix_message_count(15)` 没有边界，所以前缀状态是 `no_prefix_boundary`；已经完成的每一轮仍然满足 `round <= 15`。这些轨迹的 r15 标记因此使用了全部可用完成轮次。完成轮次不超过 5 时，r5 与 r15 覆盖同一批调用。这个审计没有改 r15 的定义。",
            "",
            f"嵌套和 50 条短轨迹核对是否通过：`{nesting_passes(nesting)}`。",
            "",
            "## 判断",
            "",
            f"1. 330 个 issue 里，同一个 scaffold/model 内同时有严格测试行为差异和已知结局差异的有 {cross['both_with_within_group_joint_issues']:,} 个 issue，对应 {cross['both_with_within_group_joint_cells']:,} 个 cells。",
            f"2. 734 个同组行为差异 issue 里，{cross['within_734_with_within_group_outcome_issues']:,} 个至少还有一个同组结局差异 cell；其中 {cross['within_734_with_same_cell_joint_issues']:,} 个是行为差异和结局差异出现在同一个 cell。",
            "3. 四组是否都有同组对照，直接看上面的 joint cells 和 discordant cells。为 0 的组没有该组内部的有效四格对照。",
            f"4. 前 5 轮有信息量的 discordant cells 是 {result['pairs_r5']['cells_any']:,}，涉及 {result['pairs_r5']['issues_any']:,} 个 issue。positive pair-products 是 {result['pairs_r5']['positive_pair_products']:,}，negative pair-products 是 {result['pairs_r5']['negative_pair_products']:,}。前 15 轮对应的 cells 是 {result['pairs_r15']['cells_any']:,}，issue 是 {result['pairs_r15']['issues_any']:,}，两个方向的 pair-products 是 {result['pairs_r15']['positive_pair_products']:,} 和 {result['pairs_r15']['negative_pair_products']:,}。",
            f"5. {dominance} 这是支持度的集中，不是模型效应。",
            "6. 短轨迹的 r15 使用了全部已完成轮次，没有因为长度不足而被写成缺失，也没有把 r5 已经阳性的短轨迹在 r15 改成阴性。"
            if nesting_passes(nesting)
            else "6. 短轨迹核对没有完全通过。上面的嵌套计数给出了不一致的位置；r15 定义没有被改写。",
            "7. 同组 joint cells 划出了后续可以设计人工阅读或 issue 内比较的集合。四组的格子数量需要按上表分别看，不能把支持度薄的组和格子多的组写成同样的共同支持。当前结果到设计讨论为止。",
            "",
            "## 没有做的事",
            "",
            "- 没有重新下载数据，没有重跑全库普查，没有改 V3.4 输出。",
            "- 没有修改 parser v2、V3 detector 或 L2。",
            "- 没有拟合回归、混合模型或预测模型，没有计算 AUPRC、ROC-AUC 或 Delta AP。",
            "- 没有开始人工标注，没有启动 L3。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _publish(summary: str) -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS / "V3_4_1_COMMON_SUPPORT_FOR_REVIEW.md"
    destination.write_text(summary, encoding="utf-8")
    for path in UPLOADS.glob("*.md"):
        if path.name != destination.name:
            path.unlink()
    if REVIEW_PATH.exists():
        stamped = REVIEW_PATH.with_name(REVIEW_PATH.stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + REVIEW_PATH.suffix)
        stamped.write_text(summary, encoding="utf-8")
    else:
        REVIEW_PATH.write_text(summary, encoding="utf-8")


def _hash_outputs(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name == "V3_4_1_SHA256_MANIFEST.csv":
            continue
        rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(rows).to_csv(run_dir / "V3_4_1_SHA256_MANIFEST.csv", index=False)


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = STAGING / "runs" / f"v3_4_1_issue_support_{stamp}"
    run_dir.mkdir(parents=True)
    problems: list[str] = []
    result: dict[str, Any] | None = None
    for label, path in INPUTS.items():
        if not path.is_file():
            problems.append(f"missing {label}: {path}")
    if not problems:
        census, meta = _read_inputs()
        problems.extend(source_problems(census, meta))
        if not problems:
            result = characterize(census)
            if result["headline"]["unexpected_groups"]:
                problems.append("unexpected scaffold/model values: " + json.dumps(result["headline"]["unexpected_groups"]))
    tests_text, tests_ok = _pytest(run_dir)
    (run_dir / "V3_4_1_TEST_RESULTS.txt").write_text(tests_text, encoding="utf-8")
    source_ok = result is not None and not problems
    if result is None:
        status = decide_status(source_ok=False, counts_complete=False, alignment_ok=False, nesting_ok=False, tests_ok=tests_ok)
        summary = "\n".join(
            [
                "# V3.4.1 Issue × Scaffold/Model Common-Support Audit",
                "",
                f"状态：`{status}`",
                "",
                "来源或连接检查没有通过，支持度计数没有当作完整结果写出。",
                "",
                *[f"- {item}" for item in problems],
                "",
            ]
        )
    else:
        nesting_ok = nesting_passes(result["nesting"])
        alignment_ok = bool(result["alignment"]["internal_match"] and result["alignment"]["published_match"])
        counts_complete = result["headline"]["trajectories"] == 207489 and result["headline"]["issues"] == 22320
        status = decide_status(
            source_ok=source_ok,
            counts_complete=counts_complete,
            alignment_ok=alignment_ok,
            nesting_ok=nesting_ok,
            tests_ok=tests_ok,
        )
        _write_csv(result["issues"], run_dir / "issue_level_support.csv")
        _write_csv(result["cells"], run_dir / "issue_group_support.csv")
        _write_csv(result["four_cell"], run_dir / "issue_group_four_cell_counts.csv")
        _write_csv(result["discordant"], run_dir / "issue_group_discordant_pairs.csv")
        _write_csv(result["by_group"], run_dir / "support_by_scaffold_model.csv")
        _write_csv(result["by_behavior"], run_dir / "support_by_behavior_definition.csv")
        _write_csv(result["short_rows"], run_dir / "short_trajectory_r15_audit.csv")
        summary = render_summary(result, status=status, run_dir=run_dir, tests_ok=tests_ok)
    (run_dir / "V3_4_1_SUMMARY.md").write_text(summary, encoding="utf-8")
    manifest = {
        "status": status,
        "created_utc": stamp,
        "dataset": "nvidia/Open-SWE-Traces",
        "revision": PINNED_REVISION,
        "source_run": str(SOURCE_RUN),
        "problems": problems,
        "tests_passed": tests_ok,
        "parameters": {
            "exposure": "v3_ran_tests high-confidence command",
            "known_nonrun": "flag 0 and exec_missing 0",
            "missing_command_excluded_from_known_nonrun": True,
            "unknown_label_excluded_from_outcome": True,
            "discordant_pairs": "a*d and b*c within an issue-scaffold-model cell",
            "models_fit": False,
            "census_rerun": False,
        },
        "input_sha256": {label: sha256_file(path) for label, path in INPUTS.items() if path.is_file()},
        "code_sha256": {path.name: sha256_file(path) for path in CODE_FILES},
    }
    if result is not None:
        manifest["headline"] = result["headline"]
        manifest["cross"] = result["cross"]
        manifest["nesting"] = result["nesting"]
        manifest["alignment"] = {
            "internal_match": result["alignment"]["internal_match"],
            "published_match": result["alignment"]["published_match"],
            "comparisons": result["alignment"]["comparisons"],
            "published": result["alignment"]["published"],
        }
        manifest["pairs_r5"] = result["pairs_r5"]
        manifest["pairs_r15"] = result["pairs_r15"]
    (run_dir / "V3_4_1_RUN_MANIFEST.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _hash_outputs(run_dir)
    _publish(summary)
    print(status)
    print(run_dir)


if __name__ == "__main__":
    main()
