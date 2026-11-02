"""Sampling-aware descriptive report for LOCKED automated annotations.

These are distributions of an automatic annotation system, not gold-standard
performance estimates. No outcomes, predictive models, or significance tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STATUSES = ("definite_test_command", "definite_no_test_command", "ambiguous", "command_unrecoverable", "insufficient_evidence")
STRATA = ("A_r5_positive", "B_r15_positive_r5_known_negative", "C_r15_known_negative", "D_r15_missing_command", "AMB_r5", "AMB_r15")
FRAMES = {
    "R5_detector_positive_frame": ("r5", (STRATA[0],)),
    "R15_detector_positive_frame": ("r15", STRATA[:2]),
    "R15_detector_known_negative_frame": ("r15", (STRATA[2],)),
    "R15_missing_command_diagnostic": ("r15", (STRATA[3],)),
    "R5_ambiguous_command_diagnostic": ("r5", (STRATA[4],)),
    "R15_ambiguous_command_diagnostic": ("r15", (STRATA[5],)),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_lock(run: Path) -> dict:
    lock = json.loads((run / "ANNOTATION_LOCK.json").read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        if Path(name).name != name or sha(run / name) != expected:
            raise ValueError("Locked annotation file changed: " + name)
    if not lock["label_lock_precedes_sampling_key_join"]:
        raise ValueError("Annotations must be locked before key join")
    return lock


def join_memberships(labels: pd.DataFrame, key: pd.DataFrame, design: pd.DataFrame) -> pd.DataFrame:
    if key.review_id.duplicated().any() or labels.duplicated(["review_id", "window"]).any():
        raise ValueError("Duplicate key or annotation")
    if set(labels.review_id) != set(key.review_id) or len(labels) != 2 * len(key):
        raise ValueError("Missing or extra annotation IDs")
    if not set(labels.test_command_status).issubset(STATUSES):
        raise ValueError("Unknown label")
    for _, group in labels.groupby("review_id"):
        if set(group.window) != {"r5", "r15"}:
            raise ValueError("Each trajectory needs both windows")
    # Do not import resolved or any detector outcome into the analysis table.
    identity = ["review_id", "issue_id", "scaffold", "model"]
    parts = []
    seen_design = set()
    if design.duplicated(["stratum", "scaffold", "model"]).any():
        raise ValueError("Duplicate design cells")
    for cell in design.itertuples(index=False):
        if cell.stratum not in STRATA:
            raise ValueError("Unexpected design stratum")
        seen_design.add((cell.stratum, cell.scaffold, cell.model))
        member = key[f"member_{cell.stratum}"]
        if not member.isin([0, 1]).all():
            raise ValueError("Non-binary membership")
        chosen = key[(member == 1) & (key.scaffold == cell.scaffold) & (key.model == cell.model)]
        N, n = int(cell.population_size), int(cell.sample_size)
        if len(chosen) != n or n > N or (N > 0 and n == 0):
            raise ValueError("Design/sample count mismatch")
        if N == 0:
            if n != 0 or not pd.isna(cell.inclusion_probability):
                raise ValueError("Empty population needs missing inclusion probability")
            continue
        expected_pi = n / N
        probabilities = chosen[f"pi_{cell.stratum}"].astype(float)
        if not np.allclose(probabilities, expected_pi, rtol=1e-12, atol=1e-15) or not math.isclose(float(cell.inclusion_probability), expected_pi, rel_tol=1e-12):
            raise ValueError("Wrong stratum-specific inclusion probability")
        part = chosen[identity].copy()
        part["stratum"] = cell.stratum
        part["weight"] = 1 / probabilities.to_numpy()
        part["population_size"] = N
        part["sample_size"] = n
        parts.append(part)
    expected_cells = {(s, a, m) for s in STRATA for a, m in key[["scaffold", "model"]].drop_duplicates().itertuples(index=False, name=None)}
    if seen_design != expected_cells:
        raise ValueError("Design lacks a stratum/group cell")
    membership = pd.concat(parts, ignore_index=True)
    # Every declared membership must have survived the design join exactly once.
    expected_memberships = sum(int(key[f"member_{s}"].sum()) for s in STRATA)
    if len(membership) != expected_memberships:
        raise ValueError("Lost membership in design join")
    primary = key[[f"member_{s}" for s in STRATA[:4]]].sum(axis=1)
    if primary.gt(1).any():
        raise ValueError("Primary strata must be mutually exclusive")
    return membership.merge(labels, on="review_id", how="inner", validate="many_to_many")


def weighted_summary(data: pd.DataFrame) -> dict:
    counts = {s: float(data.loc[data.test_command_status == s, "weight"].sum()) for s in STATUSES}
    positive, negative = counts[STATUSES[0]], counts[STATUSES[1]]
    total = float(data.weight.sum())
    uncertain = sum(counts[s] for s in STATUSES[2:])
    if total <= 0:
        raise ValueError("Cannot summarise an empty frame")
    return dict(sample_memberships=len(data), unique_sample_trajectories=data.review_id.nunique(),
                sample_issues=data.issue_id.nunique(), repeated_sample_issues=int(data.groupby("issue_id").review_id.nunique().ge(2).sum()),
                population_weight_sum=total, weighted_positive=positive, weighted_negative=negative, weighted_uncertain=uncertain,
                positive_share_all=positive / total, negative_share_all=negative / total, uncertain_share=uncertain / total,
                positive_share_determinate=positive / (positive + negative) if positive + negative else None,
                ambiguity_lower=positive / total, ambiguity_upper=(positive + uncertain) / total,
                **{f"share_{s}": counts[s] / total for s in STATUSES})


def tables(joined: pd.DataFrame):
    summaries, distributions = [], []
    for frame, (window, strata) in FRAMES.items():
        data = joined[(joined.window == window) & joined.stratum.isin(strata)]
        groups = [("all", "all", data)]
        groups.extend((a, m, part) for (a, m), part in data.groupby(["scaffold", "model"], sort=True))
        for scaffold, model, part in groups:
            info = dict(frame=frame, window=window, scaffold=scaffold, model=model)
            summaries.append(dict(**info, **weighted_summary(part)))
            for field in ("test_command_status", "execution_status", "process_status", "execution_detail", "test_output_status", "output_composition"):
                total = float(part.weight.sum())
                for value, subset in part.groupby(field, dropna=False):
                    distributions.append(dict(**info, field=field, value=value, sample_count=len(subset), weighted_count=float(subset.weight.sum()), weighted_share=float(subset.weight.sum()) / total))
    return pd.DataFrame(summaries), pd.DataFrame(distributions)


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    return "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n" + "\n".join("| " + " | ".join(fmt(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))


def plot_summary(pooled: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {
        "R5_detector_positive_frame": "R5 detector-positive frame",
        "R15_detector_positive_frame": "R15 detector-positive frame",
        "R15_detector_known_negative_frame": "R15 known-negative frame",
        "R15_missing_command_diagnostic": "R15 missing-command diagnostic",
        "R5_ambiguous_command_diagnostic": "R5 ambiguous-command diagnostic",
        "R15_ambiguous_command_diagnostic": "R15 ambiguous-command diagnostic",
    }
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    left = np.zeros(len(pooled))
    y = np.arange(len(pooled))
    for column, label, color in (("positive_share_all", "Automatic positive", "#287c8e"),
                                 ("negative_share_all", "Automatic negative", "#a7c7c0"),
                                 ("uncertain_share", "Abstention", "#efb655")):
        values = pooled[column].to_numpy() * 100
        ax.barh(y, values, left=left, label=label, color=color, height=.6)
        for index, value in enumerate(values):
            if value >= 5:
                ax.text(left[index] + value / 2, index, f"{value:.1f}%", ha="center", va="center", fontsize=9)
        left += values
    ax.set_yticks(y, [names[x] for x in pooled.frame])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Stratum-weighted share within each frame (%)")
    ax.set_title("Automated evidence annotations: positive, negative and abstained", loc="left", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.27), ncol=3, frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Not human-validated accuracy. Abstention is not counted as a negative.", x=.5, y=1.04, fontsize=10)
    fig.savefig(output / "weighted_annotation_shares.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "weighted_annotation_shares.svg", bbox_inches="tight")
    plt.close(fig)


def run(annotation_run: Path, key_path: Path, design_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Choose a new report directory")
    lock = verify_lock(annotation_run)
    labels = pd.read_csv(annotation_run / "annotations_long.csv", keep_default_na=False)
    if not labels.independent_human_review.astype(str).str.lower().eq("false").all():
        raise ValueError("This report accepts only explicitly non-human labels")
    key = pd.read_csv(key_path)
    design = pd.read_csv(design_path)
    joined = join_memberships(labels, key, design)
    estimates, distributions = tables(joined)
    output.mkdir(parents=True)
    estimates.to_csv(output / "weighted_frame_summaries.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(output / "weighted_status_distributions.csv", index=False, encoding="utf-8-sig")
    # Keep raw memberships so overlapping samples can be traced without combining pi.
    joined.to_csv(output / "analysis_memberships.csv", index=False, encoding="utf-8-sig")
    overlap = key[["review_id"] + [f"member_{s}" for s in STRATA]].copy()
    overlap = overlap[overlap.iloc[:, 1:].sum(axis=1).gt(1)]
    overlap.to_csv(output / "cross_stratum_sample_memberships.csv", index=False, encoding="utf-8-sig")
    pooled = estimates[estimates.scaffold == "all"]
    plot_summary(pooled, output)
    brief = pooled[["frame", "sample_memberships", "population_weight_sum", "positive_share_all", "uncertain_share", "positive_share_determinate", "ambiguity_lower", "ambiguity_upper"]]
    expected_population = {"R5_detector_positive_frame": 1304, "R15_detector_positive_frame": 44480, "R15_detector_known_negative_frame": 162514, "R15_missing_command_diagnostic": 495, "R5_ambiguous_command_diagnostic": 619, "R15_ambiguous_command_diagnostic": 9975}
    for row in pooled.itertuples(index=False):
        if not math.isclose(row.population_weight_sum, expected_population[row.frame], abs_tol=1e-7):
            raise ValueError("Weighted population total differs from frozen design")
    summary = dict(status="AUTOMATED DESCRIPTIVE AUDIT COMPLETE; INDEPENDENT MEASUREMENT VALIDITY NOT ESTABLISHED",
                   annotation_run=str(annotation_run.resolve()), report_run=str(output.resolve()),
                   trajectories=int(key.review_id.nunique()), windows=len(labels), issues=int(key.issue_id.nunique()),
                   issues_with_repeated_sample_trajectories=int(key.groupby("issue_id").size().ge(2).sum()),
                   stratum_memberships=len(joined) // 2, cross_stratum_sample_trajectories=len(overlap),
                   abstention_windows=lock["abstention_windows"], semantic_spot_review_windows=lock["semantic_spot_review_windows"],
                   independent_human_reviews=0, confidence_intervals_computed=False, outcomes_used=False,
                   new_predictive_or_association_models_run=False, gold_standard_metrics_computed=False,
                   annotation_lock_sha256=sha(annotation_run / "ANNOTATION_LOCK.json"), sampling_key_sha256=sha(key_path),
                   sampling_design_sha256=sha(design_path), report_code_sha256=sha(Path(__file__)),
                   pooled_summaries=json.loads(pooled.to_json(orient="records")))
    (output / "RESULTS.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = f"""# P8 V3 自动证据标注与探索性审计报告

日期：2026-11-03（Asia/Shanghai）。状态：自动标注与统计交付完成；独立测量有效性未确立。

## 结论

本轮处理全部 {len(labels)//2:,} 条抽中轨迹的 R5/R15，共 {len(labels):,} 个窗口。全部原始证据由程序读取并按静态规则分类；只有 {lock['semantic_spot_review_windows']} 个窗口经过明确记录的 AI 语义抽查，不声称逐条 LLM 语义复核。{lock['abstention_windows']} 个窗口保留弃判。实际人类复核人数为 0，没有双人一致性或人工金标准。

这些结果是两套自动规则的探索性对照。它们不能证明冻结检测器正确率、实际测试行为发生率、预测增量或因果关系。独立测量质量仍未确立，因此不启动 V3.6/L3。既有论文及原确认性结果保留原结论。

## 数据、标注与权重

原始数据 revision 为 `c144d9d160dcd9f5ebcd9520d2d54111dda8b681`；沿用 V3.5 的固定样本，不重抽。抽样包涉及 {key.issue_id.nunique():,} 个 issue，其中 {summary['issues_with_repeated_sample_trajectories']} 个 issue 有至少两条抽中轨迹。轨迹不能视为独立二项试验。

每个窗口附执行调用分类、引用索引、原文或弃判理由。先锁定标签 SHA-256，再合并密封键。AI 抽查与自动规则有不同来源。脚本不会执行证据中的命令。标注是事后探索性的，不是确认性测量验证。

每项估计只使用其对应抽样层，权重为 N_h/n_h。R15 阳性框使用互斥 A+B；R5 使用 A；C、D、AMB_r5、AMB_r15 分开。仅有 {len(overlap)} 条轨迹跨抽中层，其成员身份和权重各自保留，不合并概率。D 的 SWE-agent/Minimax 总体为 0，保留空组而不补一个概率。

## 自动标签的分层加权汇总

所有比例的分母、标签来源均为本报告所述自动标注。`positive_share_determinate` 只覆盖自动标签可判定的部分，不是人类验证后的 precision。`ambiguity_lower/upper` 仅把弃判全部视作阴性/阳性，**不是置信区间，也不覆盖自动规则错误**。富集样本的未加权阳性比例不能解释为全库发生率。

{md_table(brief)}

![自动标签的分层加权份额，非人工验证准确率](weighted_annotation_shares.png)

## 四组分别报告

{md_table(estimates[estimates.scaffold != 'all'][['frame', 'scaffold', 'model', 'sample_memberships', 'positive_share_all', 'uncertain_share', 'positive_share_determinate']])}

## 执行与输出

命令意图、进程启动、测试是否执行及测试结果分别保留。输出表在 `weighted_status_distributions.csv`；该表按框、组和字段逐项列出，不把所有字段混成一个准确率。明确测试命令可以因依赖、收集、构建或缺失运行器而没有执行测试。空命令只有能够关联同工具的在运行命令时才作为轮询；无法关联就弃判。

直接运行新写的复现脚本不因文件名为 test_ 就算项目测试。显示或搜索测试代码不是执行。Vitest 的失败文件且 `Tests no tests` 不算测试用例失败。Jest 覆盖率警告不改写已观察到的测试用例通过结果。静态解析仍可能漏掉动态脚本、复杂构建目标、安装 hooks 和输出中的引用文本；这些是自动审计的限制。

## 对原研究的影响

V3.4 全库普查中，冻结检测器的 R5 测试命令阳性为 1,304/207,489（0.6285%），R15 为 44,480/207,489（21.4373%）；这些是检测器发生率。V3.4.1 显示 R5 只有 98 个 issue、101 个同 scaffold/model 的 joint cells，且 SWE-agent/Qwen 为 0；R15 有 1,849 个 discordant cells，涉及 1,694 个 issue。共同支持存在不代表测量准确或关联成立。

原确认性研究的 ΔAP 为 0.00006589172683257871，历史 repository-cluster bootstrap 95% 区间为 [0.00000560783009556365, 0.00013282286584345433]。本轮未重新评分测试集；历史结果仅从冻结 README/报告引用。绝对增量非常小，实际用途未确立。自动标注不能为其补上独立人工核验，也没有产生新的预测结论。

## 方案偏离与限制

原 V3.5.1 计划要求两人独立首轮与第三方裁决。本次按用户要求由助手推进，改为自动静态证据标注，加有限 AI 语义抽查，来源已明确。规则从此前预标注和案例观察中改进，存在适应样本的风险，不视为独立留出验证。

没有人类金标准，不计算人工 PPV/recall、双人一致性、因果效应、显著性检验或新的预测模型。没有选定人工测量统计的 issue 聚类区间方法，所以本交付不提供抽样置信区间。未弃判的自动标签仍可能错；弃判范围不是整体误差上界。

原人类复核包保持空白，冻结 detector、parser v2、L2、旧论文和数据 revision 不改动。本交付可独立复现；若未来要恢复独立人工验证，应使用原盲化材料，不能把本次自动标签重命名为人工标签。

## 文件与复现

- `annotations_long.csv` / `annotations_wide.csv`：全量标注，见相邻标注运行目录。
- `abstentions.csv`：最终弃判清单。弃判是明确的结论类别，不是漏处理。
- `call_annotations.jsonl`：每个执行调用的分类与证据哈希。
- `ANNOTATION_LOCK.json`：标签冻结记录，先于密封键合并。
- `weighted_frame_summaries.csv`：主框及诊断框的加权标签汇总。
- `weighted_status_distributions.csv`：命令、进程、执行、输出的分开分布。
- `analysis_memberships.csv`：对应各层的分析成员与权重，不含 resolved。
- `cross_stratum_sample_memberships.csv`：跨层样本检查。
- `RESULTS.json`：机器可读结果与来源 SHA-256。

代码：`analysis_v2/v3_5_complete_annotations.py`、`analysis_v2/v3_5_ai_audit_report.py`。方案：`analysis_v2/PROTOCOL_AI_AUDIT_20261103.md`。完整复现命令和验收结果在最终交付清单。
"""
    (output / "FINAL_REPORT.md").write_text(text, encoding="utf-8")
    # Machine-generated table for paper workflows; no manual numeric transcription.
    columns = ["frame", "positive_share_all", "uncertain_share", "ambiguity_lower", "ambiguity_upper"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Frame & Positive share & Abstention & Lower & Upper \\", r"\midrule"]
    for row in pooled[columns].itertuples(index=False, name=None):
        name = row[0].replace("_", r"\_")
        lines.append(name + " & " + " & ".join(f"{x:.4f}" for x in row[1:]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "REPORT_MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in output.iterdir() if p.is_file()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.annotations, args.key, args.design, args.output), ensure_ascii=False, indent=2, allow_nan=False))
