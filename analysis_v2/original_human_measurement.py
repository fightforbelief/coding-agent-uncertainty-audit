"""V3.5.1 human measurement point estimates AFTER locked human adjudication.

Refuses the earlier AI-labelled datasets. The frozen plan did not choose an
issue-cluster interval procedure, so this entry point does not invent one.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from v3_5_ai_audit_report import join_memberships, md_table, tables


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(final_labels: Path, key: Path, design: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a fresh human measurement output directory")
    if not (final_labels / "FINAL_HUMAN_LABEL_LOCK.json").is_file():
        raise ValueError("Missing final human label lock; independent first rounds and adjudication have not been supplied")
    lock = json.loads((final_labels / "FINAL_HUMAN_LABEL_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("source_type") != "declared_human_review_not_identity_verified_by_tool":
        raise ValueError("Actual locked human first rounds and adjudication are required")
    for name, expected in lock["files"].items():
        if Path(name).name != name or sha(final_labels / name) != expected:
            raise ValueError("Final human label source has changed")
    labels = pd.read_csv(final_labels / "final_human_annotations_long.csv", keep_default_na=False)
    if not labels.independent_human_review.astype(str).str.lower().eq("true").all() or not labels.annotation_source.eq("two_declared_independent_human_first_rounds_and_adjudication").all():
        raise ValueError("AI proposals cannot be relabelled as human data")
    memberships = join_memberships(labels, pd.read_csv(key), pd.read_csv(design))
    summaries, distributions = tables(memberships)
    summaries["human_ppv_determinate"] = summaries.positive_share_determinate.where(summaries.frame.isin(["R5_detector_positive_frame", "R15_detector_positive_frame"]))
    summaries["explicit_test_share_in_known_negative_frame"] = summaries.positive_share_all.where(summaries.frame == "R15_detector_known_negative_frame")
    output.mkdir(parents=True)
    summaries.to_csv(output / "human_measurement_point_estimates.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(output / "human_execution_and_output_distributions.csv", index=False, encoding="utf-8-sig")
    memberships.to_csv(output / "human_analysis_memberships.csv", index=False, encoding="utf-8-sig")
    selected = summaries[summaries.scaffold == "all"][["frame", "sample_memberships", "population_weight_sum", "positive_share_all", "uncertain_share", "human_ppv_determinate", "explicit_test_share_in_known_negative_frame"]]
    body = "# V3.5.1 人工测量点估计\n\n只有在真实提交的两份独立首轮已锁定、分歧已由另一位声明的人类裁决后才能生成本报告。工具验证文件、引用、规则类别和来源链，不验证自然人的身份。\n\n" + md_table(selected)
    body += "\n\nA 与 A+B 分别计算 R5/R15 PPV；主点估计是确定标签中按原纳入概率加权的比值。含糊、证据不足和命令不可恢复的加权份额另报，不确定类别进入全部样本分母而不进入阳性分子的敏感性结果为 positive_share_all。C 层的分母是检测器已知阴性框，不是真实阳性总体，因此不称为假阴性率或 recall。D 与 AMB 诊断层不补入主估计。各组完整表在 CSV 中。\n\n本入口不计算区间：原冻结计划尚未选定处理 issue 内重复轨迹的方法。不使用合并二项区间，不采用事后通过阈值。获得点估计不自动通过测量质量门槛，也不自动启动 V3.6/L3。人类复核、测量方法和后续研究协议仍需按原方案验收。\n"
    (output / "HUMAN_MEASUREMENT_REPORT.md").write_text(body, encoding="utf-8")
    summary = dict(status="HUMAN MEASUREMENT POINT ESTIMATES PRODUCED; QUALITY GATE NOT AUTOMATICALLY PASSED",
                   final_human_label_lock_sha256=sha(final_labels / "FINAL_HUMAN_LABEL_LOCK.json"), key_sha256=sha(key), design_sha256=sha(design),
                   confidence_intervals_computed=False, recall_computed=False, new_v3_models_run=False, files={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / "HUMAN_MEASUREMENT_MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-labels", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.final_labels, args.key, args.design, args.output), ensure_ascii=False, indent=2))
