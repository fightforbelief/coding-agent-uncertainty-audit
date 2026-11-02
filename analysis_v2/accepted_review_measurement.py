"""Owner-accepted, AI-assisted human-reviewed first-round measurement.

Runs A and B separately. Agreement-only summaries retain disputes as uncertainty.
Rule-based candidate adjudications never become final human adjudications here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from original_review_control import CORE, check_return
from v3_5_ai_audit_report import join_memberships, md_table, tables, verify_lock

UNCERTAIN = dict(test_command_status="ambiguous", execution_status="unknown", test_output_status="unknown_output",
                 process_status="unknown", execution_detail="unknown", output_composition="unknown_output", runner_family="")
FIELDS = tuple(CORE) + ("runner_family",)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_round_rows(intake: Path, reference: Path, owner_acceptance: Path):
    approval = json.loads(owner_acceptance.read_text(encoding="utf-8"))
    if not all(approval.get(k) is True for k in ("owner_accepts_reported_review_provenance", "owner_accepts_timeline_explanation", "owner_states_no_mutual_viewing")):
        raise ValueError("Owner acceptance and independence attestation must be supplied")
    snapshot = json.loads((intake / "SUBMISSION_TECHNICAL_LOCK.json").read_text(encoding="utf-8"))
    for slot, files in snapshot["submitted_files"].items():
        for filename, expected in files.items():
            if Path(filename).name != filename or sha(intake / slot / filename) != expected:
                raise ValueError("Submitted file changed after technical snapshot")
    result = {}
    checks = {}
    # These are retained/disclosed source and citation issues, not permission to
    # ignore invalid IDs, missing fields, category errors or impossible nesting.
    disclosed = {
        "cover slot/rules/independent_first_round declaration missing or wrong",
        "a real reviewer must declare their name; automated sources cannot be imported as human",
        "valid first-round start and finish timestamps are required",
        "quotation is not present at a cited message", "event quotation not found in evidence",
    }
    for slot in ("reviewer_a", "reviewer_b"):
        report, rows = check_return(intake / slot, reference, slot)
        substantive = [error for error in report["errors"] if error.split(": ", 1)[-1] not in disclosed]
        if substantive:
            raise ValueError("Unaccepted structural/label issue: " + substantive[0])
        values = []
        for row in rows:
            for window in ("r5", "r15"):
                item = dict(review_id=row["review_id"], window=window)
                item.update({field: row[f"{field}_{window}"] for field in FIELDS})
                item.update(annotation_source="owner_attested_ai_assisted_human_review", reviewer_slot=slot,
                            declared_reviewer_name=approval["assigned_reviewers"][slot], itemwise_adjudicated=False)
                values.append(item)
        result[slot] = pd.DataFrame(values)
        checks[slot] = dict(structural_and_category_checks_pass=True, original_strict_validation_pass=report["valid"],
                            retained_strict_error_count=report["error_count"], owner_attestation_used=True)
    return result, checks


def agreed_only(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a = a.set_index(["review_id", "window"]).sort_index()
    b = b.set_index(["review_id", "window"]).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("First-round IDs/windows differ")
    result = a[list(FIELDS)].copy()
    disagreement_count = pd.Series(0, index=a.index)
    for field in FIELDS:
        different = a[field] != b[field]
        result.loc[different, field] = UNCERTAIN[field]
        disagreement_count += different.astype(int)
    result["unadjudicated_field_count"] = disagreement_count
    result["annotation_source"] = "agreed_submitted_values_with_disagreements_kept_uncertain"
    result["reviewer_slot"] = "agreement_only"
    result["itemwise_adjudicated"] = False
    return result.reset_index()


def reliability(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    a = a.set_index(["review_id", "window"]).sort_index()
    b = b.set_index(["review_id", "window"]).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Cannot compare unmatched windows")
    rows = []
    for window in ("r5", "r15"):
        av, bv = a.xs(window, level="window"), b.xs(window, level="window")
        for field in FIELDS:
            observed = float(av[field].eq(bv[field]).mean())
            ac, bc = av[field].value_counts(normalize=True), bv[field].value_counts(normalize=True)
            expected = sum(float(ac.get(value, 0)) * float(bc.get(value, 0)) for value in set(ac.index) | set(bc.index))
            kappa = (observed - expected) / (1 - expected) if 1 - expected > 1e-12 else None
            rows.append(dict(window=window, field=field, audit_sample_n=len(av), sample_agreement=observed, cohen_kappa=kappa,
                             interpretation="audit-sample first-round reliability; not population accuracy; owner-attested AI-assisted review"))
    return pd.DataFrame(rows)


def candidate_fields(a: pd.DataFrame, b: pd.DataFrame, automated: pd.DataFrame):
    am = a.set_index(["review_id", "window"]).sort_index()
    bm = b.set_index(["review_id", "window"]).sort_index()
    cm = automated.set_index(["review_id", "window"])
    if not am.index.equals(bm.index) or set(am.index) != set(cm.index):
        raise ValueError("Candidate source IDs/windows must match both first rounds")
    rows = []
    for key in am.index:
        for field in FIELDS:
            av, bv = str(am.loc[key, field]), str(bm.loc[key, field])
            if av == bv:
                continue
            candidate = str(cm.loc[key, field]) if field in cm.columns else ""
            rows.append(dict(review_id=key[0], window=key[1], field=f"{field}_{key[1]}", reviewer_a_value=av, reviewer_b_value=bv,
                             proposed_value=candidate, candidate_matches_a=candidate == av, candidate_matches_b=candidate == bv,
                             candidate_source="prior_frozen_automated_evidence_annotation_not_human_adjudication",
                             final_value="", final_source="", adjudicator_id="", adjudicated_at_utc="", notes=""))
    return pd.DataFrame(rows)


def plot_paired(pooled: pd.DataFrame, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["R5 detector-positive", "R15 detector-positive", "R15 known-negative", "R15 missing command", "R5 ambiguous command", "R15 ambiguous command"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5), sharey=True, layout="constrained")
    for ax, version, title in zip(axes, ("reviewer_a", "reviewer_b", "agreement_only"), ("A: owner-attested review", "B: owner-attested review", "Agreement-only sensitivity")):
        data = pooled[pooled.review_version == version]
        starts = np.zeros(len(data))
        for field, color, legend in (("positive_share_all", "#287c8e", "Positive"), ("negative_share_all", "#aac9c2", "Negative"), ("uncertain_share", "#efb655", "Uncertain")):
            values = data[field].to_numpy() * 100
            ax.barh(np.arange(len(data)), values, left=starts, label=legend, color=color, height=.65)
            starts += values
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Weighted share (%)")
        ax.set_yticks(np.arange(len(data)), labels)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].invert_yaxis()
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("First-round measurement: disagreements retained, not treated as adjudicated gold", fontsize=12)
    fig.savefig(output / "paired_measurement_shares.png", dpi=180)
    fig.savefig(output / "paired_measurement_shares.svg")
    plt.close(fig)


def run(intake: Path, reference: Path, approval: Path, key: Path, design: Path, automated: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a fresh owner-accepted measurement directory")
    values, checks = first_round_rows(intake, reference, approval)
    verify_lock(automated)
    auto_labels = pd.read_csv(automated / "annotations_long.csv", keep_default_na=False)
    candidates = candidate_fields(values["reviewer_a"], values["reviewer_b"], auto_labels)
    values["agreement_only"] = agreed_only(values["reviewer_a"], values["reviewer_b"])
    agreement = reliability(values["reviewer_a"], values["reviewer_b"])
    sampling_key, sampling_design = pd.read_csv(key), pd.read_csv(design)
    summaries, distributions, memberships = [], [], []
    for version, labels in values.items():
        joined = join_memberships(labels, sampling_key, sampling_design)
        summary, distribution = tables(joined)
        summary.insert(0, "review_version", version)
        distribution.insert(0, "review_version", version)
        summaries.append(summary)
        distributions.append(distribution)
        joined.insert(0, "review_version", version)
        memberships.append(joined)
    estimates = pd.concat(summaries, ignore_index=True)
    output.mkdir(parents=True)
    estimates.to_csv(output / "paired_weighted_measurement.csv", index=False, encoding="utf-8-sig")
    pd.concat(distributions, ignore_index=True).to_csv(output / "paired_execution_output_distributions.csv", index=False, encoding="utf-8-sig")
    pd.concat(memberships, ignore_index=True).to_csv(output / "paired_analysis_memberships.csv", index=False, encoding="utf-8-sig")
    pd.concat(values.values(), ignore_index=True).to_csv(output / "first_round_and_agreement_only_labels.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(output / "first_round_reliability.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output / "candidate_adjudications_not_final.csv", index=False, encoding="utf-8-sig")
    pooled = estimates[estimates.scaffold == "all"]
    columns = ["review_version", "frame", "sample_memberships", "positive_share_all", "uncertain_share", "positive_share_determinate", "ambiguity_lower", "ambiguity_upper"]
    plot_paired(pooled, output)
    facts = dict(status="OWNER-ACCEPTED ASSISTED FIRST-ROUND MEASUREMENT COMPLETE; ITEMWISE ADJUDICATION NOT COMPLETE",
                 owner_acceptance_sha256=sha(approval), input_checks=checks, confidence_intervals_computed=False,
                 final_adjudicated_gold_labels_created=False, independent_identity_or_chronology_verification_performed=False,
                 candidate_source_lock_sha256=sha(automated / "ANNOTATION_LOCK.json"), candidate_fields=len(candidates),
                 candidates_matching_a=int(candidates.candidate_matches_a.sum()), candidates_matching_b=int(candidates.candidate_matches_b.sum()),
                 candidates_matching_neither=int((~candidates.candidate_matches_a & ~candidates.candidate_matches_b).sum()),
                 original_judgements_changed=False, disagreements_converted_to_final_values=False, new_predictive_models_run=False,
                 pooled_summaries=json.loads(pooled.to_json(orient="records")), input_reference_sha256=sha(reference),
                 sampling_key_sha256=sha(key), sampling_design_sha256=sha(design), code_sha256=sha(Path(__file__)))
    (output / "PAIRED_MEASUREMENT_RESULTS.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    body = """# 已获负责人认可的双份首轮测量结果

状态：项目负责人已认可时间线说明，声明两人没有互看，并确认继续。以下按这一声明接续工作，不再要求重复解释。来源仍是 AI 草稿基础上的、据负责人陈述已由指定人员分别逐条复核的判断；不改写原 ZIP，不称为工具独立验证过身份或时间线，也不称为未经 AI 辅助的盲审。

## 分层测量结果

A、B 分别计算。`agreement_only` 仅保留两表相同的值，不同值在对应字段保留不确定；这是一份保守敏感性汇总，不是第三人裁决，不是新的人类标签。不能用一致子集的 100% 数字代替完整验证。

每项只使用冻结设计中自己的抽样层、N_h/n_h 权重；A+B、C、D、AMB 分别分析。跨层轨迹不合并纳入概率。阳性框中的 `positive_share_determinate` 是相应首轮对可判定标签的加权比例；原始来源和未裁决分歧意味着它不是已经确立的最终金标准 PPV。C 的阳性份额不是假阴性率或 recall。

`ambiguity_lower/upper` 仅把不确定全部视为 0 或 1，既不是置信区间，也不覆盖标注错误。原计划未选定正式 issue 聚类区间，本轮不事后生成一个貌似确认性的区间。

"""
    body += md_table(pooled[columns])
    body += "\n\n![双份首轮和保守一致汇总的加权分类份额](paired_measurement_shares.png)\n"
    body += "\n\n## 首轮一致程度\n\n下表是富集审计样本上的描述性一致性，不是全库准确率或检测器性能。Kappa 无定义时留空，不补 1。\n\n"
    body += md_table(agreement[["window", "field", "audit_sample_n", "sample_agreement", "cohen_kappa"]])
    body += f"\n\n## 候选裁决\n\n共有 {len(candidates):,} 个不同字段。已从此前冻结的自动证据标注中提取候选值：{facts['candidates_matching_a']:,} 个与 A 相同，{facts['candidates_matching_b']:,} 个与 B 相同，{facts['candidates_matching_neither']:,} 个与两者都不同。候选来源明确为自动标注，只是复核辅助，不是人工裁决。所有最终值、裁决人、时间继续为空。文件为 `candidate_adjudications_not_final.csv`。\n"
    body += "\n项目负责人的‘全部通过’已作为行政/来源认可记录，但未提供每个相互矛盾字段应选择的最终值。为避免替复核者制造判断，本轮用 A、B 和一致/分歧敏感性汇总继续统计；没有假造一个单一最终金标准。先前的引文索引建议与原始来源继续保留在收件目录。\n\n原确认性结果仍为历史冻结结果。本轮首轮测量和候选裁决不能单独证实新增预测价值或因果关系。原方案的逐项裁决、测量门槛及后续方案状态以实际完成的产物为准。\n"
    (output / "PAIRED_MEASUREMENT_REPORT.md").write_text(body, encoding="utf-8")
    (output / "PAIRED_MEASUREMENT_MANIFEST.json").write_text(json.dumps({p.name: sha(p) for p in output.iterdir() if p.is_file()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake", "reference", "approval", "key", "design", "automated", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.intake, args.reference, args.approval, args.key, args.design, args.automated, args.output)
    print(json.dumps({k:v for k,v in result.items() if k != "pooled_summaries"}, ensure_ascii=False, indent=2))
