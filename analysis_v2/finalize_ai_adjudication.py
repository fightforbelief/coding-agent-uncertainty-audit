"""Adopt the owner's explicitly selected AI candidates with full provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from original_review_control import CORE
from v3_5_ai_audit_report import join_memberships, md_table, tables

FIELDS = tuple(CORE) + ("runner_family",)
TEST_FIELDS = ("execution_status", "test_output_status", "process_status", "execution_detail", "output_composition")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(measurement: Path, authorization: Path, key: Path, design: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a fresh AI adjudication run")
    auth = json.loads(authorization.read_text(encoding="utf-8"))
    if auth.get("adjudication_source") != "AI_candidate_adjudication_accepted_by_project_owner" or auth.get("is_third_independent_human_adjudication") is not False:
        raise ValueError("Explicit owner choice of disclosed AI adjudication is required")
    manifest = json.loads((measurement / "PAIRED_MEASUREMENT_MANIFEST.json").read_text(encoding="utf-8"))
    for filename in ("first_round_and_agreement_only_labels.csv", "candidate_adjudications_not_final.csv"):
        if sha(measurement / filename) != manifest[filename]:
            raise ValueError("Candidate/source file changed after measurement run")
    submitted = pd.read_csv(measurement / "first_round_and_agreement_only_labels.csv", keep_default_na=False)
    a = submitted[submitted.reviewer_slot == "reviewer_a"].set_index(["review_id", "window"]).sort_index()
    b = submitted[submitted.reviewer_slot == "reviewer_b"].set_index(["review_id", "window"]).sort_index()
    candidates = pd.read_csv(measurement / "candidate_adjudications_not_final.csv", keep_default_na=False)
    if not a.index.equals(b.index) or candidates.duplicated(["review_id", "field"]).any():
        raise ValueError("Mismatched or duplicate candidate keys")
    expected = {(rid, f"{field}_{window}") for rid, window in a.index for field in FIELDS if a.loc[(rid,window),field] != b.loc[(rid,window),field]}
    if {(row.review_id, row.field) for row in candidates.itertuples(index=False)} != expected:
        raise ValueError("Candidates must cover exactly the submitted differing fields")
    final = a[list(FIELDS)].copy()
    origins = {((rid,window),field): "submitted_first_round_agreement" for rid,window in a.index for field in FIELDS}
    changes = []
    for row in candidates.itertuples(index=False):
        index = (row.review_id, row.window)
        field = row.field.removesuffix("_" + row.window)
        if a.loc[index,field] != row.reviewer_a_value or b.loc[index,field] != row.reviewer_b_value:
            raise ValueError("Candidate does not refer to the actual first-round values")
        if field in CORE and row.proposed_value not in CORE[field]:
            raise ValueError("Invalid AI candidate value")
        final.loc[index,field] = row.proposed_value
        origins[(index,field)] = "AI_candidate_adjudication_accepted_by_project_owner"
    consistency_windows = set()
    for index, row in final.iterrows():
        if row.test_command_status == "definite_no_test_command":
            for field in TEST_FIELDS:
                if row[field] != "not_applicable":
                    consistency_windows.add(index)
                    final.loc[index,field] = "not_applicable"
                    origins[(index,field)] = "AI_logical_not_applicable_after_final_no_test_command"
    for index, row in final.iterrows():
        for field in FIELDS:
            value = row[field]
            if field in CORE and value not in CORE[field]:
                raise ValueError("Invalid final category")
            changes.append(dict(review_id=index[0], window=index[1], field=field, reviewer_a_value=a.loc[index,field], reviewer_b_value=b.loc[index,field],
                                final_value=value, final_source=origins[(index,field)], is_human_adjudication=False))
        if row.test_output_status in {"explicit_pass", "explicit_failure"} and row.execution_status != "execution_confirmed":
            raise ValueError("Final explicit result lacks confirmed test execution")
        if row.output_composition == "mixed_pass_and_failure" and row.test_output_status != "explicit_failure":
            raise ValueError("Mixed output must retain explicit failure")
        if row.execution_detail in {"command_not_found", "dependency_error", "collection_failure", "build_failure", "timeout_before_tests", "other_pre_test_failure"} and row.test_output_status in {"explicit_pass", "explicit_failure"}:
            raise ValueError("Pre-test error cannot be the test result")
    for rid in final.index.get_level_values("review_id").unique():
        r5, r15 = final.loc[(rid,"r5")], final.loc[(rid,"r15")]
        if r5.test_command_status == "definite_test_command" and r15.test_command_status == "definite_no_test_command":
            raise ValueError("R5 positive cannot become R15 negative")
        if r5.test_output_status == "explicit_failure" and r15.test_output_status != "explicit_failure":
            raise ValueError("R15 must retain a final R5 failure")
    labels = final.reset_index()
    labels["annotation_source"] = "owner_authorized_ai_reconciliation_of_assisted_first_rounds"
    labels["independent_human_adjudication"] = False
    joined = join_memberships(labels, pd.read_csv(key), pd.read_csv(design))
    summary, distribution = tables(joined)
    output.mkdir(parents=True)
    labels.to_csv(output / "final_ai_adjudicated_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(output / "all_field_provenance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([r for r in changes if r["final_source"].startswith("AI_logical")]).to_csv(output / "additional_logical_consistency_changes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "final_ai_measurement.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(output / "final_ai_execution_output_distributions.csv", index=False, encoding="utf-8-sig")
    pooled = summary[summary.scaffold == "all"]
    facts = dict(status="OWNER-AUTHORIZED AI ADJUDICATION AND MEASUREMENT COMPLETE; NOT INDEPENDENT HUMAN GOLD",
                 windows=len(labels), adopted_candidate_fields=len(candidates), additional_logical_consistency_windows=len(consistency_windows),
                 additional_logical_consistency_fields=sum(r["final_source"].startswith("AI_logical") for r in changes),
                 authorization_sha256=sha(authorization), candidate_csv_sha256=sha(measurement / "candidate_adjudications_not_final.csv"),
                 owner_attested_first_round_source=True, third_independent_human_adjudication=False, confidence_intervals_computed=False,
                 source_submissions_modified=False, new_v3_test_set_scored=False, pooled_summaries=json.loads(pooled.to_json(orient="records")), code_sha256=sha(Path(__file__)))
    body = "# 负责人批准采用的 AI 裁决与测量结果\n\n最终值来自两表一致判断，或负责人明确授权使用的先前 AI 候选。不是第三位人类独立裁决，不作为独立人工金标准。两份提交保留原字节和原来源。\n\n"
    body += f"共处理 {len(labels)} 个窗口，采用 {len(candidates)} 个 AI 候选字段。逐字段采用后，{len(consistency_windows)} 个窗口出现‘最终无测试命令但仍有适用的测试执行/输出字段’的组合；按规则将这些字段置为 not_applicable，额外 {facts['additional_logical_consistency_fields']} 个字段逐项记录为 AI 逻辑一致性修正，不归为人类判断。未选择另一套测试行为定义，也未改冻结 detector。\n\n"
    body += md_table(pooled[["frame", "sample_memberships", "positive_share_all", "uncertain_share", "positive_share_determinate", "ambiguity_lower", "ambiguity_upper"]])
    body += "\n\n点估计按原分层概率计算。分类敏感性范围不是抽样置信区间；确定标签中较高的阳性比例不证明检测器准确率。原正式 issue 聚类区间方法未事先选择，因此没有新造确认性区间。此处使用 AI 裁决替代原第三人裁决，是负责人明确选择的方案偏离。\n\n最终标签、全部来源和额外一致性修改均在 CSV 中，数据锁在 AI_FINAL_LABEL_LOCK.json。没有生成 FINAL_HUMAN_LABEL_LOCK.json，也没有将本结果伪装成独立人工裁决。\n"
    (output / "FINAL_AI_MEASUREMENT_REPORT.md").write_text(body, encoding="utf-8")
    facts["files"] = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output / "AI_FINAL_LABEL_LOCK.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    return facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("measurement", "authorization", "key", "design", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.measurement, args.authorization, args.key, args.design, args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in {"files", "pooled_summaries"}}, ensure_ascii=False, indent=2))
