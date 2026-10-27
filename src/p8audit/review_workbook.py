"""Build a student review package without altering the source queue."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REVIEW_LABELS = (
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
    "ambiguous",
    "insufficient_evidence",
    "uncertain",
)
UNRESOLVED_LABELS = ("ambiguous", "insufficient_evidence", "uncertain")
FORBIDDEN_COLUMNS = frozenset(
    {
        "resolved",
        "reference_patch",
        "model_patch",
        "reasoning_content",
        "execution_failure",
    }
)
PROTECTED_OUTPUT_NAMES = frozenset(
    {
        "marker_review_queue.csv",
        "marker_review_queue_v2.csv",
        "manual_review_queue_manifest.json",
        "manual_review_queue_v2_manifest.json",
    }
)
IDENTITY_FIELDS = ("sample_id", "observation_key", "signal_family", "review_purpose")
LEADING_COLUMNS = (
    "sample_id",
    "observation_key",
    "repeated_observation_slot",
    "review_purpose",
    "signal_family",
    "review_scope",
    "reviewer_label",
    "reviewer_notes",
    "student_verified",
    "config",
    "split",
    "parquet_shard",
    "language",
    "matched_pattern",
    "parser_predicted_label",
    "sampling_probability",
    "sampling_stratum",
    "source_shard_selection_method",
    "duplicate_signal_families",
    "excerpt_truncated",
    "trajectory_id",
    "instance_id",
    "repo",
    "observation_index",
    "row_index",
    "n_chars",
    "content_sha1_16",
    "queue_version",
    "queue_seed",
    "dataset_revision",
    "context_excerpt",
)

INSTRUCTIONS = """# P8 人工复核说明

这份文件是给学生看的复核包，不是模型结果。代理生成的标签不算学生核验。`student_verified` 必须保持 `false`，直到你本人看完上下文并亲自确认。不要把解析器的 `parser_predicted_label` 抄进 `reviewer_label`。

## 研究问题

How well do published trajectory-reliability signals transfer to repository-level coding-agent traces, and do observable execution-feedback features add incremental predictive value beyond simple structural baselines under repository-held-out evaluation?

复核的目的，是判断解析器标成阳性的工具观察，是否真的是它声称的那一种执行反馈。这一步不估计 Delta_AUPRC，也不冻结特征。

## 先看上下文，再填标签

每一行的 `context_excerpt` 是当时保存的观察摘录。先读摘录，再填 `reviewer_label`。摘录不够时，不要根据“没有失败标记”推断测试成功，填 `insufficient_evidence`。`excerpt_truncated` 为 `true` 的行，摘录在生成队列时已被截断；本复核包没有再缩短它。截断后的文本如果不能判定，就填 `insufficient_evidence`，不要猜。

允许的 `reviewer_label`：

- `true_positive`：摘录里确实有该信号家族所声称的现象。
- `false_positive`：解析器标了阳性，但现象并不存在。
- `true_negative`：这行本来就不是该阳性现象，而且摘录足以确认它不存在。诊断行可能用到。
- `false_negative`：解析器没标出，但摘录里其实有该现象。诊断行可能用到。精度表的行本身是预测阳性，通常不会是 false_negative。
- `ambiguous`：摘录里有相关痕迹，但两种读法都说得通，你不能判定。
- `insufficient_evidence`：摘录太短、被截断，或缺少判定所需的命令结果。
- `uncertain`：你不愿意在 ambiguous 和 insufficient_evidence 之间再细分。

空的 `reviewer_label` 表示还没复核。空着不等于阴性，也不等于已核验。

## 不要把不同现象合成一个失败

- 工具退出状态：OpenHands 观察里的 scaffold 包装写出非 0、且不是 124 或 130 的退出码。它不是测试断言的结果。SWE-agent 在已审计的分片里没有同一种包装；缺退出码不是 0，也不是成功。
- 测试失败：pytest、Go 或 cargo 的失败断言。退出码非 0 本身不是测试失败。
- 异常：Python traceback，或一行明确的 `Error:` / `Exception:`。普通散文里的 error 不算。
- 测试通过：只有出现通过标记，并且同一观察里没有失败标记。没有失败标记，不等于通过。
- 中断或超时：退出码 124 或 130，或明确的中断/超时文字。它不是普通测试失败。
- 未知：文本自相矛盾，或只有无法归类的退出码写法。什么标记都没有，不是未知，也不要把它填成成功或失败。

## 四张表分别是什么

1. `01_precision_explicit_exception.csv`：异常信号的 50 行精度样本。它抽自八个补充分片里解析器判为阳性的观察，不是 84 个分片的全体。
2. `02_precision_explicit_test_failure.csv`：测试失败信号的 50 行，范围同上。
3. `03_openhands_exit_status.csv`：OpenHands 非零退出码的 50 行。这只是可能的 OpenHands 二次分析，不是跨 scaffold 的主特征。
4. `04_diagnostic_review.csv`：分层诊断行。它不是概率样本，不能用来估计总体召回，也不能并进精度区间。

`repeated_observation_slot` 为 `yes` 时，同一个观察槽在队列里出现了不止一行。请按每一行自己的 `signal_family` 分别判断，不要因为另一行已经填过就复制标签。

## 不要覆盖原始队列

把标签写在这些 CSV 副本上，或按 `06_label_import_schema.csv` 另存一份。不要保存到 `marker_review_queue_v2.csv` 或 `marker_review_queue.csv`。`05_review_summary.csv` 在没有人工标签时，未复核行就是未复核；不要把它们写成阳性或已核验。

现有环境没有安装 openpyxl，所以这里不是带下拉框的 xlsx。在表格软件里请自行冻结首行并打开筛选。标签只能用上面列出的词。
"""


def _reject_protected_destination(destination: Path) -> None:
    if destination.name in PROTECTED_OUTPUT_NAMES or "marker_review_queue" in destination.name:
        raise ValueError(f"refusing to write {destination.name}")
    if destination.exists() and not destination.is_dir():
        raise ValueError("destination must be a directory")


def repeated_slot_keys(rows: list[dict]) -> set[str]:
    counts = Counter(str(row.get("observation_key") or "") for row in rows)
    return {key for key, count in counts.items() if key and count > 1}


def prepare_row(row: dict, repeated: set[str], *, review_scope: str = "") -> dict:
    cleaned = {key: value for key, value in row.items() if key not in FORBIDDEN_COLUMNS}
    observation_key = str(cleaned.get("observation_key") or "")
    cleaned["repeated_observation_slot"] = "yes" if observation_key in repeated else "no"
    cleaned["review_scope"] = review_scope
    cleaned["reviewer_label"] = str(cleaned.get("reviewer_label") or "")
    cleaned["reviewer_notes"] = str(cleaned.get("reviewer_notes") or "")
    cleaned["student_verified"] = str(cleaned.get("student_verified") or "false")
    return cleaned


def column_order(rows: list[dict]) -> list[str]:
    present = set()
    for row in rows:
        present.update(row)
    ordered = [name for name in LEADING_COLUMNS if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = column_order(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def select_sheet(rows: list[dict], purpose: str, family: str | None = None) -> list[dict]:
    selected = []
    for row in rows:
        if row.get("review_purpose") != purpose:
            continue
        if family is not None and row.get("signal_family") != family:
            continue
        selected.append(row)
    return selected


def summarize_review(rows: list[dict]) -> dict[str, int]:
    """Count labels. Blank labels stay unreviewed and are not verification."""
    label_counts: Counter[str] = Counter()
    for row in rows:
        label = str(row.get("reviewer_label") or "").strip()
        if label and label not in REVIEW_LABELS:
            raise ValueError(f"unknown reviewer_label: {label}")
        label_counts[label] += 1
    student_verified = 0
    for row in rows:
        label = str(row.get("reviewer_label") or "").strip()
        flag = str(row.get("student_verified") or "").strip().lower()
        if label and flag == "true":
            student_verified += 1
    blank = label_counts.get("", 0)
    reviewed = sum(count for label, count in label_counts.items() if label)
    keys = {str(row.get("observation_key") or "") for row in rows}
    return {
        "total_rows": len(rows),
        "unique_observations": len(keys - {""}),
        "reviewed_rows": reviewed,
        "true_positive": label_counts.get("true_positive", 0),
        "false_positive": label_counts.get("false_positive", 0),
        "true_negative": label_counts.get("true_negative", 0),
        "false_negative": label_counts.get("false_negative", 0),
        "ambiguous": label_counts.get("ambiguous", 0),
        "insufficient_evidence": label_counts.get("insufficient_evidence", 0),
        "uncertain": label_counts.get("uncertain", 0),
        "unreviewed_rows": blank,
        "student_verified_rows": student_verified,
    }


def apply_human_labels(template_rows: list[dict], label_rows: list[dict]) -> list[dict]:
    """Copy human labels onto a new list. Does not set student_verified by itself."""
    incoming: dict[str, dict] = {}
    for row in label_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("label row is missing sample_id")
        if sample_id in incoming:
            raise ValueError(f"duplicate sample_id in labels: {sample_id}")
        incoming[sample_id] = row
    template_ids = {str(row.get("sample_id") or "") for row in template_rows}
    unknown = sorted(set(incoming) - template_ids)
    if unknown:
        raise ValueError(f"unknown sample_id: {unknown[0]}")
    merged = []
    for source in template_rows:
        copied = dict(source)
        sample_id = str(source.get("sample_id") or "")
        update = incoming.get(sample_id)
        if update is None:
            merged.append(copied)
            continue
        for field in IDENTITY_FIELDS:
            if str(update.get(field) or "") != str(source.get(field) or ""):
                raise ValueError(f"{sample_id} does not match on {field}")
        label = str(update.get("reviewer_label") or "").strip()
        if label and label not in REVIEW_LABELS:
            raise ValueError(f"unknown reviewer_label: {label}")
        copied["reviewer_label"] = label
        if "reviewer_notes" in update:
            copied["reviewer_notes"] = update.get("reviewer_notes") or ""
        if "student_verified" in update and str(update.get("student_verified") or "").strip() != "":
            flag = str(update.get("student_verified")).strip().lower()
            if flag not in {"true", "false"}:
                raise ValueError("student_verified must be true or false")
            copied["student_verified"] = flag
        merged.append(copied)
    return merged


def write_review_package(rows: list[dict], destination: Path) -> dict:
    """Write the CSV review package. Never writes the historical queue files."""
    destination = Path(destination)
    _reject_protected_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)
    repeated = repeated_slot_keys(rows)
    sheets = {
        "01_precision_explicit_exception.csv": (
            select_sheet(rows, "precision_validation", "explicit_exception"),
            "",
        ),
        "02_precision_explicit_test_failure.csv": (
            select_sheet(rows, "precision_validation", "explicit_test_failure"),
            "",
        ),
        "03_openhands_exit_status.csv": (
            select_sheet(rows, "precision_validation", "tool_exit_nonzero"),
            "OpenHands-only secondary signal; not in the pooled cross-scaffold primary feature set",
        ),
        "04_diagnostic_review.csv": (
            select_sheet(rows, "diagnostic_coverage"),
            "diagnostic coverage; not a probability sample and not a recall estimate",
        ),
    }
    prepared_all = []
    sheet_counts = {}
    for name, (selected, scope) in sheets.items():
        prepared = [prepare_row(row, repeated, review_scope=scope) for row in selected]
        for row in prepared:
            if any(column in row for column in FORBIDDEN_COLUMNS):
                raise ValueError("forbidden column survived")
        write_csv(destination / name, prepared)
        prepared_all.extend(prepared)
        sheet_counts[name] = len(prepared)
    summary = summarize_review(prepared_all)
    summary_path = destination / "05_review_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["measure", "count", "interpretation"],
        )
        writer.writeheader()
        notes = {
            "total_rows": "Rows in the four review sheets. Not a population.",
            "unique_observations": "Distinct observation slots. A slot can appear on more than one row.",
            "reviewed_rows": "Rows whose reviewer_label is one of the allowed labels.",
            "true_positive": "Human label true_positive. Empty labels are not counted here.",
            "false_positive": "Human label false_positive.",
            "true_negative": "Human label true_negative.",
            "false_negative": "Human label false_negative.",
            "ambiguous": "Human label ambiguous. Not a confirmed true positive.",
            "insufficient_evidence": "Human label insufficient_evidence. Not a confirmed true positive.",
            "uncertain": "Human label uncertain. Not a confirmed true positive.",
            "unreviewed_rows": "Blank reviewer_label. Not labeled and not verified.",
            "student_verified_rows": "Nonblank label and student_verified true. The template starts at zero.",
        }
        for measure, count in summary.items():
            writer.writerow(
                {
                    "measure": measure,
                    "count": count,
                    "interpretation": notes[measure],
                }
            )
    schema_path = destination / "06_label_import_schema.csv"
    with schema_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["column", "required_on_import", "rule"],
        )
        writer.writeheader()
        rules = [
            ("sample_id", "yes", "Stable id. Must match the template row."),
            ("observation_key", "yes", "shard|trajectory_id|observation_index. Must match."),
            ("signal_family", "yes", "Must match the template row."),
            ("review_purpose", "yes", "Must match the template row."),
            ("reviewer_label", "yes", "Blank, or one of: " + ", ".join(REVIEW_LABELS)),
            ("reviewer_notes", "no", "Free text. Do not paste outcome labels."),
            (
                "student_verified",
                "no",
                "Leave false. Importing the word true does not create verification by itself; the student has to have done the review.",
            ),
        ]
        for column, required, rule in rules:
            writer.writerow(
                {"column": column, "required_on_import": required, "rule": rule}
            )
    (destination / "00_INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    truncated = [
        str(row.get("sample_id"))
        for row in rows
        if str(row.get("excerpt_truncated")).lower() == "true"
    ]
    empty_pattern = [
        str(row.get("sample_id"))
        for row in rows
        if row.get("review_purpose") == "precision_validation"
        and not str(row.get("matched_pattern") or "").strip()
    ]
    manifest = {
        "package": "gate08-csv-review-package",
        "openpyxl": False,
        "xlsx_written": False,
        "reason": "openpyxl is not installed in the existing .venv; no package was installed",
        "source_queue_modified": False,
        "sheet_counts": sheet_counts,
        "summary": summary,
        "repeated_observation_slots": sorted(repeated),
        "excerpt_truncated_sample_ids": truncated,
        "precision_rows_missing_matched_pattern": empty_pattern,
        "forbidden_columns_omitted": sorted(FORBIDDEN_COLUMNS),
        "student_labels_present": summary["reviewed_rows"] > 0,
    }
    (destination / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (destination / "PACKAGE_README.md").write_text(
        "\n".join(
            [
                "# Gate 0.8 review package",
                "",
                "This directory is a copy for human labels. It is not the source queue.",
                "Do not save edits over `artifacts/manual_review/marker_review_queue_v2.csv`.",
                "`openpyxl` was not available, so the package is CSV rather than xlsx.",
                "Unreviewed rows in `05_review_summary.csv` are not labels and are not student verification.",
                "Diagnostic rows are not a probability sample.",
                "Sheet 03 is an OpenHands-only secondary signal.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest
