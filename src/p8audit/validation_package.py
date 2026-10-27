"""Write the parser-v2 validation package. Does not label rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path

FORBIDDEN_COLUMNS = frozenset(
    {
        "resolved",
        "reference_patch",
        "model_patch",
        "reasoning_content",
        "execution_failure",
        "ai_label",
        "reviewer_label_proposed",
    }
)
PROTECTED_NAMES = frozenset(
    {
        "marker_review_queue.csv",
        "marker_review_queue_v2.csv",
        "manual_review_queue_manifest.json",
        "manual_review_queue_v2_manifest.json",
    }
)

INSTRUCTIONS = """# Parser v2 fresh-shard validation

These rows are unlabeled. Do not treat an empty `reviewer_label` as a decision.
`student_verified` stays false. AI adjudication of this package has not been done.
Do not copy a parser field into `reviewer_label`.

`explicit_exception_validation.csv`, `explicit_test_failure_validation.csv`, and
`openhands_exit_status_validation.csv` are precision samples of parser-v2
predicted positives on the eight fresh shards named in `package_manifest.json`.
They are not a sample of all 84 shards.

`diagnostic_validation.csv` is not a probability sample. Do not estimate
precision or recall from it.

`random_observation_validation.csv` is a small exploratory draw. It is not a
recall sample.

The operational freeze gate, after a later adjudication, remains a one-sided
95% lower precision bound of 0.90. Ambiguous and insufficient-evidence rows
stay in that denominator as not confirmed. The binomial Clopper-Pearson
interval is not by itself an exact design-based interval for shard sampling.
Neither interval, on these eight shards, proves full-snapshot precision.

OpenHands exit status is a secondary feature. It is not in the pooled
cross-scaffold primary set. A missing SWE-agent exit code is not zero.

Do not save labels over this package's source copy or over the Gate 0.5 or
Gate 0.75 queues.
"""


def assert_clean_row(row: dict) -> None:
    leaked = FORBIDDEN_COLUMNS.intersection(row)
    if leaked:
        raise ValueError(f"forbidden field: {sorted(leaked)[0]}")
    if str(row.get("reviewer_label") or "").strip():
        raise ValueError("reviewer_label must be empty")
    if str(row.get("student_verified") or "false").strip().lower() != "false":
        raise ValueError("student_verified must be false")


def write_validation_package(destination: Path, sheets: dict[str, list[dict]], manifest: dict) -> None:
    destination = Path(destination)
    if destination.name in PROTECTED_NAMES or "marker_review_queue" in destination.name:
        raise ValueError(f"refusing to write {destination.name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in sheets.items():
        path = destination / name
        fields: list[str] = []
        for row in rows:
            assert_clean_row(row)
            for key in row:
                if key not in fields:
                    fields.append(key)
        if not fields:
            fields = ["sample_id", "signal_family", "reviewer_label", "student_verified"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    (destination / "VALIDATION_INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    (destination / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    readme = [
        "# Gate 0.9 parser v2 validation package",
        "",
        "Parser version: v2. No reviewer labels are filled.",
        "This directory is not `marker_review_queue_v2.csv`.",
        "Precision sheets are predicted positives on eight fresh shards only.",
        "Diagnostic and random sheets are not precision samples.",
        "",
    ]
    (destination / "README.md").write_text("\n".join(readme), encoding="utf-8")
