"""The generated Gate 0.9 package stays unlabeled and separate from history."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "artifacts" / "manual_review" / "gate09_parser_v2_validation"
FORBIDDEN = {
    "resolved",
    "reference_patch",
    "model_patch",
    "reasoning_content",
    "execution_failure",
    "ai_label",
}
HISTORICAL = {
    "artifacts/manual_review/marker_review_queue_v2.csv":
        "6da2d7cb345ce4c2a8494707e940679b15f2c845ee43b2f29b69bfb19d4264a2",
    "artifacts/manual_review/MANUAL_REVIEW_GUIDE.md":
        "0e33f6906a4569b64df548aefbaae09ae708ab7ba6b70f51b22ca8333af77e07",
    "artifacts/manual_review/MANUAL_REVIEW_GUIDE_GATE05.md":
        "fbec4d8a4975563ba0b0440601c0be673275ffac37e754c73ea408b057a538d0",
    "docs/PROTOCOL_AMENDMENT_2026-10-26.md":
        "7d2a0ba501a7ef0f9a7d5b043d06c1473e1a8ab9445da83b8c37407b8e36313a",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_historical_review_files_keep_their_hashes() -> None:
    for relative, expected in HISTORICAL.items():
        assert _sha256(ROOT / relative) == expected


def test_generated_validation_package_has_no_labels_or_outcomes() -> None:
    manifest = json.loads((PKG / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels_filled"] is False
    assert manifest["confidence_bounds_calculated"] is False
    assert manifest["models_fit"] is False
    assert manifest["delta_auprc_calculated"] is False
    assert manifest["parser_version"] == "v2"
    assert len(manifest["shards"]) == 8
    sheets = {
        "explicit_exception_validation.csv": "precision_validation",
        "explicit_test_failure_validation.csv": "precision_validation",
        "openhands_exit_status_validation.csv": "precision_validation",
        "diagnostic_validation.csv": "diagnostic_coverage",
        "random_observation_validation.csv": "exploratory_error_analysis",
    }
    for name, purpose in sheets.items():
        with (PKG / name).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            assert FORBIDDEN.isdisjoint(reader.fieldnames or [])
            rows = list(reader)
        assert rows
        assert {row["review_purpose"] for row in rows} == {purpose}
        assert all(row["reviewer_label"] == "" for row in rows)
        assert all(row["student_verified"] == "false" for row in rows)
        if name == "openhands_exit_status_validation.csv":
            assert {row["config"] for row in rows} == {"openhands"}
    gate08 = ROOT / "artifacts" / "manual_review" / "gate08_review_package"
    for name in (
        "01_precision_explicit_exception.csv",
        "02_precision_explicit_test_failure.csv",
        "03_openhands_exit_status.csv",
        "04_diagnostic_review.csv",
    ):
        with (gate08 / name).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                assert row["reviewer_label"] == ""
                assert row["student_verified"] == "false"
