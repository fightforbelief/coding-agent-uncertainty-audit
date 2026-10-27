"""Copy the v2 queue into a student CSV package. Does not modify the source."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p8audit.review_workbook import write_review_package

SOURCE = ROOT / "artifacts" / "manual_review" / "marker_review_queue_v2.csv"
DESTINATION = ROOT / "artifacts" / "manual_review" / "gate08_review_package"
EXPECTED = {
    "01_precision_explicit_exception.csv": 50,
    "02_precision_explicit_test_failure.csv": 50,
    "03_openhands_exit_status.csv": 50,
    "04_diagnostic_review.csv": 216,
}


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"missing source queue: {SOURCE}")
    before = SOURCE.stat().st_size
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if SOURCE.stat().st_size != before:
        raise SystemExit("source queue size changed while reading")
    manifest = write_review_package(rows, DESTINATION)
    counts = manifest["sheet_counts"]
    if counts != EXPECTED:
        raise SystemExit(f"unexpected sheet counts: {counts}")
    if manifest["summary"]["total_rows"] != 366:
        raise SystemExit("package did not contain 366 review rows")
    if manifest["summary"]["unreviewed_rows"] != 366:
        raise SystemExit("template summary is not entirely unreviewed")
    if manifest["summary"]["student_verified_rows"] != 0:
        raise SystemExit("template is not student verified")
    if manifest["precision_rows_missing_matched_pattern"]:
        raise SystemExit("precision row is missing matched_pattern")
    if SOURCE.stat().st_size != before:
        raise SystemExit("source queue was modified")
    print(
        "REVIEW_PACKAGE_DONE"
        f" rows={manifest['summary']['total_rows']}"
        f" repeated_slots={len(manifest['repeated_observation_slots'])}"
        f" truncated={len(manifest['excerpt_truncated_sample_ids'])}"
    )


if __name__ == "__main__":
    main()
