"""Synthetic examples only; no test data are actual reviewer judgements."""
import csv
import json
from datetime import datetime, timezone

import pytest

from original_review_control import check_return, compare_returns, finalize_returns, lock_return, verify_locked


def write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def submitted(tmp_path, slot="reviewer_a", failed=False):
    folder = tmp_path / slot
    folder.mkdir()
    call = dict(round=1, action_index=2, observation_index=3, extracted_command="pytest", arguments='{"command":"pytest"}', observation="1 passed; 1 failed")
    evidence = dict(review_id="SYNTHETIC-1", completed_rounds=15, prefix5_boundary="match", prefix15_boundary="match", prefix_parse_status="ok", r5=[call], r15=[call])
    path = tmp_path / "synthetic_evidence.jsonl"
    path.write_text(json.dumps(evidence) + "\n", encoding="utf8")
    queue = {k: str(evidence[k]) for k in ("review_id", "completed_rounds", "prefix5_boundary", "prefix15_boundary", "prefix_parse_status")}
    ext = {"review_id": evidence["review_id"]}
    events = []
    for window in ("r5", "r15"):
        values = dict(test_command_status="definite_test_command", execution_status="execution_confirmed",
                      test_output_status="explicit_failure" if failed else "explicit_pass", supporting_message_indexes="2;3",
                      runner_family="pytest", evidence_quotation="1 failed" if failed else "1 passed", reviewer_confidence="high", adjudication_notes="")
        queue.update({f"{k}_{window}": v for k, v in values.items()})
        ext.update({f"process_status_{window}": "started", f"execution_detail_{window}": "tests_executed",
                    f"output_composition_{window}": "mixed_pass_and_failure" if failed else "all_explicit_pass"})
        events.append(dict(review_id=evidence["review_id"], window=window, event_order="1", message_index="3", process_started="started", execution_detail="tests_executed", output_event=values["test_output_status"], quotation=values["evidence_quotation"]))
    write(folder / "review_queue.csv", [queue])
    write(folder / "window_extension.csv", [ext])
    write(folder / "event_worksheet.csv", events)
    write(folder / "COVER_SHEET.csv", [dict(reviewer_slot=slot, reviewer_name="Synthetic Reviewer " + slot,
          rules_version="v3.5.1-rules-1", started_at_utc="2020-01-01T00:00:00Z", finished_at_utc="2020-01-01T01:00:00Z", packet_role="independent_first_round")])
    return folder, path


def test_complete_return_locks_source_bytes_and_not_identity(tmp_path):
    folder, evidence = submitted(tmp_path)
    summary, rows = check_return(folder, evidence, "reviewer_a")
    assert summary["valid"]
    assert not summary["reviewer_identity_verified_by_tool"]
    output = tmp_path / "locked"
    lock_return(folder, evidence, "reviewer_a", output)
    verify_locked(output)
    assert (folder / "review_queue.csv").read_bytes() == (output / "review_queue.csv").read_bytes()
    with pytest.raises(FileExistsError):
        lock_return(folder, evidence, "reviewer_a", output)


def test_blank_fields_cannot_be_locked(tmp_path):
    folder, evidence = submitted(tmp_path)
    path = folder / "review_queue.csv"
    text = path.read_text(encoding="utf-8-sig").replace("definite_test_command", "")
    path.write_text(text, encoding="utf-8-sig")
    summary, _ = check_return(folder, evidence, "reviewer_a")
    assert not summary["valid"]
    assert summary["incomplete_windows"] == 2
    with pytest.raises(ValueError, match="invalid/incomplete"):
        lock_return(folder, evidence, "reviewer_a", tmp_path / "invalid-lock")
    assert not (tmp_path / "invalid-lock").exists()


def test_automated_label_schema_rejected_even_with_named_cover(tmp_path):
    folder, evidence = submitted(tmp_path)
    with (folder / "review_queue.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    row["annotation_source"] = "automated"
    write(folder / "review_queue.csv", [row])
    summary, _ = check_return(folder, evidence, "reviewer_a")
    assert not summary["valid"]
    assert any("automated annotation schema" in error for error in summary["errors"])


def test_quote_and_prefix_validation(tmp_path):
    folder, evidence = submitted(tmp_path)
    path = folder / "review_queue.csv"
    path.write_text(path.read_text(encoding="utf-8-sig").replace("1 passed", "invented result"), encoding="utf-8-sig")
    summary, _ = check_return(folder, evidence, "reviewer_a")
    assert not summary["valid"]
    assert any("quotation" in error for error in summary["errors"])


def test_change_to_locked_first_round_is_detected(tmp_path):
    folder, evidence = submitted(tmp_path)
    locked = tmp_path / "locked"
    lock_return(folder, evidence, "reviewer_a", locked)
    with (locked / "review_queue.csv").open("a", encoding="utf8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="changed"):
        verify_locked(locked)


def test_comparison_only_emits_real_disagreements_and_no_final_values(tmp_path):
    a, evidence = submitted(tmp_path)
    b, _ = submitted(tmp_path, "reviewer_b", failed=True)
    al, bl = tmp_path / "a-lock", tmp_path / "b-lock"
    lock_return(a, evidence, "reviewer_a", al)
    lock_return(b, evidence, "reviewer_b", bl)
    output = tmp_path / "comparison"
    summary = compare_returns(al, bl, evidence, output)
    assert summary["disagreement_fields"] == 4
    with (output / "disagreements_for_adjudicator.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert all(not row["final_value"] and not row["final_source"] for row in rows)


def test_finalisation_requires_all_disagreements_and_separate_adjudicator(tmp_path):
    a, evidence = submitted(tmp_path)
    b, _ = submitted(tmp_path, "reviewer_b", failed=True)
    al, bl = tmp_path / "a-lock", tmp_path / "b-lock"
    lock_return(a, evidence, "reviewer_a", al)
    lock_return(b, evidence, "reviewer_b", bl)
    compare_returns(al, bl, evidence, tmp_path / "compare")
    path = tmp_path / "compare/disagreements_for_adjudicator.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with pytest.raises(ValueError, match="distinct declared human adjudicator"):
        finalize_returns(al, bl, evidence, path, tmp_path / "missing-adjudicator")
    assert not (tmp_path / "missing-adjudicator").exists()
    for row in rows:
        row.update(final_value=row["reviewer_b_value"], final_source="adjudicator", adjudicator_id="Synthetic Third Reviewer",
                   adjudicated_at_utc=datetime.now(timezone.utc).isoformat(), notes="Synthetic fixture: explicit failure visible in observation 3")
    write(path, rows)
    result = finalize_returns(al, bl, evidence, path, tmp_path / "final")
    assert result["adjudicated_disagreement_fields"] == 4
    assert result["windows"] == 2
    with (tmp_path / "final/final_human_annotations_long.csv").open(encoding="utf-8-sig", newline="") as stream:
        final = list(csv.DictReader(stream))
    assert all(row["test_output_status"] == "explicit_failure" for row in final)


def test_human_measurement_will_not_consume_ai_output_directory(tmp_path):
    from original_human_measurement import run
    with pytest.raises(ValueError, match="Missing final human label lock"):
        run(tmp_path, tmp_path / "key.csv", tmp_path / "design.csv", tmp_path / "stats")
    assert not (tmp_path / "stats").exists()
