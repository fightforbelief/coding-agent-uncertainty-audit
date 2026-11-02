"""Validate, lock and compare REAL independent first-round reviewer returns.

This tool does not create labels. Blank, incomplete and explicitly automated
returns fail validation. The reviewer identity is a declared identity, not an
identity check. Locks retain the actual submitted bytes and attestations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

COMMAND = {"definite_test_command", "definite_no_test_command", "ambiguous", "insufficient_evidence", "command_unrecoverable"}
EXECUTION = {"not_applicable", "command_issued", "execution_not_confirmed", "execution_confirmed", "unknown"}
OUTPUT = {"explicit_pass", "explicit_failure", "unknown_output", "no_output_observed", "not_applicable"}
PROCESS = {"not_started", "started", "unknown", "not_applicable"}
DETAIL = {"tests_executed", "command_not_found", "dependency_error", "collection_failure", "build_failure", "timeout_before_tests", "timeout_during_tests", "other_pre_test_failure", "not_applicable", "unknown"}
COMPOSITION = {"all_explicit_pass", "all_explicit_failure", "mixed_pass_and_failure", "unknown_output", "no_output_observed", "not_applicable"}
CORE = {"test_command_status": COMMAND, "execution_status": EXECUTION, "test_output_status": OUTPUT,
        "process_status": PROCESS, "execution_detail": DETAIL, "output_composition": COMPOSITION}
REQUIRED_FILES = ("review_queue.csv", "window_extension.csv", "event_worksheet.csv", "COVER_SHEET.csv")
RULES = "v3.5.1-rules-1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing or duplicate CSV headers: " + str(path))
        result = list(reader)
    if any(None in row for row in result):
        raise ValueError("Extra CSV values: " + str(path))
    return result


def id_map(rows, label):
    values = {row.get("review_id"): row for row in rows}
    if len(values) != len(rows) or "" in values or None in values:
        raise ValueError("Duplicate or missing review_id: " + label)
    return values


def parse_time(value):
    value = value.replace("Z", "+00:00")
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Reviewer timestamps need UTC offsets")
    return stamp.astimezone(timezone.utc)


def check_return(folder: Path, reference_evidence: Path, slot: str) -> tuple[dict, list[dict]]:
    errors = []
    for name in REQUIRED_FILES:
        if not (folder / name).is_file():
            errors.append("missing file: " + name)
    if errors:
        return dict(valid=False, errors=errors, reviewer_slot=slot), []
    evidence = id_map([json.loads(line) for line in reference_evidence.read_text(encoding="utf-8-sig").splitlines() if line.strip()], "evidence")
    queues = read_csv(folder / "review_queue.csv")
    extensions = read_csv(folder / "window_extension.csv")
    events = read_csv(folder / "event_worksheet.csv")
    covers = read_csv(folder / "COVER_SHEET.csv")
    queue = id_map(queues, "queue")
    extension = id_map(extensions, "extension")
    if set(queue) != set(evidence) or set(extension) != set(evidence):
        errors.append("review IDs must match the full evidence set exactly")
    cover = covers[0] if len(covers) == 1 else {}
    if len(covers) != 1:
        errors.append("cover sheet must contain exactly one reviewer")
    if cover.get("reviewer_slot") != slot or cover.get("rules_version") != RULES or cover.get("packet_role") != "independent_first_round":
        errors.append("cover slot/rules/independent_first_round declaration missing or wrong")
    name = cover.get("reviewer_name", "").strip()
    if not name or re.search(r"\b(?:codex|chatgpt|gpt|automated|model|ai)\b", name, re.I):
        errors.append("a real reviewer must declare their name; automated sources cannot be imported as human")
    try:
        start, finish = parse_time(cover.get("started_at_utc", "")), parse_time(cover.get("finished_at_utc", ""))
        if finish < start or finish > datetime.now(timezone.utc):
            errors.append("invalid first-round start/finish ordering or future finish")
    except (ValueError, TypeError):
        errors.append("valid first-round start and finish timestamps are required")
    forbidden_sources = {"annotation_source", "annotation_version", "semantic_spot_review", "requires_review"}
    if any(forbidden_sources.intersection(row) for row in queues + extensions):
        errors.append("automated annotation schema is not a human return")
    if (folder / "review_evidence.jsonl").exists() and sha(folder / "review_evidence.jsonl") != sha(reference_evidence):
        errors.append("reviewer evidence differs from reference")
    merged_rows = []
    incomplete = 0
    for review_id in sorted(set(evidence).intersection(queue).intersection(extension)):
        record, q, ext = evidence[review_id], queue[review_id], extension[review_id]
        for column in ("completed_rounds", "prefix5_boundary", "prefix15_boundary", "prefix_parse_status"):
            if str(q.get(column)) != str(record.get(column)):
                errors.append(f"{review_id}: evidence metadata changed: {column}")
        row = {"review_id": review_id}
        for window in ("r5", "r15"):
            prefix = f"{review_id}/{window}: "
            cells = {field: (q if field in {"test_command_status", "execution_status", "test_output_status"} else ext).get(f"{field}_{window}", "").strip() for field in CORE}
            if any(not value for value in cells.values()):
                incomplete += 1
            for field, valid in CORE.items():
                if cells[field] not in valid:
                    errors.append(prefix + "missing/invalid " + field)
            confidence = q.get(f"reviewer_confidence_{window}", "").strip()
            if confidence not in {"high", "moderate", "low"}:
                errors.append(prefix + "missing/invalid reviewer confidence")
            calls = record[window]
            valid_indexes = {call[key] for call in calls for key in ("action_index", "observation_index") if call.get(key) is not None}
            ref = q.get(f"supporting_message_indexes_{window}", "").strip()
            try:
                indexes = {int(x) for x in ref.split(";") if x.strip()}
                if not indexes.issubset(valid_indexes) or (calls and not indexes):
                    errors.append(prefix + "supporting indexes missing or outside prefix")
            except ValueError:
                indexes = set()
                errors.append(prefix + "supporting indexes must be semicolon-separated integers")
            quote = q.get(f"evidence_quotation_{window}", "").strip()
            if quote:
                found = any((call.get("action_index") in indexes and any(quote in (call.get(k) or "") for k in ("arguments", "extracted_command"))) or (call.get("observation_index") in indexes and quote in (call.get("observation") or "")) for call in calls)
                if not found:
                    errors.append(prefix + "quotation is not present at a cited message")
            elif cells["test_command_status"] == "definite_test_command":
                errors.append(prefix + "positive command judgement needs an evidence quotation")
            if cells["test_command_status"] == "definite_no_test_command":
                if any(cells[x] != "not_applicable" for x in CORE if x != "test_command_status"):
                    errors.append(prefix + "known negative requires not_applicable execution/output fields")
            if cells["test_output_status"] in {"explicit_pass", "explicit_failure"} and cells["execution_status"] != "execution_confirmed":
                errors.append(prefix + "explicit test result requires confirmed test execution")
            if cells["output_composition"] == "mixed_pass_and_failure" and cells["test_output_status"] != "explicit_failure":
                errors.append(prefix + "mixed results must retain explicit_failure")
            if cells["execution_detail"] in {"command_not_found", "dependency_error", "collection_failure", "build_failure", "timeout_before_tests", "other_pre_test_failure"} and cells["test_output_status"] in {"explicit_pass", "explicit_failure"}:
                errors.append(prefix + "pre-test failure cannot itself be a test-case result")
            if cells["test_command_status"] in {"ambiguous", "command_unrecoverable", "insufficient_evidence"} and not q.get(f"adjudication_notes_{window}", "").strip():
                errors.append(prefix + "uncertain judgement requires a reason")
            row.update({f"{field}_{window}": value for field, value in cells.items()})
            row[f"runner_family_{window}"] = q.get(f"runner_family_{window}", "").strip()
        if row["test_command_status_r5"] == "definite_test_command" and row["test_command_status_r15"] == "definite_no_test_command":
            errors.append(review_id + ": R5 positive cannot become R15 negative")
        if row["test_output_status_r5"] == "explicit_failure" and row["test_output_status_r15"] not in {"explicit_failure", ""}:
            errors.append(review_id + ": R15 must retain an explicit R5 failure")
        merged_rows.append(row)
    event_pairs = set()
    seen_event_order = set()
    for event in events:
        review_id, window = event.get("review_id"), event.get("window")
        if review_id not in evidence or window not in {"r5", "r15"}:
            errors.append("event worksheet contains unknown ID/window")
            continue
        event_pairs.add((review_id, window))
        try:
            order, index = int(event.get("event_order", "")), int(event.get("message_index", ""))
            if order < 1 or (review_id, window, order) in seen_event_order:
                errors.append("event order must be unique positive integer within window")
            seen_event_order.add((review_id, window, order))
            calls = evidence[review_id][window]
            possible = [c for c in calls if index in {c.get("action_index"), c.get("observation_index")}]
            if not possible:
                errors.append("event message index outside prefix")
            quotation = event.get("quotation", "").strip()
            if quotation and not any(quotation in (c.get(k) or "") for c in possible for k in ("arguments", "extracted_command", "observation")):
                errors.append("event quotation not found in evidence")
        except ValueError:
            errors.append("event order/message index must be integer")
        if event.get("execution_detail") not in DETAIL or event.get("output_event") not in OUTPUT:
            errors.append("event execution/output category invalid")
    for row in merged_rows:
        for window in ("r5", "r15"):
            if row[f"test_command_status_{window}"] == "definite_test_command" and (row["review_id"], window) not in event_pairs:
                errors.append(f"{row['review_id']}/{window}: positive command needs event worksheet evidence")
    summary = dict(valid=not errors, reviewer_slot=slot, declared_reviewer_name=name, reviewer_identity_verified_by_tool=False,
                   trajectories=len(evidence), windows=2 * len(evidence), incomplete_windows=incomplete,
                   error_count=len(errors), errors=errors, rules_version=RULES,
                   reference_evidence_sha256=sha(reference_evidence), files={name: sha(folder / name) for name in REQUIRED_FILES})
    return summary, merged_rows


def lock_return(folder: Path, evidence: Path, slot: str, output: Path) -> dict:
    summary, rows = check_return(folder, evidence, slot)
    if not summary["valid"]:
        raise ValueError(f"Cannot lock an invalid/incomplete return ({summary['error_count']} errors)")
    if output.exists():
        raise FileExistsError("Never overwrite locked first-round files")
    output.mkdir(parents=True)
    for name in REQUIRED_FILES:
        shutil.copy2(folder / name, output / name)
    summary["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["record_type"] = "submitted_independent_first_round_lock"
    (output / "FIRST_ROUND_LOCK.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "validated_core.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def verify_locked(folder: Path):
    lock = json.loads((folder / "FIRST_ROUND_LOCK.json").read_text(encoding="utf-8"))
    if not lock["valid"] or lock["record_type"] != "submitted_independent_first_round_lock":
        raise ValueError("Invalid first-round lock")
    for name, expected in lock["files"].items():
        if Path(name).name != name or sha(folder / name) != expected:
            raise ValueError("Locked first-round file changed")
    return lock


def compare_returns(a: Path, b: Path, evidence: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Use a new comparison directory")
    alock, block = verify_locked(a), verify_locked(b)
    if alock["declared_reviewer_name"].casefold() == block["declared_reviewer_name"].casefold():
        raise ValueError("The same declared reviewer cannot fill both independent slots")
    asummary, arows = check_return(a, evidence, "reviewer_a")
    bsummary, brows = check_return(b, evidence, "reviewer_b")
    if not asummary["valid"] or not bsummary["valid"]:
        raise ValueError("A returned first round no longer validates")
    am, bm = id_map(arows, "A core"), id_map(brows, "B core")
    differences, agreements = [], []
    for review_id in sorted(am):
        for window in ("r5", "r15"):
            for field in tuple(CORE) + ("runner_family",):
                column = f"{field}_{window}"
                av, bv = am[review_id][column], bm[review_id][column]
                equal = av == bv
                row = dict(review_id=review_id, field=column, reviewer_a_value=av, reviewer_b_value=bv,
                           disagreement=not equal, final_value="", final_source="", adjudicator_id="", adjudicated_at_utc="", rules_version=RULES, notes="")
                (agreements if equal else differences).append(row)
    output.mkdir(parents=True)
    fields = list((differences or agreements)[0])
    for name, values in (("disagreements_for_adjudicator.csv", differences), ("agreed_fields.csv", agreements)):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    summary = dict(status="FIRST ROUNDS COMPARED; DISAGREEMENTS REQUIRE ADJUDICATION", disagreement_fields=len(differences), agreed_fields=len(agreements),
                   reviewer_a_name=alock["declared_reviewer_name"], reviewer_b_name=block["declared_reviewer_name"],
                   reviewer_a_lock_sha256=sha(a / "FIRST_ROUND_LOCK.json"), reviewer_b_lock_sha256=sha(b / "FIRST_ROUND_LOCK.json"),
                   evidence_sha256=sha(evidence), no_final_values_invented=True)
    (output / "COMPARISON_MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def finalize_returns(a: Path, b: Path, evidence: Path, resolutions: Path, output: Path) -> dict:
    """Finalise agreed fields and HUMAN-adjudicated disagreements, then weight.

    Does not supply invented final values or silently choose reviewer A on a
    disagreement. No formal confidence interval or automatic measurement gate.
    """
    if output.exists():
        raise FileExistsError("Use a new final adjudication directory")
    alock, block = verify_locked(a), verify_locked(b)
    if alock["declared_reviewer_name"].casefold() == block["declared_reviewer_name"].casefold():
        raise ValueError("Two distinct declared reviewers are required")
    asummary, arows = check_return(a, evidence, "reviewer_a")
    bsummary, brows = check_return(b, evidence, "reviewer_b")
    if not asummary["valid"] or not bsummary["valid"]:
        raise ValueError("Both human first rounds must validate before finalisation")
    am, bm = id_map(arows, "A"), id_map(brows, "B")
    provided = read_csv(resolutions)
    rm = {(r.get("review_id"), r.get("field")): r for r in provided}
    if len(rm) != len(provided):
        raise ValueError("Duplicate adjudication field")
    expected = {(review_id, field) for review_id in am for field in am[review_id] if field != "review_id" and am[review_id][field] != bm[review_id][field]}
    if set(rm) != expected:
        raise ValueError("Resolutions must cover exactly the actual disagreement fields")
    rows, provenance = [], []
    human_names = {alock["declared_reviewer_name"].casefold(), block["declared_reviewer_name"].casefold()}
    for review_id in sorted(am):
        for window in ("r5", "r15"):
            row = dict(review_id=review_id, window=window)
            for field in tuple(CORE) + ("runner_family",):
                column = f"{field}_{window}"
                av, bv = am[review_id][column], bm[review_id][column]
                resolution = rm.get((review_id, column))
                value = av
                source = "independent_reviewer_a"
                if resolution:
                    if resolution.get("reviewer_a_value") != av or resolution.get("reviewer_b_value") != bv:
                        raise ValueError("Adjudication does not describe the locked first-round values")
                    adjudicator = resolution.get("adjudicator_id", "").strip()
                    if (not adjudicator or adjudicator.casefold() in human_names or re.search(r"\b(?:ai|gpt|codex|chatgpt|automated)\b", adjudicator, re.I)
                        or resolution.get("final_source") != "adjudicator" or resolution.get("rules_version") != RULES or not resolution.get("notes", "").strip()):
                        raise ValueError("Disagreements require a distinct declared human adjudicator, source, rules and rationale")
                    stamp = parse_time(resolution.get("adjudicated_at_utc", ""))
                    if stamp > datetime.now(timezone.utc) or stamp < max(parse_time(alock["locked_at_utc"]), parse_time(block["locked_at_utc"])):
                        raise ValueError("Adjudication must occur after both first-round locks and cannot be future-dated")
                    value = resolution.get("final_value", "").strip()
                    if field in CORE and value not in CORE[field]:
                        raise ValueError("Missing or invalid final adjudicated value")
                    source = "adjudicator"
                row[field] = value
                provenance.append(dict(review_id=review_id, field=column, reviewer_a_value=av, reviewer_b_value=bv,
                                       final_value=value, final_source=source, adjudicator_id=resolution.get("adjudicator_id", "") if resolution else "",
                                       rationale=resolution.get("notes", "") if resolution else "independent first-round agreement"))
            if row["test_command_status"] == "definite_no_test_command" and any(row[f] != "not_applicable" for f in CORE if f != "test_command_status"):
                raise ValueError("Final negative has applicable test execution/output fields")
            if row["test_output_status"] in {"explicit_pass", "explicit_failure"} and row["execution_status"] != "execution_confirmed":
                raise ValueError("Final explicit output requires confirmed execution")
            if row["output_composition"] == "mixed_pass_and_failure" and row["test_output_status"] != "explicit_failure":
                raise ValueError("Final mixed output must preserve explicit_failure")
            row.update(annotation_source="two_declared_independent_human_first_rounds_and_adjudication", independent_human_review=True)
            rows.append(row)
    for review_id in am:
        pair = {r["window"]: r for r in rows if r["review_id"] == review_id}
        if pair["r5"]["test_command_status"] == "definite_test_command" and pair["r15"]["test_command_status"] == "definite_no_test_command":
            raise ValueError("Final R5/R15 command nesting violation")
        if pair["r5"]["test_output_status"] == "explicit_failure" and pair["r15"]["test_output_status"] != "explicit_failure":
            raise ValueError("Final R15 must retain R5 explicit failure")
    output.mkdir(parents=True)
    for filename, values in (("final_human_annotations_long.csv", rows), ("final_field_provenance.csv", provenance)):
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = dict(status="SUBMITTED HUMAN JUDGEMENTS RECONCILED; MEASUREMENT GATE REQUIRES REVIEW",
                   source_type="declared_human_review_not_identity_verified_by_tool", windows=len(rows), adjudicated_disagreement_fields=len(expected),
                   reviewer_a_lock_sha256=sha(a / "FIRST_ROUND_LOCK.json"), reviewer_b_lock_sha256=sha(b / "FIRST_ROUND_LOCK.json"),
                   resolutions_sha256=sha(resolutions), evidence_sha256=sha(evidence), new_v3_models_run=False)
    summary["files"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    (output / "FINAL_HUMAN_LABEL_LOCK.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("inspect", "lock"):
        item = sub.add_parser(mode)
        item.add_argument("--return-dir", type=Path, required=True)
        item.add_argument("--evidence", type=Path, required=True)
        item.add_argument("--slot", choices=("reviewer_a", "reviewer_b"), required=True)
        if mode == "inspect":
            item.add_argument("--report", type=Path, required=True)
        else:
            item.add_argument("--output", type=Path, required=True)
    item = sub.add_parser("compare")
    item.add_argument("--reviewer-a", type=Path, required=True)
    item.add_argument("--reviewer-b", type=Path, required=True)
    item.add_argument("--evidence", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item = sub.add_parser("finalize")
    item.add_argument("--reviewer-a", type=Path, required=True)
    item.add_argument("--reviewer-b", type=Path, required=True)
    item.add_argument("--evidence", type=Path, required=True)
    item.add_argument("--resolutions", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "inspect":
        summary, _ = check_return(args.return_dir, args.evidence, args.slot)
        if args.report.exists():
            raise FileExistsError("Use a new inspection report filename")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if summary["valid"] else 2)
    if args.mode == "lock":
        summary = lock_return(args.return_dir, args.evidence, args.slot, args.output)
    elif args.mode == "compare":
        summary = compare_returns(args.reviewer_a, args.reviewer_b, args.evidence, args.output)
    else:
        summary = finalize_returns(args.reviewer_a, args.reviewer_b, args.evidence, args.resolutions, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
