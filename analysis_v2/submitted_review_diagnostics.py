"""Diagnose submitted A/B returns without asserting human provenance.

Preserves all decisions. Joined quotation fragments are checked individually
against cited evidence; missing spans are not fabricated. Suggested poll-index
corrections are separate from submitted worksheets and never applied here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from original_review_control import CORE, check_return, id_map, read_csv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(text):
    return text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")


def sources(calls, indexes):
    result = []
    for call in calls:
        for key in ("arguments", "extracted_command"):
            if call.get("action_index") in indexes and call.get(key):
                result.append(dict(message_index=call["action_index"], source_field=key, text=canonical(call[key])))
        if call.get("observation_index") in indexes and call.get("observation"):
            result.append(dict(message_index=call["observation_index"], source_field="observation", text=canonical(call["observation"])))
    return result


def match_excerpts(quotation, source_items):
    """Prefer whole quotations, then longest exact parts at explicit separators.

    Does not tokenize arbitrary words, ignore missing text or collapse whitespace.
    A literal shell pipe remains within one fragment whenever it matches evidence.
    """
    text = canonical(quotation).strip()
    if not text:
        return dict(status="empty", fragments=[])
    for source in source_items:
        if text in source["text"]:
            return dict(status="whole_exact", fragments=[dict(text=text, message_index=source["message_index"], source_field=source["source_field"])])
    if not re.sub(r"\[\u2026\]|\||\s", "", text):
        return dict(status="empty_separator_only", fragments=[])
    # Delimiters can overlap a literal trailing shell/compiler pipe. Lookahead
    # retains both possible boundaries instead of skipping the true separator.
    boundaries = [(match.start(), match.start() + len(match.group(1))) for match in re.finditer(r"(?=( \[\u2026\] | \| ))", text)]

    @lru_cache(None)
    def partition(start):
        ends = [(len(text), len(text))]
        ends.extend((left, right) for left, right in boundaries if left >= start)
        for end, next_start in sorted(ends, reverse=True):
            part = text[start:end].strip()
            if not part:
                continue
            found = next((source for source in source_items if part in source["text"]), None)
            if found is None:
                continue
            fragment = dict(text=part, message_index=found["message_index"], source_field=found["source_field"])
            if end == len(text):
                return (fragment,)
            tail = partition(next_start)
            if tail is not None:
                return (fragment,) + tail
        return None

    fragments = partition(0)
    return dict(status="joined_exact_fragments" if fragments else "unmatched", fragments=list(fragments or []))


def poll_origins(calls):
    active, origins = {}, {}
    for call in calls:
        tool = call.get("tool")
        if tool not in {"execute_bash", "bash"}:
            continue
        command = call.get("extracted_command")
        try:
            args = json.loads(call.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        is_poll = not command and args.get("command") == "" and tool in active
        if is_poll:
            origins[call["action_index"]] = active[tool]
        observation = (call.get("observation") or "").lower()
        pending = any(token in observation for token in ("no new output", "may wait longer", "command timed out", "still running"))
        if command and command not in {"C-c", "C-d", "C-z", "y", "n", "yes", "no"}:
            if pending:
                active[tool] = call["action_index"]
            else:
                active.pop(tool, None)
        elif is_poll and not pending:
            active.pop(tool, None)
    return origins


def write(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(intake: Path, evidence_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Use a fresh diagnostics directory")
    evidence = id_map([json.loads(line) for line in evidence_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()], "reference")
    output.mkdir(parents=True)
    parts, excerpt_checks, event_checks, corrections, validation, cores = [], [], [], [], {}, {}
    queues = {}
    for slot in ("reviewer_a", "reviewer_b"):
        folder = intake / slot
        report, values = check_return(folder, evidence_path, slot)
        validation[slot] = report
        cores[slot] = id_map(values, slot)
        queues[slot] = id_map(read_csv(folder / "review_queue.csv"), slot)
        (output / f"{slot}_strict_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for review_id, row in queues[slot].items():
            for window in ("r5", "r15"):
                refs = {int(x) for x in row[f"supporting_message_indexes_{window}"].split(";") if x.strip()}
                matched = match_excerpts(row[f"evidence_quotation_{window}"], sources(evidence[review_id][window], refs))
                excerpt_checks.append(dict(slot=slot, review_id=review_id, window=window, status=matched["status"], fragment_count=len(matched["fragments"])))
                if matched["fragments"]:
                    parts.append(dict(slot=slot, review_id=review_id, window=window, submitted_quotation=row[f"evidence_quotation_{window}"], **matched))
        for event in read_csv(folder / "event_worksheet.csv"):
            review_id, window, index = event["review_id"], event["window"], int(event["message_index"])
            calls = evidence[review_id][window]
            matched = match_excerpts(event.get("quotation", ""), sources(calls, {index}))
            kind = matched["status"]
            correction_index = None
            if kind == "unmatched":
                original = poll_origins(calls).get(index)
                if original is not None:
                    parent_match = match_excerpts(event.get("quotation", ""), sources(calls, {original}))
                    if parent_match["status"] not in {"unmatched", "empty"}:
                        kind = "poll_command_quote_matches_origin_not_cited_poll"
                        correction_index = original
                        corrections.append(dict(slot=slot, review_id=review_id, window=window, event_order=event["event_order"], submitted_message_index=index,
                                                proposed_command_origin_index=original, quotation=event["quotation"], reason="Exact command quotation at the verified pending-process origin; submitted index is the empty polling call", applied=False))
            event_checks.append(dict(slot=slot, review_id=review_id, window=window, event_order=event["event_order"], message_index=index, status=kind,
                                     proposed_origin_index=correction_index))
    am, bm = cores["reviewer_a"], cores["reviewer_b"]
    if set(am) != set(evidence) or set(bm) != set(evidence):
        raise ValueError("Cannot compare incomplete ID sets")
    differences, agreement = [], []
    counts = Counter()
    disagreement_windows = set()
    fields = tuple(CORE) + ("runner_family",)
    for review_id in sorted(evidence):
        for window in ("r5", "r15"):
            for field in fields:
                column = f"{field}_{window}"
                av, bv = am[review_id][column], bm[review_id][column]
                counts[(column, av == bv)] += 1
                if av != bv:
                    disagreement_windows.add((review_id, window))
                    differences.append(dict(review_id=review_id, window=window, field=column, submitted_a_value=av, submitted_b_value=bv,
                                            submitted_a_quotation=queues["reviewer_a"][review_id][f"evidence_quotation_{window}"],
                                            submitted_b_quotation=queues["reviewer_b"][review_id][f"evidence_quotation_{window}"],
                                            final_value="", final_source="", adjudicator_id="", adjudicated_at_utc="", notes=""))
    for field in fields:
        for window in ("r5", "r15"):
            column = f"{field}_{window}"
            agreed, disagreed = counts[(column, True)], counts[(column, False)]
            agreement.append(dict(field=column, same_value_count=agreed, different_value_count=disagreed, same_value_fraction=agreed/(agreed+disagreed), interpretation="submitted-table agreement only; human independence and measurement validity not established"))
    write(output / "quotation_format_diagnostics.csv", excerpt_checks)
    write(output / "event_reference_diagnostics.csv", event_checks)
    write(output / "event_index_correction_proposals.csv", corrections, ["slot", "review_id", "window", "event_order", "submitted_message_index", "proposed_command_origin_index", "quotation", "reason", "applied"])
    write(output / "submitted_field_agreement.csv", agreement)
    write(output / "disagreements_for_review.csv", differences)
    with (output / "verified_quotation_fragments.jsonl").open("w", encoding="utf-8") as stream:
        for part in parts:
            stream.write(json.dumps(part, ensure_ascii=False) + "\n")
    summary = dict(status="RETURNS RECEIVED; TECHNICAL DIAGNOSTICS COMPLETE; HUMAN PROVENANCE AND ADJUDICATION UNRESOLVED",
                   trajectories_per_submission=len(evidence), windows_per_submission=2*len(evidence),
                   incomplete_windows={slot: validation[slot]["incomplete_windows"] for slot in validation},
                   evidence_byte_identical={slot: sha(intake/slot/"review_evidence.jsonl") == sha(evidence_path) for slot in validation},
                   quote_status_counts={slot: dict(Counter(x["status"] for x in excerpt_checks if x["slot"] == slot)) for slot in validation},
                   event_status_counts={slot: dict(Counter(x["status"] for x in event_checks if x["slot"] == slot)) for slot in validation},
                   differing_fields=len(differences), windows_with_any_different_core_or_runner_field=len(disagreement_windows),
                   trajectories_with_any_difference=len({x[0] for x in disagreement_windows}),
                   citation_changes_applied=False, judgement_values_changed=False, original_zip_files_modified=False,
                   independent_human_first_round_lock_created=False, formal_human_measurement_statistics_run=False,
                   reference_evidence_sha256=sha(evidence_path), submitted_file_hashes={slot:{p.name:sha(p) for p in (intake/slot).iterdir() if p.is_file()} for slot in validation},
                   code_sha256=sha(Path(__file__)))
    summary["outputs"] = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output / "DIAGNOSTICS_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.intake, args.evidence, args.output)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"outputs", "submitted_file_hashes"}}, ensure_ascii=False, indent=2))
