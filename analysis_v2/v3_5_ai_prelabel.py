"""Conservative rule-assisted proposals, NEVER independent human adjudication.

Reads only the blinded evidence. Does not import the detector being audited,
read sealed keys, execute trajectory commands, or compute validation metrics.
All proposals require review, including apparently definite negatives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
from collections import Counter
from pathlib import Path

VERSION = "ai-rule-proposals-4"
SOURCE = "automated_rule_proposal_not_human"
READ_ONLY = {"ls", "cat", "head", "tail", "sed", "grep", "rg", "find", "pwd", "cd", "wc", "which", "file", "stat", "echo", "printf", "true", "false", "sleep", "sort", "uniq", "cut", "awk", "diff"}
SHELLS = {"bash", "sh", "zsh"}
SEPARATORS = {";", "&&", "||", "|", "&"}
FAIL = re.compile(r"(?im)^\s*(?:FAILED\s+\S+::|--- FAIL:|test result: FAILED|Tests:\s+\d+ failed|FAILURES!|FAILED\s*\(failures=[1-9])|\b[1-9]\d* (?:failed|failing)\b")
PASS = re.compile(r"(?im)\b[1-9]\d* passed\b|^ok\s+\S+\s+[\d.]+s|test result: ok\.|^OK\s*$|Tests:\s+[1-9]\d* passed")
STARTED_TESTS = re.compile(r"(?im)collected [1-9]\d* items?|^running [1-9]\d* tests|^=== RUN\s+|^Ran [1-9]\d* tests? in")
PREFAIL = [
    ("command_not_found", r"(?:command not found|pytest: not found|No such file or directory.*(?:pytest|cargo|npm))"),
    ("collection_failure", r"(?:ERROR collecting|errors? during collection|Interrupted:.*errors?)"),
    ("dependency_error", r"(?:ModuleNotFoundError|ImportError|No module named|Cannot find module)"),
    ("build_failure", r"(?:could not compile|\[build failed\]|BUILD FAILED)"),
    ("other_pre_test_failure", r"Missing script:\s*[\"']test[\"']"),
]


def command_kind(command: str, depth: int = 0) -> tuple[str, str, str]:
    """Recognise a narrow set; abstain on embedded code and unknown scripts."""
    if depth > 3 or any(x in command for x in ("<<", "$(", "`", "${")):
        return "ambiguous", "", "embedded_code_or_substitution"
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lex.whitespace += "\n"
        tokens = list(lex)
    except ValueError:
        return "ambiguous", "", "shell_parse_uncertain"
    # Newlines are command boundaries, but shlex cannot distinguish quoted
    # newlines after tokenisation. Abstain rather than silently lose a command.
    if "\n" in command:
        return "ambiguous", "", "multiline_command"
    segments, current = [], []
    skip_redirect_target = False
    for token in tokens + [";"]:
        if skip_redirect_target:
            skip_redirect_target = False
            continue
        if re.fullmatch(r"(?:[<>]{1,2}|[<>]&|&>)", token):
            if current and current[-1].isdigit():
                current.pop()
            skip_redirect_target = True
            continue
        if token in SEPARATORS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    found, unknown = [], []
    for words in segments:
        while words and (re.match(r"^[A-Za-z_]\w*=", words[0]) or words[0] in {"env", "time", "command"}):
            words = words[1:]
        if words and words[0] == "timeout":
            words = words[1:]
            while words and words[0].startswith("-"):
                words = words[1:]
            words = words[1:]
        if not words:
            continue
        exe = words[0].rsplit("/", 1)[-1]
        args = words[1:]
        if exe in SHELLS and len(args) == 2 and args[0] in {"-c", "-lc"}:
            kind, runner, reason = command_kind(args[1], depth + 1)
            if kind == "test":
                found.append(runner)
            elif kind != "non_test":
                unknown.append(reason)
        elif exe in {"pytest", "py.test", "jest", "mocha", "vitest", "rspec", "phpunit", "ctest", "tox"}:
            if any(x in args for x in ("--version", "--help", "-h", "--collect-only", "--collectonly")):
                unknown.append("runner_information_or_collection_only")
            else:
                found.append(exe)
        elif re.fullmatch(r"python[\d.]*", exe) and len(args) >= 2 and args[0] == "-m" and args[1] in {"pytest", "unittest"}:
            if any(x in args for x in ("--help", "--version", "--collect-only", "--collectonly")):
                unknown.append("runner_information_or_collection_only")
            else:
                found.append(args[1])
        elif exe in {"cargo", "go", "npm", "yarn", "pnpm", "mvn", "gradle", "gradlew", "make"}:
            if args and (args[0] in {"test", "tests", "test:unit", "test:integration"} or args[:2] == ["run", "test"]):
                found.append(exe + " test")
            elif exe == "cargo" and args and args[0] in {"build", "check", "fmt", "new", "metadata", "package"}:
                pass
            elif exe in {"npm", "yarn", "pnpm"} and args and args[0] in {"install", "add", "ci"}:
                # Install hooks can run arbitrary scripts. Never infer a negative.
                unknown.append("package_install_hooks")
            else:
                unknown.append("custom_build_or_runner_command")
        elif exe == "git" and args and args[0] in {"status", "diff", "log", "show", "grep", "ls-files", "rev-parse"}:
            pass
        elif exe in READ_ONLY:
            # awk/sed/find may themselves execute programs. Keep these cases open.
            if (exe == "find" and any(x in args for x in {"-exec", "-execdir"})) or exe == "awk" or (exe == "sed" and any(re.search(r"system\s*\(|[; ]e(?:[ ;]|$)", x) for x in args)):
                unknown.append("read_command_can_execute")
        elif exe in {"pip", "pip3"} and args and args[0] in {"show", "list", "freeze"}:
            pass
        else:
            unknown.append("unrecognised_command_or_script")
    if found:
        return "test", ";".join(sorted(set(found))), ";".join(sorted(set(unknown)))
    return ("ambiguous", "", ";".join(sorted(set(unknown)))) if unknown else ("non_test", "", "recognised_non_test_commands")


def output_proposal(text: str | None, runner: str) -> dict:
    text = text or ""
    if re.search(r"(?im)^\s*Tests\s+no tests\s*$", text):
        return dict(execution_status="execution_not_confirmed", process_status="started", execution_detail="collection_failure",
                    test_output_status="no_output_observed", output_composition="no_output_observed", quote=re.search(r"(?im)^\s*Tests\s+no tests\s*$", text).group(0),
                    reason="no_tests_executed_requires_collection_context_review")
    # Suite/file failures alone can reflect imports or collection, not failed cases.
    text_for_results = re.sub(r"(?im)^.*(?:Test Files|Test Suites).*$", "", text)
    failure, passed = FAIL.search(text_for_results), PASS.search(text_for_results)
    # These are suggestions; summaries can be quoted or produced by scripts.
    if failure or passed:
        match = failure or passed
        return dict(execution_status="execution_confirmed", process_status="started",
                    execution_detail="tests_executed", test_output_status="explicit_failure" if failure else "explicit_pass",
                    output_composition="mixed_pass_and_failure" if failure and passed else ("all_explicit_failure" if failure else "all_explicit_pass"),
                    quote=match.group(0).strip(), reason="test_summary_pattern_requires_context_review")
    for detail, pattern in PREFAIL:
        match = re.search(pattern, text, re.I)
        if match and not STARTED_TESTS.search(text):
            return dict(execution_status="execution_not_confirmed", process_status="started", execution_detail=detail,
                        test_output_status="no_output_observed", output_composition="no_output_observed",
                        quote=match.group(0), reason="pre_test_error_pattern_requires_context_review")
    if STARTED_TESTS.search(text):
        match = STARTED_TESTS.search(text)
        return dict(execution_status="execution_confirmed", process_status="started", execution_detail="tests_executed",
                    test_output_status="unknown_output", output_composition="unknown_output", quote=match.group(0), reason="test_started_without_result_summary")
    return dict(execution_status="unknown" if text else "command_issued", process_status="unknown", execution_detail="unknown",
                test_output_status="unknown_output" if text else "no_output_observed", output_composition="unknown_output" if text else "no_output_observed",
                quote="", reason="execution_or_output_requires_context_review")


def label_window(record: dict, window: str) -> tuple[dict, list[dict]]:
    events, kinds, reasons = [], [], []
    previous = None
    for call in record[window]:
        if call.get("command_extraction_status") == "non_execution":
            continue
        cmd = call.get("extracted_command")
        origin = call.get("action_index")
        poll = False
        if not cmd:
            try:
                arguments = json.loads(call.get("arguments") or "{}")
            except (ValueError, TypeError):
                arguments = {}
            # Only explicit terminal-input polls; never infer from empty text alone.
            if isinstance(arguments, dict) and arguments.get("command") == "" and str(arguments.get("is_input", "")).lower() == "true" and previous and previous[2] == call.get("tool"):
                cmd, origin, _tool = previous
                poll = True
            else:
                kinds.append("command_unrecoverable")
                reasons.append("unrecoverable_command")
                previous = None
                events.append(dict(review_id=record["review_id"], window=window, action_index=call.get("action_index"), observation_index=call.get("observation_index"), kind="command_unrecoverable", reason="unrecoverable_command", command="", command_origin_index="", output_excerpt=(call.get("observation") or "")[:500], runner=""))
                continue
        elif call.get("tool") in {"execute_bash", "bash"}:
            previous = (cmd, origin, call.get("tool"))
        kind, runner, reason = command_kind(cmd)
        kinds.append(kind)
        if call.get("observation_index") is None:
            reasons.append("missing_observation")
        output = output_proposal(call.get("observation"), runner) if kind == "test" else {}
        output_reason = output.pop("reason", "")
        events.append(dict(review_id=record["review_id"], window=window, action_index=call.get("action_index"), observation_index=call.get("observation_index"), kind=kind, reason=";".join(x for x in (reason, output_reason) if x), command=cmd, command_origin_index=origin, continued_poll=poll, runner=runner, output_excerpt=(call.get("observation") or "")[:500], **output))
        if kind == "ambiguous" or (kind == "test" and reason):
            reasons.append(reason)
    tests = [event for event in events if event["kind"] == "test"]
    if tests:
        status = "definite_test_command"
    elif "ambiguous" in kinds:
        status = "ambiguous"
    elif "command_unrecoverable" in kinds:
        status = "command_unrecoverable"
    elif reasons or record.get("prefix_parse_status") != "ok" or record.get(f"prefix{window[1:]}_boundary") != "match":
        status = "insufficient_evidence"
    else:
        status = "definite_no_test_command"
    decisive = next((x for x in tests if x.get("test_output_status") == "explicit_failure"), None)
    decisive = decisive or next((x for x in tests if x.get("test_output_status") == "explicit_pass"), None)
    decisive = decisive or next((x for x in tests if x.get("execution_status") == "execution_confirmed"), None)
    decisive = decisive or (tests[-1] if tests else None)
    if decisive:
        result = {key: decisive[key] for key in ("execution_status", "process_status", "execution_detail", "test_output_status", "output_composition")}
        if any(x.get("test_output_status") == "explicit_failure" for x in tests) and any(x.get("test_output_status") == "explicit_pass" or x.get("output_composition") == "mixed_pass_and_failure" for x in tests):
            result["output_composition"] = "mixed_pass_and_failure"
        quote = decisive.get("quote") or decisive["command"][:300]
        quote_index = decisive["observation_index"] if decisive.get("quote") else decisive["command_origin_index"]
        indexes = ";".join(str(i) for i in sorted({x["command_origin_index"] for x in tests if x["command_origin_index"] is not None} | ({quote_index} if quote_index is not None else set())))
        runner = ";".join(sorted({x["runner"] for x in tests}))
        reasons.extend(x["reason"] for x in tests if x["reason"])
    else:
        na = status == "definite_no_test_command"
        result = dict(execution_status="not_applicable" if na else "unknown", process_status="not_applicable" if na else "unknown", execution_detail="not_applicable" if na else "unknown", test_output_status="not_applicable" if na else "unknown_output", output_composition="not_applicable" if na else "unknown_output")
        quote = " | ".join(x["command"][:100] for x in events[:3])
        indexes = ";".join(str(x["action_index"]) for x in events if x["action_index"] is not None)
        runner = ""
    result.update(test_command_status=status, supporting_message_indexes=indexes, runner_family=runner, evidence_quotation=quote,
                  reviewer_confidence="low", adjudication_notes="AUTOMATED PROPOSAL; NOT HUMAN; REVIEW REQUIRED: " + ";".join(sorted(set(reasons))),
                  annotation_source=SOURCE, requires_review=True)
    return result, events


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(evidence: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Use a new output directory; never overwrite review work")
    records = [json.loads(line) for line in evidence.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = [x["review_id"] for x in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate review IDs")
    forbidden = {"resolved", "stratum", "trajectory_id", "issue_id", "inclusion_probability", "scaffold", "model"}
    def check(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("Unblinded input rejected")
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(records)
    output.mkdir(parents=True)
    queue, extensions, long_rows, events = [], [], [], []
    for record in records:
        row = {key: record[key] for key in ("review_id", "completed_rounds", "prefix5_boundary", "prefix15_boundary", "prefix_parse_status")}
        ext = {"review_id": record["review_id"]}
        for window in ("r5", "r15"):
            result, window_events = label_window(record, window)
            events.extend(window_events)
            long_rows.append(dict(review_id=record["review_id"], window=window, **result))
            for key, value in result.items():
                if key in {"annotation_source", "requires_review"}:
                    continue
                target = ext if key in {"process_status", "execution_detail", "output_composition"} else row
                target[f"{key}_{window}"] = value
        row.update(annotation_source=SOURCE, requires_review=True)
        queue.append(row)
        extensions.append(ext)
    write_csv(output / "ai_proposed_review_queue.csv", queue)
    write_csv(output / "ai_proposed_window_extension.csv", extensions)
    write_csv(output / "ai_proposals_long.csv", long_rows)
    with (output / "ai_call_proposals.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    uncertain = [x for x in long_rows if x["test_command_status"] not in {"definite_test_command", "definite_no_test_command"}]
    write_csv(output / "priority_review.csv", uncertain)
    summary = dict(status="AUTOMATED PRELABEL COMPLETE — HUMAN VALIDATION PENDING", version=VERSION, annotation_source=SOURCE,
                   records=len(records), windows=len(long_rows), all_require_review=True, uncertain_command_windows=len(uncertain),
                   counts={window: dict(Counter(x["test_command_status"] for x in long_rows if x["window"] == window)) for window in ("r5", "r15")},
                   continued_poll_events=sum(bool(x.get("continued_poll")) for x in events),
                   evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   detector_metrics_computed=False, independent_human_reviews=0)
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.evidence, args.output), ensure_ascii=False, indent=2))
