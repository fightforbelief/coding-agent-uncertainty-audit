"""Finish a conservative automated evidence annotation, not a human gold set.

The program reads every blinded call and uses static command/code inspection.
It never executes evidence, imports project code from evidence, or reads keys.
The separate, explicitly scoped Codex semantic spot review is applied as an
override with its own provenance. Ambiguity is an allowed final annotation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import posixpath
import re
import shlex
from collections import Counter
from pathlib import Path

from v3_5_ai_prelabel import output_proposal, write_csv

VERSION = "automated-evidence-annotation-1"
SOURCE = "automated_static_evidence_annotation_not_human"
FORBIDDEN = {"resolved", "stratum", "trajectory_id", "issue_id", "inclusion_probability", "scaffold", "model", "sampling_stratum"}
NONEXEC = {"str_replace_editor", "think", "finish", "open", "edit", "create", "submit"}
SIMPLE = {"ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "cd", "wc", "which", "file", "stat", "echo", "printf", "true", "false", "sleep", "sort", "uniq", "cut", "diff", "mkdir", "rm", "rmdir", "mv", "cp", "touch", "chmod", "ln", "ps", "kill", "pkill", "pgrep", "tr", "tee", "readlink", "realpath", "du", "df", "date", "uname", "type", "command", "export", "unset", "git", "curl", "wget", "tar", "unzip", "rustc", "rustfmt", "javac", "tsc"}
RUNNERS = {"pytest", "py.test", "jest", "mocha", "vitest", "rspec", "phpunit", "ctest", "tox", "nosetests", "ava", "tap", "tape"}
INFO = {"--help", "--version", "-h", "--collect-only", "--collectonly", "--list-tests", "--listTests"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_blind(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("Unblinded evidence rejected")
        for item in value.values():
            check_blind(item)
    elif isinstance(value, list):
        for item in value:
            check_blind(item)


def segments(command: str) -> list[list[str]]:
    # Heredoc contents are not shell commands. Abstain rather than tokenize them.
    if any(x in command for x in ("<<", "$(", "`")):
        raise ValueError("heredoc_or_command_substitution")
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace = " \t\r"
    lexer.commenters = "#"
    tokens = list(lexer)
    result, current, redirect = [], [], False
    for token in tokens + [";"]:
        if redirect:
            redirect = False
            continue
        if re.fullmatch(r"(?:[<>]{1,2}|[<>]&|&>)", token):
            if current and current[-1].isdigit():
                current.pop()
            redirect = True
        elif token in {";", "&&", "||", "|", "&", "\n"}:
            if current:
                result.append(current)
                current = []
        else:
            current.append(token)
    return result


def inspect_python(code: str) -> tuple[str, str, str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "ambiguous", "", "python_code_not_statically_parseable"
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(x.name.split(".")[0] for x in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
    # Explicit programmatic runners are a positive command-intent observation.
    if names.intersection({"pytest", "unittest", "nose", "nose2"}):
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in {"main", "run", "runTests", "runModule", "run_suite"} for n in ast.walk(tree)):
            return "test", "python-programmatic-runner", "visible_programmatic_test_runner"
        return "ambiguous", "", "test_library_present_without_clear_invocation"
    if names.intersection({"exec", "eval", "compile", "__import__", "importlib", "subprocess", "system", "popen", "runpy"}):
        return "ambiguous", "", "dynamic_code_or_child_process"
    return "non_test", "", "visible_python_snippet_has_no_formal_test_runner"


def inspect_node(code: str) -> tuple[str, str, str]:
    if re.search(r"\b(?:jest|mocha|vitest|node:test|child_process|exec|eval|spawn)\b", code):
        return "ambiguous", "", "javascript_runner_or_dynamic_execution_needs_context"
    return "non_test", "", "visible_inline_javascript_reproduction_without_runner"


def one_command(words: list[str], created: dict[str, str], depth=0) -> tuple[str, str, str]:
    if depth > 5:
        return "ambiguous", "", "wrapper_depth"
    words = list(words)
    while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0], re.S):
        words.pop(0)
    if not words:
        return "non_test", "", "environment_assignment_only"
    exe, args = posixpath.basename(words[0]), words[1:]
    if exe in {"env", "time", "command", "sudo", "nice"}:
        while args and (args[0].startswith("-") or re.fullmatch(r"[A-Za-z_]\w*=.*", args[0], re.S)):
            args = args[1:]
        return one_command(args, created, depth + 1)
    if exe == "timeout":
        if len(args) < 2 or args[0].startswith("-"):
            return "ambiguous", "", "timeout_option_syntax"
        return one_command(args[1:], created, depth + 1)
    if exe in {"bash", "sh", "zsh"} and len(args) >= 2 and args[0] in {"-c", "-lc"}:
        return classify(args[1], created, depth + 1)
    if exe in RUNNERS:
        if set(args).intersection(INFO) or (exe == "ctest" and "-N" in args):
            return "ambiguous", exe, "runner_information_or_collection_only"
        return "test", exe, "visible_formal_test_runner"
    if exe in {"npx", "bunx", "bundle"}:
        if exe == "bundle" and args and args[0] == "exec":
            args = args[1:]
        while args and args[0] in {"--yes", "-y", "--no-install"}:
            args = args[1:]
        if args and args[0].split("@")[0] in RUNNERS:
            return one_command([args[0].split("@")[0]] + args[1:], created, depth + 1)
        return "ambiguous", "", "npx_or_bundle_custom_program"
    if re.fullmatch(r"python[\d.]*", exe):
        if len(args) >= 2 and args[0] == "-m" and args[1] in {"pytest", "unittest", "nose", "nose2"}:
            if set(args).intersection(INFO):
                return "ambiguous", args[1], "runner_information_or_collection_only"
            return "test", args[1], "python_module_test_runner"
        if len(args) >= 2 and args[0] == "-c":
            return inspect_python(args[1])
        if args and posixpath.basename(args[0]) in {"manage.py", "runtests.py"} and ("test" in args or posixpath.basename(args[0]) == "runtests.py"):
            return "test", "django-or-project-runtests", "visible_project_test_entrypoint"
        if args:
            code = created.get(args[0])
            if code is not None:
                kind, runner, reason = inspect_python(code)
                if kind == "non_test":
                    return kind, runner, "agent_created_reproduction_script_without_runner"
                return kind, runner, "agent_created_script:" + reason
        return "ambiguous", "", "script_contents_or_original_test_provenance_unavailable"
    if exe in {"node", "nodejs"} and len(args) >= 2 and args[0] in {"-e", "--eval"}:
        return inspect_node(args[1])
    if exe in {"node", "nodejs"} and "--test" in args:
        return "test", "node-test", "visible_node_test_runner"
    if exe in {"go", "cargo"}:
        if args and args[0] == "test":
            if exe == "cargo" and "--no-run" in args:
                return "ambiguous", exe, "compile_tests_without_running"
            return "test", exe + " test", "visible_language_test_command"
        if args and args[0] in {"build", "check", "fmt", "new", "metadata", "package", "mod", "version", "env", "tool", "doc", "list", "vet", "clean"}:
            return "non_test", "", "language_build_inspection_or_dependency_command"
        return "ambiguous", "", "custom_language_program_or_subcommand"
    if exe in {"npm", "yarn", "pnpm"}:
        if args and (args[0] in {"test", "test:unit", "test:integration"} or args[:2] in (["run", "test"], ["run", "test:unit"], ["run", "test:integration"])):
            return "test", exe + " test", "visible_package_test_script"
        return "ambiguous", "", "package_script_or_install_hooks_need_context"
    if exe in {"mvn", "mvnw"}:
        if "test" in args and not any(x.startswith(("-DskipTests", "-Dmaven.test.skip")) for x in args):
            return "test", "maven", "maven_test_goal"
        if "compile" in args or any(x.startswith(("-DskipTests", "-Dmaven.test.skip")) for x in args):
            return "non_test", "", "maven_compile_or_tests_explicitly_skipped"
        return "ambiguous", "", "maven_lifecycle_may_include_tests"
    if exe in {"gradle", "gradlew", "make"}:
        if any(x in {"test", "tests", "check", "test-unit"} for x in args):
            return "test", exe, "visible_build_system_test_target"
        return "ambiguous", "", "custom_build_target"
    if exe == "xargs":
        # Only default or plainly specified commands; uncertain options abstain.
        if not args:
            return "non_test", "", "xargs_default_echo"
        if args[0].startswith("-"):
            return "ambiguous", "", "xargs_options_need_context"
        return one_command(args, created, depth + 1)
    if exe == "find" and set(args).intersection({"-exec", "-execdir"}):
        start = next(i for i, x in enumerate(args) if x in {"-exec", "-execdir"}) + 1
        nested = args[start:]
        if nested and nested[-1] in {";", "+"}:
            nested = nested[:-1]
        return one_command(nested, created, depth + 1)
    if exe in {"awk", "sed", "perl"}:
        return "ambiguous", "", "text_processor_with_embedded_code"
    if exe in SIMPLE:
        if exe == "git" and args and args[0] in {"run", "bisect"}:
            return "ambiguous", "", "git_can_run_custom_commands"
        return "non_test", "", "visible_file_inspection_or_non_test_utility"
    return "ambiguous", "", "custom_program_or_unseen_script"


def classify(command: str, created: dict[str, str], depth=0, cwd: str | None = None) -> tuple[str, str, str]:
    if command in {"C-c", "C-d", "C-z", "y", "n", "yes", "no"}:
        return "control", "", "terminal_control_or_prompt_response"
    try:
        parsed = segments(command)
    except ValueError as exc:
        return "ambiguous", "", str(exc)
    decisions = []
    for words in parsed:
        if len(words) == 2 and words[0] == "cd" and not words[1].startswith(("-", "$", "~")):
            cwd = posixpath.normpath(words[1] if words[1].startswith("/") else posixpath.join(cwd, words[1])) if (words[1].startswith("/") or cwd) else None
        visible_created = dict(created)
        if cwd:
            visible_created.update({posixpath.relpath(path, cwd): code for path, code in created.items() if path.startswith("/")})
        decisions.append(one_command(words, visible_created, depth))
    tests = [x for x in decisions if x[0] == "test"]
    if tests:
        return "test", ";".join(sorted({x[1] for x in tests})), ";".join(sorted({x[2] for x in decisions}))
    if any(x[0] == "ambiguous" for x in decisions):
        return "ambiguous", "", ";".join(sorted({x[2] for x in decisions if x[0] == "ambiguous"}))
    return "non_test", "", ";".join(sorted({x[2] for x in decisions})) or "empty_shell_command"


def extract_created(call: dict, created: dict[str, str]):
    try:
        data = json.loads(call.get("arguments") or "{}")
    except (ValueError, TypeError):
        return
    if isinstance(data, dict) and data.get("command") == "create" and isinstance(data.get("file_text"), str):
        path = data.get("path")
        if isinstance(path, str) and path.startswith("/"):
            created[posixpath.normpath(path)] = data["file_text"]


def annotate_window(record: dict, window: str) -> tuple[dict, list[dict]]:
    created, active, working_dirs = {}, {}, {}
    events = []
    calls = record[window]
    for call in calls:
        extract_created(call, created)
        if call.get("command_extraction_status") == "non_execution" or call.get("tool") in NONEXEC:
            continue
        command = call.get("extracted_command")
        tool = call.get("tool")
        origin = call.get("action_index")
        continued = False
        try:
            arguments = json.loads(call.get("arguments") or "{}")
        except (ValueError, TypeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        if not command and isinstance(arguments.get("command"), str) and arguments["command"]:
            command = arguments["command"]
        if not command and arguments.get("command") == "" and tool in active:
            command, origin = active[tool]
            continued = True
        if not command:
            kind, runner, reason = "command_unrecoverable", "", "no_visible_command_or_linkable_active_process"
        else:
            kind, runner, reason = classify(command, created, cwd=working_dirs.get(tool))
        event = dict(review_id=record["review_id"], window=window, round=call.get("round"), action_index=call.get("action_index"),
                     observation_index=call.get("observation_index"), command_origin_index=origin,
                     command=command or "", command_kind=kind, runner_family=runner, reason=reason, continued_poll=continued,
                     observation_sha256=hashlib.sha256((call.get("observation") or "").encode("utf-8")).hexdigest())
        if kind == "test":
            result = output_proposal(call.get("observation"), runner)
            result["output_reason"] = result.pop("reason")
            event.update(result)
        events.append(event)
        observation = call.get("observation") or ""
        working_dir_match = re.search(r"\[Current working directory: ([^\]\n]+)\]", observation)
        if working_dir_match:
            working_dirs[tool] = working_dir_match.group(1)
        pending = any(x in observation.lower() for x in ("no new output", "may wait longer", "command timed out", "still running"))
        if command and kind != "control" and pending:
            active[tool] = (command, origin)
        elif not pending:
            active.pop(tool, None)
    tests = [x for x in events if x["command_kind"] == "test"]
    kinds = {x["command_kind"] for x in events}
    missing = any(c.get("observation_index") is None and c.get("command_extraction_status") != "non_execution" for c in calls)
    boundary = record.get(f"prefix{window[1:]}_boundary")
    if tests:
        status = "definite_test_command"
    elif "ambiguous" in kinds:
        status = "ambiguous"
    elif "command_unrecoverable" in kinds:
        status = "command_unrecoverable"
    elif missing or boundary != "match" or record.get("prefix_parse_status") != "ok":
        status = "insufficient_evidence"
    else:
        status = "definite_no_test_command"
    execution = "not_applicable" if status == "definite_no_test_command" else "unknown"
    output = "not_applicable" if status == "definite_no_test_command" else "unknown_output"
    detail, process, composition = execution, execution, output
    quote, quote_index, quote_kind = "", "", ""
    if tests:
        confirmed = [x for x in tests if x["execution_status"] == "execution_confirmed"]
        failed = [x for x in tests if x["test_output_status"] == "explicit_failure"]
        passed = [x for x in tests if x["test_output_status"] == "explicit_pass" or x["output_composition"] == "mixed_pass_and_failure"]
        prefail = [x for x in tests if x["execution_status"] == "execution_not_confirmed"]
        decisive = (failed or passed or confirmed or prefail or tests)[0]
        execution = "execution_confirmed" if confirmed else ("execution_not_confirmed" if prefail else decisive["execution_status"])
        process = "started" if any(x["process_status"] == "started" for x in tests) else "unknown"
        detail = "tests_executed" if confirmed else decisive["execution_detail"]
        output = "explicit_failure" if failed else ("explicit_pass" if passed else decisive["test_output_status"])
        composition = "mixed_pass_and_failure" if failed and passed else decisive["output_composition"]
        quote = decisive.get("quote") or decisive["command"]
        quote_index = decisive["observation_index"] if decisive.get("quote") else decisive["command_origin_index"]
        quote_kind = "observation" if decisive.get("quote") else "command"
    else:
        uncertain = [x for x in events if x["command_kind"] in {"ambiguous", "command_unrecoverable"}]
        if uncertain:
            decisive = uncertain[0]
            quote = decisive["command"]
            quote_index = decisive["command_origin_index"]
            quote_kind = "command" if quote else ""
    row = dict(review_id=record["review_id"], window=window, test_command_status=status, execution_status=execution,
               process_status=process, execution_detail=detail, test_output_status=output, output_composition=composition,
               runner_family=";".join(sorted({x["runner_family"] for x in tests})),
               supporting_message_indexes=";".join(str(x) for x in sorted({c["action_index"] for c in calls if c.get("action_index") is not None} | ({quote_index} if isinstance(quote_index, int) else set()))),
               evidence_quotation=quote, quotation_message_index=quote_index, quotation_kind=quote_kind,
               annotation_source=SOURCE, annotation_version=VERSION, semantic_spot_review=False, independent_human_review=False,
               decision_status="annotated_with_abstention" if status in {"ambiguous", "command_unrecoverable", "insufficient_evidence"} else "automated_annotation",
               uncertainty_reason=";".join(sorted({x["reason"] for x in events if x["command_kind"] in {"ambiguous", "command_unrecoverable"}})),
               reviewed_call_count=len(calls), execution_call_count=len(events), prefix_boundary=boundary)
    return row, events


def apply_spot_reviews(rows: list[dict], reviews: dict, evidence: dict):
    lookup = {(r["review_id"], r["window"]): r for r in rows}
    for review in reviews["reviews"]:
        for window in review["windows"]:
            row = lookup[(review["review_id"], window)]
            for field in ("test_command_status", "execution_status", "test_output_status", "process_status", "execution_detail", "output_composition"):
                if field in review:
                    row[field] = review[field]
            if row["test_command_status"] == "definite_no_test_command":
                row.update(process_status="not_applicable", execution_detail="not_applicable", output_composition="not_applicable", runner_family="")
            row.update(annotation_source=reviews["annotation_source"], semantic_spot_review=True, decision_status="ai_semantic_spot_review", uncertainty_reason=review["rationale"])
            # The prior notes cite different parts of each window. Verify each.
            refs = review["supporting_message_indexes_by_window"][window]
            valid = {c[k] for c in evidence[row["review_id"]][window] for k in ("action_index", "observation_index") if c.get(k) is not None}
            if not set(refs).issubset(valid):
                raise ValueError("Spot review cites an index outside its window")
            row["supporting_message_indexes"] = ";".join(map(str, refs))
            row["evidence_quotation"] = review.get("evidence_quotation", "")
            row["quotation_message_index"], row["quotation_kind"] = "", ""
            if row["evidence_quotation"]:
                matched = [c for c in evidence[row["review_id"]][window] if row["evidence_quotation"] in (c.get("observation") or "")]
                if not matched:
                    raise ValueError("Spot review quotation does not match evidence")
                row["quotation_message_index"] = matched[0]["observation_index"]
                row["quotation_kind"] = "observation"


def validate(rows: list[dict], evidence: dict):
    if len(rows) != 2 * len(evidence) or len({(r["review_id"], r["window"]) for r in rows}) != len(rows):
        raise ValueError("Incomplete or duplicated annotation coverage")
    statuses = {"definite_test_command", "definite_no_test_command", "ambiguous", "command_unrecoverable", "insufficient_evidence"}
    for row in rows:
        if row["test_command_status"] not in statuses or row["independent_human_review"] is not False:
            raise ValueError("Invalid label or provenance")
        if row["test_command_status"] == "definite_no_test_command" and any(row[x] != "not_applicable" for x in ("execution_status", "process_status", "execution_detail", "test_output_status", "output_composition")):
            raise ValueError("Negative window has applicable test output")
        calls = evidence[row["review_id"]][row["window"]]
        valid = {c[k] for c in calls for k in ("action_index", "observation_index") if c.get(k) is not None}
        refs = {int(x) for x in row["supporting_message_indexes"].split(";") if x}
        if not refs.issubset(valid):
            raise ValueError("Citation outside prefix")
        quote = row["evidence_quotation"]
        if quote:
            if row["quotation_kind"] == "observation":
                matched = any(c.get("observation_index") == row["quotation_message_index"] and quote in (c.get("observation") or "") for c in calls)
            else:
                matched = any(c.get("action_index") == row["quotation_message_index"] and (quote == c.get("extracted_command") or quote in (c.get("arguments") or "")) for c in calls)
            if not matched:
                raise ValueError(f"Quotation does not match its cited evidence: {row['review_id']} {row['window']} {row['quotation_kind']} {row['quotation_message_index']} {quote[:100]!r}")
    lookup = {(r["review_id"], r["window"]): r for r in rows}
    for review_id in evidence:
        if lookup[(review_id, "r5")]["test_command_status"] == "definite_test_command" and lookup[(review_id, "r15")]["test_command_status"] == "definite_no_test_command":
            raise ValueError("R5 positive cannot become R15 negative")


def run(evidence_path: Path, spots: Path, output: Path, protocol: Path) -> dict:
    if output.exists():
        raise FileExistsError("Choose a fresh run directory")
    evidence = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    check_blind(evidence)
    lookup = {x["review_id"]: x for x in evidence}
    if len(lookup) != len(evidence):
        raise ValueError("Duplicate evidence IDs")
    rows, events = [], []
    for record in evidence:
        if record["r5"] != [c for c in record["r15"] if c["round"] <= 5]:
            # order fields are local to windows; the actual bundle nests exactly.
            raise ValueError("R5/R15 evidence nesting failed")
        for window in ("r5", "r15"):
            row, window_events = annotate_window(record, window)
            rows.append(row)
            events.extend(window_events)
    raw_rows = [dict(x) for x in rows]
    apply_spot_reviews(rows, json.loads(spots.read_text(encoding="utf-8")), lookup)
    validate(rows, lookup)
    output.mkdir(parents=True)
    write_csv(output / "automated_annotations_before_spot_review.csv", raw_rows)
    write_csv(output / "annotations_long.csv", rows)
    wide = []
    for record in evidence:
        item = {k: record[k] for k in ("review_id", "completed_rounds", "prefix5_boundary", "prefix15_boundary", "prefix_parse_status")}
        for row in rows:
            if row["review_id"] == record["review_id"]:
                item.update({f"{k}_{row['window']}": v for k, v in row.items() if k not in {"review_id", "window"}})
        wide.append(item)
    write_csv(output / "annotations_wide.csv", wide)
    # Always include a header for an empty abstention list.
    uncertain = [r for r in rows if r["test_command_status"] not in {"definite_test_command", "definite_no_test_command"}]
    if uncertain:
        write_csv(output / "abstentions.csv", uncertain)
    else:
        (output / "abstentions.csv").write_text("review_id,window,test_command_status\n", encoding="utf-8")
    with (output / "call_annotations.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    summary = dict(status="AUTOMATED ANNOTATION DELIVERY COMPLETE; INDEPENDENT HUMAN VALIDATION NOT ESTABLISHED",
                   version=VERSION, trajectories=len(evidence), windows=len(rows), semantic_spot_review_windows=sum(r["semantic_spot_review"] for r in rows),
                   full_per_item_llm_semantic_review=False, independent_human_reviews=0, abstention_windows=len(uncertain),
                   counts={w: dict(Counter(r["test_command_status"] for r in rows if r["window"] == w)) for w in ("r5", "r15")},
                   source_evidence_sha256=sha(evidence_path), spot_review_sha256=sha(spots), protocol_sha256=sha(protocol),
                   code_sha256=sha(Path(__file__)), output_helper_sha256=sha(Path(__file__).with_name("v3_5_ai_prelabel.py")),
                   files={p.name: sha(p) for p in output.iterdir() if p.is_file()}, label_lock_precedes_sampling_key_join=True)
    (output / "ANNOTATION_LOCK.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--spot-reviews", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.evidence, args.spot_reviews, args.output, args.protocol), ensure_ascii=False, indent=2))
