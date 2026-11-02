"""Guard against circular validation and unsafe negative proposals."""
import json

import pytest

from v3_5_ai_prelabel import command_kind, label_window, output_proposal, run


def record(calls, boundary="match"):
    return dict(review_id="V35-000001", completed_rounds=15, prefix5_boundary="match", prefix15_boundary=boundary,
                prefix_parse_status="ok", r5=calls, r15=calls)


def call(command, output="", index=2, arguments=None):
    return dict(tool="execute_bash", extracted_command=command, action_index=index, observation_index=index + 1,
                command_extraction_status="json_command" if command else "json_neither",
                arguments=arguments or json.dumps({"command": command}), observation=output)


@pytest.mark.parametrize("command", [
    "grep 'pytest tests/a.py' README.md", "echo 'pytest tests/a.py'", "cat tests/test_x.py", "git diff", "cargo build 2>&1 | tail -20",
])
def test_reading_test_names_is_not_running_tests(command):
    assert command_kind(command)[0] == "non_test"


@pytest.mark.parametrize("command", [
    "cd /work && python -m pytest tests/a.py", "timeout 300 pytest -q", "bash -lc 'cargo test -q'", "go test ./...", "pytest 2>&1 | tail -20",
])
def test_runner_proposals(command):
    assert command_kind(command)[0] == "test"


@pytest.mark.parametrize("command", [
    "python reproduce.py", "python tests/test_x.py", "python - <<'PY'\nprint('pytest')\nPY",
    "echo $(pytest)", "find . -exec pytest {} ;", "pytest --collect-only", "npm install", "make verify",
])
def test_uncertain_commands_never_become_known_negatives(command):
    assert command_kind(command)[0] == "ambiguous"


def test_missing_command_is_not_negative():
    row, _ = label_window(record([call(None)]), "r15")
    assert row["test_command_status"] == "command_unrecoverable"


def test_explicit_poll_links_to_prior_command():
    poll = call(None, "1 passed", 4, json.dumps({"command": "", "is_input": "true"}))
    row, events = label_window(record([call("pytest"), poll]), "r15")
    assert row["test_command_status"] == "definite_test_command"
    assert events[-1]["continued_poll"]
    assert events[-1]["command_origin_index"] == 2
    assert row["test_output_status"] == "explicit_pass"
    assert row["requires_review"] is True


def test_pre_test_error_not_test_failure():
    row = output_proposal("ERROR collecting test_a.py\nModuleNotFoundError: package", "pytest")
    assert row["test_output_status"] == "no_output_observed"
    assert row["execution_detail"] == "collection_failure"


def test_failed_files_without_executed_tests_are_not_case_failures():
    row = output_proposal("Test Files  2 failed (2)\n      Tests  no tests\n", "vitest")
    assert row["test_output_status"] == "no_output_observed"
    assert row["execution_status"] == "execution_not_confirmed"


def test_fail_then_pass_keeps_failure_and_mixed_composition():
    row, _ = label_window(record([call("pytest", "1 failed", 2), call("pytest", "1 passed", 4)]), "r15")
    assert row["test_output_status"] == "explicit_failure"
    assert row["output_composition"] == "mixed_pass_and_failure"


def test_short_window_is_not_known_negative():
    row, _ = label_window(record([call("ls")], "no_prefix_boundary"), "r15")
    assert row["test_command_status"] == "insufficient_evidence"


def test_no_test_is_not_applicable():
    row, _ = label_window(record([call("ls")]), "r15")
    assert row["execution_status"] == "not_applicable"


def test_run_guards_and_provenance(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(json.dumps(record([call("pytest", "1 passed")])) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    result = run(evidence, output)
    assert result["independent_human_reviews"] == 0
    assert result["detector_metrics_computed"] is False
    assert "automated_rule_proposal_not_human" in (output / "ai_proposed_review_queue.csv").read_text(encoding="utf-8-sig")
    with pytest.raises(FileExistsError):
        run(evidence, output)
    evidence.write_text(json.dumps(dict(record([]), resolved=1)), encoding="utf-8")
    with pytest.raises(ValueError, match="Unblinded"):
        run(evidence, tmp_path / "other")
    assert not (tmp_path / "other").exists()
