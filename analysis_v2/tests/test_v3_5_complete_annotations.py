import json

import pytest

from v3_5_complete_annotations import annotate_window, check_blind, classify, inspect_python, segments, validate


def call(command, index=2, output="", status="json_command", arguments=None):
    return dict(round=index // 2, tool="execute_bash", extracted_command=command, action_index=index,
                observation_index=index + 1, observation=output, command_extraction_status=status,
                arguments=arguments or json.dumps({"command": command}))


def record(calls):
    return dict(review_id="id", r5=calls, r15=calls, prefix5_boundary="match", prefix15_boundary="match", prefix_parse_status="ok")


@pytest.mark.parametrize("cmd", [
    'grep "pytest test_x.py" README.md', 'printf "1 failed"', 'cat tests/test_x.py',
    'cd /repo && cargo build 2>&1 | tail -20', 'go mod tidy', 'mvn compile -q',
    'xargs grep -l test', 'find . -exec grep -l test {} \\;',
    'python -c "import re; print(re.findall(\'a\', \'abc\'))"',
    'node -e "console.log(\'FAIL\')"',
])
def test_non_test_commands(cmd):
    assert classify(cmd, {})[0] == "non_test"


@pytest.mark.parametrize("cmd", [
    'python -m pytest tests', 'cd /repo && npx --yes jest -q',
    'bash -lc "go test ./..."', 'npm run test:unit -- issue.ts',
    'pytest 2>&1 | tail -20', 'node --test file.js',
    'python -c "import pytest; pytest.main([\'-q\'])"',
])
def test_visible_runners(cmd):
    assert classify(cmd, {})[0] == "test"


@pytest.mark.parametrize("cmd", [
    'python unknown.py', 'python tests/test_x.py', 'python -c "import subprocess; subprocess.run([\'pytest\'])"',
    'echo $(pytest)', 'cat > foo.py <<EOF\nprint(1)\nEOF', 'npx unknown-custom-tool',
    'npm install', 'cargo test --no-run', 'pytest --collect-only',
])
def test_abstentions_are_not_negative(cmd):
    assert classify(cmd, {})[0] == "ambiguous"


def test_newline_inside_quotes_is_not_new_shell_command():
    assert classify('python -c "import re\nprint(1)"', {})[0] == "non_test"
    assert classify('ls\npytest', {})[0] == "test"


def test_created_reproduction_requires_visible_contents():
    assert classify("python test_issue.py", {"test_issue.py": "print('FAIL')"})[0] == "non_test"
    assert classify("python test_issue.py", {})[0] == "ambiguous"
    assert inspect_python("import unittest\nunittest.main() ")[0] == "test"


def test_poll_without_is_input_links_only_pending_process():
    pending = call("pytest", output="The command has no new output. You may wait longer")
    poll = call(None, 4, "1 passed", "json_neither", '{"command": ""}')
    row, events = annotate_window(record([pending, poll]), "r15")
    assert events[1]["continued_poll"]
    assert row["test_output_status"] == "explicit_pass"
    assert row["quotation_message_index"] == 5
    validate([dict(row, window="r5"), row], {"id": record([pending, poll])})


def test_poll_after_completed_command_is_unrecoverable():
    completed = call("ls", output="done")
    poll = call(None, 4, "", "json_neither", '{"command": ""}')
    row, events = annotate_window(record([completed, poll]), "r15")
    assert events[1]["command_kind"] == "command_unrecoverable"
    assert row["test_command_status"] == "command_unrecoverable"


def test_blinding_and_quotation_validation():
    with pytest.raises(ValueError, match="Unblinded"):
        check_blind([{"r15": [{"resolved": 1}]}])
    evidence = record([call("pytest", output="1 passed")])
    row, _ = annotate_window(evidence, "r15")
    row["evidence_quotation"] = "invented result"
    with pytest.raises(ValueError, match="Quotation"):
        validate([dict(row, window="r5"), row], {"id": evidence})
