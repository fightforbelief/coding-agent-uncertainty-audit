"""Parser, marker, and leakage tests that do not need the remote dataset."""

from __future__ import annotations

import json

from p8audit.markers import error_signature, mark_observation
from p8audit.schema import PROHIBITED_FEATURE_FIELDS, failure_label
from p8audit.trajectory_parser import observable_text, parse_trajectory


def _call(name: str, arguments: str = "{}") -> dict:
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _assistant(content: str, calls: list[dict], reasoning: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "think": False,
        "tool_calls": calls,
    }


def _tool(content: str) -> dict:
    return {
        "role": "tool",
        "content": content,
        "reasoning_content": "",
        "think": False,
        "tool_calls": [],
    }


def test_serialized_and_malformed_trajectories() -> None:
    payload = [_assistant("run", [_call("bash")]), _tool("ok")]
    parsed = parse_trajectory(json.dumps(payload))
    assert parsed.status == "ok"
    assert parsed.notes[0] == "input:json_string"
    assert parsed.n_complete_cycles == 1

    broken = parse_trajectory("{not json")
    assert broken.status == "malformed_json"
    assert broken.messages == []

    wrong = parse_trajectory(json.dumps({"role": "assistant"}))
    assert wrong.status == "json_not_list"

    partial = parse_trajectory([{"content": "no role"}, payload[0], payload[1]])
    assert partial.status == "partial"
    assert partial.n_malformed_messages == 1
    assert partial.n_complete_cycles == 1


def test_cycle_pairing_and_prefix_truncation() -> None:
    trajectory = [
        {"role": "system", "content": "sys", "reasoning_content": "", "think": False, "tool_calls": []},
        {"role": "user", "content": "issue", "reasoning_content": "", "think": False, "tool_calls": []},
        _assistant("one", [_call("bash")], reasoning="hidden thought"),
        _tool("file list"),
        _assistant("two", [_call("bash"), _call("bash")]),
        _tool("output-a"),
        _tool("Traceback (most recent call last):\nValueError: bad\n"),
        _assistant("three", [_call("finish")]),
        _tool("submitted"),
    ]
    parsed = parse_trajectory(trajectory)
    assert [cycle.status for cycle in parsed.cycles] == ["complete", "complete", "complete"]
    assert parsed.prefix_message_count(1) == 4
    assert parsed.prefix_message_count(2) == 7
    assert parsed.prefix_message_count(5) is None
    prefix_end = parsed.prefix_message_count(1)
    assert prefix_end is not None
    visible = [message.content for message in parsed.messages[:prefix_end]]
    assert "Traceback" not in " ".join(visible)
    assert "submitted" not in " ".join(visible)
    assert observable_text(parsed.messages[2]) == "one"
    assert parsed.messages[2].has_reasoning_content is True
    assert "hidden thought" not in observable_text(parsed.messages[2])


def test_assistant_prose_does_not_count_as_tool_error() -> None:
    trajectory = [
        _assistant("The word error appears in my plan, and Traceback is only mentioned.", [_call("bash")]),
        _tool("total 12\nREADME.md\n"),
    ]
    parsed = parse_trajectory(trajectory)
    assert parsed.cycles[0].markers[0].explicit_error is False
    assert mark_observation(trajectory[0]["content"]).explicit_error is True or "Traceback" in trajectory[0]["content"]
    # The assistant string contains "Traceback" as prose. The marker function
    # itself is content-based; the parser must not apply it to assistant text.
    assert all(message.role == "tool" for message in _observation_messages(parsed))


def _observation_messages(parsed):
    indexes = [index for cycle in parsed.cycles for index in cycle.observation_indexes]
    return [parsed.messages[index] for index in indexes]


def test_explicit_error_and_test_markers() -> None:
    traceback = "Traceback (most recent call last):\n  File \"a.py\", line 1\nValueError: nope\n"
    marked = mark_observation(traceback)
    assert marked.explicit_error is True
    assert "python_traceback" in marked.error_reasons
    assert "exception_line" in marked.error_reasons

    exit_text = (
        "bash: go: command not found\n"
        "[The command completed with exit code 127.]\n"
        "[Command finished with exit code 127]\n"
    )
    exit_marked = mark_observation(exit_text)
    assert exit_marked.explicit_error is True
    assert "nonzero_scaffold_exit" in exit_marked.error_reasons

    interrupted = (
        "^C\n"
        "[The command completed with exit code 130. CTRL+C was sent.]\n"
        "[Command finished with exit code 130]\n"
    )
    assert mark_observation(interrupted).explicit_error is True

    zero = (
        "Output: pair=('a', 'b')\n"
        "Exit code: 0\n"
        "[The command completed with exit code 0.]\n"
    )
    assert mark_observation(zero).explicit_error is False

    pytest_fail = "=========================== short test summary info ============================\nFAILED tests/test_app.py::test_one - AssertionError\n========== 1 failed, 2 passed in 0.10s ==========\n"
    failed = mark_observation(pytest_fail)
    assert failed.explicit_test_failure is True
    assert failed.explicit_test_pass is False

    pytest_pass = "============================== 3 passed in 0.02s ==============================\n"
    assert mark_observation(pytest_pass).explicit_test_pass is True
    assert mark_observation(pytest_pass).explicit_test_failure is False

    listing = "Here's the files and directories up to 2 levels deep in /workspace/error-handler, excluding hidden items:\n"
    assert mark_observation(listing).explicit_error is False
    assert error_signature(listing) is None
    assert error_signature(traceback) is not None


def test_missing_observation_is_not_silently_complete() -> None:
    parsed = parse_trajectory([_assistant("stop", [_call("finish")])])
    assert parsed.cycles[0].status == "missing_observation"
    assert parsed.n_complete_cycles == 0
    assert parsed.prefix_message_count(1) is None


def test_failure_label_and_prohibited_fields() -> None:
    assert failure_label(0) == 1
    assert failure_label(1) == 0
    assert failure_label(-1) is None
    for column in ("resolved", "reasoning_content", "repo", "metadata.reference_patch"):
        assert column in PROHIBITED_FEATURE_FIELDS
