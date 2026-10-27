"""Separated execution signals. These tests do not read the dataset."""

from p8audit.execution_signals import classify_execution
from p8audit.markers import mark_observation
from p8audit.trajectory_parser import parse_trajectory


def test_exit_zero_does_not_imply_test_pass() -> None:
    text = "[The command completed with exit code 0.]\n"
    signals = classify_execution(text)
    assert signals.explicit_test_pass is False
    assert signals.tool_exit_nonzero is False
    assert signals.explicit_test_failure is False


def test_exit_zero_can_still_contain_a_test_failure() -> None:
    text = (
        "FAILED tests/test_app.py::test_one - AssertionError\n"
        "[The command completed with exit code 0.]\n"
    )
    signals = classify_execution(text)
    assert signals.explicit_test_failure is True
    assert signals.tool_exit_nonzero is False
    assert signals.explicit_test_pass is False
    assert "discordant" not in signals.predicted_label()


def test_interrupt_is_not_a_test_failure_or_ordinary_nonzero_exit() -> None:
    text = "[The command completed with exit code 130. CTRL+C was sent.]\n"
    signals = classify_execution(text)
    assert signals.interrupt_or_timeout is True
    assert signals.tool_exit_nonzero is False
    assert signals.explicit_test_failure is False
    historical = mark_observation(text)
    assert historical.explicit_error is True


def test_generic_exit_line_is_unknown_rather_than_a_tool_failure() -> None:
    text = "Usage: tool [OPTIONS]\nExit code: 1\n"
    signals = classify_execution(text)
    assert signals.tool_exit_nonzero is False
    assert signals.execution_status_unknown is True
    assert "generic_exit_line_not_scaffold" in signals.reasons


def test_the_word_error_is_not_a_tool_failure() -> None:
    text = "The error handler file was copied into the workspace.\n"
    signals = classify_execution(text)
    assert signals.explicit_exception is False
    assert signals.tool_exit_nonzero is False
    assert signals.execution_status_unknown is False
    assert signals.predicted_label() == "none"


def test_traceback_is_an_exception_and_not_an_exit_code() -> None:
    text = "Traceback (most recent call last):\nValueError: bad\n"
    signals = classify_execution(text)
    assert signals.explicit_exception is True
    assert signals.tool_exit_nonzero is False
    assert signals.interrupt_or_timeout is False


def test_known_missed_test_formats_stay_exploratory() -> None:
    jest = "Test Suites: 8 passed, 8 total\nTests: 1164 passed, 1164 total\n"
    signals = classify_execution(jest)
    assert signals.explicit_test_pass is False
    assert signals.exploratory_test_format is True
    assert signals.execution_status_unknown is True
    rust = "test parser::case ... ok\n"
    assert classify_execution(rust).explicit_test_pass is False
    assert classify_execution(rust).exploratory_test_format is True


def test_fields_are_not_collapsed_into_execution_failure() -> None:
    signals = classify_execution("[The command completed with exit code 1.]\n")
    assert signals.tool_exit_nonzero is True
    assert not hasattr(signals, "execution_failure")


def test_parser_does_not_apply_execution_markers_to_assistant_prose() -> None:
    trajectory = [
        {
            "role": "assistant",
            "content": "Traceback (most recent call last):\nValueError: only a plan",
            "reasoning_content": "hidden",
            "think": False,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "content": "total 1\nREADME.md\n",
            "reasoning_content": "",
            "think": False,
            "tool_calls": [],
        },
    ]
    parsed = parse_trajectory(trajectory)
    assert parsed.cycles[0].execution[0].explicit_exception is False
    assert parsed.cycles[0].markers[0].explicit_error is False


def test_bracket_form_without_a_period_is_a_scaffold_exit() -> None:
    text = "[Command finished with exit code 127]\n"
    signals = classify_execution(text)
    assert signals.scaffold_exit_codes == (127,)
    assert signals.tool_exit_nonzero is True
