"""Alternative exit spellings stay out of the OpenHands feature."""

from p8audit.execution_signals import classify_execution
from p8audit.exit_formats import describe_exit_formats, returncode_tag_is_not_scaffold_exit


def test_openhands_wrapper_is_reported_separately_from_other_spellings() -> None:
    text = "[The command completed with exit code 1.]\nls: not found"
    found = describe_exit_formats(text)
    assert found["openhands_scaffold_wrapper"] == [1]
    assert found["returncode_tag"] == []
    assert classify_execution(text).tool_exit_nonzero is True


def test_returncode_tag_does_not_become_tool_exit_nonzero() -> None:
    text = "OBSERVATION:\npytest failed\n<returncode>1</returncode>"
    assert describe_exit_formats(text)["returncode_tag"] == [1]
    assert returncode_tag_is_not_scaffold_exit(text) is True
    signals = classify_execution(text)
    assert signals.tool_exit_nonzero is False
    assert "execution_failure" not in signals.__dict__


def test_assigned_exit_field_is_not_scaffold_exit() -> None:
    text = "OBSERVATION:\nexit_code: 1\n"
    found = describe_exit_formats(text)
    assert found["assigned_exit_field"] == [1]
    assert found["openhands_scaffold_wrapper"] == []
    assert classify_execution(text).tool_exit_nonzero is False


def test_generic_exit_line_is_not_counted_as_the_scaffold_wrapper() -> None:
    text = "Exit code: 2\n"
    found = describe_exit_formats(text)
    assert found["generic_exit_line"] == [2]
    assert found["openhands_scaffold_wrapper"] == []
    assert classify_execution(text).tool_exit_nonzero is False
    assert classify_execution(text).execution_status_unknown is True
