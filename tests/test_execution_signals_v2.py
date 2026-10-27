"""Parser v2. Historical classify_execution is not modified by these tests."""

from p8audit.execution_signals import classify_execution
from p8audit.execution_signals_v2 import PARSER_VERSION, classify_execution_v2


def test_rejected_parse_error_name_is_not_an_exception_event() -> None:
    text = "messages.rejectedParseError\n======= 1 passed in 0.01s =======\n"
    signals = classify_execution_v2(text)
    assert signals.explicit_exception is False
    assert signals.explicit_test_pass is True


def test_traceback_remains_an_exception_event() -> None:
    text = "Traceback (most recent call last):\n  File \"a.py\", line 1\nValueError: bad\n"
    signals = classify_execution_v2(text)
    assert signals.parser_version == PARSER_VERSION
    assert signals.explicit_exception is True


def test_source_snippet_and_config_key_are_not_exception_events() -> None:
    source = "class ParseError:\n    pass\nexcept ValueError:\n    raise\n"
    config = "messages.rejectedParseError: true\n"
    prose = "The handler explains ValueError without an event.\n"
    for text in (source, config, prose):
        assert classify_execution_v2(text).explicit_exception is False


def test_historical_parser_still_flags_a_broad_exception_line() -> None:
    text = "messages.rejectedParseError: true\n"
    assert classify_execution(text).explicit_exception is True
    assert classify_execution_v2(text).explicit_exception is False


def test_package_fail_is_not_an_explicit_test_assertion() -> None:
    text = "FAIL\tgithub.com/example/pkg\t0.012s\n"
    signals = classify_execution_v2(text)
    assert signals.explicit_test_failure is False
    assert signals.package_level_failure_unresolved is True
    assert classify_execution(text).explicit_test_failure is True


def test_named_go_test_failure_stays_explicit() -> None:
    text = "--- FAIL: TestParse (0.01s)\n"
    signals = classify_execution_v2(text)
    assert signals.explicit_test_failure is True
    assert signals.package_level_failure_unresolved is False


def test_build_and_collection_are_separate_from_test_failure() -> None:
    build = "error: could not compile `demo`\n"
    collect = "ERROR collecting tests/test_app.py\n"
    assert classify_execution_v2(build).build_or_compile_failure is True
    assert classify_execution_v2(build).explicit_test_failure is False
    assert classify_execution_v2(collect).test_setup_or_collection_failure is True
    assert classify_execution_v2(collect).explicit_test_failure is False


def test_pass_requires_an_explicit_signal_and_no_contradiction() -> None:
    assert classify_execution_v2("[The command completed with exit code 0.]\n").explicit_test_pass is False
    assert classify_execution_v2("======= 0 passed in 0.01s =======\n").explicit_test_pass is False
    passed = classify_execution_v2("======= 3 passed in 0.01s =======\n")
    assert passed.explicit_test_pass is True
    both = "======= 1 failed, 3 passed in 0.01s =======\n"
    assert classify_execution_v2(both).explicit_test_pass is False
    assert classify_execution_v2(both).explicit_test_failure is True


def test_source_mentions_are_not_interrupt_events() -> None:
    for text in (
        "signal.SIGINT\n",
        "setTimeout(fn, 10)\n",
        "status: DEADLINE_EXCEEDED\n",
        "timeout: 30\n",
    ):
        signals = classify_execution_v2(text)
        assert signals.interrupt_or_timeout is False
        assert signals.interrupt_mention_not_event is True


def test_real_interrupt_is_not_a_nonzero_tool_exit() -> None:
    text = "[The command completed with exit code 130.]\nCTRL+C was sent\n"
    signals = classify_execution_v2(text)
    assert signals.interrupt_or_timeout is True
    assert signals.tool_exit_nonzero is False
    assert signals.interrupt_mention_not_event is False


def test_openhands_nonzero_exit_and_missing_swe_exit() -> None:
    wrapped = classify_execution_v2("[Command finished with exit code 2]\n")
    assert wrapped.tool_exit_nonzero is True
    missing = classify_execution_v2("OBSERVATION:\ncommand output\n")
    assert missing.tool_exit_nonzero is False
    generic = classify_execution_v2("Exit code: 2\n")
    assert generic.tool_exit_nonzero is False
    assert generic.execution_status_unknown is True


def test_discordant_exit_is_not_unknown_status() -> None:
    text = "FAILED tests/test_app.py::test_one\n[The command completed with exit code 0.]\n"
    signals = classify_execution_v2(text)
    assert signals.discordant_exit_status_and_test_result is True
    assert signals.execution_status_unknown is False
    assert not hasattr(signals, "execution_failure")


def test_absence_of_a_marker_is_not_unknown() -> None:
    signals = classify_execution_v2("listing files\n")
    assert signals.execution_status_unknown is False
    assert signals.explicit_test_pass is False
    assert signals.predicted_label() == "none"
