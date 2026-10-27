"""Parser v2 execution events.

Gate 0.5 ``classify_execution`` is unchanged and remains the historical
definition. This module does not rewrite it. Nothing here defines
``execution_failure``. A missing scaffold exit code is not zero and is not
success. ``tool_exit_nonzero`` is true only for an OpenHands wrapper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PARSER_VERSION = "v2"

_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")
_EXCEPTION_IN_THREAD = re.compile(r"Exception in thread\b")
_UNCAUGHT = re.compile(
    r"Uncaught (?:Exception|Error)\b|UnhandledPromiseRejection|uncaughtException",
    re.IGNORECASE,
)
# Final component must be a Capitalized class. This rejects property names
# such as messages.rejectedParseError and prose that only contains the word.
_EXCEPTION_EVENT_LINE = re.compile(
    r"(?m)^(?!\s*\d+\s)(?:[A-Za-z_][\w]*\.)*[A-Z][\w]*(?:Error|Exception):\s+\S"
)

_SCAFFOLD_EXIT = re.compile(
    r"\[(?:The command completed|Command finished) with exit code (-?\d+)(?:\.|\])"
)
_GENERIC_EXIT_LINE = re.compile(r"(?m)^(?!\s*\[)\s*Exit code:\s*(-?\d+)\b")
_INTERRUPT_EXIT_CODES = {124, 130}
_INTERRUPT_EVENT = re.compile(
    r"CTRL\+C was sent|^KeyboardInterrupt\b|KeyboardInterrupt:|"
    r"command timed out|timed out after \d|TimeoutExpired\b",
    re.IGNORECASE | re.MULTILINE,
)
# Mentions that are not evidence the observed command was interrupted.
_INTERRUPT_MENTION = re.compile(
    r"\bSIGINT\b|setTimeout\b|DEADLINE_EXCEEDED|signal\.signal|"
    r"signal handler|timeout(?:_seconds|_ms)?\s*[:=]",
    re.IGNORECASE,
)

_PYTEST_FAIL_LINE = re.compile(r"(?m)^FAILED\s+\S")
_PYTEST_SUMMARY_FAIL = re.compile(r"(?m)^={3,}.*\b([1-9]\d*)\s+failed\b.*={3,}\s*$")
_UNITTEST_FAIL = re.compile(r"(?m)^FAIL:\s+test_\S")
_GO_TEST_FAIL = re.compile(r"(?m)^--- FAIL:\s+\S")
_RUST_TEST_FAIL = re.compile(r"(?m)^test\s+\S+\s+\.\.\.\s+FAILED\b")
_CARGO_RESULT_FAIL = re.compile(r"(?m)^test result: FAILED\b|^error: test failed\b")

_PYTEST_SUMMARY_PASS = re.compile(r"(?m)^={3,}.*\b([1-9]\d*)\s+passed\b.*={3,}\s*$")
_CARGO_RESULT_PASS = re.compile(r"(?m)^test result: ok\b")
_GO_TEST_PASS = re.compile(r"(?m)^--- PASS:\s+\S")
_GO_PKG_OK = re.compile(r"(?m)^ok\s+\S+")
_RUST_TEST_OK = re.compile(r"(?m)^test\s+\S+\s+\.\.\.\s+ok\b")
_ZERO_TESTS = re.compile(
    r"(?i)no tests ran|(?<!\d)0 passed|collected 0 items"
)

_BUILD = re.compile(
    r"^(?:error: could not compile|error\[\w+\d*\]:|compilation failed|"
    r"build failed|failed to compile|error: linking with\b)|^error TS\d+:",
    re.IGNORECASE | re.MULTILINE,
)
_SETUP = re.compile(
    r"^ERROR collecting\b|^ERROR: test_\S|^E\s+ERROR at setup\b|"
    r"errors during collection|fixture '.+' not found",
    re.IGNORECASE | re.MULTILINE,
)
_PACKAGE_FAIL = re.compile(r"(?m)^FAIL(?:\t|\s+\S+|\s*$)")
_PYTEST_SUMMARY_ERROR_ONLY = re.compile(
    r"(?m)^={3,}.*\b([1-9]\d*)\s+errors?\b(?!.*\bfailed\b).*={3,}\s*$"
)

_EXPLORATORY_TEST = (
    ("jest_or_node_summary", re.compile(r"(?m)^(?:Test Suites|Tests):\s+.*\bpassed\b")),
    ("vitest_summary", re.compile(r"(?m)^Test Files\s+.*\bpassed\b")),
    ("prose_suite_passed", re.compile(
        r"Ran all test suites|End-to-end test PASSED|All error handling tests passed",
        re.IGNORECASE,
    )),
    ("junit_tests_run", re.compile(r"(?m)^Tests run:\s*[1-9]\d*")),
)


@dataclass(frozen=True)
class ExecutionSignalsV2:
    """Versioned events. Not a frozen primary feature set."""

    parser_version: str
    explicit_exception: bool
    explicit_test_failure: bool
    build_or_compile_failure: bool
    test_setup_or_collection_failure: bool
    package_level_failure_unresolved: bool
    explicit_test_pass: bool
    tool_exit_nonzero: bool
    interrupt_or_timeout: bool
    interrupt_mention_not_event: bool
    discordant_exit_status_and_test_result: bool
    execution_status_unknown: bool
    exploratory_test_format: bool
    scaffold_exit_codes: tuple[int, ...]
    reasons: tuple[str, ...]

    def predicted_label(self) -> str:
        names = [
            name
            for name, flag in (
                ("explicit_exception", self.explicit_exception),
                ("explicit_test_failure", self.explicit_test_failure),
                ("build_or_compile_failure", self.build_or_compile_failure),
                ("test_setup_or_collection_failure", self.test_setup_or_collection_failure),
                ("package_level_failure_unresolved", self.package_level_failure_unresolved),
                ("explicit_test_pass", self.explicit_test_pass),
                ("tool_exit_nonzero", self.tool_exit_nonzero),
                ("interrupt_or_timeout", self.interrupt_or_timeout),
                ("discordant_exit_status_and_test_result", self.discordant_exit_status_and_test_result),
                ("execution_status_unknown", self.execution_status_unknown),
                ("exploratory_test_format", self.exploratory_test_format),
            )
            if flag
        ]
        return "|".join(names) if names else "none"


def _exception_event(text: str, reasons: list[str]) -> bool:
    if _TRACEBACK.search(text):
        reasons.append("python_traceback")
        return True
    if _EXCEPTION_IN_THREAD.search(text):
        reasons.append("exception_in_thread")
        return True
    if _UNCAUGHT.search(text):
        reasons.append("uncaught_exception")
        return True
    if _EXCEPTION_EVENT_LINE.search(text):
        reasons.append("exception_event_line")
        return True
    return False


def _scaffold_codes(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in _SCAFFOLD_EXIT.findall(text)}))


def classify_execution_v2(content: str) -> ExecutionSignalsV2:
    """Classify one tool observation. Do not call this on assistant prose."""
    text = content or ""
    reasons: list[str] = []
    exception = _exception_event(text, reasons)
    codes = _scaffold_codes(text)
    interrupt_exit = any(code in _INTERRUPT_EXIT_CODES for code in codes)
    ordinary_nonzero = len(codes) == 1 and codes[0] not in {0, *_INTERRUPT_EXIT_CODES}
    if ordinary_nonzero:
        reasons.append("nonzero_scaffold_exit")
    if interrupt_exit:
        reasons.append("interrupt_scaffold_exit")
    interrupt_event = bool(_INTERRUPT_EVENT.search(text))
    if interrupt_event:
        reasons.append("interrupt_or_timeout_event")
    interrupt = interrupt_exit or interrupt_event
    mention = bool(_INTERRUPT_MENTION.search(text)) and not interrupt
    if mention:
        reasons.append("interrupt_mention_not_event")

    test_failure = False
    for pattern, reason in (
        (_PYTEST_FAIL_LINE, "pytest_failed_line"),
        (_PYTEST_SUMMARY_FAIL, "pytest_summary_failed"),
        (_UNITTEST_FAIL, "unittest_fail_line"),
        (_GO_TEST_FAIL, "go_test_fail"),
        (_RUST_TEST_FAIL, "rust_test_fail"),
        (_CARGO_RESULT_FAIL, "cargo_test_failed"),
    ):
        if pattern.search(text):
            test_failure = True
            reasons.append(reason)

    build = bool(_BUILD.search(text))
    if build:
        reasons.append("build_or_compile_failure")
    setup = bool(_SETUP.search(text) or _PYTEST_SUMMARY_ERROR_ONLY.search(text))
    if setup:
        reasons.append("test_setup_or_collection_failure")
    package_line = bool(_PACKAGE_FAIL.search(text))
    named_test_fail = test_failure
    package_unresolved = package_line and not named_test_fail
    if package_unresolved:
        reasons.append("package_level_failure_unresolved")

    contradictory = test_failure or build or setup or package_unresolved
    explicit_pass = False
    if not contradictory and not _ZERO_TESTS.search(text):
        for pattern, reason in (
            (_PYTEST_SUMMARY_PASS, "pytest_summary_passed"),
            (_CARGO_RESULT_PASS, "cargo_ok"),
            (_GO_TEST_PASS, "go_test_pass"),
            (_RUST_TEST_OK, "rust_test_ok"),
        ):
            if pattern.search(text):
                explicit_pass = True
                reasons.append(reason)
        if (
            not explicit_pass
            and _GO_PKG_OK.search(text)
            and not re.search(r"(?m)^FAIL\b", text)
        ):
            explicit_pass = True
            reasons.append("go_package_ok")

    exploratory = False
    for name, pattern in _EXPLORATORY_TEST:
        if pattern.search(text):
            exploratory = True
            reasons.append(f"exploratory:{name}")

    exit_zero = codes == (0,)
    discordant = (exit_zero and test_failure) or (ordinary_nonzero and explicit_pass)
    if discordant:
        reasons.append("discordant_exit_status_and_test_result")

    unknown = False
    if len(codes) > 1:
        unknown = True
        reasons.append("conflicting_scaffold_exit_codes")
    if _GENERIC_EXIT_LINE.search(text) and not codes:
        unknown = True
        reasons.append("generic_exit_line_not_scaffold")

    return ExecutionSignalsV2(
        parser_version=PARSER_VERSION,
        explicit_exception=exception,
        explicit_test_failure=test_failure,
        build_or_compile_failure=build,
        test_setup_or_collection_failure=setup,
        package_level_failure_unresolved=package_unresolved,
        explicit_test_pass=explicit_pass,
        tool_exit_nonzero=ordinary_nonzero,
        interrupt_or_timeout=interrupt,
        interrupt_mention_not_event=mention,
        discordant_exit_status_and_test_result=discordant,
        execution_status_unknown=unknown,
        exploratory_test_format=exploratory,
        scaffold_exit_codes=codes,
        reasons=tuple(reasons),
    )
