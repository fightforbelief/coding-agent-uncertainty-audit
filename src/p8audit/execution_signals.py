"""Separated execution signals for the Gate 0.5 amendment.

Gate 0 ``mark_observation`` remains the historical marker. It folds a nonzero
scaffold exit, including exit 130, into ``explicit_error``. This module does
not do that. Each field below is recorded on its own. Nothing here defines a
combined ``execution_failure`` feature.

``explicit_test_pass`` uses the narrow Gate 0 patterns and is not a primary
feature. Broader test-result formats are flagged as exploratory so they can
enter a review queue; they are not frozen features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")
_EXCEPTION_LINE = re.compile(
    r"(?m)^(?!\s*\d+\s)[A-Za-z_][\w.]*(?:Error|Exception):\s+\S"
)
# Period or closing bracket. Exit 130 with trailing "CTRL+C was sent" still matches.
_SCAFFOLD_EXIT = re.compile(
    r"\[(?:The command completed|Command finished) with exit code (-?\d+)(?:\.|\])"
)
# Click-style program stdout. This is not the scaffold wrapper.
_GENERIC_EXIT_LINE = re.compile(r"(?m)^(?!\s*\[)\s*Exit code:\s*(-?\d+)\b")
_INTERRUPT_TEXT = re.compile(
    r"CTRL\+C was sent|KeyboardInterrupt|\bSIGINT\b|"
    r"command timed out|timed out after|TimeoutExpired|deadline exceeded",
    re.IGNORECASE,
)
_PYTEST_FAIL_LINE = re.compile(r"(?m)^FAILED\s+\S")
_PYTEST_SUMMARY_FAIL = re.compile(r"(?m)^={3,}.*\b\d+\s+failed\b.*={3,}\s*$")
_GO_FAIL = re.compile(r"(?m)^--- FAIL:|^FAIL(?:\t|\s+\S|\s*$)")
_CARGO_FAIL = re.compile(r"(?m)^test result: FAILED\b|^error: test failed\b")
_PYTEST_SUMMARY_PASS = re.compile(r"(?m)^={3,}.*\b\d+\s+passed\b.*={3,}\s*$")
_CARGO_PASS = re.compile(r"(?m)^test result: ok\b")
_GO_OK_PKG = re.compile(r"(?m)^ok\s+\S+")

# Known misses from the Gate 0 review. Exploratory only.
_EXPLORATORY_TEST = (
    ("jest_or_node_summary", re.compile(r"(?m)^(?:Test Suites|Tests):\s+.*\bpassed\b")),
    ("vitest_summary", re.compile(r"(?m)^Test Files\s+.*\bpassed\b")),
    ("rust_test_ok_line", re.compile(r"(?m)^test\s+\S+\s+\.\.\.\s+ok\b")),
    ("prose_suite_passed", re.compile(
        r"Ran all test suites|End-to-end test PASSED|All error handling tests passed",
        re.IGNORECASE,
    )),
    ("junit_tests_run", re.compile(r"(?m)^Tests run:\s*\d+")),
)

# Scaffold codes that mean interrupt or the GNU timeout convention.
# 137 and 143 are not included: without timeout text they stay ordinary
# nonzero exits, because a kill is not automatically a test failure or a timeout.
_INTERRUPT_EXIT_CODES = {124, 130}


@dataclass(frozen=True)
class ExecutionSignals:
    """Candidate measurements. Not a frozen model feature set."""

    tool_exit_nonzero: bool
    explicit_test_failure: bool
    explicit_test_pass: bool
    interrupt_or_timeout: bool
    execution_status_unknown: bool
    explicit_exception: bool
    exploratory_test_format: bool
    scaffold_exit_codes: tuple[int, ...]
    reasons: tuple[str, ...]

    def predicted_label(self) -> str:
        names = [
            name
            for name, flag in (
                ("tool_exit_nonzero", self.tool_exit_nonzero),
                ("explicit_exception", self.explicit_exception),
                ("explicit_test_failure", self.explicit_test_failure),
                ("explicit_test_pass", self.explicit_test_pass),
                ("interrupt_or_timeout", self.interrupt_or_timeout),
                ("execution_status_unknown", self.execution_status_unknown),
                ("exploratory_test_format", self.exploratory_test_format),
            )
            if flag
        ]
        return "|".join(names) if names else "none"


def classify_execution(content: str) -> ExecutionSignals:
    """Classify one tool observation. Do not call this on assistant prose."""
    text = content or ""
    reasons: list[str] = []

    exception = False
    if _TRACEBACK.search(text):
        exception = True
        reasons.append("python_traceback")
    if _EXCEPTION_LINE.search(text):
        exception = True
        reasons.append("exception_line")

    scaffold_codes = tuple(int(value) for value in _SCAFFOLD_EXIT.findall(text))
    distinct_codes = tuple(sorted(set(scaffold_codes)))
    interrupt = any(code in _INTERRUPT_EXIT_CODES for code in distinct_codes)
    nonzero = any(code != 0 and code not in _INTERRUPT_EXIT_CODES for code in distinct_codes)
    if nonzero:
        reasons.append("nonzero_scaffold_exit")
    if any(code in _INTERRUPT_EXIT_CODES for code in distinct_codes):
        reasons.append("interrupt_scaffold_exit")
    if _INTERRUPT_TEXT.search(text):
        interrupt = True
        reasons.append("interrupt_or_timeout_text")

    test_failure = False
    if _PYTEST_FAIL_LINE.search(text):
        test_failure = True
        reasons.append("pytest_failed_line")
    if _PYTEST_SUMMARY_FAIL.search(text):
        test_failure = True
        reasons.append("pytest_summary_failed")
    if _GO_FAIL.search(text):
        test_failure = True
        reasons.append("go_or_package_fail")
    if _CARGO_FAIL.search(text):
        test_failure = True
        reasons.append("cargo_failed")

    test_pass = False
    if not test_failure and _PYTEST_SUMMARY_PASS.search(text):
        test_pass = True
        reasons.append("pytest_summary_passed")
    if not test_failure and _CARGO_PASS.search(text):
        test_pass = True
        reasons.append("cargo_ok")
    if not test_failure and _GO_OK_PKG.search(text) and not re.search(r"(?m)^FAIL\b", text):
        test_pass = True
        reasons.append("go_ok")

    exploratory = False
    for name, pattern in _EXPLORATORY_TEST:
        if pattern.search(text):
            exploratory = True
            reasons.append(f"exploratory:{name}")

    generic_exit = bool(_GENERIC_EXIT_LINE.search(text))
    unknown = False
    if len(distinct_codes) > 1:
        unknown = True
        reasons.append("conflicting_scaffold_exit_codes")
    if generic_exit and not distinct_codes:
        unknown = True
        reasons.append("generic_exit_line_not_scaffold")
    if exploratory and not test_failure and not test_pass:
        unknown = True
        reasons.append("unclassified_test_format")

    return ExecutionSignals(
        tool_exit_nonzero=nonzero,
        explicit_test_failure=test_failure,
        explicit_test_pass=test_pass,
        interrupt_or_timeout=interrupt,
        execution_status_unknown=unknown,
        explicit_exception=exception,
        exploratory_test_format=exploratory,
        scaffold_exit_codes=distinct_codes,
        reasons=tuple(reasons),
    )
