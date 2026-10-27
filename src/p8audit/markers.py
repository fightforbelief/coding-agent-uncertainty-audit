"""Conservative markers for explicit execution feedback.

Rules apply only to tool-observation text supplied by the caller. A mention
of the word "error" inside assistant prose is not an observation, so it cannot
fire these markers. Patterns are anchored to command or test summaries rather
than to arbitrary substrings inside source listings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")
# Numbered editor listings (`     1\tcode`) are not exception reports.
_EXCEPTION_LINE = re.compile(
    r"(?m)^(?!\s*\d+\s)[A-Za-z_][\w.]*(?:Error|Exception):\s+\S"
)
# Scaffold wrappers observed in OpenHands tool observations.
_SCAFFOLD_EXIT = re.compile(
    r"\[(?:The command completed|Command finished) with exit code (-?\d+)\."
)
_PYTEST_FAIL_LINE = re.compile(r"(?m)^FAILED\s+\S")
_PYTEST_SUMMARY_FAIL = re.compile(
    r"(?m)^={3,}.*\b\d+\s+failed\b.*={3,}\s*$"
)
_GO_FAIL = re.compile(r"(?m)^--- FAIL:|^FAIL(?:\t|\s+\S|\s*$)")
_CARGO_FAIL = re.compile(r"(?m)^test result: FAILED\b|^error: test failed\b")
_PYTEST_SUMMARY_PASS = re.compile(
    r"(?m)^={3,}.*\b\d+\s+passed\b.*={3,}\s*$"
)
_CARGO_PASS = re.compile(r"(?m)^test result: ok\b")
_GO_OK_PKG = re.compile(r"(?m)^ok\s+\S+")


@dataclass(frozen=True)
class ObservationMarkers:
    explicit_error: bool
    explicit_test_failure: bool
    explicit_test_pass: bool
    error_reasons: tuple[str, ...]
    test_failure_reasons: tuple[str, ...]
    test_pass_reasons: tuple[str, ...]

    @property
    def any_explicit_signal(self) -> bool:
        return self.explicit_error or self.explicit_test_failure or self.explicit_test_pass


def mark_observation(content: str) -> ObservationMarkers:
    text = content or ""
    error_reasons: list[str] = []
    fail_reasons: list[str] = []
    pass_reasons: list[str] = []

    if _TRACEBACK.search(text):
        error_reasons.append("python_traceback")
    if _EXCEPTION_LINE.search(text):
        error_reasons.append("exception_line")
    exit_codes = [int(value) for value in _SCAFFOLD_EXIT.findall(text)]
    if any(code != 0 for code in exit_codes):
        error_reasons.append("nonzero_scaffold_exit")

    if _PYTEST_FAIL_LINE.search(text):
        fail_reasons.append("pytest_failed_line")
    if _PYTEST_SUMMARY_FAIL.search(text):
        fail_reasons.append("pytest_summary_failed")
    if _GO_FAIL.search(text):
        fail_reasons.append("go_or_package_fail")
    if _CARGO_FAIL.search(text):
        fail_reasons.append("cargo_failed")

    failed = bool(fail_reasons)
    if not failed and _PYTEST_SUMMARY_PASS.search(text):
        pass_reasons.append("pytest_summary_passed")
    if not failed and _CARGO_PASS.search(text):
        pass_reasons.append("cargo_ok")
    if not failed and _GO_OK_PKG.search(text) and not re.search(r"(?m)^FAIL\b", text):
        pass_reasons.append("go_ok")

    return ObservationMarkers(
        explicit_error=bool(error_reasons),
        explicit_test_failure=bool(fail_reasons),
        explicit_test_pass=bool(pass_reasons),
        error_reasons=tuple(error_reasons),
        test_failure_reasons=tuple(fail_reasons),
        test_pass_reasons=tuple(pass_reasons),
    )


def error_signature(content: str) -> str | None:
    """Stable signature for an explicit error, used only after a marker fires."""
    markers = mark_observation(content)
    if not markers.explicit_error and not markers.explicit_test_failure:
        return None
    lines = []
    for line in (content or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            _TRACEBACK.search(stripped)
            or _EXCEPTION_LINE.search(stripped)
            or _SCAFFOLD_EXIT.search(stripped)
            or _PYTEST_FAIL_LINE.search(stripped)
            or _GO_FAIL.search(stripped)
            or _CARGO_FAIL.search(stripped)
        ):
            lines.append(stripped[:180])
        if len(lines) >= 3:
            break
    if not lines:
        return "|".join(markers.error_reasons + markers.test_failure_reasons)
    return " || ".join(lines)
