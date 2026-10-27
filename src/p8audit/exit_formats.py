"""Candidate exit-code spellings for the bounded SWE-agent audit.

A match here is not a validated feature. ``classify_execution`` remains the
only function that can set ``tool_exit_nonzero``, and it still requires the
OpenHands scaffold wrapper.
"""

from __future__ import annotations

import re

from p8audit.execution_signals import _SCAFFOLD_EXIT, classify_execution

_RETURNCODE_TAG = re.compile(r"(?i)<returncode>\s*(-?\d+)\s*</returncode>")
_ASSIGNED_EXIT = re.compile(
    r"(?i)\b(?:exit_code|returncode|return_code)\b\s*[:=]\s*(-?\d+)"
)
_COMMAND_EXITED = re.compile(
    r"(?i)\bcommand exited with(?:\s+(?:status|code))?\s+(-?\d+)\b"
)
_GENERIC_EXIT = re.compile(r"(?m)^(?!\s*\[)\s*Exit code:\s*(-?\d+)\b")


def describe_exit_formats(text: str) -> dict[str, list[int]]:
    """Return integer codes found by each candidate spelling.

    Empty lists mean that spelling was not present. This function does not
    decide whether the observation failed.
    """
    raw = text or ""
    return {
        "openhands_scaffold_wrapper": [int(value) for value in _SCAFFOLD_EXIT.findall(raw)],
        "returncode_tag": [int(value) for value in _RETURNCODE_TAG.findall(raw)],
        "assigned_exit_field": [int(value) for value in _ASSIGNED_EXIT.findall(raw)],
        "command_exited_phrase": [int(value) for value in _COMMAND_EXITED.findall(raw)],
        "generic_exit_line": [int(value) for value in _GENERIC_EXIT.findall(raw)],
    }


def returncode_tag_is_not_scaffold_exit(text: str) -> bool:
    """A tagged return code must not be stored as ``tool_exit_nonzero``."""
    described = describe_exit_formats(text)
    signals = classify_execution(text)
    return bool(described["returncode_tag"]) and not signals.tool_exit_nonzero
