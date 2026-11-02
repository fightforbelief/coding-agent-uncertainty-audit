"""Prefix evidence for a human reviewer.

The frozen parser and adapter supply the window. Detector hit/miss labels are
not copied into the evidence record.
"""

from __future__ import annotations

from typing import Any

from p8audit.trajectory_parser import coerce_trajectory, parse_trajectory
from v3_3_pilot import argument_category, prefix_index_comparison
from v3_4_lib import build_calls

FORBIDDEN_KEYS = {
    "resolved",
    "reasoning_content",
    "reference_patch",
    "model_patch",
    "v3_ran_tests_r5",
    "v3_ran_tests_r15",
    "v3_ran_tests_amb_r5",
    "v3_ran_tests_amb_r15",
    "parser_v2_strict_fail_r5",
    "stratum",
    "sampling_stratum",
}


def _boundary(parsed: Any, calls: list[dict[str, Any]], checkpoint: int) -> str:
    status = str(prefix_index_comparison(parsed, calls, checkpoint)["status"])
    if checkpoint != 5 and status == "no_five_round_boundary":
        return "no_prefix_boundary"
    return status


def _call(call: dict[str, Any], order: int) -> dict[str, Any]:
    arguments = call.get("input") or ""
    return {
        "order": order,
        "round": int(call["round"]),
        "tool": call.get("tool"),
        "action_index": call.get("action_index"),
        "observation_index": call.get("observation_index"),
        "arguments": arguments,
        "extracted_command": call.get("command"),
        "command_extraction_status": argument_category(call.get("tool"), arguments),
        "observation": call.get("output"),
    }


def extract_prefix_evidence(trajectory: Any, scaffold: str) -> dict[str, Any]:
    """Tool calls inside the first 5 and first 15 completed rounds."""
    records, _kind = coerce_trajectory(trajectory)
    parsed = parse_trajectory(records if records is not None else trajectory)
    calls = build_calls(scaffold, parsed, records)
    windows = {}
    for checkpoint in (5, 15):
        chosen = [call for call in calls if int(call["round"]) <= checkpoint]
        windows[f"r{checkpoint}"] = [_call(call, order) for order, call in enumerate(chosen)]
    evidence = {
        "completed_rounds": None if parsed.n_complete_cycles is None else int(parsed.n_complete_cycles),
        "prefix_parse_status": parsed.status,
        "prefix5_boundary": _boundary(parsed, calls, 5),
        "prefix15_boundary": _boundary(parsed, calls, 15),
        "r5": windows["r5"],
        "r15": windows["r15"],
    }
    leaked = sorted(forbidden_keys(evidence))
    if leaked:
        raise RuntimeError(f"evidence contains forbidden keys: {leaked}")
    return evidence


def forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                found.add(str(key))
            found.update(forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(forbidden_keys(item))
    return found
