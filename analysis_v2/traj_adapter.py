"""Adapt OpenHands and SWE-agent trajectories to one call list.

Round numbers follow parser v2 completed cycles. An incomplete cycle that
occurs before the next completed cycle receives that next round number, so
``round <= k`` matches messages inside ``prefix_message_count(k)``. A trailing
incomplete cycle is numbered ``n_complete_cycles + 1`` and stays outside every
completed prefix.
"""

from __future__ import annotations

import json
from typing import Any

from p8audit.prefix_features import tool_family
from p8audit.trajectory_parser import coerce_trajectory, parse_trajectory

_COMMAND_KEYS = ("command", "cmd")


def _field(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    getter = getattr(row, "get", None)
    if callable(getter):
        return getter(name)
    return None


def _arguments_text(raw_arguments: Any) -> str:
    if isinstance(raw_arguments, str):
        return raw_arguments
    if isinstance(raw_arguments, dict):
        return json.dumps(raw_arguments, ensure_ascii=False, separators=(",", ":"))
    return ""


def _command_text(config: str, tool: str | None, arguments: str) -> str | None:
    """Shell command text for execute-family tools. Other tools stay None.

    A non-empty ``command`` string wins, then a non-empty ``cmd`` string.
    A valid JSON object with neither field yields ``None`` so the object is
    not sent to test-command detection. A non-empty raw string that is not a
    JSON object stays the command. Empty arguments stay missing.
    """
    if tool_family(tool) != "execute":
        return None
    if config not in {"openhands", "sweagent"}:
        raise ValueError(f"unsupported scaffold: {config}")
    if not arguments.strip():
        return None
    text = arguments.strip()
    parsed: Any = None
    if text[0] in "{[":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return arguments
        if isinstance(parsed, dict):
            for key in _COMMAND_KEYS:
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return None
        return arguments
    return arguments


def _round_numbers(statuses: list[str]) -> list[int]:
    completed = 0
    pending = 1
    numbers: list[int] = []
    for status in statuses:
        if status == "complete":
            completed += 1
            numbers.append(completed)
            pending = completed + 1
        else:
            numbers.append(pending)
    return numbers


def iter_calls(row: Any) -> list[dict[str, Any]]:
    """Return parser-aligned calls for one trajectory row.

    Each item has ``round``, ``tool``, ``command``, ``input``, and ``output``.
    ``command`` is set only for shell-execution tools.
    """
    config = _field(row, "config") or _field(row, "scaffold")
    if config not in {"openhands", "sweagent"}:
        raise ValueError("row needs config or scaffold, one of openhands or sweagent")
    trajectory = _field(row, "trajectory")
    parsed = parse_trajectory(trajectory)
    records, _kind = coerce_trajectory(trajectory)
    by_index = {message.index: message for message in parsed.messages}
    rounds = _round_numbers([cycle.status for cycle in parsed.cycles])
    calls: list[dict[str, Any]] = []
    for cycle, round_number in zip(parsed.cycles, rounds, strict=True):
        action = by_index.get(cycle.action_index)
        raw_action = None
        if records is not None and 0 <= cycle.action_index < len(records) and isinstance(records[cycle.action_index], dict):
            raw_action = records[cycle.action_index]
        raw_calls = raw_action.get("tool_calls") if raw_action is not None else None
        if not isinstance(raw_calls, list):
            raw_calls = []
        observations = [by_index[index].content for index in cycle.observation_indexes if index in by_index]
        for position, tool_name in enumerate(cycle.tool_names):
            raw_call = raw_calls[position] if position < len(raw_calls) and isinstance(raw_calls[position], dict) else {}
            function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
            arguments = _arguments_text(function.get("arguments"))
            if action is not None and position < len(action.tool_calls) and not arguments:
                arguments = action.tool_calls[position].arguments
            observation_index = cycle.observation_indexes[position] if position < len(cycle.observation_indexes) else None
            output = observations[position] if position < len(observations) else None
            calls.append(
                {
                    "round": round_number,
                    "tool": tool_name,
                    "command": _command_text(config, tool_name, arguments),
                    "input": arguments,
                    "output": output,
                    "action_index": cycle.action_index,
                    "observation_index": observation_index,
                }
            )
    return calls
