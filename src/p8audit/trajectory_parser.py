"""Deterministic parser for pinned Open-SWE-Traces trajectories.

Messages are normalized without copying ``reasoning_content`` into the
feature view. Action-observation cycles are assistant messages that carry
tool calls, followed by the contiguous ``tool`` observations that answer them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from p8audit.execution_signals import ExecutionSignals, classify_execution
from p8audit.markers import ObservationMarkers, mark_observation


@dataclass(frozen=True)
class ToolCall:
    call_id: str | None
    name: str | None
    arguments: str
    parse_status: str


@dataclass(frozen=True)
class NormalizedMessage:
    index: int
    role: str
    content: str
    think: bool | None
    tool_calls: tuple[ToolCall, ...]
    has_reasoning_content: bool
    parse_status: str


@dataclass(frozen=True)
class ActionObservationCycle:
    cycle_index: int
    action_index: int
    observation_indexes: tuple[int, ...]
    tool_names: tuple[str | None, ...]
    status: str
    markers: tuple[ObservationMarkers, ...]
    execution: tuple[ExecutionSignals, ...]


@dataclass
class ParseResult:
    status: str
    messages: list[NormalizedMessage] = field(default_factory=list)
    cycles: list[ActionObservationCycle] = field(default_factory=list)
    n_malformed_messages: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_complete_cycles(self) -> int:
        return sum(1 for cycle in self.cycles if cycle.status == "complete")

    def prefix_message_count(self, n_cycles: int) -> int | None:
        """Number of messages available after ``n_cycles`` completed cycles.

        The prefix ends at the last observation of the n-th completed cycle.
        Incomplete cycles do not count. Returns ``None`` when the trajectory
        never reaches that many completed cycles.
        """
        if n_cycles < 1:
            raise ValueError("n_cycles must be positive")
        seen = 0
        for cycle in self.cycles:
            if cycle.status != "complete":
                continue
            seen += 1
            if seen == n_cycles:
                if not cycle.observation_indexes:
                    return None
                return cycle.observation_indexes[-1] + 1
        return None


def coerce_trajectory(trajectory: Any) -> tuple[list[Any] | None, str]:
    """Accept a list or a JSON string. Reject anything else."""
    if isinstance(trajectory, list):
        return trajectory, "list"
    if isinstance(trajectory, str):
        try:
            parsed = json.loads(trajectory)
        except json.JSONDecodeError:
            return None, "malformed_json"
        if isinstance(parsed, list):
            return parsed, "json_string"
        return None, "json_not_list"
    return None, f"unsupported_type:{type(trajectory).__name__}"


def _parse_tool_call(raw: Any) -> ToolCall:
    if not isinstance(raw, dict):
        return ToolCall(None, None, "", "malformed_tool_call")
    function = raw.get("function")
    if not isinstance(function, dict):
        return ToolCall(
            _as_optional_str(raw.get("id")),
            None,
            "",
            "missing_function",
        )
    name = function.get("name")
    arguments = function.get("arguments")
    return ToolCall(
        call_id=_as_optional_str(raw.get("id")),
        name=name if isinstance(name, str) else None,
        arguments=arguments if isinstance(arguments, str) else "",
        parse_status="ok" if isinstance(name, str) and name else "missing_name",
    )


def _as_optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def normalize_message(index: int, raw: Any) -> tuple[NormalizedMessage | None, str | None]:
    if not isinstance(raw, dict):
        return None, "message_not_object"
    role = raw.get("role")
    if not isinstance(role, str) or not role:
        return None, "missing_role"
    content = raw.get("content")
    if content is None:
        content_text = ""
        content_status = "null_content"
    elif isinstance(content, str):
        content_text = content
        content_status = "ok"
    else:
        content_text = ""
        content_status = "non_string_content"
    reasoning = raw.get("reasoning_content")
    has_reasoning = isinstance(reasoning, str) and reasoning != ""
    think_raw = raw.get("think")
    think = think_raw if isinstance(think_raw, bool) else None
    raw_calls = raw.get("tool_calls") or []
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        calls: tuple[ToolCall, ...] = ()
        call_status = "tool_calls_not_list"
    else:
        calls = tuple(_parse_tool_call(item) for item in raw_calls)
        call_status = "ok"
    status = content_status if call_status == "ok" else f"{content_status}+{call_status}"
    return (
        NormalizedMessage(
            index=index,
            role=role,
            content=content_text,
            think=think,
            tool_calls=calls,
            has_reasoning_content=has_reasoning,
            parse_status=status,
        ),
        None,
    )


def parse_trajectory(trajectory: Any) -> ParseResult:
    records, kind = coerce_trajectory(trajectory)
    if records is None:
        return ParseResult(status=kind, notes=[kind])
    messages: list[NormalizedMessage] = []
    malformed = 0
    notes = [f"input:{kind}"]
    for index, raw in enumerate(records):
        message, error = normalize_message(index, raw)
        if message is None:
            malformed += 1
            notes.append(f"message {index}: {error}")
            continue
        messages.append(message)
    cycles = _pair_cycles(messages)
    if malformed and not messages:
        status = "unparseable"
    elif malformed:
        status = "partial"
    elif not cycles:
        status = "no_cycles"
    else:
        status = "ok"
    return ParseResult(
        status=status,
        messages=messages,
        cycles=cycles,
        n_malformed_messages=malformed,
        notes=notes,
    )


def _pair_cycles(messages: list[NormalizedMessage]) -> list[ActionObservationCycle]:
    cycles: list[ActionObservationCycle] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != "assistant" or not message.tool_calls:
            index += 1
            continue
        observations: list[NormalizedMessage] = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].role == "tool":
            observations.append(messages[cursor])
            cursor += 1
        expected = len(message.tool_calls)
        observed = len(observations)
        if observed == 0:
            status = "missing_observation"
        elif observed == expected:
            status = "complete"
        else:
            status = "count_mismatch"
        markers = tuple(mark_observation(item.content) for item in observations)
        execution = tuple(classify_execution(item.content) for item in observations)
        cycles.append(
            ActionObservationCycle(
                cycle_index=len(cycles),
                action_index=message.index,
                observation_indexes=tuple(item.index for item in observations),
                tool_names=tuple(call.name for call in message.tool_calls),
                status=status,
                markers=markers,
                execution=execution,
            )
        )
        index = cursor
    return cycles


def observable_text(message: NormalizedMessage) -> str:
    """Text that is allowed to enter feature extraction.

    ``reasoning_content`` is deliberately omitted.
    """
    return message.content
