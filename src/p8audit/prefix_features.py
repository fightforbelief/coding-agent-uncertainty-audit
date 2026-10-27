"""Prefix features for the frozen L2 run.

Tool families and the checkpoint rule are fixed before any model is fit.
``resolved`` is not an input. Messages after the selected completed-round
cut are not counted. ``reasoning_content`` is not part of ``NormalizedMessage.content``.
"""

from __future__ import annotations

from p8audit.execution_signals_v2 import classify_execution_v2
from p8audit.trajectory_parser import ParseResult

TOOL_FAMILIES = (
    "execute",
    "edit",
    "read",
    "search",
    "think",
    "submit",
    "other",
    "unknown",
)
USABLE_PARSE_STATUSES = frozenset({"ok", "no_cycles"})
STRUCTURAL_FIELDS = (
    "prefix_completed_rounds",
    "prefix_checkpoint_reached",
    "prefix_assistant_message_count",
    "prefix_tool_call_count",
    "prefix_tool_observation_count",
    "prefix_distinct_tool_count",
    "prefix_user_content_chars",
    "prefix_assistant_content_chars",
    "prefix_tool_observation_chars",
    "prefix_tool_observation_lines",
) + tuple(f"prefix_tool_family_{name}" for name in TOOL_FAMILIES)
PRIMARY_EVENT = "prefix_explicit_test_failure_any"
SECONDARY_EVENT = "prefix_tool_exit_nonzero_any"
AUDIT_ONLY_FIELDS = ("n_complete_cycles", "parse_status", "parse_eligible")


def tool_family(name: str | None) -> str:
    """Map one tool name to a family frozen before model fitting."""
    if name is None or not str(name).strip():
        return "unknown"
    text = str(name).lower()
    if any(token in text for token in ("bash", "shell", "terminal", "execute", "cmd")):
        return "execute"
    if any(token in text for token in ("edit", "str_replace", "patch", "write")):
        return "edit"
    if any(token in text for token in ("read", "view", "open_file")):
        return "read"
    if any(token in text for token in ("grep", "search", "find", "glob")):
        return "search"
    if "think" in text or text == "plan":
        return "think"
    if any(token in text for token in ("finish", "submit")):
        return "submit"
    return "other"


def choose_checkpoint(n_known_label: int, n_reaching_five: int) -> int:
    """Return 5 when known-label coverage is at least 70%, otherwise 3."""
    if n_known_label < 1:
        raise ValueError("known-label coverage requires at least one trajectory")
    if n_reaching_five / n_known_label >= 0.70:
        return 5
    return 3


def predictor_fields(checkpoint: int, *, include_exit: bool) -> list[str]:
    if checkpoint not in (3, 5):
        raise ValueError("checkpoint must be 3 or 5")
    fields = [f"k{checkpoint}_{name}" for name in STRUCTURAL_FIELDS]
    fields.append(f"k{checkpoint}_{PRIMARY_EVENT}")
    if include_exit:
        fields.append(f"k{checkpoint}_{SECONDARY_EVENT}")
    return fields


def _empty_features(checkpoint: int) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for name in STRUCTURAL_FIELDS:
        values[f"k{checkpoint}_{name}"] = None
    values[f"k{checkpoint}_{PRIMARY_EVENT}"] = None
    values[f"k{checkpoint}_{SECONDARY_EVENT}"] = None
    return values


def prefix_feature_row(parsed: ParseResult, *, config: str) -> dict[str, float | str | int | None]:
    """Features at both candidate checkpoints. Raw cycle count is audit-only."""
    row: dict[str, float | str | int | None] = {
        "parse_status": parsed.status,
        "parse_eligible": parsed.status in USABLE_PARSE_STATUSES,
        "n_complete_cycles": None if parsed.status not in USABLE_PARSE_STATUSES else parsed.n_complete_cycles,
    }
    for checkpoint in (3, 5):
        if parsed.status not in USABLE_PARSE_STATUSES:
            row.update(_empty_features(checkpoint))
            continue
        row.update(_features_at_checkpoint(parsed, checkpoint, config=config))
    return row


def _features_at_checkpoint(parsed: ParseResult, checkpoint: int, *, config: str) -> dict[str, float]:
    completed = parsed.n_complete_cycles
    used = min(completed, checkpoint)
    if used == 0:
        messages = []
    else:
        end = parsed.prefix_message_count(used)
        if end is None:
            raise ValueError("completed cycles did not yield a prefix end")
        messages = [message for message in parsed.messages if message.index < end]
    assistant = [message for message in messages if message.role == "assistant"]
    users = [message for message in messages if message.role == "user"]
    observations = [message for message in messages if message.role == "tool"]
    names: list[str | None] = []
    for message in assistant:
        names.extend(call.name for call in message.tool_calls)
    families = {family: 0 for family in TOOL_FAMILIES}
    for name in names:
        families[tool_family(name)] += 1
    test_failure = 0
    exit_nonzero = 0
    for message in observations:
        signals = classify_execution_v2(message.content)
        if signals.explicit_test_failure:
            test_failure = 1
        if signals.tool_exit_nonzero:
            exit_nonzero = 1
    values: dict[str, float] = {
        f"k{checkpoint}_prefix_completed_rounds": float(used),
        f"k{checkpoint}_prefix_checkpoint_reached": float(completed >= checkpoint),
        f"k{checkpoint}_prefix_assistant_message_count": float(len(assistant)),
        f"k{checkpoint}_prefix_tool_call_count": float(len(names)),
        f"k{checkpoint}_prefix_tool_observation_count": float(len(observations)),
        f"k{checkpoint}_prefix_distinct_tool_count": float(len({name for name in names if name})),
        f"k{checkpoint}_prefix_user_content_chars": float(sum(len(message.content) for message in users)),
        f"k{checkpoint}_prefix_assistant_content_chars": float(sum(len(message.content) for message in assistant)),
        f"k{checkpoint}_prefix_tool_observation_chars": float(sum(len(message.content) for message in observations)),
        f"k{checkpoint}_prefix_tool_observation_lines": float(
            sum(message.content.count("\n") + (1 if message.content else 0) for message in observations)
        ),
        f"k{checkpoint}_{PRIMARY_EVENT}": float(test_failure),
    }
    for family, count in families.items():
        values[f"k{checkpoint}_prefix_tool_family_{family}"] = float(count)
    if config == "openhands":
        values[f"k{checkpoint}_{SECONDARY_EVENT}"] = float(exit_nonzero)
    else:
        values[f"k{checkpoint}_{SECONDARY_EVENT}"] = None
    return values
