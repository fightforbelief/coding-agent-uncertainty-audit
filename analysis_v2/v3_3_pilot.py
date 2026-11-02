"""V3.3 fresh-shard pilot helpers.

Sampling and measurement only. This module does not read ``resolved``,
does not score a model, and does not download data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from p8audit.execution_signals_v2 import classify_execution_v2
from p8audit.prefix_features import tool_family
from p8audit.trajectory_parser import parse_trajectory
from traj_adapter import iter_calls

DATASET_ID = "nvidia/Open-SWE-Traces"
PINNED_REVISION = "c144d9d160dcd9f5ebcd9520d2d54111dda8b681"
SELECTION_SEED = 20261103
CHECKPOINT = 5

GROUPS = (
    ("data/minimax_m25_openhands_trajectories", "openhands", "minimax_m25"),
    ("data/qwen35_openhands_trajectories", "openhands", "qwen35_122b"),
    ("data/minimax_m25_sweagent_trajectories", "sweagent", "minimax_m25"),
    ("data/qwen35_sweagent_trajectories", "sweagent", "qwen35_122b"),
)

EXCLUDED = {
    ("data/minimax_m25_openhands_trajectories", "train-00000-of-00020.parquet"),
    ("data/minimax_m25_openhands_trajectories", "train-00001-of-00020.parquet"),
    ("data/minimax_m25_openhands_trajectories", "train-00004-of-00020.parquet"),
    ("data/minimax_m25_openhands_trajectories", "train-00012-of-00020.parquet"),
    ("data/minimax_m25_openhands_trajectories", "train-00016-of-00020.parquet"),
    ("data/qwen35_openhands_trajectories", "train-00000-of-00023.parquet"),
    ("data/qwen35_openhands_trajectories", "train-00003-of-00023.parquet"),
    ("data/qwen35_openhands_trajectories", "train-00007-of-00023.parquet"),
    ("data/qwen35_openhands_trajectories", "train-00009-of-00023.parquet"),
    ("data/qwen35_openhands_trajectories", "train-00018-of-00023.parquet"),
    ("data/minimax_m25_sweagent_trajectories", "train-00000-of-00023.parquet"),
    ("data/minimax_m25_sweagent_trajectories", "train-00011-of-00023.parquet"),
    ("data/minimax_m25_sweagent_trajectories", "train-00015-of-00023.parquet"),
    ("data/minimax_m25_sweagent_trajectories", "train-00017-of-00023.parquet"),
    ("data/minimax_m25_sweagent_trajectories", "train-00018-of-00023.parquet"),
    ("data/qwen35_sweagent_trajectories", "train-00000-of-00018.parquet"),
    ("data/qwen35_sweagent_trajectories", "train-00010-of-00018.parquet"),
    ("data/qwen35_sweagent_trajectories", "train-00011-of-00018.parquet"),
    ("data/qwen35_sweagent_trajectories", "train-00012-of-00018.parquet"),
    ("data/qwen35_sweagent_trajectories", "train-00015-of-00018.parquet"),
}

READ_COLUMNS = ("trajectory_id", "instance_id", "repo", "language", "trajectory")
PROHIBITED_COLUMNS = ("resolved", "metadata", "reference_patch", "model_patch")


def columns_to_read(schema_names: list[str] | tuple[str, ...]) -> list[str]:
    """Return the parser columns. Prohibited fields are never included."""
    present = set(schema_names)
    missing = [name for name in READ_COLUMNS if name not in present]
    if missing:
        raise ValueError(f"required columns missing: {missing}")
    chosen = list(READ_COLUMNS)
    if any(name in chosen for name in PROHIBITED_COLUMNS):
        raise ValueError("prohibited column was requested")
    return chosen
FAILURE_REASONS = (
    "pytest_failed_line",
    "pytest_summary_failed",
    "unittest_fail_line",
    "go_test_fail",
    "rust_test_fail",
    "cargo_test_failed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    actual_bytes = path.stat().st_size if path.is_file() else None
    actual_sha = sha256_file(path) if path.is_file() else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "expected_bytes": int(expected_bytes),
        "actual_bytes": actual_bytes,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha,
        "pass": actual_bytes == int(expected_bytes) and actual_sha == expected_sha256,
    }


def _parent(path: str) -> str:
    return str(Path(path).parent).replace("\\", "/")


def select_shards(files: list[dict[str, Any]], *, seed: int = SELECTION_SEED) -> dict[str, Any]:
    """Sample two lexicographically sorted candidates from each group.

    The manifest revision and the excluded shard names are checked first.
    NumPy's generator is consumed once per group, in the fixed group order.
    """
    if len(EXCLUDED) != 20:
        raise ValueError("the exclusion list must contain 20 shards")
    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group, _, _ in GROUPS}
    for item in files:
        parent = _parent(str(item["path"]))
        if parent not in grouped:
            raise ValueError(f"path is outside SHARD_GROUPS: {item['path']}")
        grouped[parent].append(item)
    found = {key: False for key in EXCLUDED}
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    candidates: dict[str, int] = {}
    for group, scaffold, model in GROUPS:
        ordered = sorted(grouped[group], key=lambda item: str(item["path"]))
        for item in ordered:
            key = (group, Path(str(item["path"])).name)
            if key in found:
                found[key] = True
        missing = [name for (parent, name), present in found.items() if parent == group and not present]
        if missing:
            raise ValueError(f"excluded shards missing from {group}: {missing}")
        remaining = [item for item in ordered if (group, Path(str(item["path"])).name) not in EXCLUDED]
        candidates[group] = len(remaining)
        if len(remaining) < 2:
            raise ValueError(f"{group} has fewer than 2 fresh shards")
        picks = rng.choice(len(remaining), size=2, replace=False)
        for index in picks.tolist():
            item = remaining[int(index)]
            if str(item.get("config")) != scaffold or str(item.get("split")) != model:
                raise ValueError(f"manifest scaffold/model does not match the path: {item['path']}")
            selected.append(
                {
                    "path": str(item["path"]).replace("\\", "/"),
                    "bytes": int(item["bytes"]),
                    "sha256": str(item["sha256"]),
                    "scaffold": scaffold,
                    "model": model,
                    "group": group,
                }
            )
    if not all(found.values()):
        raise ValueError("not every excluded shard was found in the manifest")
    if len(selected) != 8:
        raise ValueError("selection must contain 8 shards")
    return {
        "seed": seed,
        "dataset": DATASET_ID,
        "revision": PINNED_REVISION,
        "excluded_count": 20,
        "candidates_per_group": candidates,
        "selected": selected,
    }


def argument_category(tool: str | None, arguments: str) -> str:
    """Classify one argument payload. This does not decide that a test ran."""
    if tool_family(tool) != "execute":
        return "non_execution"
    if not arguments.strip():
        return "empty"
    text = arguments.strip()
    if text[0] in "{[":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "invalid_json"
        if not isinstance(parsed, dict):
            return "json_non_object"
        command = parsed.get("command")
        cmd = parsed.get("cmd")
        if isinstance(command, str) and command.strip():
            return "json_command"
        if isinstance(cmd, str) and cmd.strip():
            return "json_cmd"
        return "json_neither"
    return "raw_string"


def argument_notes(tool: str | None, arguments: str) -> list[str]:
    """Formats that the current extractor does not treat as a new command key."""
    if tool_family(tool) != "execute" or not arguments.strip() or arguments.strip()[0] not in "{[":
        return []
    try:
        parsed = json.loads(arguments.strip())
    except json.JSONDecodeError:
        return []
    notes: list[str] = []
    if not isinstance(parsed, dict):
        notes.append("json_non_object")
        return notes
    for key in ("command", "cmd"):
        if key in parsed and not isinstance(parsed[key], str):
            notes.append(f"non_string_{key}")
    return notes


def prefix_index_comparison(parsed: Any, calls: list[dict[str, Any]], checkpoint: int = CHECKPOINT) -> dict[str, Any]:
    """Compare observation indexes, not their counts alone."""
    round_indexes = [
        call["observation_index"]
        for call in calls
        if int(call["round"]) <= checkpoint and call["observation_index"] is not None
    ]
    end = parsed.prefix_message_count(checkpoint)
    if end is None:
        return {
            "status": "no_five_round_boundary",
            "end": None,
            "round_indexes": round_indexes,
            "prefix_indexes": None,
        }
    prefix_indexes = [message.index for message in parsed.messages if message.role == "tool" and message.index < end]
    status = "match" if round_indexes == prefix_indexes else "mismatch"
    return {
        "status": status,
        "end": end,
        "round_indexes": round_indexes,
        "prefix_indexes": prefix_indexes,
    }


def frozen_prefix_observations(parsed: Any, checkpoint: int = CHECKPOINT) -> list[Any]:
    """Tool observations inside the frozen ``min(n_complete, checkpoint)`` prefix."""
    used = min(parsed.n_complete_cycles, checkpoint)
    if used == 0:
        return []
    end = parsed.prefix_message_count(used)
    if end is None:
        return []
    return [message for message in parsed.messages if message.role == "tool" and message.index < end]


def _excerpt(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def inspect_trajectory(
    *,
    scaffold: str,
    model: str,
    shard_path: str,
    trajectory_id: str,
    instance_id: str,
    repo: str,
    language: str | None,
    trajectory: Any,
) -> dict[str, Any]:
    """Measure one trajectory. Numeric fields stay missing when parsing fails."""
    base = {
        "trajectory_id": str(trajectory_id),
        "instance_id": str(instance_id),
        "repo": str(repo),
        "language": None if language is None else str(language),
        "scaffold": scaffold,
        "model": model,
        "shard_path": shard_path,
    }
    try:
        parsed = parse_trajectory(trajectory)
        calls = iter_calls({"scaffold": scaffold, "trajectory": trajectory})
    except Exception as exc:
        base.update(
            {
                "n_rounds_total": None,
                "parser_status": "exception",
                "prefix_completed_rounds": None,
                "prefix_explicit_test_failure_any": None,
                "command_extraction_status": None,
                "exception_type": type(exc).__name__,
            }
        )
        return {"meta": base, "error": True, "edges": [{"kind": "exception", "detail": type(exc).__name__}]}

    categories: dict[str, int] = {}
    notes: list[str] = []
    json_keys: dict[str, int] = {}
    for call in calls:
        category = argument_category(call["tool"], call["input"] or "")
        categories[category] = categories.get(category, 0) + 1
        notes.extend(argument_notes(call["tool"], call["input"] or ""))
        text = (call["input"] or "").strip()
        if category.startswith("json_") and text[:1] in "{[":
            try:
                parsed_arguments = json.loads(text)
            except json.JSONDecodeError:
                parsed_arguments = None
            if isinstance(parsed_arguments, dict):
                key_name = ",".join(sorted(str(key) for key in parsed_arguments))
                json_keys[key_name] = json_keys.get(key_name, 0) + 1
    comparison = prefix_index_comparison(parsed, calls, CHECKPOINT)
    observations = frozen_prefix_observations(parsed, CHECKPOINT)
    signal_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    excerpts: list[dict[str, Any]] = []
    for message in observations:
        signals = classify_execution_v2(message.content)
        for name in (
            "explicit_test_failure",
            "build_or_compile_failure",
            "test_setup_or_collection_failure",
            "package_level_failure_unresolved",
            "explicit_test_pass",
            "execution_status_unknown",
        ):
            if getattr(signals, name):
                signal_counts[name] = signal_counts.get(name, 0) + 1
        for reason in signals.reasons:
            if reason in FAILURE_REASONS:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if signals.explicit_test_failure:
            excerpts.append(
                {
                    "trajectory_id": str(trajectory_id),
                    "shard_path": shard_path,
                    "scaffold": scaffold,
                    "model": model,
                    "message_index": message.index,
                    "reasons": "|".join(reason for reason in signals.reasons if reason in FAILURE_REASONS),
                    "excerpt": _excerpt(message.content),
                }
            )
    edges: list[dict[str, str]] = []
    if parsed.n_malformed_messages:
        edges.append({"kind": "malformed_messages", "detail": str(parsed.n_malformed_messages)})
    cycle_counts: dict[str, int] = {}
    for cycle in parsed.cycles:
        cycle_counts[cycle.status] = cycle_counts.get(cycle.status, 0) + 1
        if cycle.status != "complete":
            edges.append({"kind": cycle.status, "detail": f"action_index={cycle.action_index}"})
    if comparison["status"] == "mismatch":
        edges.append({"kind": "prefix_mismatch", "detail": f"end={comparison['end']}"})
    if notes:
        edges.append({"kind": "unexpected_argument", "detail": ",".join(sorted(set(notes)))})
    valid = parsed.status in {"ok", "no_cycles"}
    base.update(
        {
            "n_rounds_total": parsed.n_complete_cycles,
            "parser_status": parsed.status,
            "prefix_completed_rounds": min(parsed.n_complete_cycles, CHECKPOINT),
            "prefix_explicit_test_failure_any": int(signal_counts.get("explicit_test_failure", 0) > 0),
            "command_extraction_status": "unexpected" if notes or "json_non_object" in categories else "known",
            "exception_type": None,
        }
    )
    return {
        "meta": base,
        "error": not valid,
        "categories": categories,
        "json_keys": json_keys,
        "notes": notes,
        "comparison": comparison,
        "signal_counts": signal_counts,
        "reason_counts": reason_counts,
        "excerpts": excerpts,
        "edges": edges,
        "n_assistant": sum(1 for message in parsed.messages if message.role == "assistant"),
        "n_tool_calls": sum(len(message.tool_calls) for message in parsed.messages if message.role == "assistant"),
        "n_observations": sum(1 for message in parsed.messages if message.role == "tool"),
        "n_malformed": parsed.n_malformed_messages,
        "cycle_counts": cycle_counts,
        "tool_names": [call["tool"] for call in calls],
    }
