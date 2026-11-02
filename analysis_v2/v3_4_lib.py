"""V3.4 measurement helpers.

The frozen parser, parser v2, and ``p8v3.testcmd`` are used as they are.
``resolved`` is not an argument of trajectory measurement. A failed parse stays
missing and is never written as zero.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from census_metrics import census_flags
from p8audit.execution_signals_v2 import classify_execution_v2
from p8audit.prefix_features import tool_family
from p8audit.schema import PINNED_REVISION, SHARD_GROUPS
from p8audit.trajectory_parser import coerce_trajectory, parse_trajectory
from p8v3.testcmd import detect_test_command, parse_test_output
from traj_adapter import _arguments_text, _command_text, _round_numbers
from v3_3_pilot import (
    argument_category,
    frozen_prefix_observations,
    prefix_index_comparison,
)

EXPECTED_MANIFEST_SHA256 = "97eb461d752b03782434b7eacb4c6a6290c769d513be00084a2716d2444c4b0b"
EXPECTED_GROUP_COUNTS = {
    "data/minimax_m25_openhands_trajectories": 20,
    "data/qwen35_openhands_trajectories": 23,
    "data/minimax_m25_sweagent_trajectories": 23,
    "data/qwen35_sweagent_trajectories": 18,
}
EXPECTED_CORPUS = {
    "trajectories": 207489,
    "resolved_1": 65244,
    "resolved_0": 95487,
    "resolved_minus_1": 46758,
    "repositories": 2654,
    "issues": 22320,
    "train_rows": 150190,
    "validation_rows": 32124,
    "test_rows": 25175,
}
LABEL_SHA256 = "4255603dce2461ffe46ed5c66a04113439fcff729e58db10e8cfee88cc3e88e6"
FEATURE_SHA256 = "eae8f46c52965534ded6f8d38931d1430423828f0917a35ad207837586c24485"
V33_SHARD_PATHS = (
    "data/minimax_m25_openhands_trajectories/train-00017-of-00020.parquet",
    "data/minimax_m25_openhands_trajectories/train-00005-of-00020.parquet",
    "data/qwen35_openhands_trajectories/train-00015-of-00023.parquet",
    "data/qwen35_openhands_trajectories/train-00019-of-00023.parquet",
    "data/minimax_m25_sweagent_trajectories/train-00022-of-00023.parquet",
    "data/minimax_m25_sweagent_trajectories/train-00007-of-00023.parquet",
    "data/qwen35_sweagent_trajectories/train-00002-of-00018.parquet",
    "data/qwen35_sweagent_trajectories/train-00016-of-00018.parquet",
)
V33_EXPECTED_TOTALS = {
    "n_total": 19768,
    "n_valid": 19768,
    "n_error": 0,
    "prefix_mismatch_r5": 0,
    "explicit_test_failure": 8,
    "explicit_test_pass": 63,
    "build_or_compile_failure": 4,
    "package_level_failure_unresolved": 7,
    "test_setup_or_collection_failure": 0,
    "execution_status_unknown": 0,
}
V33_EXPECTED_SHARDS = {
    "data/minimax_m25_openhands_trajectories/train-00017-of-00020.parquet": {
        "rows": 2500,
        "explicit_test_failure": 2,
        "explicit_test_pass": 8,
        "build_or_compile_failure": 0,
        "package_level_failure_unresolved": 0,
    },
    "data/minimax_m25_openhands_trajectories/train-00005-of-00020.parquet": {
        "rows": 2500,
        "explicit_test_failure": 1,
        "explicit_test_pass": 4,
        "build_or_compile_failure": 0,
        "package_level_failure_unresolved": 0,
    },
    "data/qwen35_openhands_trajectories/train-00015-of-00023.parquet": {
        "rows": 2500,
        "explicit_test_failure": 0,
        "explicit_test_pass": 1,
        "build_or_compile_failure": 1,
        "package_level_failure_unresolved": 0,
    },
    "data/qwen35_openhands_trajectories/train-00019-of-00023.parquet": {
        "rows": 2500,
        "explicit_test_failure": 0,
        "explicit_test_pass": 1,
        "build_or_compile_failure": 3,
        "package_level_failure_unresolved": 0,
    },
    "data/minimax_m25_sweagent_trajectories/train-00022-of-00023.parquet": {
        "rows": 2268,
        "explicit_test_failure": 4,
        "explicit_test_pass": 26,
        "build_or_compile_failure": 0,
        "package_level_failure_unresolved": 6,
    },
    "data/minimax_m25_sweagent_trajectories/train-00007-of-00023.parquet": {
        "rows": 2500,
        "explicit_test_failure": 0,
        "explicit_test_pass": 23,
        "build_or_compile_failure": 0,
        "package_level_failure_unresolved": 1,
    },
    "data/qwen35_sweagent_trajectories/train-00002-of-00018.parquet": {
        "rows": 2500,
        "explicit_test_failure": 0,
        "explicit_test_pass": 0,
        "build_or_compile_failure": 0,
        "package_level_failure_unresolved": 0,
    },
    "data/qwen35_sweagent_trajectories/train-00016-of-00018.parquet": {
        "rows": 2500,
        "explicit_test_failure": 1,
        "explicit_test_pass": 0,
        "build_or_compile_failure": 0,
        "package_level_failure_unresolved": 0,
    },
}
V2_SIGNALS = (
    "explicit_test_failure",
    "explicit_test_pass",
    "build_or_compile_failure",
    "test_setup_or_collection_failure",
    "package_level_failure_unresolved",
    "execution_status_unknown",
)
CHECKPOINTS = (5, 15)
CENSUS_COLUMNS = ("trajectory_id", "instance_id", "repo", "trajectory")
PROHIBITED_TOP_LEVEL = ("resolved", "metadata", "reference_patch", "model_patch")
USABLE_PARSE_STATUSES = frozenset({"ok", "no_cycles"})
STATUS_COMPLETE = "V3.4 FULL CENSUS COMPLETE — DESCRIPTIVE RESULTS ONLY"
STATUS_PARTIAL = "V3.4 PARTIAL — DATA OR PARSER ISSUES"
STATUS_BLOCKED = "V3.4 BLOCKED — PROVENANCE OR METADATA FAILURE"
PARAMS = {
    "revision": PINNED_REVISION,
    "checkpoints": list(CHECKPOINTS),
    "disagreement_seed": 20261102,
    "resolved_passed_to_detector": False,
    "reasoning_content_used_as_evidence": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_token(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def params_token(params: dict | None = None) -> str:
    payload = params if params is not None else PARAMS
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def manifest_problems(payload: Any, *, digest: str | None = None) -> list[str]:
    problems: list[str] = []
    if digest is not None and digest != EXPECTED_MANIFEST_SHA256:
        problems.append(f"manifest sha256 is {digest}")
    if not isinstance(payload, dict):
        return problems + ["manifest is not an object"]
    if payload.get("revision") != PINNED_REVISION:
        problems.append("manifest revision is not the pinned revision")
    files = payload.get("files")
    if not isinstance(files, list):
        return problems + ["manifest files is not a list"]
    if len(files) != 84:
        problems.append(f"manifest has {len(files)} files")
    seen: set[str] = set()
    groups: Counter[str] = Counter()
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            problems.append(f"file {index} is not an object")
            continue
        relative = str(item.get("path", "")).replace("\\", "/")
        if not relative or relative in seen:
            problems.append(f"missing or duplicate path at {index}: {relative}")
        seen.add(relative)
        parent = str(Path(relative).parent).replace("\\", "/")
        groups[parent] += 1
        if parent not in SHARD_GROUPS:
            problems.append(f"path outside pinned groups: {relative}")
        if "bytes" not in item:
            problems.append(f"missing bytes: {relative}")
        else:
            try:
                if int(item["bytes"]) <= 0:
                    problems.append(f"non-positive bytes: {relative}")
            except (TypeError, ValueError):
                problems.append(f"bytes is not an integer: {relative}")
        sha = item.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            problems.append(f"missing sha256: {relative}")
    for name, expected in EXPECTED_GROUP_COUNTS.items():
        if groups[name] != expected:
            problems.append(f"{name} has {groups[name]} files, expected {expected}")
    return problems


def locate_shard(data_root: Path, relative: str) -> Path | None:
    """Return the one file for a manifest path.

    A path that exists both with and without the leading ``data/`` is accepted
    only when both entries are the same file. Two different files are an error.
    """
    rel = Path(str(relative).replace("\\", "/"))
    candidates = [Path(data_root) / rel]
    if rel.parts and rel.parts[0] == "data":
        candidates.append(Path(data_root) / Path(*rel.parts[1:]))
    existing: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.is_file():
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                existing.append(candidate)
    if not existing:
        return None
    if len(existing) == 1:
        return existing[0]
    hashes = {sha256_file(path) for path in existing}
    if len(hashes) == 1:
        return existing[0]
    raise FileExistsError(f"multiple different files for {relative}: {existing}")


def verify_local_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    actual_bytes = path.stat().st_size if path.is_file() else None
    actual_sha = sha256_file(path) if path.is_file() else None
    return {
        "exists": path.is_file(),
        "expected_bytes": int(expected_bytes),
        "actual_bytes": actual_bytes,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha,
        "pass": actual_bytes == int(expected_bytes) and actual_sha == expected_sha256,
    }


def group_for_relative(relative: str) -> tuple[str, str]:
    parent = str(Path(str(relative).replace("\\", "/")).parent).replace("\\", "/")
    if parent not in SHARD_GROUPS:
        raise KeyError(parent)
    return SHARD_GROUPS[parent]


def outcome_kind(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "invalid"
    if isinstance(value, str):
        return "invalid"
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "invalid"
    if float(value) != float(number):
        return "invalid"
    if number in (0, 1):
        return "known"
    if number == -1:
        return "unknown"
    return "invalid"


def _integral(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    return int(number)


def _binary_flag(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    number = _integral(value)
    if number not in (0, 1):
        return None
    return number


def assemble_metadata(
    shard_ids: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    repo_to_split: dict[str, str],
    expected: dict[str, int] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Join shard identifiers to the historical L2 label and feature tables.

    ``issue_id`` is the stored ``instance_id``. It is not built by concatenating
    repository and issue number. ``split`` is the saved label column, checked
    against the saved repository map. Null parser flags stay null.
    """
    expected = EXPECTED_CORPUS if expected is None else expected
    problems: list[str] = []
    detail: dict[str, Any] = {"problems": problems}

    def need(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            problems.append(f"{name} missing columns {missing}")

    need(shard_ids, ("trajectory_id", "instance_id", "repo", "shard"), "shard ids")
    need(labels, ("trajectory_id", "instance_id", "repo", "config", "resolved", "split", "shard"), "labels")
    need(
        features,
        (
            "trajectory_id",
            "instance_id",
            "repo",
            "config",
            "dataset_split",
            "n_complete_cycles",
            "k5_prefix_explicit_test_failure_any",
            "shard",
        ),
        "features",
    )
    if problems:
        return None, detail

    shards = shard_ids.copy()
    stored = labels.copy()
    feats = features.copy()
    for frame in (shards, stored, feats):
        frame["trajectory_id"] = frame["trajectory_id"].astype(str)
        frame["instance_id"] = frame["instance_id"].astype(str)
        frame["repo"] = frame["repo"].astype(str)
        frame["shard"] = frame["shard"].astype(str).str.replace("\\", "/", regex=False)

    def duplicates(frame: pd.DataFrame, name: str) -> None:
        found = int(frame["trajectory_id"].duplicated().sum())
        if found:
            problems.append(f"{name} has {found} duplicate trajectory ids")

    duplicates(shards, "shard ids")
    duplicates(stored, "labels")
    duplicates(feats, "features")
    shard_set = set(shards["trajectory_id"])
    label_set = set(stored["trajectory_id"])
    feature_set = set(feats["trajectory_id"])
    if shard_set != label_set or shard_set != feature_set:
        problems.append(
            "trajectory id sets differ: "
            f"shards {len(shard_set)}, labels {len(label_set)}, features {len(feature_set)}, "
            f"missing_from_shards {len(label_set - shard_set)}, "
            f"missing_from_labels {len(shard_set - label_set)}, "
            f"missing_from_features {len(shard_set - feature_set)}"
        )
    if problems:
        detail["rows"] = {"shards": len(shards), "labels": len(stored), "features": len(feats)}
        return None, detail

    merged = shards.merge(stored, on="trajectory_id", how="inner", validate="one_to_one", suffixes=("", "_label"))
    merged = merged.merge(feats, on="trajectory_id", how="inner", validate="one_to_one", suffixes=("", "_feature"))
    if len(merged) != len(shards):
        problems.append("join changed the row count")
    for left, right, label in (
        ("instance_id", "instance_id_label", "instance_id labels"),
        ("repo", "repo_label", "repo labels"),
        ("shard", "shard_label", "shard labels"),
        ("instance_id", "instance_id_feature", "instance_id features"),
        ("repo", "repo_feature", "repo features"),
        ("shard", "shard_feature", "shard features"),
        ("config", "config_feature", "config"),
    ):
        mismatch = int((merged[left].astype(str) != merged[right].astype(str)).sum())
        if mismatch:
            problems.append(f"{label} disagree on {mismatch} rows")

    scaffolds: list[str] = []
    models: list[str] = []
    group_mismatch = 0
    for relative, config, dataset_split in zip(merged["shard"], merged["config"], merged["dataset_split"], strict=True):
        try:
            expected_scaffold, expected_model = group_for_relative(str(relative))
        except KeyError:
            group_mismatch += 1
            scaffolds.append("")
            models.append("")
            continue
        if str(config) != expected_scaffold or str(dataset_split) != expected_model:
            group_mismatch += 1
        scaffolds.append(expected_scaffold)
        models.append(expected_model)
    if group_mismatch:
        problems.append(f"group mapping mismatches: {group_mismatch}")
    if problems:
        return None, detail

    splits_ok = True
    for repo, split_name in zip(merged["repo"].astype(str), merged["split"].astype(str), strict=True):
        recorded = repo_to_split.get(repo)
        if recorded != split_name:
            splits_ok = False
            break
    if not splits_ok:
        problems.append("a repository split differs from the saved repo_to_split map")
    if merged.groupby("repo")["split"].nunique().max() != 1:
        problems.append("a repository crosses splits")
    if merged.groupby("instance_id")["split"].nunique().max() != 1:
        problems.append("an issue crosses splits")

    kinds = [outcome_kind(value) for value in merged["resolved"].tolist()]
    invalid_outcomes = sum(kind == "invalid" for kind in kinds)
    if invalid_outcomes:
        problems.append(f"resolved has {invalid_outcomes} values outside -1, 0, 1")
    resolved_counts = Counter(int(value) for value, kind in zip(merged["resolved"].tolist(), kinds, strict=True) if kind != "invalid")
    split_counts = Counter(str(value) for value in merged["split"].tolist())
    observed = {
        "trajectories": int(len(merged)),
        "resolved_1": int(resolved_counts[1]),
        "resolved_0": int(resolved_counts[0]),
        "resolved_minus_1": int(resolved_counts[-1]),
        "repositories": int(merged["repo"].nunique()),
        "issues": int(merged["instance_id"].nunique()),
        "train_rows": int(split_counts["train"]),
        "validation_rows": int(split_counts["validation"]),
        "test_rows": int(split_counts["test"]),
    }
    detail["observed"] = observed
    for key, value in expected.items():
        if observed.get(key) != value:
            problems.append(f"{key} is {observed.get(key)}, expected {value}")

    parser_flags: list[int | None] = []
    bad_flags = 0
    for value in merged["k5_prefix_explicit_test_failure_any"].tolist():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            parser_flags.append(None)
            continue
        flag = _binary_flag(value)
        if flag is None:
            bad_flags += 1
            parser_flags.append(None)
        else:
            parser_flags.append(flag)
    if bad_flags:
        problems.append(f"parser flag has {bad_flags} values outside 0, 1, and null")
    rounds: list[int | None] = []
    bad_rounds = 0
    for value in merged["n_complete_cycles"].tolist():
        number = _integral(value)
        if number is None or number < 0:
            bad_rounds += 1
            rounds.append(None)
        else:
            rounds.append(number)
    if bad_rounds:
        problems.append(f"n_complete_cycles has {bad_rounds} non-integral values")
    if problems:
        detail["parser_flag_nulls"] = sum(flag is None for flag in parser_flags)
        return None, detail

    meta = pd.DataFrame(
        {
            "trajectory_id": merged["trajectory_id"].astype(str).tolist(),
            "issue_id": merged["instance_id"].astype(str).tolist(),
            "instance_id": merged["instance_id"].astype(str).tolist(),
            "repo": merged["repo"].astype(str).tolist(),
            "scaffold": scaffolds,
            "model": models,
            "resolved": [int(value) for value in merged["resolved"].tolist()],
            "split": merged["split"].astype(str).tolist(),
            "n_rounds_total": pd.array(rounds, dtype="Int64"),
            "parser_v2_strict_fail_r5": pd.array(parser_flags, dtype="Int64"),
            "shard": merged["shard"].astype(str).tolist(),
        }
    )
    detail["parser_flag_nulls"] = int(meta["parser_v2_strict_fail_r5"].isna().sum())
    detail["parser_flag_positives"] = int((meta["parser_v2_strict_fail_r5"] == 1).sum())
    detail["rows"] = int(len(meta))
    return meta, detail


def build_calls(config: str, parsed: Any, records: list[Any] | None) -> list[dict[str, Any]]:
    """Adapter calls from one already-parsed trajectory.

    The loop matches ``traj_adapter.iter_calls``. Tests keep the two aligned.
    """
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


def _blank_measure(flag_keys: list[str]) -> dict[str, Any]:
    row = {key: None for key in flag_keys}
    row.update(
        {
            "parser_status": None,
            "census_parse_ok": False,
            "prefix5_status": None,
            "prefix15_status": None,
            "n_rounds_observed": None,
            "n_malformed_messages": None,
        }
    )
    for checkpoint in CHECKPOINTS:
        row[f"exec_calls_r{checkpoint}"] = None
        row[f"exec_missing_r{checkpoint}"] = None
        row[f"v3_test_fail_obs_r{checkpoint}"] = None
        for name in V2_SIGNALS:
            row[f"v2_{name}_r{checkpoint}"] = None
    return row


def _prefix_status(parsed: Any, calls: list[dict[str, Any]], checkpoint: int) -> str:
    status = prefix_index_comparison(parsed, calls, checkpoint)["status"]
    if status == "no_five_round_boundary" and checkpoint != 5:
        return "no_prefix_boundary"
    return str(status)


def _v2_counts(parsed: Any, checkpoint: int) -> dict[str, int]:
    counts = {name: 0 for name in V2_SIGNALS}
    for message in frozen_prefix_observations(parsed, checkpoint):
        signals = classify_execution_v2(message.content)
        for name in V2_SIGNALS:
            if getattr(signals, name):
                counts[name] += 1
    return counts


def measure_trajectory(trajectory: Any, config: str) -> tuple[dict[str, Any], Counter]:
    """Census one trajectory. The signature intentionally has no ``resolved``."""
    flag_keys = list(census_flags([]).keys())
    categories: Counter = Counter()
    try:
        records, _kind = coerce_trajectory(trajectory)
        parsed = parse_trajectory(records if records is not None else trajectory)
        calls = build_calls(config, parsed, records)
    except Exception as exc:
        row = _blank_measure(flag_keys)
        row["parser_status"] = f"exception:{type(exc).__name__}"
        return row, categories
    if parsed.status not in USABLE_PARSE_STATUSES:
        row = _blank_measure(flag_keys)
        row["parser_status"] = parsed.status
        row["n_malformed_messages"] = int(parsed.n_malformed_messages)
        return row, categories
    row = census_flags(calls)
    row["parser_status"] = parsed.status
    row["census_parse_ok"] = True
    row["prefix5_status"] = _prefix_status(parsed, calls, 5)
    row["prefix15_status"] = _prefix_status(parsed, calls, 15)
    row["n_rounds_observed"] = int(parsed.n_complete_cycles)
    row["n_malformed_messages"] = int(parsed.n_malformed_messages)
    for call in calls:
        categories[argument_category(call.get("tool"), call.get("input") or "")] += 1
    for checkpoint in CHECKPOINTS:
        window = [call for call in calls if int(call["round"]) <= checkpoint]
        execute = [call for call in window if tool_family(call.get("tool")) == "execute"]
        row[f"exec_calls_r{checkpoint}"] = len(execute)
        row[f"exec_missing_r{checkpoint}"] = sum(call.get("command") is None for call in execute)
        fail_obs = 0
        for call in window:
            hit = detect_test_command(call.get("command"))
            if hit is not None and hit.confidence == "high" and parse_test_output(call.get("output")).test_failed:
                fail_obs += 1
        row[f"v3_test_fail_obs_r{checkpoint}"] = fail_obs
        for name, count in _v2_counts(parsed, checkpoint).items():
            row[f"v2_{name}_r{checkpoint}"] = count
    return row, categories


def columns_to_read(schema_names: list[str] | tuple[str, ...]) -> list[str]:
    present = set(schema_names)
    missing = [name for name in CENSUS_COLUMNS if name not in present]
    if missing:
        raise ValueError(f"required columns missing: {missing}")
    if any(name in CENSUS_COLUMNS for name in PROHIBITED_TOP_LEVEL):
        raise ValueError("prohibited column was requested")
    return list(CENSUS_COLUMNS)


def checkpoint_reusable(payload: dict[str, Any], *, code_token_value: str, params_token_value: str, input_sha256: str) -> bool:
    return (
        payload.get("status") == "verified"
        and payload.get("code_token") == code_token_value
        and payload.get("params_token") == params_token_value
        and payload.get("input_sha256") == input_sha256
    )


def _integer_frame(frame: pd.DataFrame) -> pd.DataFrame:
    skip = {
        "trajectory_id",
        "instance_id",
        "repo",
        "parser_status",
        "prefix5_status",
        "prefix15_status",
        "high_runners_r5",
        "high_runners_r15",
        "shard",
    }
    out = frame.copy()
    for column in out.columns:
        if column in skip or column == "census_parse_ok":
            continue
        if out[column].dtype == object or str(out[column].dtype).startswith("int") or str(out[column].dtype).startswith("float"):
            numeric = pd.to_numeric(out[column], errors="coerce")
            if numeric.notna().any() or out[column].isna().any():
                out[column] = numeric.astype("Int64")
    return out


def process_shard(
    *,
    parquet_path: Path,
    relative_path: str,
    scaffold: str,
    output_parquet: Path,
    output_errors: Path,
    output_checkpoint: Path,
    code_token_value: str,
    params_token_value: str,
    input_sha256: str,
    input_bytes: int,
    batch_size: int = 25,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Stream one shard. A verified checkpoint with the same code and bytes is reused."""
    parquet_path = Path(parquet_path)
    if not parquet_path.is_file() or parquet_path.stat().st_size != int(input_bytes):
        raise RuntimeError(f"shard bytes changed before processing: {relative_path}")
    actual_sha = sha256_file(parquet_path)
    if actual_sha != input_sha256:
        raise RuntimeError(f"shard sha256 changed before processing: {relative_path}")
    if output_checkpoint.is_file() and output_parquet.is_file() and output_errors.is_file():
        payload = json.loads(output_checkpoint.read_text(encoding="utf-8"))
        if (
            checkpoint_reusable(
                payload,
                code_token_value=code_token_value,
                params_token_value=params_token_value,
                input_sha256=actual_sha,
            )
            and payload.get("output_sha256") == sha256_file(output_parquet)
            and payload.get("errors_sha256") == sha256_file(output_errors)
        ):
            payload["reused"] = True
            return payload

    reader = pq.ParquetFile(parquet_path)
    chosen = columns_to_read(reader.schema_arrow.names)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    categories: Counter = Counter()
    seen: set[str] = set()

    def _note(message: str) -> None:
        print(message, flush=True)
        if log_path is not None:
            with Path(log_path).open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    _note(f"SHARD_START {relative_path} rows={reader.metadata.num_rows}")
    for batch in reader.iter_batches(batch_size=batch_size, columns=chosen):
        for record in batch.to_pylist():
            trajectory_id = str(record.get("trajectory_id"))
            if trajectory_id in seen:
                raise RuntimeError(f"duplicate trajectory id inside {relative_path}: {trajectory_id}")
            seen.add(trajectory_id)
            try:
                measured, counted = measure_trajectory(record.get("trajectory"), scaffold)
            except Exception as exc:
                measured = _blank_measure(list(census_flags([]).keys()))
                measured["parser_status"] = f"exception:{type(exc).__name__}"
                counted = Counter()
                errors.append(
                    {
                        "trajectory_id": trajectory_id,
                        "shard": relative_path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if measured.get("census_parse_ok") is not True and not any(item["trajectory_id"] == trajectory_id for item in errors):
                errors.append(
                    {
                        "trajectory_id": trajectory_id,
                        "shard": relative_path,
                        "error_type": "parse_status",
                        "error": str(measured.get("parser_status")),
                    }
                )
            categories.update(counted)
            measured["trajectory_id"] = trajectory_id
            measured["instance_id"] = None if record.get("instance_id") is None else str(record.get("instance_id"))
            measured["repo"] = None if record.get("repo") is None else str(record.get("repo"))
            measured["shard"] = relative_path
            rows.append(measured)
            if len(rows) % 500 == 0:
                _note(f"SHARD_PROGRESS {relative_path} rows={len(rows)}")
        del batch
    if len(rows) != reader.metadata.num_rows:
        raise RuntimeError(f"row accounting failed for {relative_path}: {len(rows)} != {reader.metadata.num_rows}")
    frame = _integer_frame(pd.DataFrame(rows))
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    partial = output_parquet.with_suffix(".parquet.partial")
    frame.to_parquet(partial, index=False)
    partial.replace(output_parquet)
    error_partial = output_errors.with_suffix(".jsonl.partial")
    error_partial.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors),
        encoding="utf-8",
    )
    error_partial.replace(output_errors)
    payload = {
        "status": "verified",
        "reused": False,
        "relative_path": relative_path,
        "input_sha256": actual_sha,
        "input_bytes": int(input_bytes),
        "code_token": code_token_value,
        "params_token": params_token_value,
        "rows": len(rows),
        "errors": len(errors),
        "output_sha256": sha256_file(output_parquet),
        "errors_sha256": sha256_file(output_errors),
        "command_categories": dict(categories),
        "duplicate_output_rows": int(frame["trajectory_id"].duplicated().sum()),
    }
    if payload["duplicate_output_rows"]:
        raise RuntimeError(f"duplicate ids written for {relative_path}")
    output_checkpoint.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def concat_unique(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    if not frames:
        return pd.DataFrame(), ["no shard outputs"]
    combined = pd.concat(frames, ignore_index=True)
    problems = []
    duplicate = int(combined["trajectory_id"].astype(str).duplicated().sum())
    if duplicate:
        problems.append(f"merged census has {duplicate} duplicate trajectory ids")
    return combined, problems


def metric_summary(series: pd.Series, *, count: bool) -> dict[str, Any]:
    total = int(len(series))
    valid = series.notna()
    n_valid = int(valid.sum())
    n_error = total - n_valid
    if count:
        observed = series.loc[valid]
        return {
            "n_total": total,
            "n_valid": n_valid,
            "n_error": n_error,
            "positive_count": int(observed.sum()) if n_valid else 0,
            "rate": None,
        }
    positives = int((series.loc[valid] == 1).sum())
    return {
        "n_total": total,
        "n_valid": n_valid,
        "n_error": n_error,
        "positive_count": positives,
        "rate": None if n_valid == 0 else positives / n_valid,
    }


def is_true_flag(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    try:
        missing = pd.isna(value)
    except TypeError:
        return False
    if missing is True:
        return False
    try:
        return int(value) == 1 and float(value) == 1.0
    except (TypeError, ValueError):
        return False


def high_confidence_buckets(frame: pd.DataFrame, checkpoint: int) -> dict[str, int]:
    """Separate a high-confidence hit, a known non-hit, and a missing command."""
    ran = pd.to_numeric(frame[f"v3_ran_tests_r{checkpoint}"], errors="coerce")
    missing = pd.to_numeric(frame[f"exec_missing_r{checkpoint}"], errors="coerce")
    ok = frame["census_parse_ok"].map(is_true_flag).astype(bool)
    high = ok & ran.eq(1)
    known = ok & ran.eq(0) & missing.eq(0)
    absent_command = ok & ran.eq(0) & missing.gt(0)
    return {
        "high_confidence_trajectories": int(high.sum()),
        "known_no_high_confidence_command": int(known.sum()),
        "no_high_confidence_with_missing_command": int(absent_command.sum()),
        "parse_error": int((~ok).sum()),
    }


def issue_support(frame: pd.DataFrame) -> dict[str, Any]:
    """Describe within-issue variation. This does not estimate an association."""
    overall = frame.groupby("issue_id", dropna=False).size()
    valid = frame.loc[frame["census_parse_ok"] == True].copy()  # noqa: E712
    valid = valid.loc[valid["v3_ran_tests_r5"].notna() & valid["exec_missing_r5"].notna()]
    behavior = valid.groupby("issue_id", dropna=False)["v3_ran_tests_r5"].agg(["min", "max", "count"])
    behavior_var_ids = set(behavior.index[(behavior["min"] == 0) & (behavior["max"] == 1)])
    strict = valid.loc[(valid["v3_ran_tests_r5"] == 1) | ((valid["v3_ran_tests_r5"] == 0) & (valid["exec_missing_r5"] == 0))]
    strict_span = strict.groupby("issue_id")["v3_ran_tests_r5"].agg(["min", "max"])
    strict_ids = set(strict_span.index[(strict_span["min"] == 0) & (strict_span["max"] == 1)])
    known = frame.loc[frame["resolved"].map(outcome_kind) == "known"]
    outcomes = known.groupby("issue_id")["resolved"].agg(["min", "max"])
    outcome_ids = set(outcomes.index[(outcomes["min"] == 0) & (outcomes["max"] == 1)])
    grouped = valid.groupby(["issue_id", "scaffold", "model"])["v3_ran_tests_r5"].agg(["min", "max"])
    within_group = grouped[(grouped["min"] == 0) & (grouped["max"] == 1)].reset_index()
    within_group_issues = set(within_group["issue_id"]) if len(within_group) else set()
    combo_counts = (
        within_group.groupby("issue_id").size().value_counts().sort_index().to_dict() if len(within_group) else {}
    )
    return {
        "issues": int(overall.shape[0]),
        "trajectories": int(len(frame)),
        "issues_with_one_trajectory": int((overall == 1).sum()),
        "issues_with_two_or_more": int((overall >= 2).sum()),
        "min_trajectories_per_issue": int(overall.min()) if len(overall) else 0,
        "max_trajectories_per_issue": int(overall.max()) if len(overall) else 0,
        "issues_with_r5_test_flag_variation": len(behavior_var_ids),
        "issues_with_strict_run_versus_known_nonrun": len(strict_ids),
        "issues_with_outcome_variation": len(outcome_ids),
        "issues_behavior_only": len(behavior_var_ids - outcome_ids),
        "issues_outcome_only": len(outcome_ids - behavior_var_ids),
        "issues_behavior_and_outcome": len(behavior_var_ids & outcome_ids),
        "issues_strict_behavior_and_outcome": len(strict_ids & outcome_ids),
        "issues_with_within_group_test_variation": len(within_group_issues),
        "within_group_variation_issue_counts_by_n_groups": {str(key): int(value) for key, value in combo_counts.items()},
    }


def compare_v33(frame: pd.DataFrame) -> dict[str, Any]:
    subset = frame.loc[frame["shard"].isin(V33_SHARD_PATHS)].copy()
    differences: list[dict[str, Any]] = []
    observed_totals = {
        "n_total": int(len(subset)),
        "n_valid": int((subset["census_parse_ok"] == True).sum()) if len(subset) else 0,  # noqa: E712
        "n_error": int((subset["census_parse_ok"] != True).sum()) if len(subset) else 0,  # noqa: E712
        "prefix_mismatch_r5": int((subset["prefix5_status"] == "mismatch").sum()) if len(subset) else 0,
    }
    for name in V2_SIGNALS:
        column = f"v2_{name}_r5"
        observed_totals[name] = int(subset[column].dropna().sum()) if len(subset) else 0
    for key, expected in V33_EXPECTED_TOTALS.items():
        if observed_totals.get(key) != expected:
            differences.append({"scope": "total", "metric": key, "expected": expected, "observed": observed_totals.get(key)})
    by_shard = []
    for relative, expected in V33_EXPECTED_SHARDS.items():
        group = subset.loc[subset["shard"] == relative]
        observed = {"rows": int(len(group))}
        for name in (
            "explicit_test_failure",
            "explicit_test_pass",
            "build_or_compile_failure",
            "package_level_failure_unresolved",
        ):
            observed[name] = int(group[f"v2_{name}_r5"].dropna().sum()) if len(group) else 0
        by_shard.append({"shard": relative, "expected": expected, "observed": observed})
        for key, value in expected.items():
            if observed[key] != value:
                differences.append({"scope": relative, "metric": key, "expected": value, "observed": observed[key]})
    return {"pass": not differences, "differences": differences, "observed_totals": observed_totals, "shards": by_shard}


def decide_status(*, provenance_ok: bool, metadata_ok: bool, census_finished: bool, problems: list[str]) -> str:
    if not provenance_ok or not metadata_ok:
        return STATUS_BLOCKED
    if census_finished and not problems:
        return STATUS_COMPLETE
    return STATUS_PARTIAL


def safe_shard_name(relative: str) -> str:
    return str(relative).replace("\\", "/").replace("/", "__")
