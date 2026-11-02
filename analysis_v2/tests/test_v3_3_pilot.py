"""Pilot sampling and measurement. No real shard is opened."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v3_3_pilot import (
    CHECKPOINT,
    EXCLUDED,
    GROUPS,
    PINNED_REVISION,
    argument_category,
    columns_to_read,
    inspect_trajectory,
    prefix_index_comparison,
    select_shards,
    verify_file,
)
from p8audit.trajectory_parser import parse_trajectory
from traj_adapter import iter_calls


def _call(name: str, arguments: str) -> dict:
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}


def _assistant(content: str, calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": "DO_NOT_USE",
        "think": True,
        "tool_calls": calls,
    }


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content, "reasoning_content": "DO_NOT_USE", "think": False, "tool_calls": []}


def _cycle(name: str, arguments: str, output: str) -> list[dict]:
    return [_assistant("run", [_call(name, arguments)]), _tool(output)]


def _manifest_files() -> list[dict]:
    files = []
    for group, scaffold, model in GROUPS:
        total = {
            "data/minimax_m25_openhands_trajectories": 20,
            "data/qwen35_openhands_trajectories": 23,
            "data/minimax_m25_sweagent_trajectories": 23,
            "data/qwen35_sweagent_trajectories": 18,
        }[group]
        width = f"{total:05d}"
        for index in range(total):
            name = f"train-{index:05d}-of-{width}.parquet"
            files.append(
                {
                    "path": f"{group}/{name}",
                    "bytes": 1000 + index,
                    "sha256": f"{index:064x}",
                    "config": scaffold,
                    "split": model,
                }
            )
    return files


def test_selection_is_two_fresh_shards_per_group_and_stable() -> None:
    files = _manifest_files()
    first = select_shards(files)
    second = select_shards(files)
    assert first == second
    assert first["revision"] == PINNED_REVISION
    assert first["seed"] == 20261103
    assert len(first["selected"]) == 8
    by_group: dict[str, list[str]] = {}
    for item in first["selected"]:
        by_group.setdefault(item["group"], []).append(Path(item["path"]).name)
        assert (item["group"], Path(item["path"]).name) not in EXCLUDED
    assert [len(names) for names in by_group.values()] == [2, 2, 2, 2]
    assert first["candidates_per_group"]["data/minimax_m25_openhands_trajectories"] == 15
    assert first["candidates_per_group"]["data/qwen35_openhands_trajectories"] == 18
    assert first["candidates_per_group"]["data/minimax_m25_sweagent_trajectories"] == 18
    assert first["candidates_per_group"]["data/qwen35_sweagent_trajectories"] == 13


def test_selection_rejects_a_missing_excluded_shard() -> None:
    files = [item for item in _manifest_files() if not item["path"].endswith("train-00000-of-00020.parquet")]
    with pytest.raises(ValueError, match="excluded shards missing"):
        select_shards(files)


def test_command_categories_follow_v311() -> None:
    assert argument_category("str_replace_editor", json.dumps({"command": "pytest -q"})) == "non_execution"
    assert argument_category("execute_bash", json.dumps({"path": "a.py"})) == "json_neither"
    assert argument_category("bash", json.dumps({"path": "a.py"})) == "json_neither"
    assert argument_category("execute_bash", json.dumps({"command": "pytest -q"})) == "json_command"
    assert argument_category("bash", json.dumps({"cmd": "go test ./..."})) == "json_cmd"
    assert argument_category("bash", "pytest -q") == "raw_string"
    assert argument_category("bash", "   ") == "empty"
    assert argument_category("bash", "{not json") == "invalid_json"


def test_five_complete_rounds_have_identical_observation_indexes() -> None:
    trajectory = []
    for index in range(6):
        output = "ok" if index else "FAILED tests/test_a.py::test_one\n"
        trajectory.extend(_cycle("bash", "pytest -q", output))
    trajectory.extend(_cycle("bash", "pytest -q", "later"))
    parsed = parse_trajectory(trajectory)
    calls = iter_calls({"scaffold": "sweagent", "trajectory": trajectory})
    comparison = prefix_index_comparison(parsed, calls, CHECKPOINT)
    assert parsed.n_complete_cycles == 7
    assert comparison["status"] == "match"
    assert comparison["round_indexes"] == comparison["prefix_indexes"]
    assert len(comparison["round_indexes"]) == 5


def test_incomplete_middle_round_stays_inside_the_same_prefix() -> None:
    trajectory = []
    trajectory.extend(_cycle("bash", "pytest -q", "ok"))
    trajectory.append(_assistant("edit", [_call("bash", "ls")]))
    trajectory.extend(_cycle("bash", "pytest -q", "ok"))
    trajectory.extend(_cycle("bash", "pytest -q", "ok"))
    trajectory.extend(_cycle("bash", "pytest -q", "ok"))
    trajectory.extend(_cycle("bash", "pytest -q", "ok"))
    parsed = parse_trajectory(trajectory)
    calls = iter_calls({"scaffold": "sweagent", "trajectory": trajectory})
    comparison = prefix_index_comparison(parsed, calls, CHECKPOINT)
    assert comparison["status"] == "match"
    assert comparison["round_indexes"] == comparison["prefix_indexes"]


def test_short_trajectory_has_no_five_round_boundary() -> None:
    trajectory = []
    trajectory.extend(_cycle("bash", "pytest -q", "ok"))
    trajectory.extend(_cycle("bash", "pytest -q", "ok"))
    parsed = parse_trajectory(trajectory)
    calls = iter_calls({"scaffold": "openhands", "trajectory": trajectory})
    comparison = prefix_index_comparison(parsed, calls, CHECKPOINT)
    assert comparison["status"] == "no_five_round_boundary"
    assert comparison["prefix_indexes"] is None


def test_missing_observation_is_not_stored_as_zero() -> None:
    trajectory = [_assistant("run", [_call("bash", "pytest -q")])]
    result = inspect_trajectory(
        scaffold="sweagent",
        model="minimax_m25",
        shard_path="data/minimax_m25_sweagent_trajectories/train-00002-of-00023.parquet",
        trajectory_id="t1",
        instance_id="i1",
        repo="example/repo",
        language="Python",
        trajectory=trajectory,
    )
    assert result["meta"]["parser_status"] == "ok"
    assert result["meta"]["n_rounds_total"] == 0
    assert result["cycle_counts"]["missing_observation"] == 1
    assert result["meta"]["prefix_explicit_test_failure_any"] == 0
    assert "resolved" not in result["meta"]


def test_failure_excerpt_uses_observation_text_not_reasoning() -> None:
    trajectory = _cycle("execute_bash", json.dumps({"command": "pytest -q"}), "FAILED tests/test_a.py::test_one\n")
    for _ in range(4):
        trajectory.extend(_cycle("execute_bash", "ls", "ok"))
    result = inspect_trajectory(
        scaffold="openhands",
        model="qwen35_122b",
        shard_path="data/qwen35_openhands_trajectories/train-00001-of-00023.parquet",
        trajectory_id="t2",
        instance_id="i2",
        repo="example/repo",
        language=None,
        trajectory=trajectory,
    )
    assert result["meta"]["prefix_explicit_test_failure_any"] == 1
    assert result["excerpts"][0]["excerpt"].startswith("FAILED tests/test_a.py")
    assert "DO_NOT_USE" not in result["excerpts"][0]["excerpt"]
    assert result["reason_counts"]["pytest_failed_line"] == 1
    assert "resolved" not in result["meta"]


def test_parse_exception_leaves_measurements_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(trajectory: object) -> object:
        raise RuntimeError("bad row")

    monkeypatch.setattr("v3_3_pilot.parse_trajectory", _boom)
    result = inspect_trajectory(
        scaffold="openhands",
        model="minimax_m25",
        shard_path="shard",
        trajectory_id="t3",
        instance_id="i3",
        repo="example/repo",
        language="Python",
        trajectory=[],
    )
    assert result["error"] is True
    assert result["meta"]["parser_status"] == "exception"
    assert result["meta"]["n_rounds_total"] is None
    assert result["meta"]["prefix_explicit_test_failure_any"] is None


def test_reader_never_requests_outcome_or_reasoning_columns() -> None:
    chosen = columns_to_read(["trajectory_id", "instance_id", "repo", "language", "trajectory", "resolved", "metadata"])
    assert chosen == ["trajectory_id", "instance_id", "repo", "language", "trajectory"]
    assert "resolved" not in chosen


def test_hash_gate_rejects_a_different_byte(tmp_path: Path) -> None:
    path = tmp_path / "shard.parquet"
    path.write_bytes(b"abc")
    passed = verify_file(path, 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    failed = verify_file(path, 3, "0" * 64)
    assert passed["pass"] is True
    assert failed["pass"] is False
