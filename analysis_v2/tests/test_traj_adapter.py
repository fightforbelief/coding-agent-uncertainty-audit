"""Adapter checks against parser v2. Real shards are optional."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from p8audit.trajectory_parser import parse_trajectory
from traj_adapter import iter_calls

ROOT = Path(__file__).resolve().parents[2]


def _call(name: str, arguments: str) -> dict:
    return {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


def _assistant(content: str, calls: list[dict]) -> dict:
    return {"role": "assistant", "content": content, "reasoning_content": "hidden", "think": False, "tool_calls": calls}


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content, "reasoning_content": "", "think": False, "tool_calls": []}


def _row(config: str, trajectory: list[dict]) -> dict:
    return {"config": config, "trajectory": trajectory}


def _parsed_calls(trajectory: list[dict]) -> list[tuple[str | None, str]]:
    parsed = parse_trajectory(trajectory)
    found = []
    by_index = {message.index: message for message in parsed.messages}
    for cycle in parsed.cycles:
        action = by_index[cycle.action_index]
        for call in action.tool_calls:
            found.append((call.name, call.arguments))
    return found


def test_openhands_rounds_and_command_field() -> None:
    trajectory = [
        {"role": "user", "content": "issue", "tool_calls": []},
        _assistant("run", [_call("execute_bash", json.dumps({"command": "pytest -q"}))]),
        _tool("FAILED tests/test_a.py::test_one\n"),
        _assistant("edit", [_call("str_replace_editor", json.dumps({"path": "a.py", "old_str": "x", "new_str": "y"}))]),
        _tool("edited"),
    ]
    calls = iter_calls(_row("openhands", trajectory))
    parsed = parse_trajectory(trajectory)
    assert max(call["round"] for call in calls) == parsed.n_complete_cycles == 2
    assert [(call["tool"], call["command"]) for call in calls] == [
        ("execute_bash", "pytest -q"),
        ("str_replace_editor", None),
    ]
    assert calls[0]["output"].startswith("FAILED")
    assert "hidden" not in calls[0]["output"]
    assert calls[0]["input"] == json.dumps({"command": "pytest -q"})


def test_sweagent_plain_command_and_non_shell_tools() -> None:
    trajectory = [
        _assistant("look", [_call("bash", "open src/app.py")]),
        _tool("file contents"),
        _assistant("edit", [_call("str_replace_editor", json.dumps({"command": "str_replace", "path": "a.py"}))]),
        _tool("done"),
        _assistant("send", [_call("submit", "{}")]),
        _tool("submitted"),
    ]
    calls = iter_calls(_row("sweagent", trajectory))
    assert [call["command"] for call in calls] == ["open src/app.py", None, None]
    assert [call["round"] for call in calls] == [1, 2, 3]
    assert parse_trajectory(trajectory).n_complete_cycles == 3


def test_every_parser_tool_call_is_present_for_both_scaffolds() -> None:
    samples = {
        "openhands": [
            _assistant("a", [_call("execute_bash", json.dumps({"command": "ls"})), _call("think", "{}")]),
            _tool("listing"),
            _tool("thought"),
            _assistant("b", [_call("finish", "{}")]),
            _tool("done"),
        ],
        "sweagent": [
            _assistant("a", [_call("bash", json.dumps({"command": "pytest tests"}))]),
            _tool("===== 1 failed in 0.1s"),
            _assistant("b", [_call("bash", "create reproduce.py"), _call("submit", "{}")]),
            _tool("created"),
        ],
    }
    for config, trajectory in samples.items():
        adapted = [(call["tool"], call["input"]) for call in iter_calls(_row(config, trajectory))]
        assert _parsed_calls(trajectory) == adapted


def test_incomplete_cycle_uses_the_next_completed_round() -> None:
    trajectory = [
        _assistant("one", [_call("bash", "pytest -q")]),
        _tool("===== 1 passed in 0.1s"),
        _assistant("gap", [_call("bash", "pytest tests/test_gap.py")]),
        _assistant("two", [_call("execute_bash", json.dumps({"command": "pytest tests"}))]),
        _tool("FAILED tests/test_a.py::test_one\n"),
        _assistant("tail", [_call("bash", "pytest tail")]),
    ]
    calls = iter_calls(_row("sweagent", trajectory))
    parsed = parse_trajectory(trajectory)
    assert parsed.n_complete_cycles == 2
    assert [call["round"] for call in calls] == [1, 2, 2, 3]
    for k in (1, 2):
        end = parsed.prefix_message_count(k)
        assert end is not None
        inside = [call for call in calls if call["round"] <= k]
        assert inside
        assert all(call["round"] <= k for call in inside)
        assert len(inside) == sum(1 for call in calls if call["round"] <= k)


def test_count_mismatch_still_keeps_every_call() -> None:
    trajectory = [
        _assistant("two calls", [_call("bash", "pytest a"), _call("bash", "pytest b")]),
        _tool("only one observation"),
    ]
    calls = iter_calls(_row("sweagent", trajectory))
    assert len(calls) == 2
    assert calls[0]["output"] == "only one observation"
    assert calls[1]["output"] is None
    assert parse_trajectory(trajectory).n_complete_cycles == 0
    assert [call["round"] for call in calls] == [1, 1]


def _message(role: str, content: str, calls: list[dict] | None = None, reasoning: str = "") -> dict:
    return {
        "role": role,
        "content": content,
        "reasoning_content": reasoning,
        "think": False,
        "tool_calls": calls or [],
    }


def _bash(arguments: str, name: str = "bash") -> dict:
    return {"id": "call", "type": "function", "function": {"name": name, "arguments": arguments}}


def assert_round_matches_prefix(trajectory: list[dict], config: str) -> None:
    parsed = parse_trajectory(trajectory)
    calls = iter_calls(_row(config, trajectory))
    for call in calls:
        output = call["output"] or ""
        assert "ASSISTANT_REASON" not in output
        assert "TOOL_REASON" not in output
        assert "USER_REASON" not in output
        assert "SYS_REASON" not in output
    assert parsed.n_complete_cycles >= 1
    for k in range(1, parsed.n_complete_cycles + 1):
        end = parsed.prefix_message_count(k)
        assert end is not None
        selected = [call for call in calls if call["round"] <= k]
        inside = [call for call in calls if call["action_index"] < end]
        assert selected == inside
    for k in (5, 15):
        if parsed.n_complete_cycles < k:
            continue
        end = parsed.prefix_message_count(k)
        assert end is not None
        assert [call for call in calls if call["round"] <= k] == [call for call in calls if call["action_index"] < end]


def _completed(index: int, output: str = "ok", calls: list[dict] | None = None) -> list[dict]:
    tool_calls = calls or [_bash(f"echo {index}")]
    messages = [_message("assistant", "run", tool_calls, reasoning="ASSISTANT_REASON")]
    for _call in tool_calls:
        messages.append(_message("tool", output, reasoning="TOOL_REASON"))
    return messages


def test_command_extraction_for_both_scaffolds() -> None:
    cases = [
        ("pytest -q", "pytest -q"),
        (json.dumps({"command": "pytest -q"}), "pytest -q"),
        (json.dumps({"cmd": "go test ./..."}), "go test ./..."),
        (json.dumps({"command": "pytest -q", "cmd": "go test"}), "pytest -q"),
        (json.dumps({"path": "a.py"}), None),
        ("{not json", "{not json"),
        ("", None),
        ("   ", None),
    ]
    for config in ("openhands", "sweagent"):
        tool_name = "execute_bash" if config == "openhands" else "bash"
        for arguments, expected in cases:
            trajectory = [
                _message("assistant", "run", [_bash(arguments, tool_name)], reasoning="ASSISTANT_REASON"),
                _message("tool", "ok", reasoning="TOOL_REASON"),
            ]
            calls = iter_calls(_row(config, trajectory))
            assert calls[0]["command"] == expected
            assert calls[0]["input"] == arguments
        editor = [
            _message(
                "assistant",
                "edit",
                [_bash(json.dumps({"command": "pytest -q"}), "str_replace_editor")],
                reasoning="ASSISTANT_REASON",
            ),
            _message("tool", "edited", reasoning="TOOL_REASON"),
        ]
        assert iter_calls(_row(config, editor))[0]["command"] is None


def test_prefix_membership_covers_incomplete_cycles() -> None:
    def trajectory_for(pattern: str) -> list[dict]:
        messages = [
            _message("system", "sys", reasoning="SYS_REASON"),
            _message("user", "issue", reasoning="USER_REASON"),
        ]
        if pattern == "before_first":
            messages.extend(_completed(0)[:1])
        elif pattern == "several_before_first":
            messages.extend(_completed(0)[:1])
            messages.extend(_completed(1)[:1])
        completed = []
        for index in range(15):
            if pattern == "between" and index == 1:
                messages.extend(_completed(90)[:1])
            if pattern == "after_fifth" and index == 5:
                messages.extend(_completed(91)[:1])
            if pattern == "multi" and index == 2:
                messages.extend(
                    _completed(
                        index,
                        calls=[_bash("pytest -q"), _bash("echo side")],
                    )
                )
                continue
            if pattern == "mismatch" and index == 3:
                messages.append(_message("assistant", "run", [_bash("pytest a"), _bash("pytest b")], reasoning="ASSISTANT_REASON"))
                messages.append(_message("tool", "only-one", reasoning="TOOL_REASON"))
                continue
            if pattern == "missing" and index == 4:
                messages.append(_message("assistant", "run", [_bash("pytest missing")], reasoning="ASSISTANT_REASON"))
                messages.append(_message("assistant", "note", [], reasoning="ASSISTANT_REASON"))
                continue
            messages.extend(_completed(index))
            completed.append(index)
        if pattern == "trailing":
            messages.extend(_completed(99)[:1])
        return messages

    for pattern in (
        "before_first",
        "several_before_first",
        "between",
        "after_fifth",
        "trailing",
        "multi",
        "mismatch",
        "missing",
    ):
        assert_round_matches_prefix(trajectory_for(pattern), "openhands")
        assert_round_matches_prefix(trajectory_for(pattern), "sweagent")


def test_real_shards_when_data_is_mounted() -> None:
    data_root = os.environ.get("P8_DATA_ROOT")
    if not data_root:
        pytest.skip("P8_DATA_ROOT is not configured for the real-shard check")
    root = Path(data_root)
    matched = [
        path
        for path in (root.rglob("*.parquet") if root.is_dir() else [])
        if any(token in path.as_posix() for token in ("openhands", "sweagent"))
    ]
    if len(matched) < 4:
        pytest.skip("pinned trajectory shards are not on this machine")
    groups = {
        "openhands": ["minimax_m25_openhands", "qwen35_openhands"],
        "sweagent": ["minimax_m25_sweagent", "qwen35_sweagent"],
    }
    import pyarrow.parquet as pq

    seen = {config: 0 for config in groups}
    for path in sorted(root.rglob("*.parquet")):
        config = next((name for name, tokens in groups.items() if any(token in path.as_posix() for token in tokens)), None)
        if config is None or seen[config] >= 3:
            continue
        table = next(pq.ParquetFile(path).iter_batches(batch_size=1, columns=["trajectory"]))
        trajectory = table.column("trajectory")[0].as_py()
        calls = iter_calls({"config": config, "trajectory": trajectory})
        parsed = parse_trajectory(trajectory)
        complete_rounds = {call["round"] for call in calls if call["round"] <= parsed.n_complete_cycles}
        if parsed.n_complete_cycles:
            assert max(complete_rounds) == parsed.n_complete_cycles
        assert len(calls) == len(_parsed_calls(trajectory if isinstance(trajectory, list) else json.loads(trajectory)))
        seen[config] += 1
    assert all(count == 3 for count in seen.values())


def test_three_real_trajectories_per_scaffold_model_group() -> None:
    """Original Prompt 0 requires 3 examples in EACH of the four groups."""
    value = os.environ.get("P8_DATA_ROOT")
    if not value:
        pytest.skip("P8_DATA_ROOT is not configured for the real-trajectory acceptance check")
    root = Path(value)
    import pyarrow.parquet as pq

    for config, token in (
        ("openhands", "minimax_m25_openhands"),
        ("openhands", "qwen35_openhands"),
        ("sweagent", "minimax_m25_sweagent"),
        ("sweagent", "qwen35_sweagent"),
    ):
        paths = sorted(path for path in root.rglob("*.parquet") if token in path.as_posix())
        assert paths, f"Missing pinned group: {token}"
        table = next(pq.ParquetFile(paths[0]).iter_batches(batch_size=3, columns=["trajectory"]))
        assert table.num_rows == 3
        for item in table.column("trajectory"):
            trajectory = item.as_py()
            records = trajectory if isinstance(trajectory, list) else json.loads(trajectory)
            assert_round_matches_prefix(records, config)
            assert len(iter_calls({"config": config, "trajectory": records})) == len(_parsed_calls(records))
