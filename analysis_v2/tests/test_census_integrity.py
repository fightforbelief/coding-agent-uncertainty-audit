import json

import pandas as pd
import pytest

from census_integrity import (
    BLOCKED,
    compare_trajectory_ids,
    metric_line,
    parser_v2_failure_trigger,
    render_report,
    require_data_root,
    stratified_sample,
    v3_failure_trigger,
    verify_dataset_revision,
)
from p8audit.schema import PINNED_REVISION


def test_empty_data_root_is_rejected_before_scanning(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="must not be empty"):
        require_data_root("", repository_root=tmp_path)
    with pytest.raises(SystemExit, match="must not be empty"):
        require_data_root(None, repository_root=tmp_path)


def test_repository_root_is_not_a_data_root(tmp_path) -> None:
    with pytest.raises(SystemExit, match="repository root"):
        require_data_root(str(tmp_path), repository_root=tmp_path)


def test_missing_manifest_blocks_revision(tmp_path) -> None:
    with pytest.raises(SystemExit, match=BLOCKED):
        verify_dataset_revision(tmp_path / "pinned_parquet_files.json", tmp_path)


def test_manifest_without_revision_is_not_proof(tmp_path) -> None:
    manifest = tmp_path / "pinned_parquet_files.json"
    manifest.write_text(json.dumps({"files": [{"path": "data/minimax_m25_openhands_trajectories/a.parquet", "bytes": 1}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match=BLOCKED):
        verify_dataset_revision(manifest, tmp_path)


def test_revision_alone_does_not_accept_a_short_inventory(tmp_path) -> None:
    manifest = tmp_path / "pinned_parquet_files.json"
    relative = "data/minimax_m25_openhands_trajectories/a.parquet"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abc")
    manifest.write_text(
        json.dumps({"revision": PINNED_REVISION, "files": [{"path": relative, "bytes": 3}]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match=BLOCKED):
        verify_dataset_revision(manifest, tmp_path)


def test_parse_errors_are_not_zeros() -> None:
    frame = pd.DataFrame(
        [
            {
                "scaffold": "openhands",
                "model": "minimax_m25",
                "resolved": 0,
                "parser_v2_strict_fail_r5": 0,
                "v3_test_fail_r5": None,
                "v3_n_test_runs_r5": None,
                "v3_ran_tests_r5": None,
                "v3_ran_tests_amb_r5": None,
                "v3_collect_build_fail_r5": None,
                "v3_anyobs_fail_r5": None,
                "loose_r5": None,
                "v3_test_fail_r15": None,
                "v3_n_test_runs_r15": None,
                "v3_ran_tests_r15": None,
                "v3_ran_tests_amb_r15": None,
                "v3_collect_build_fail_r15": None,
                "v3_anyobs_fail_r15": None,
                "loose_r15": None,
                "high_runners_r5": None,
                "high_unknown_outputs_r5": None,
                "high_runners_r15": None,
                "high_unknown_outputs_r15": None,
            },
            {
                "scaffold": "openhands",
                "model": "minimax_m25",
                "resolved": 2,
                "parser_v2_strict_fail_r5": 1,
                "v3_test_fail_r5": 1,
                "v3_n_test_runs_r5": 4,
                "v3_ran_tests_r5": 1,
                "v3_ran_tests_amb_r5": 0,
                "v3_collect_build_fail_r5": 0,
                "v3_anyobs_fail_r5": 1,
                "loose_r5": 1,
                "v3_test_fail_r15": 1,
                "v3_n_test_runs_r15": 4,
                "v3_ran_tests_r15": 1,
                "v3_ran_tests_amb_r15": 0,
                "v3_collect_build_fail_r15": 0,
                "v3_anyobs_fail_r15": 1,
                "loose_r15": 1,
                "high_runners_r5": "pytest",
                "high_unknown_outputs_r5": 0,
                "high_runners_r15": "pytest",
                "high_unknown_outputs_r15": 0,
            },
        ]
    )
    summary = metric_line(frame, "v3_n_test_runs_r5", count=True)
    assert summary["n_total"] == 2
    assert summary["n_valid"] == 1
    assert summary["n_error"] == 1
    assert summary["positive_or_sum"] == 4
    binary = metric_line(frame, "v3_test_fail_r5", count=False)
    assert binary["positive_or_sum"] == 1
    assert binary["rate"] == 1
    text, status = render_report(frame, error_count=1)
    assert status == "PARTIAL"
    assert "not full-corpus" in text
    assert "known-label" in text
    assert "| v3_n_test_runs_r5 | 0 |" not in text


def test_id_mismatch_is_explicit() -> None:
    meta = pd.DataFrame({"trajectory_id": ["a", "a", "b"]})
    census = pd.DataFrame({"trajectory_id": ["b", "c"]})
    comparison = compare_trajectory_ids(meta, census)
    assert comparison["meta_duplicate_ids"] == 1
    assert comparison["missing_from_census"] == ["a"]
    assert comparison["extra_in_census"] == ["c"]


def test_sample_is_stratified_and_deterministic() -> None:
    rows = []
    for scaffold, model, count in (
        ("openhands", "minimax_m25", 5),
        ("openhands", "qwen35_122b", 1),
        ("sweagent", "minimax_m25", 4),
        ("sweagent", "qwen35_122b", 0),
    ):
        for index in range(count):
            rows.append({"scaffold": scaffold, "model": model, "trajectory_id": f"{scaffold}-{model}-{index}"})
    frame = pd.DataFrame(rows)
    first, strata = stratified_sample(frame, 6, 20261102)
    second, _ = stratified_sample(frame, 6, 20261102)
    assert first["trajectory_id"].tolist() == second["trajectory_id"].tolist()
    counts = {(item["scaffold"], item["model"]): item for item in strata}
    assert counts[("openhands", "qwen35_122b")]["population"] == 1
    assert counts[("openhands", "qwen35_122b")]["sampled"] == 1
    assert sum(item["sampled"] for item in strata) == 6
    assert len(first) == 6


def test_triggers_use_the_responsible_observation() -> None:
    unrelated = "hello from a directory listing"
    parser_only = "FAILED tests/a.py\n"
    later = "FAILED tests/z.py::test_late\n===== 1 failed in 0.2s"
    trajectory = [
        {"role": "system", "content": "sys", "reasoning_content": "SYS_REASON", "think": False, "tool_calls": []},
        {"role": "user", "content": "issue", "reasoning_content": "USER_REASON", "think": False, "tool_calls": []},
    ]
    def add(name, arguments, output):
        trajectory.append(
            {
                "role": "assistant",
                "content": "run",
                "reasoning_content": "ASSISTANT_REASON",
                "think": False,
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": name, "arguments": arguments}}],
            }
        )
        trajectory.append(
            {"role": "tool", "content": output, "reasoning_content": "TOOL_REASON", "think": False, "tool_calls": []}
        )
    add("execute_bash", json.dumps({"command": "ls"}), unrelated)
    add("execute_bash", json.dumps({"command": "pytest -q"}), parser_only)
    for index in range(3):
        add("execute_bash", json.dumps({"command": f"echo {index}"}), "ok")
    add("execute_bash", json.dumps({"command": "pytest -q"}), later)
    trigger = parser_v2_failure_trigger(trajectory, "openhands")
    assert trigger["status"] == "identified"
    assert trigger["output"] == parser_only
    assert later not in trigger["output"]
    assert unrelated not in trigger["output"]
    assert trigger["command"] == "pytest -q"
    v3 = v3_failure_trigger(
        [
            {"round": 1, "tool": "bash", "command": "ls", "output": unrelated, "observation_index": 1},
            {"round": 4, "tool": "bash", "command": "pytest -q", "output": later, "observation_index": 4},
        ]
    )
    assert v3["status"] == "identified"
    assert v3["output"] == later
    assert v3_failure_trigger([{"round": 1, "tool": "bash", "command": "ls", "output": unrelated}])["status"] == "unresolved"
