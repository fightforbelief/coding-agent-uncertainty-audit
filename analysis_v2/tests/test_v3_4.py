"""V3.4 census rules. These tests do not fit a model or read the full corpus."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from census_integrity import parser_v2_failure_trigger, v3_failure_trigger
from p8audit.schema import PINNED_REVISION
from traj_adapter import iter_calls
from v3_4_lib import (
    EXPECTED_MANIFEST_SHA256,
    assemble_metadata,
    build_calls,
    checkpoint_reusable,
    concat_unique,
    decide_status,
    high_confidence_buckets,
    locate_shard,
    manifest_problems,
    measure_trajectory,
    metric_summary,
    outcome_kind,
    process_shard,
    sha256_file,
)
from v3_4_report import summary_rows

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"


def _call(name: str, arguments: str) -> dict:
    return {"id": f"call-{name}-{arguments[:12]}", "type": "function", "function": {"name": name, "arguments": arguments}}


def _assistant(content: str, calls: list[dict]) -> dict:
    return {"role": "assistant", "content": content, "reasoning_content": "hidden", "think": False, "tool_calls": calls}


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content, "reasoning_content": "hidden", "think": False, "tool_calls": []}


def _rounds(n: int, test_rounds: set[int], fail_rounds: set[int] | None = None) -> list[dict]:
    fail_rounds = fail_rounds or set()
    trajectory = [{"role": "user", "content": "issue", "reasoning_content": "hidden", "think": False, "tool_calls": []}]
    for index in range(1, n + 1):
        command = "pytest -q" if index in test_rounds else "pwd"
        output = "FAILED tests/test_a.py::test_one\n" if index in fail_rounds else "ok\n"
        trajectory.append(_assistant("run", [_call("execute_bash", json.dumps({"command": command}))]))
        trajectory.append(_tool(output))
    return trajectory


def test_parse_failure_stays_missing() -> None:
    measured, _categories = measure_trajectory("{not json", "openhands")
    assert measured["census_parse_ok"] is False
    assert measured["v3_ran_tests_r5"] is None
    assert measured["v3_n_test_runs_r5"] is None
    assert measured["v2_explicit_test_failure_r5"] is None
    partial = [{"content": "no role"}, _assistant("run", [_call("execute_bash", json.dumps({"command": "pytest -q"}))]), _tool("FAILED tests/test_a.py::test_one\n")]
    failed, _categories = measure_trajectory(partial, "openhands")
    assert failed["census_parse_ok"] is False
    assert failed["v3_ran_tests_r5"] is None


def test_rate_uses_valid_rows_only() -> None:
    summary = metric_summary(pd.Series([1, None, 0]), count=False)
    assert summary == {"n_total": 3, "n_valid": 2, "n_error": 1, "positive_count": 1, "rate": 0.5}
    counted = metric_summary(pd.Series([2, None, 3]), count=True)
    assert counted["positive_count"] == 5
    assert counted["rate"] is None
    assert counted["n_error"] == 1


def test_known_label_is_zero_or_one() -> None:
    assert outcome_kind(0) == "known"
    assert outcome_kind(1) == "known"
    assert outcome_kind(-1) == "unknown"
    assert outcome_kind(2) == "invalid"
    assert outcome_kind(None) == "invalid"
    assert outcome_kind(True) == "invalid"


def test_metadata_join_and_null_parser_flag() -> None:
    shard = pd.DataFrame(
        [
            {"trajectory_id": "a", "instance_id": "issue-a", "repo": "org/a", "shard": "data/minimax_m25_openhands_trajectories/train.parquet"},
            {"trajectory_id": "b", "instance_id": "issue-b", "repo": "org/b", "shard": "data/qwen35_sweagent_trajectories/train.parquet"},
        ]
    )
    labels = shard.assign(config=["openhands", "sweagent"], resolved=[1, -1], split=["train", "test"])
    features = shard.assign(
        config=["openhands", "sweagent"],
        dataset_split=["minimax_m25", "qwen35_122b"],
        n_complete_cycles=[5, 6],
        k5_prefix_explicit_test_failure_any=[None, 1],
    )
    expected = {
        "trajectories": 2,
        "resolved_1": 1,
        "resolved_0": 0,
        "resolved_minus_1": 1,
        "repositories": 2,
        "issues": 2,
        "train_rows": 1,
        "validation_rows": 0,
        "test_rows": 1,
    }
    meta, detail = assemble_metadata(shard, labels, features, {"org/a": "train", "org/b": "test"}, expected)
    assert detail["problems"] == []
    assert meta is not None
    assert meta["issue_id"].tolist() == ["issue-a", "issue-b"]
    assert pd.isna(meta.loc[0, "parser_v2_strict_fail_r5"])
    assert int(meta.loc[1, "parser_v2_strict_fail_r5"]) == 1
    broken = labels.copy()
    broken.loc[0, "trajectory_id"] = "c"
    rejected, rejected_detail = assemble_metadata(shard, broken, features, {"org/a": "train", "org/b": "test"}, expected)
    assert rejected is None
    assert rejected_detail["problems"]


def test_pinned_manifest_hash_groups_and_byte_check(tmp_path: Path) -> None:
    digest = sha256_file(MANIFEST)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert digest == EXPECTED_MANIFEST_SHA256
    assert manifest_problems(payload, digest=digest) == []
    assert payload["revision"] == PINNED_REVISION
    target = tmp_path / "data" / "minimax_m25_openhands_trajectories" / "a.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abc")
    other = tmp_path / "minimax_m25_openhands_trajectories" / "a.parquet"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        locate_shard(tmp_path, "data/minimax_m25_openhands_trajectories/a.parquet")
    other.unlink()
    found = locate_shard(tmp_path, "data/minimax_m25_openhands_trajectories/a.parquet")
    assert found == target
    checked = __import__("v3_4_lib", fromlist=["verify_local_file"]).verify_local_file(target, 3, sha256_file(target))
    assert checked["pass"] is True
    assert __import__("v3_4_lib", fromlist=["verify_local_file"]).verify_local_file(target, 4, sha256_file(target))["pass"] is False


def test_prefix_boundaries_and_command_extraction() -> None:
    trajectory = _rounds(16, {5, 15, 16}, {15})
    measured, categories = measure_trajectory(trajectory, "openhands")
    assert measured["census_parse_ok"] is True
    assert measured["v3_ran_tests_r5"] == 1
    assert measured["v3_test_fail_r5"] == 0
    assert measured["v3_ran_tests_r15"] == 1
    assert measured["v3_test_fail_r15"] == 1
    assert measured["v3_n_test_runs_r5"] == 1
    assert measured["v3_n_test_runs_r15"] == 2
    assert measured["first_test_round"] == 5
    assert categories["json_command"] == 16
    missing = [
        {"role": "user", "content": "issue", "tool_calls": []},
        _assistant("run", [_call("execute_bash", json.dumps({"timeout": 1}))]),
        _tool("ok"),
    ]
    missing_row, missing_categories = measure_trajectory(missing, "openhands")
    assert missing_row["v3_ran_tests_r5"] == 0
    assert missing_row["exec_missing_r5"] == 1
    assert missing_categories["json_neither"] == 1
    buckets = high_confidence_buckets(pd.DataFrame([measured, missing_row]), 5)
    assert buckets["high_confidence_trajectories"] == 1
    assert buckets["no_high_confidence_with_missing_command"] == 1


def test_adapter_calls_match_iter_calls() -> None:
    samples = [
        _rounds(2, {1}, {1}),
        [
            _assistant("look", [_call("bash", "pytest tests")]),
            _tool("1 failed"),
            _assistant("edit", [_call("str_replace_editor", json.dumps({"command": "str_replace"}))]),
            _tool("done"),
        ],
    ]
    for trajectory in samples:
        from p8audit.trajectory_parser import coerce_trajectory, parse_trajectory

        records, _kind = coerce_trajectory(trajectory)
        parsed = parse_trajectory(trajectory)
        assert build_calls("openhands" if trajectory is samples[0] else "sweagent", parsed, records) == iter_calls(
            {"scaffold": "openhands" if trajectory is samples[0] else "sweagent", "trajectory": trajectory}
        )


def test_disagreement_uses_the_triggering_observation() -> None:
    trajectory = [
        {"role": "user", "content": "issue", "tool_calls": []},
        _assistant("ls", [_call("execute_bash", json.dumps({"command": "ls"}))]),
        _tool("hello"),
        _assistant("test", [_call("execute_bash", json.dumps({"command": "pytest -q"}))]),
        _tool("FAILED tests/test_a.py::test_one\n"),
    ]
    calls = iter_calls({"scaffold": "openhands", "trajectory": trajectory})
    v3_trigger = v3_failure_trigger(calls)
    parser_trigger = parser_v2_failure_trigger(trajectory, "openhands")
    assert v3_trigger["status"] == "identified"
    assert v3_trigger["observation_index"] == 4
    assert v3_trigger["command"] == "pytest -q"
    assert parser_trigger["status"] == "identified"
    assert parser_trigger["observation_index"] == 4
    assert "pytest_failed_line" in parser_trigger["reasons"]
    unresolved = parser_v2_failure_trigger([
        {"role": "user", "content": "issue", "tool_calls": []},
        _assistant("ls", [_call("execute_bash", json.dumps({"command": "ls"}))]),
        _tool("hello"),
    ], "openhands")
    assert unresolved["status"] == "unresolved"


def test_detector_does_not_accept_resolved() -> None:
    assert "resolved" not in inspect.signature(measure_trajectory).parameters


def test_shard_resume_and_duplicate_merge(tmp_path: Path) -> None:
    trajectory = _rounds(1, set(), set())
    source = tmp_path / "shard.parquet"
    pd.DataFrame(
        [{"trajectory_id": "t1", "instance_id": "issue-1", "repo": "org/repo", "trajectory": trajectory}]
    ).to_parquet(source, index=False)
    digest = sha256_file(source)
    output = tmp_path / "out.parquet"
    errors = tmp_path / "out.jsonl"
    checkpoint = tmp_path / "out.json"
    first = process_shard(
        parquet_path=source,
        relative_path="data/minimax_m25_openhands_trajectories/a.parquet",
        scaffold="openhands",
        output_parquet=output,
        output_errors=errors,
        output_checkpoint=checkpoint,
        code_token_value="token-a",
        params_token_value="params-a",
        input_sha256=digest,
        input_bytes=source.stat().st_size,
    )
    assert first["reused"] is False
    assert first["rows"] == 1
    second = process_shard(
        parquet_path=source,
        relative_path="data/minimax_m25_openhands_trajectories/a.parquet",
        scaffold="openhands",
        output_parquet=output,
        output_errors=errors,
        output_checkpoint=checkpoint,
        code_token_value="token-a",
        params_token_value="params-a",
        input_sha256=digest,
        input_bytes=source.stat().st_size,
    )
    assert second["reused"] is True
    stored = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_reusable(stored, code_token_value="token-b", params_token_value="params-a", input_sha256=digest) is False
    third = process_shard(
        parquet_path=source,
        relative_path="data/minimax_m25_openhands_trajectories/a.parquet",
        scaffold="openhands",
        output_parquet=output,
        output_errors=errors,
        output_checkpoint=checkpoint,
        code_token_value="token-b",
        params_token_value="params-a",
        input_sha256=digest,
        input_bytes=source.stat().st_size,
    )
    assert third["reused"] is False
    assert len(pd.read_parquet(output)) == 1
    duplicated, problems = concat_unique(
        [pd.DataFrame({"trajectory_id": ["t1"]}), pd.DataFrame({"trajectory_id": ["t1"]})]
    )
    assert problems
    assert len(duplicated) == 2


def test_v33_metric_names_match_measurement() -> None:
    trajectory = _rounds(1, {1}, {1})
    measured, _categories = measure_trajectory(trajectory, "openhands")
    assert measured["v2_explicit_test_failure_r5"] == 1
    assert measured["v3_test_fail_r5"] == 1
    assert measured["v3_test_fail_obs_r5"] == 1
    rows = summary_rows(
        pd.DataFrame(
            [
                {
                    **measured,
                    "scaffold": "openhands",
                    "model": "minimax_m25",
                    "resolved": 0,
                    "census_parse_ok": True,
                }
            ]
        )
    )
    found = [row for row in rows if row["group"] == "all" and row["metric"] == "v3_test_fail_r5"]
    assert found[0]["positive_count"] == 1
    assert found[0]["n_error"] == 0


def test_status_gates() -> None:
    assert decide_status(provenance_ok=False, metadata_ok=True, census_finished=False, problems=[]) .endswith("PROVENANCE OR METADATA FAILURE")
    assert "PARTIAL" in decide_status(provenance_ok=True, metadata_ok=True, census_finished=True, problems=["parse errors: 1"])
    assert decide_status(provenance_ok=True, metadata_ok=True, census_finished=True, problems=[]).endswith("DESCRIPTIVE RESULTS ONLY")


def test_runner_source_does_not_fit_models() -> None:
    text = (ROOT / "analysis_v2" / "v3_4_run.py").read_text(encoding="utf-8")
    library = (ROOT / "analysis_v2" / "v3_4_lib.py").read_text(encoding="utf-8")
    for source in (text, library):
        assert "00_reproduce" not in source
        assert "average_precision" not in source
        assert "LogisticRegression" not in source
        assert "fillna(0)" not in source
