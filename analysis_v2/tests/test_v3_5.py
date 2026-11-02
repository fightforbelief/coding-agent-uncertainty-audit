"""V3.5 sampling and unlabeled evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v3_5_design import (
    FORBIDDEN_REVIEWER_COLUMNS,
    REVIEWER_LABEL_COLUMNS,
    SEED,
    assign_review_ids,
    blank_review_queue,
    design_sample,
    sealed_records,
)
from v3_5_evidence import extract_prefix_evidence, forbidden_keys
from v3_5_run import write_reviewer_zip

ROOT = Path(__file__).resolve().parents[2]


def _call(name: str, arguments: str) -> dict:
    return {"id": "call", "type": "function", "function": {"name": name, "arguments": arguments}}


def _assistant(content: str, calls: list[dict]) -> dict:
    return {"role": "assistant", "content": content, "reasoning_content": "hidden-reasoning", "think": False, "tool_calls": calls}


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content, "reasoning_content": "hidden-reasoning", "think": False, "tool_calls": []}


def _trajectory(rounds: list[tuple[int, str]]) -> list[dict]:
    messages = [{"role": "user", "content": "issue", "reasoning_content": "hidden-reasoning", "think": False, "tool_calls": []}]
    for index, (_round, command) in enumerate(rounds, start=1):
        messages.append(_assistant("run", [_call("execute_bash", json.dumps({"command": command}))]))
        messages.append(_tool(f"output-{index}\n"))
    return messages


def _population() -> pd.DataFrame:
    rows = []
    groups = (
        ("openhands", "minimax_m25", 60, 40, 30, 3, 10, 8),
        ("openhands", "qwen35_122b", 10, 40, 30, 2, 6, 8),
        ("sweagent", "minimax_m25", 80, 40, 30, 0, 10, 8),
        ("sweagent", "qwen35_122b", 2, 40, 30, 1, 1, 8),
    )
    number = 0
    for scaffold, model, n_a, n_b, n_c, n_d, n_amb5, n_amb15 in groups:
        for _ in range(n_a):
            number += 1
            rows.append(_row(number, scaffold, model, ran5=1, ran15=1, miss5=0, miss15=0, amb5=0, amb15=0, resolved=number % 2))
        for index in range(n_b):
            number += 1
            rows.append(_row(number, scaffold, model, ran5=0, ran15=1, miss5=0, miss15=0, amb5=int(index < n_amb5), amb15=0, resolved=-1))
        for index in range(n_c):
            number += 1
            rows.append(_row(number, scaffold, model, ran5=0, ran15=0, miss5=0, miss15=0, amb5=0, amb15=int(index < n_amb15), resolved=0))
        for _ in range(n_d):
            number += 1
            rows.append(_row(number, scaffold, model, ran5=0, ran15=0, miss5=1, miss15=2, amb5=0, amb15=0, resolved=1))
    return pd.DataFrame(rows)


def _row(number, scaffold, model, ran5, ran15, miss5, miss15, amb5, amb15, resolved) -> dict:
    return {
        "trajectory_id": f"t{number:04d}",
        "issue_id": f"issue-{number % 7}",
        "scaffold": scaffold,
        "model": model,
        "shard": f"data/{model}_{scaffold}/train.parquet",
        "census_parse_ok": True,
        "v3_ran_tests_r5": ran5,
        "v3_ran_tests_r15": ran15,
        "exec_missing_r5": miss5,
        "exec_missing_r15": miss15,
        "v3_ran_tests_amb_r5": amb5,
        "v3_ran_tests_amb_r15": amb15,
        "resolved": resolved,
        "reference_patch": "patch",
        "reasoning_content": "hidden",
    }


def test_sampling_is_reproducible_and_covers_four_groups() -> None:
    frame = _population()
    first = design_sample(frame)
    second = design_sample(frame)
    assert first["review_ids"] == second["review_ids"]
    left = set(zip(first["selected"]["stratum"], first["selected"]["trajectory_id"], strict=True))
    right = set(zip(second["selected"]["stratum"], second["selected"]["trajectory_id"], strict=True))
    assert left == right
    assert set(first["summary"]["scaffold"] + "/" + first["summary"]["model"]) == {
        "openhands/minimax_m25",
        "openhands/qwen35_122b",
        "sweagent/minimax_m25",
        "sweagent/qwen35_122b",
    }
    assert int(first["summary"]["sampling_seed"].iloc[0]) == SEED


def test_r5_positive_sample_caps_at_50_and_keeps_small_groups() -> None:
    selected = design_sample(_population())["selected"]
    part = selected.loc[selected["stratum"] == "A_r5_positive"]
    counts = part.groupby(["scaffold", "model"]).size()
    assert int(counts.loc[("openhands", "minimax_m25")]) == 50
    assert int(counts.loc[("sweagent", "minimax_m25")]) == 50
    assert int(counts.loc[("openhands", "qwen35_122b")]) == 10
    assert int(counts.loc[("sweagent", "qwen35_122b")]) == 2
    small = part.loc[(part["scaffold"] == "sweagent") & (part["model"] == "qwen35_122b")]
    assert float(small["inclusion_probability"].iloc[0]) == 1


def test_r15_positive_requires_known_r5_negative() -> None:
    frame = _population()
    selected = design_sample(frame)["selected"]
    chosen = set(selected.loc[selected["stratum"] == "B_r15_positive_r5_known_negative", "trajectory_id"])
    source = frame.set_index("trajectory_id")
    assert chosen
    assert set(source.loc[list(chosen), "v3_ran_tests_r15"]) == {1}
    assert set(source.loc[list(chosen), "v3_ran_tests_r5"]) == {0}
    assert set(source.loc[list(chosen), "exec_missing_r5"]) == {0}
    counts = selected.loc[selected["stratum"] == "B_r15_positive_r5_known_negative"].groupby(["scaffold", "model"]).size()
    assert int(counts.max()) == 40


def test_known_negative_excludes_missing_commands_and_missing_is_complete() -> None:
    frame = _population()
    result = design_sample(frame)
    selected = result["selected"]
    known = set(selected.loc[selected["stratum"] == "C_r15_known_negative", "trajectory_id"])
    missing = set(selected.loc[selected["stratum"] == "D_r15_missing_command", "trajectory_id"])
    source = frame.set_index("trajectory_id")
    assert set(source.loc[list(known), "exec_missing_r15"]) == {0}
    assert set(source.loc[list(known), "v3_ran_tests_r15"]) == {0}
    population_missing = set(frame.loc[frame["exec_missing_r15"].gt(0) & frame["v3_ran_tests_r15"].eq(0), "trajectory_id"])
    assert missing == population_missing
    assert known.isdisjoint(missing)
    assert result["masks_overlap"]["C_and_D"] == 0


def test_ambiguous_sample_stays_marked_diagnostic() -> None:
    result = design_sample(_population())
    summary = result["summary"]
    ambiguous = summary.loc[summary["stratum"].isin(["AMB_r5", "AMB_r15"])]
    assert set(ambiguous["role"]) == {"diagnostic"}
    assert int(ambiguous.loc[ambiguous["stratum"] == "AMB_r5", "sample_size"].max()) <= 25
    selected = result["selected"]
    both = set(selected.loc[selected["stratum"] == "A_r5_positive", "trajectory_id"]) & set(
        selected.loc[selected["stratum"] == "AMB_r5", "trajectory_id"]
    )
    assert both == set()


def test_dedup_preserves_each_inclusion_probability() -> None:
    frame = _population()
    result = design_sample(frame)
    attributes = frame.drop(columns=["resolved", "reference_patch", "reasoning_content"])
    sealed = sealed_records(result["selected"], attributes, result["review_ids"])
    assert sealed["trajectory_id"].is_unique
    assert sealed["review_id"].is_unique
    overlap = sealed.loc[(sealed["member_C_r15_known_negative"] == 1) & (sealed["member_AMB_r15"] == 1)]
    if len(overlap):
        assert overlap["pi_C_r15_known_negative"].notna().all()
        assert overlap["pi_AMB_r15"].notna().all()
    assert len(sealed) == len(result["review_ids"])


def test_outcome_labels_do_not_change_the_sample() -> None:
    frame = _population()
    flipped = frame.copy()
    flipped["resolved"] = 1 - frame["resolved"].clip(lower=0)
    flipped["reference_patch"] = "other"
    first = set(design_sample(frame)["review_ids"])
    second = set(design_sample(flipped)["review_ids"])
    assert first == second


def test_prefix_windows_message_order_and_short_trajectories() -> None:
    commands = ["pwd"] * 16
    commands[1] = "pytest -q"
    commands[5] = "npm test"
    commands[15] = "cargo test"
    evidence = extract_prefix_evidence(_trajectory(list(enumerate(commands, start=1))), "openhands")
    assert [call["round"] for call in evidence["r5"]] == [1, 2, 3, 4, 5]
    assert [call["round"] for call in evidence["r15"]] == list(range(1, 16))
    assert [call["order"] for call in evidence["r5"]] == [0, 1, 2, 3, 4]
    assert evidence["r5"][0]["observation_index"] == 2
    assert evidence["r5"][1]["observation_index"] == 4
    assert evidence["r5"][1]["extracted_command"] == "pytest -q"
    assert evidence["r5"][1]["arguments"].startswith("{")
    assert evidence["r15"][5]["extracted_command"] == "npm test"
    assert all(call["round"] <= 15 for call in evidence["r15"])
    short = extract_prefix_evidence(_trajectory([(1, "pwd"), (2, "ls"), (3, "pytest -q"), (4, "pwd")]), "openhands")
    assert short["completed_rounds"] == 4
    assert short["prefix15_boundary"] == "no_prefix_boundary"
    assert [call["round"] for call in short["r15"]] == [1, 2, 3, 4]
    assert [call["round"] for call in short["r5"]] == [1, 2, 3, 4]
    assert short["prefix_parse_status"] == "ok"


def test_evidence_and_queue_do_not_carry_labels_or_reasoning() -> None:
    trajectory = _trajectory([(1, "pytest -q")])
    evidence = extract_prefix_evidence(trajectory, "openhands")
    assert forbidden_keys(evidence) == set()
    assert "hidden-reasoning" not in json.dumps(evidence)
    assert "reference_patch" not in json.dumps(evidence)
    sealed = pd.DataFrame(
        [
            {
                "review_id": "V35-000001",
                "completed_rounds": evidence["completed_rounds"],
                "prefix5_boundary": evidence["prefix5_boundary"],
                "prefix15_boundary": evidence["prefix15_boundary"],
                "prefix_parse_status": evidence["prefix_parse_status"],
            }
        ]
    )
    queue = blank_review_queue(sealed)
    assert FORBIDDEN_REVIEWER_COLUMNS.isdisjoint(queue.columns)
    assert set(queue[list(REVIEWER_LABEL_COLUMNS)].iloc[0]) == {""}


def test_reviewer_zip_excludes_the_sealed_key(tmp_path: Path) -> None:
    queue = pd.DataFrame([{"review_id": "V35-000001", "test_command_status_r5": ""}])
    evidence = tmp_path / "review_evidence.jsonl"
    evidence.write_text('{"review_id":"V35-000001","r5":[]}\n', encoding="utf-8")
    instructions = tmp_path / "validation_instructions.md"
    rules = tmp_path / "adjudication_rules.md"
    instructions.write_text("instructions\n", encoding="utf-8")
    rules.write_text("rules\n", encoding="utf-8")
    sealed = tmp_path / "sealed_sampling_key.csv"
    sealed.write_text("review_id,trajectory_id\nV35-000001,secret\n", encoding="utf-8")
    archive = write_reviewer_zip(
        tmp_path / "validation_bundle.zip",
        {
            "review_queue.csv": queue,
            "review_evidence.jsonl": evidence,
            "validation_instructions.md": instructions,
            "adjudication_rules.md": rules,
        },
    )
    import zipfile

    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        assert "sealed_sampling_key.csv" not in names
        assert handle.testzip() is None
        assert b"secret" not in handle.read("review_queue.csv")
        assert b"trajectory_id" not in handle.read("review_evidence.jsonl")


def test_runner_source_has_no_model_fit() -> None:
    text = (ROOT / "analysis_v2" / "v3_5_design.py").read_text(encoding="utf-8")
    text += (ROOT / "analysis_v2" / "v3_5_evidence.py").read_text(encoding="utf-8")
    text += (ROOT / "analysis_v2" / "v3_5_run.py").read_text(encoding="utf-8")
    for token in ("LogisticRegression", "mixedlm", "average_precision", "roc_auc", "Clopper", "fillna(0)"):
        assert token not in text
