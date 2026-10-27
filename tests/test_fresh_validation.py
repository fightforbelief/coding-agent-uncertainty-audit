"""Fresh shard draw and unlabeled validation package."""

from pathlib import Path

import pytest

from p8audit.fresh_shards import (
    GATE075_SUPPLEMENTAL_SHARDS,
    GATE0_FIRST_SHARDS,
    select_fresh_shards,
)
from p8audit.sampling import select_supplemental_shards
from p8audit.validation_package import write_validation_package


def _records() -> list[dict]:
    rows = []
    groups = {
        ("openhands", "minimax_m25", "data/minimax_m25_openhands_trajectories", 20),
        ("openhands", "qwen35_122b", "data/qwen35_openhands_trajectories", 23),
        ("sweagent", "minimax_m25", "data/minimax_m25_sweagent_trajectories", 23),
        ("sweagent", "qwen35_122b", "data/qwen35_sweagent_trajectories", 18),
    }
    for config, split, prefix, count in groups:
        width = 5
        for index in range(count):
            rows.append(
                {
                    "config": config,
                    "dataset_split": split,
                    "path": f"{prefix}/train-{index:0{width}d}-of-{count:0{width}d}.parquet",
                    "bytes": 1000 + index,
                }
            )
    return rows


def test_fresh_draw_excludes_previously_read_shards_and_is_reproducible() -> None:
    records = _records()
    supplemental = select_supplemental_shards(
        records,
        seed=20261075,
        per_group=2,
        excluded_paths=set(GATE0_FIRST_SHARDS),
    )
    assert [item["path"] for item in supplemental] == list(GATE075_SUPPLEMENTAL_SHARDS)
    first = select_fresh_shards(records)
    second = select_fresh_shards(records)
    assert [item["path"] for item in first] == [item["path"] for item in second]
    chosen = {item["path"] for item in first}
    assert chosen.isdisjoint(GATE0_FIRST_SHARDS)
    assert chosen.isdisjoint(GATE075_SUPPLEMENTAL_SHARDS)
    assert len(chosen) == 8
    by_group: dict[tuple[str, str], int] = {}
    for item in first:
        key = (item["config"], item["dataset_split"])
        by_group[key] = by_group.get(key, 0) + 1
        assert "resolved" not in item
        assert "outcomes" in item["selection_method"]
    assert set(by_group.values()) == {2}


def test_package_rejects_outcome_fields_and_prefilled_labels(tmp_path: Path) -> None:
    clean = {
        "sample_id": "g09-00001",
        "signal_family": "explicit_exception",
        "reviewer_label": "",
        "student_verified": "false",
        "context_excerpt": "Traceback (most recent call last):",
    }
    write_validation_package(
        tmp_path / "pkg",
        {"explicit_exception_validation.csv": [clean]},
        {"rows": 1},
    )
    text = (tmp_path / "pkg" / "explicit_exception_validation.csv").read_text(encoding="utf-8-sig")
    assert "resolved" not in text.splitlines()[0]
    assert "g09-00001" in text
    dirty = dict(clean)
    dirty["resolved"] = "0"
    with pytest.raises(ValueError, match="forbidden"):
        write_validation_package(tmp_path / "bad", {"explicit_exception_validation.csv": [dirty]}, {})
    labeled = dict(clean)
    labeled["reviewer_label"] = "true_positive"
    with pytest.raises(ValueError, match="reviewer_label"):
        write_validation_package(tmp_path / "labeled", {"explicit_exception_validation.csv": [labeled]}, {})
    with pytest.raises(ValueError, match="refusing"):
        write_validation_package(tmp_path / "marker_review_queue_v2.csv", {"a.csv": [clean]}, {})
