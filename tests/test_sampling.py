"""Supplemental shards are chosen without looking at outcomes."""

import numpy as np

from p8audit.sampling import inclusion_probability, select_supplemental_shards


def _records() -> list[dict]:
    rows = []
    for config in ("openhands", "sweagent"):
        for dataset_split in ("minimax_m25", "qwen35_122b"):
            for index in range(5):
                rows.append(
                    {
                        "path": f"data/{config}_{dataset_split}/train-{index:05d}.parquet",
                        "config": config,
                        "dataset_split": dataset_split,
                        "resolved_rate_must_be_ignored": index / 10,
                    }
                )
    return rows


def test_selection_excludes_named_first_shards_and_is_balanced() -> None:
    records = _records()
    excluded = {
        record["path"]
        for record in records
        if record["path"].endswith("train-00000.parquet")
    }
    selected = select_supplemental_shards(
        records,
        seed=20261075,
        per_group=2,
        excluded_paths=excluded,
    )
    assert len(selected) == 8
    assert not any(item["path"] in excluded for item in selected)
    groups = {(item["config"], item["dataset_split"]) for item in selected}
    assert groups == {
        ("openhands", "minimax_m25"),
        ("openhands", "qwen35_122b"),
        ("sweagent", "minimax_m25"),
        ("sweagent", "qwen35_122b"),
    }
    assert all(sum(1 for item in selected if (item["config"], item["dataset_split"]) == key) == 2 for key in groups)
    again = select_supplemental_shards(
        records,
        seed=20261075,
        per_group=2,
        excluded_paths=excluded,
    )
    assert [item["path"] for item in again] == [item["path"] for item in selected]


def test_a_different_seed_is_allowed_to_differ() -> None:
    records = _records()
    excluded = {record["path"] for record in records if record["path"].endswith("train-00000.parquet")}
    first = [item["path"] for item in select_supplemental_shards(records, seed=1, per_group=2, excluded_paths=excluded)]
    second = [item["path"] for item in select_supplemental_shards(records, seed=2, per_group=2, excluded_paths=excluded)]
    assert first != second


def test_inclusion_probability_matches_equal_probability_sampling() -> None:
    assert inclusion_probability(50, 200) == 0.25
    assert inclusion_probability(10, 10) == 1.0


def test_numpy_choice_documents_the_generator_used_by_selection() -> None:
    # Guard against silently switching the generator implementation.
    rng = np.random.default_rng(20261075)
    picks = sorted(int(value) for value in rng.choice(19, size=2, replace=False).tolist())
    assert picks == sorted(int(value) for value in np.random.default_rng(20261075).choice(19, size=2, replace=False).tolist())
