"""Fresh-shard draw for parser v2 validation.

The draw uses sorted paths only. It does not read outcomes, marker counts,
or feature performance.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

FRESH_SHARD_SEED = 20261090
FRESH_SHARDS_PER_GROUP = 2
OBSERVATION_SEED = 20261091
PRECISION_SAMPLE_SIZE = 50
DIAGNOSTIC_SAMPLE_SIZE = 12
RANDOM_SAMPLE_SIZE = 24

# Gate 0 first shards. Trajectory bodies were read then.
GATE0_FIRST_SHARDS = (
    "data/minimax_m25_openhands_trajectories/train-00000-of-00020.parquet",
    "data/qwen35_openhands_trajectories/train-00000-of-00023.parquet",
    "data/minimax_m25_sweagent_trajectories/train-00000-of-00023.parquet",
    "data/qwen35_sweagent_trajectories/train-00000-of-00018.parquet",
)

# Gate 0.75 supplemental draw, seed 20261075. Recomputed before the fresh draw.
GATE075_SUPPLEMENTAL_SHARDS = (
    "data/minimax_m25_openhands_trajectories/train-00001-of-00020.parquet",
    "data/minimax_m25_openhands_trajectories/train-00012-of-00020.parquet",
    "data/qwen35_openhands_trajectories/train-00003-of-00023.parquet",
    "data/qwen35_openhands_trajectories/train-00007-of-00023.parquet",
    "data/minimax_m25_sweagent_trajectories/train-00015-of-00023.parquet",
    "data/minimax_m25_sweagent_trajectories/train-00018-of-00023.parquet",
    "data/qwen35_sweagent_trajectories/train-00011-of-00018.parquet",
    "data/qwen35_sweagent_trajectories/train-00012-of-00018.parquet",
)


def previously_read_shard_paths() -> set[str]:
    return set(GATE0_FIRST_SHARDS) | set(GATE075_SUPPLEMENTAL_SHARDS)


def select_fresh_shards(
    records: list[dict],
    *,
    seed: int = FRESH_SHARD_SEED,
    per_group: int = FRESH_SHARDS_PER_GROUP,
    excluded_paths: set[str] | None = None,
) -> list[dict]:
    """Two shards from each configuration, excluding shards already read."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    excluded = set(excluded_paths) if excluded_paths is not None else previously_read_shard_paths()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        path = record["path"]
        if path in excluded:
            continue
        grouped[(record["config"], record["dataset_split"])].append(record)
    expected = {
        ("openhands", "minimax_m25"),
        ("openhands", "qwen35_122b"),
        ("sweagent", "minimax_m25"),
        ("sweagent", "qwen35_122b"),
    }
    if set(grouped) != expected:
        raise ValueError(f"expected four groups, found {sorted(grouped)}")
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    method = (
        f"numpy.random.default_rng({seed}).choice without replacement; "
        f"{per_group} paths per (config, dataset_split); sorted eligible paths; "
        "excluded the four Gate 0 first shards and the eight Gate 0.75 "
        "supplemental shards; outcomes, marker counts, and feature performance "
        "were not used"
    )
    for key in sorted(grouped):
        pool = sorted(grouped[key], key=lambda item: item["path"])
        if len(pool) < per_group:
            raise ValueError(f"{key} has {len(pool)} eligible shards, need {per_group}")
        picks = rng.choice(len(pool), size=per_group, replace=False)
        for index in sorted(int(value) for value in picks.tolist()):
            chosen = dict(pool[index])
            chosen["selection_method"] = method
            chosen["selection_seed"] = seed
            selected.append(chosen)
    return selected
