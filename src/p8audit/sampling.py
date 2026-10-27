"""Shard and observation sampling plans for the pre-freeze review.

Shard choice uses only file identity. It does not look at ``resolved`` or at
any marker count. Observation inclusion probabilities are recorded so a later
precision interval is not applied to a sample that was not designed for it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

SUPPLEMENTAL_SHARD_SEED = 20261075
SUPPLEMENTAL_SHARDS_PER_GROUP = 2
PRECISION_SAMPLE_SIZE = 50
# Families whose precision can later be compared with the pre-specified bound.
# The bound is not computed from these rows until a student reviews them.
PRECISION_FAMILIES = (
    "explicit_exception",
    "explicit_test_failure",
)
# Measured on OpenHands observations only. SWE-agent rows are not filled with zero.
OPENHANDS_ONLY_PRECISION_FAMILIES = ("tool_exit_nonzero",)


def select_supplemental_shards(
    records: list[dict],
    *,
    seed: int,
    per_group: int,
    excluded_paths: set[str],
) -> list[dict]:
    """Draw the same number of shards from each config/model group.

    ``records`` need ``path``, ``config``, and ``dataset_split``. Paths in
    ``excluded_paths`` are the already audited first shards. They are removed
    before the draw so this stage does not treat them as a fresh random sample.
    """
    if per_group < 1:
        raise ValueError("per_group must be positive")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        path = record["path"]
        if path in excluded_paths:
            continue
        grouped[(record["config"], record["dataset_split"])].append(record)
    if not grouped:
        raise ValueError("no eligible shards")
    rng = np.random.default_rng(seed)
    selected: list[dict] = []
    for key in sorted(grouped):
        pool = sorted(grouped[key], key=lambda item: item["path"])
        if len(pool) < per_group:
            raise ValueError(f"{key} has {len(pool)} eligible shards, need {per_group}")
        picks = rng.choice(len(pool), size=per_group, replace=False)
        for index in sorted(int(value) for value in picks.tolist()):
            chosen = dict(pool[index])
            chosen["selection_method"] = (
                f"numpy.random.default_rng({seed}).choice without replacement; "
                f"{per_group} paths per (config, dataset_split) after removing "
                "the Gate 0 first shards; file path order only"
            )
            selected.append(chosen)
    return selected


def inclusion_probability(sample_size: int, population_size: int) -> float:
    """Equal inclusion probability for a simple random sample without replacement.

    This is the probability required by a binomial or Clopper-Pearson description
    of that finite population. It is not valid for a language-stratified queue.
    """
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if sample_size < 0 or sample_size > population_size:
        raise ValueError("sample_size must be between 0 and population_size")
    return sample_size / population_size
