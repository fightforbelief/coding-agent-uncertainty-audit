"""Trajectory-level outcomes with repository dependence.

The analysis unit is the trajectory. This module does not vote labels down
to one issue outcome. Bootstrap resampling draws repositories and keeps every
trajectory that belongs to a drawn repository.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def repository_cluster_bootstrap_indices(
    repos: list[str],
    rng: np.random.Generator,
    n_replicates: int,
) -> list[list[int]]:
    """Resample repositories with replacement.

    Each replicate lists row indexes. Rows from one repository stay together.
    A repository drawn twice contributes its rows twice. Rows are not drawn
    independently.
    """
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, repo in enumerate(repos):
        groups[repo].append(index)
    names = np.array(sorted(groups), dtype=object)
    if len(names) == 0:
        return [[] for _ in range(n_replicates)]
    replicates: list[list[int]] = []
    for _ in range(n_replicates):
        drawn = rng.choice(names, size=len(names), replace=True)
        indexes: list[int] = []
        for name in drawn.tolist():
            indexes.extend(groups[str(name)])
        replicates.append(indexes)
    return replicates


def issue_equal_trajectory_weights(instance_ids: list[str]) -> list[float]:
    """Weight each trajectory by ``1 / n_j`` inside one analysis set.

    ``n_j`` is the number of trajectories of issue ``j`` in ``instance_ids``.
    Pass only the rows that enter that sensitivity analysis, such as the
    known-label trajectories in one repository-held-out partition. Do not pass
    trajectories from another partition, and do not pass ``resolved == -1``
    rows that were excluded from the metric. The weights do not replace the
    unweighted trajectory-level primary estimand and do not change labels.
    """
    counts = Counter(instance_ids)
    if any(count <= 0 for count in counts.values()):
        raise ValueError("instance counts must be positive")
    return [1.0 / counts[instance_id] for instance_id in instance_ids]


def paired_contrast(augmented: list[float], baseline: list[float]) -> list[float]:
    """Difference within each already paired replicate.

    Both series must be computed on the same repository-cluster indexes from
    one call to ``repository_cluster_bootstrap_indices``. This function does
    not compute AUPRC and does not draw a second sample.
    """
    if len(augmented) != len(baseline):
        raise ValueError("paired replicates must use the same resample")
    if not augmented:
        raise ValueError("at least one replicate is required")
    return [float(left) - float(right) for left, right in zip(augmented, baseline, strict=True)]


def trajectories_per_issue(instance_ids: list[str]) -> dict[str, int]:
    return dict(Counter(instance_ids))
