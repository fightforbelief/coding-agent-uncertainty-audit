"""Repository-grouped splits that also keep an instance in one partition."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def connected_components(
    repos: list[str],
    instance_ids: list[str],
) -> list[str]:
    """Return one component id per row.

    Rows that share a repository or an instance id are joined, including
    chains through either field. The component id is the smallest repository
    name in the component, which is stable across runs.
    """
    if len(repos) != len(instance_ids):
        raise ValueError("repos and instance_ids must have the same length")
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for repo, instance_id in zip(repos, instance_ids, strict=True):
        union(f"repo:{repo}", f"instance:{instance_id}")
    roots = []
    members: dict[str, list[str]] = defaultdict(list)
    for repo in repos:
        root = find(f"repo:{repo}")
        roots.append(root)
        members[root].append(repo)
    labels = {}
    for root, names in members.items():
        labels[root] = min(names)
    return [labels[root] for root in roots]


def assign_group_splits(
    component_ids: list[str],
    *,
    seed: int = 20261024,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, str]:
    """Assign each component to train, validation, or test.

    Fractions are applied to the sorted unique component list after a seeded
    shuffle. The test fraction is the remainder.
    """
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test share")
    unique = sorted(set(component_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    shuffled = [unique[int(index)] for index in order]
    n = len(shuffled)
    n_train = int(round(n * train_fraction))
    n_valid = int(round(n * validation_fraction))
    if n_train + n_valid >= n:
        n_valid = max(1, n - n_train - 1)
    assignment = {}
    for index, component in enumerate(shuffled):
        if index < n_train:
            assignment[component] = "train"
        elif index < n_train + n_valid:
            assignment[component] = "validation"
        else:
            assignment[component] = "test"
    return assignment


def crossing_groups(
    keys: list[str],
    split_names: list[str],
) -> dict[str, list[str]]:
    """Return keys that appear in more than one split.

    ``keys`` and ``split_names`` are aligned rows. An empty result means the
    key does not cross partitions. This check stays even when the current
    census finds no instance stored under two repositories.
    """
    if len(keys) != len(split_names):
        raise ValueError("keys and split_names must have the same length")
    seen: dict[str, set[str]] = defaultdict(set)
    for key, split_name in zip(keys, split_names, strict=True):
        seen[key].add(split_name)
    return {key: sorted(names) for key, names in seen.items() if len(names) > 1}
