"""Repository bootstrap keeps issue structure. Labels are not collapsed."""

import numpy as np
import pytest

from p8audit.dependence import (
    issue_equal_trajectory_weights,
    paired_contrast,
    repository_cluster_bootstrap_indices,
)
from p8audit.schema import failure_label
from p8audit.splits import assign_group_splits, connected_components, crossing_groups


def test_bootstrap_keeps_every_row_of_a_drawn_repository() -> None:
    repos = ["alpha", "alpha", "alpha", "beta"]
    replicates = repository_cluster_bootstrap_indices(
        repos,
        np.random.default_rng(20261026),
        n_replicates=30,
    )
    assert len(replicates) == 30
    for indexes in replicates:
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for index in indexes:
            counts[index] += 1
        assert counts[0] == counts[1] == counts[2]


def test_issue_weights_do_not_change_trajectory_labels() -> None:
    labels = [failure_label(0), failure_label(1), failure_label(0)]
    weights = issue_equal_trajectory_weights(["issue-1", "issue-1", "issue-2"])
    assert labels == [1, 0, 1]
    assert weights == [0.5, 0.5, 1.0]


def test_same_instance_cannot_cross_the_assigned_split() -> None:
    repos = ["alpha", "alpha", "beta", "gamma"]
    instances = ["i1", "i2", "i2", "i3"]
    components = connected_components(repos, instances)
    assignment = assign_group_splits(components, seed=20261024)
    split_names = [assignment[component] for component in components]
    assert crossing_groups(repos, split_names) == {}
    assert crossing_groups(instances, split_names) == {}
    assert split_names[1] == split_names[2]


def test_paired_contrast_subtracts_within_the_same_replicate() -> None:
    assert paired_contrast([0.40, 0.55], [0.30, 0.50]) == [pytest.approx(0.10), pytest.approx(0.05)]


def test_paired_contrast_rejects_unpaired_replicates() -> None:
    with pytest.raises(ValueError):
        paired_contrast([0.2], [0.1, 0.0])


def test_issue_weights_use_only_the_ids_that_were_passed() -> None:
    # The same issue can have more trajectories outside this analysis set.
    # Those rows are not in the argument, so they do not change 1/n_j here.
    assert issue_equal_trajectory_weights(["issue-1"]) == [1.0]


def test_crossing_check_still_fires_when_a_key_is_placed_in_two_splits() -> None:
    found = crossing_groups(["issue-1", "issue-1"], ["train", "test"])
    assert found == {"issue-1": ["test", "train"]}
