"""Group-split tests."""

from p8audit.splits import assign_group_splits, connected_components


def test_shared_instance_joins_repositories() -> None:
    repos = ["alpha", "alpha", "beta", "gamma"]
    instances = ["i1", "i2", "i2", "i3"]
    components = connected_components(repos, instances)
    assert components[0] == components[1] == components[2]
    assert components[3] != components[0]


def test_split_assignment_has_no_component_overlap() -> None:
    components = [f"repo-{index}" for index in range(50)]
    assignment = assign_group_splits(components, seed=20261024)
    assert set(assignment) == set(components)
    assert set(assignment.values()) == {"train", "validation", "test"}
    again = assign_group_splits(components, seed=20261024)
    assert assignment == again
    other = assign_group_splits(components, seed=11)
    assert other != assignment
