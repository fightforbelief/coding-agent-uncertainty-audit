"""Schema pin tests."""

from p8audit.schema import (
    PINNED_REVISION,
    SHARD_GROUPS,
    TOP_LEVEL_FIELDS,
    group_for_shard,
)


def test_pinned_revision_is_the_historical_candidate() -> None:
    assert PINNED_REVISION == "c144d9d160dcd9f5ebcd9520d2d54111dda8b681"
    assert "trajectory" in TOP_LEVEL_FIELDS
    assert "messages" not in TOP_LEVEL_FIELDS


def test_shard_groups_cover_the_v1_card() -> None:
    assert group_for_shard(
        "data/minimax_m25_openhands_trajectories/train-00000-of-00020.parquet"
    ) == ("openhands", "minimax_m25")
    assert group_for_shard(
        "data/qwen35_sweagent_trajectories/train-00001-of-00018.parquet"
    ) == ("sweagent", "qwen35_122b")
    assert len(SHARD_GROUPS) == 4
