"""Pinned schema for Open-SWE-Traces historical v1.0.

The field list was checked against parquet metadata at revision
``c144d9d160dcd9f5ebcd9520d2d54111dda8b681``. Nested trajectory fields are
part of the verified schema. Later ``main`` layouts are intentionally absent.
"""

from __future__ import annotations

PINNED_REVISION = "c144d9d160dcd9f5ebcd9520d2d54111dda8b681"
DATASET_ID = "nvidia/Open-SWE-Traces"
DATASET_CARD_URL = (
    "https://huggingface.co/datasets/nvidia/Open-SWE-Traces/blob/"
    f"{PINNED_REVISION}/README.md"
)
# Commit timestamp returned by the Hugging Face commits API for this revision.
PINNED_COMMIT_TIME_UTC = "2026-07-15T06:29:42Z"
AS_OF_DATE = "2026-10-24"

CONFIGS = ("openhands", "sweagent")
SPLITS = ("minimax_m25", "qwen35_122b")

# path prefix -> (config, split). Verified from the pinned dataset card.
SHARD_GROUPS = {
    "data/minimax_m25_openhands_trajectories": ("openhands", "minimax_m25"),
    "data/qwen35_openhands_trajectories": ("openhands", "qwen35_122b"),
    "data/minimax_m25_sweagent_trajectories": ("sweagent", "minimax_m25"),
    "data/qwen35_sweagent_trajectories": ("sweagent", "qwen35_122b"),
}

TOP_LEVEL_FIELDS = (
    "instance_id",
    "repo",
    "license",
    "language",
    "trajectory_id",
    "trajectory",
    "tools",
    "resolved",
    "metadata",
)

TRAJECTORY_MESSAGE_FIELDS = (
    "role",
    "content",
    "reasoning_content",
    "think",
    "tool_calls",
)

# Present in the file, but never eligible as model inputs.
PROHIBITED_FEATURE_FIELDS = (
    "resolved",
    "metadata.reference_patch",
    "metadata.model_patch",
    "reasoning_content",
    "repo",
    "instance_id",
    "trajectory_id",
)

LABEL_COLUMNS = (
    "instance_id",
    "repo",
    "license",
    "language",
    "trajectory_id",
    "resolved",
)

KNOWN_RESOLVED = (0, 1)
UNKNOWN_RESOLVED = -1


def failure_label(resolved: int) -> int | None:
    """Map the dataset resolution label to the failure class.

    ``None`` means the row is excluded from supervised analysis.
    """
    if resolved == 0:
        return 1
    if resolved == 1:
        return 0
    if resolved == -1:
        return None
    raise ValueError(f"unexpected resolved value: {resolved!r}")


def group_for_shard(path: str) -> tuple[str, str]:
    norm = path.replace("\\", "/")
    for prefix, group in SHARD_GROUPS.items():
        if norm.startswith(prefix) or f"/{prefix}/" in f"/{norm}":
            return group
    raise KeyError(f"shard path is not part of the pinned v1.0 layout: {path}")
