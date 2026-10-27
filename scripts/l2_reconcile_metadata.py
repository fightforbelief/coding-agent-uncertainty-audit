"""Census and repository split from label columns only.

Does not read trajectory bodies and does not extract features.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from p8audit.schema import DATASET_ID, PINNED_REVISION, SHARD_GROUPS
from p8audit.splits import assign_group_splits, connected_components, crossing_groups

ROOT = Path(r"C:\Users\hanqi\Desktop\projects\p8-coding-agent-uncertainty-audit")
PINNED = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
RUN = ROOT / "artifacts" / "l2_run_20261027_01"
EXPECTED = {
    "trajectories": 207489,
    "resolved_1": 65244,
    "resolved_0": 95487,
    "resolved_minus_1": 46758,
    "known_label": 160731,
    "repositories": 2654,
    "issues": 22320,
    "conflicting_issues": 7161,
    "train_rows": 150190,
    "train_repos": 1858,
    "validation_rows": 32124,
    "validation_repos": 398,
    "test_rows": 25175,
    "test_repos": 398,
}


def records() -> list[dict]:
    payload = json.loads(PINNED.read_text(encoding="utf-8"))
    files = payload["files"] if isinstance(payload, dict) and "files" in payload else payload
    rows = []
    for item in files:
        config, dataset_split = SHARD_GROUPS[str(Path(item["path"]).parent).replace("\\", "/")]
        rows.append({"path": item["path"], "bytes": int(item["bytes"]), "config": config, "dataset_split": dataset_split})
    return rows


def main() -> None:
    if RUN.exists():
        raise SystemExit(f"run directory already exists: {RUN}")
    info = HfApi().dataset_info(DATASET_ID, revision=PINNED_REVISION)
    if getattr(info, "sha", None) != PINNED_REVISION:
        raise SystemExit("pinned revision check failed")
    shards = records()
    if len(shards) != 84:
        raise SystemExit(f"manifest has {len(shards)} files")
    fs = HfFileSystem()
    repos: list[str] = []
    issues: list[str] = []
    resolved: list[int] = []
    shard_names: list[str] = []
    row_indexes: list[int] = []
    trajectory_ids: list[str] = []
    configs: list[str] = []
    for shard in shards:
        remote = f"datasets/{DATASET_ID}@{PINNED_REVISION}/{shard['path']}"
        with fs.open(remote, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=["instance_id", "repo", "trajectory_id", "resolved"])
        if "trajectory" in table.column_names:
            raise SystemExit("trajectory body was read during the label pass")
        n = table.num_rows
        repos.extend(str(value) for value in table.column("repo").to_pylist())
        issues.extend(str(value) for value in table.column("instance_id").to_pylist())
        resolved.extend(int(value) for value in table.column("resolved").to_pylist())
        trajectory_ids.extend(str(value) for value in table.column("trajectory_id").to_pylist())
        shard_names.extend([shard["path"]] * n)
        row_indexes.extend(range(n))
        configs.extend([shard["config"]] * n)
        print(f"LABELS {shard['path']} rows={n}", flush=True)
    counts = Counter(resolved)
    by_issue: dict[str, set[int]] = defaultdict(set)
    for issue, value in zip(issues, resolved, strict=True):
        by_issue[issue].add(value)
    conflicting = sum(1 for values in by_issue.values() if len(values) > 1)
    components = connected_components(repos, issues)
    assignment = assign_group_splits(components, seed=20261024)
    splits = [assignment[component] for component in components]
    if crossing_groups(repos, splits) or crossing_groups(issues, splits):
        raise SystemExit("a repository or issue crosses splits")
    split_rows = Counter(splits)
    split_repos = {name: len({repo for repo, split_name in zip(repos, splits, strict=True) if split_name == name}) for name in ("train", "validation", "test")}
    observed = {
        "trajectories": len(resolved),
        "resolved_1": counts[1],
        "resolved_0": counts[0],
        "resolved_minus_1": counts[-1],
        "known_label": counts[0] + counts[1],
        "repositories": len(set(repos)),
        "issues": len(set(issues)),
        "conflicting_issues": conflicting,
        "train_rows": split_rows["train"],
        "train_repos": split_repos["train"],
        "validation_rows": split_rows["validation"],
        "validation_repos": split_repos["validation"],
        "test_rows": split_rows["test"],
        "test_repos": split_repos["test"],
    }
    RUN.mkdir(parents=True)
    (RUN / "L2_DATASET_CENSUS.json").write_text(json.dumps({"expected": EXPECTED, "observed": observed, "match": observed == EXPECTED}, indent=2), encoding="utf-8")
    if observed != EXPECTED:
        raise SystemExit(f"census mismatch: {json.dumps(observed)}")
    table = pa.table(
        {
            "shard": shard_names,
            "row_index": row_indexes,
            "trajectory_id": trajectory_ids,
            "instance_id": issues,
            "repo": repos,
            "config": configs,
            "resolved": resolved,
            "split": splits,
        }
    )
    pq.write_table(table, RUN / "labels_for_join.parquet")
    print("CENSUS_MATCH", flush=True)


if __name__ == "__main__":
    main()
