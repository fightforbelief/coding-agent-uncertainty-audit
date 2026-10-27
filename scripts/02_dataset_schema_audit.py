"""Label-column census and shard-sample parse coverage for Gate 0.

This script reads identifier and label columns from every pinned parquet
file. It does not download trajectory text for the census. A separate mode
reads the first shard of each config/split for parser coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from p8audit.schema import (
    DATASET_ID,
    LABEL_COLUMNS,
    PINNED_COMMIT_TIME_UTC,
    PINNED_REVISION,
    group_for_shard,
)
from p8audit.splits import assign_group_splits, connected_components
from p8audit.review_view import review_view
from p8audit.trajectory_parser import parse_trajectory

ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_parquet_files() -> list[dict]:
    api = HfApi()
    rows = []
    for item in api.list_repo_tree(
        DATASET_ID,
        revision=PINNED_REVISION,
        repo_type="dataset",
        recursive=True,
    ):
        path = item.path
        if not path.endswith(".parquet"):
            continue
        lfs = getattr(item, "lfs", None)
        if isinstance(lfs, dict):
            size = lfs.get("size")
            sha = lfs.get("sha256")
        else:
            size = getattr(lfs, "size", None) if lfs is not None else None
            sha = getattr(lfs, "sha256", None) if lfs is not None else None
        size = size or getattr(item, "size", None) or 0
        config, split = group_for_shard(path)
        rows.append(
            {
                "path": path,
                "bytes": int(size),
                "sha256": sha,
                "config": config,
                "split": split,
            }
        )
    return rows


def census(files: list[dict]) -> dict:
    fs = HfFileSystem()
    per_group = {}
    repos = []
    instances = []
    resolved_values = []
    languages = Counter()
    licenses = Counter()
    duplicate_trajectory_ids = 0
    seen_trajectories = set()
    for file_info in files:
        remote = f"datasets/{DATASET_ID}@{PINNED_REVISION}/{file_info['path']}"
        with fs.open(remote, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=list(LABEL_COLUMNS))
        key = f"{file_info['config']}/{file_info['split']}"
        bucket = per_group.setdefault(
            key,
            {
                "config": file_info["config"],
                "split": file_info["split"],
                "rows": 0,
                "resolved": Counter(),
                "repos": set(),
                "instances": set(),
                "files": 0,
                "bytes": 0,
            },
        )
        bucket["files"] += 1
        bucket["bytes"] += file_info["bytes"]
        bucket["rows"] += table.num_rows
        resolved = table.column("resolved").to_pylist()
        repo_values = table.column("repo").to_pylist()
        instance_values = table.column("instance_id").to_pylist()
        language_values = table.column("language").to_pylist()
        license_values = table.column("license").to_pylist()
        trajectory_ids = table.column("trajectory_id").to_pylist()
        bucket["resolved"].update(resolved)
        bucket["repos"].update(repo_values)
        bucket["instances"].update(instance_values)
        languages.update(language_values)
        licenses.update(license_values)
        repos.extend(repo_values)
        instances.extend(instance_values)
        resolved_values.extend(resolved)
        for trajectory_id in trajectory_ids:
            if trajectory_id in seen_trajectories:
                duplicate_trajectory_ids += 1
            seen_trajectories.add(trajectory_id)
        print(
            f"CENSUS {key} file={file_info['path']} rows={table.num_rows}",
            flush=True,
        )

    instance_to_repos = defaultdict(set)
    repo_to_instances = defaultdict(set)
    instance_outcomes = defaultdict(set)
    for repo, instance_id, resolved in zip(repos, instances, resolved_values, strict=True):
        instance_to_repos[instance_id].add(repo)
        repo_to_instances[repo].add(instance_id)
        instance_outcomes[instance_id].add(resolved)
    multi_repo_instances = sum(1 for names in instance_to_repos.values() if len(names) > 1)
    conflicting_outcomes = sum(1 for values in instance_outcomes.values() if len(values) > 1)

    components = connected_components(repos, instances)
    assignment = assign_group_splits(components, seed=20261024)
    split_counts = {}
    for component, repo, instance_id, resolved in zip(
        components, repos, instances, resolved_values, strict=True
    ):
        part = assignment[component]
        slot = split_counts.setdefault(
            part,
            {
                "rows": 0,
                "resolved": Counter(),
                "repos": set(),
                "instances": set(),
                "instances_by_class": {0: set(), 1: set(), -1: set()},
            },
        )
        slot["rows"] += 1
        slot["resolved"][resolved] += 1
        slot["repos"].add(repo)
        slot["instances"].add(instance_id)
        slot["instances_by_class"].setdefault(resolved, set()).add(instance_id)

    def pack_group(bucket: dict) -> dict:
        return {
            "config": bucket["config"],
            "split": bucket["split"],
            "files": bucket["files"],
            "bytes": bucket["bytes"],
            "rows": bucket["rows"],
            "resolved": {str(k): int(v) for k, v in sorted(bucket["resolved"].items())},
            "unique_repos": len(bucket["repos"]),
            "unique_instances": len(bucket["instances"]),
        }

    def pack_split(slot: dict) -> dict:
        return {
            "rows": slot["rows"],
            "resolved": {str(k): int(v) for k, v in sorted(slot["resolved"].items())},
            "unique_repos": len(slot["repos"]),
            "unique_instances": len(slot["instances"]),
            "unique_instances_by_resolved": {
                str(k): len(v) for k, v in sorted(slot["instances_by_class"].items())
            },
        }

    known = sum(1 for value in resolved_values if value in (0, 1))
    return {
        "generated_at_utc": _now(),
        "revision": PINNED_REVISION,
        "pinned_commit_time_utc": PINNED_COMMIT_TIME_UTC,
        "rows": len(resolved_values),
        "resolved": dict(Counter(str(value) for value in resolved_values)),
        "known_label_rows": known,
        "unknown_label_rows": sum(1 for value in resolved_values if value == -1),
        "unique_repos": len(set(repos)),
        "unique_instances": len(set(instances)),
        "unique_trajectory_ids": len(seen_trajectories),
        "duplicate_trajectory_id_rows": duplicate_trajectory_ids,
        "instances_in_multiple_repos": multi_repo_instances,
        "instances_with_conflicting_resolved": conflicting_outcomes,
        "languages": {str(k): int(v) for k, v in languages.items()},
        "licenses": {str(k): int(v) for k, v in licenses.items()},
        "groups": [pack_group(per_group[key]) for key in sorted(per_group)],
        "feasibility_split_seed_20261024": {
            key: pack_split(value) for key, value in sorted(split_counts.items())
        },
        "note": (
            "The split counts are a Gate 0 feasibility diagnostic on repository "
            "components. They are not a frozen test evaluation and no model was fit."
        ),
    }


def _first_shard(files: list[dict], config: str, split: str) -> dict:
    matches = [
        item
        for item in files
        if item["config"] == config and item["split"] == split and "train-00000-" in item["path"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one first shard for {config}/{split}, found {matches}")
    return matches[0]


def parse_sample(files: list[dict], out_dir: Path) -> dict:
    fs = HfFileSystem()
    summary = []
    chosen_observations: list[dict] = []
    for config in ("openhands", "sweagent"):
        for split in ("minimax_m25", "qwen35_122b"):
            file_info = _first_shard(files, config, split)
            remote = f"datasets/{DATASET_ID}@{PINNED_REVISION}/{file_info['path']}"
            print(f"PARSE_SAMPLE {config}/{split}", flush=True)
            with fs.open(remote, "rb") as handle:
                table = pq.ParquetFile(handle).read(
                    columns=["resolved", "trajectory", "instance_id", "repo", "language"]
                )
            resolved = table.column("resolved").to_pylist()
            trajectories = table.column("trajectory")
            status_counts = Counter()
            complete_hist = Counter()
            reached = {3: 0, 5: 0, 10: 0}
            known = 0
            known_reached = {3: 0, 5: 0, 10: 0}
            role_counts = Counter()
            tool_names = Counter()
            marker_counts = Counter()
            n_tool = 0
            group_seed = 20261024 + (0 if config == "openhands" else 2) + (
                0 if split == "minimax_m25" else 1
            )
            rng = random.Random(group_seed)
            sample_rows = set(rng.sample(range(len(resolved)), k=min(80, len(resolved))))
            group_observations: list[dict] = []
            instances = table.column("instance_id").to_pylist()
            repos = table.column("repo").to_pylist()
            languages = table.column("language").to_pylist()
            for row_index, resolved_value in enumerate(resolved):
                parsed = parse_trajectory(trajectories[row_index].as_py())
                status_counts[parsed.status] += 1
                n_complete = parsed.n_complete_cycles
                complete_hist[n_complete] += 1
                for checkpoint in reached:
                    if n_complete >= checkpoint:
                        reached[checkpoint] += 1
                if resolved_value in (0, 1):
                    known += 1
                    for checkpoint in known_reached:
                        if n_complete >= checkpoint:
                            known_reached[checkpoint] += 1
                for message in parsed.messages:
                    role_counts[message.role] += 1
                    for call in message.tool_calls:
                        tool_names[call.name] += 1
                for cycle in parsed.cycles:
                    for markers in cycle.markers:
                        n_tool += 1
                        if markers.explicit_error:
                            marker_counts["explicit_error"] += 1
                        if markers.explicit_test_failure:
                            marker_counts["explicit_test_failure"] += 1
                        if markers.explicit_test_pass:
                            marker_counts["explicit_test_pass"] += 1
            # Collect review rows in a second walk over 80 rows. Re-parsing
            # them is cheap relative to reading the shard.
            for row_index in sorted(sample_rows):
                parsed = parse_trajectory(trajectories[row_index].as_py())
                for cycle in parsed.cycles:
                    for obs_index, markers in zip(
                        cycle.observation_indexes, cycle.markers, strict=True
                    ):
                        message = parsed.messages[obs_index]
                        digest = hashlib.sha1(
                            (message.content or "").encode("utf-8")
                        ).hexdigest()[:16]
                        group_observations.append(
                            {
                                "config": config,
                                "split": split,
                                "shard": file_info["path"],
                                "row_index": row_index,
                                "instance_id": instances[row_index],
                                "repo": repos[row_index],
                                "language": languages[row_index],
                                "resolved": resolved[row_index],
                                "message_index": message.index,
                                "tool_names": list(cycle.tool_names),
                                "cycle_status": cycle.status,
                                "review": review_view(message.content),
                                "content_sha1_16": digest,
                                "n_chars": len(message.content or ""),
                                "predicted_explicit_error": markers.explicit_error,
                                "predicted_test_failure": markers.explicit_test_failure,
                                "predicted_test_pass": markers.explicit_test_pass,
                                "error_reasons": list(markers.error_reasons),
                                "test_failure_reasons": list(markers.test_failure_reasons),
                                "test_pass_reasons": list(markers.test_pass_reasons),
                            }
                        )
            rng.shuffle(group_observations)
            chosen_observations.extend(group_observations[:25])
            summary.append(
                {
                    "config": config,
                    "split": split,
                    "path": file_info["path"],
                    "rows": len(resolved),
                    "parse_status": dict(status_counts),
                    "role_counts": dict(role_counts),
                    "tool_names": dict(tool_names),
                    "tool_observations": n_tool,
                    "marker_counts": dict(marker_counts),
                    "known_label_rows": known,
                    "fraction_reaching_checkpoint": {
                        str(k): (reached[k] / len(resolved) if resolved else None) for k in reached
                    },
                    "fraction_known_label_reaching_checkpoint": {
                        str(k): (known_reached[k] / known if known else None) for k in known_reached
                    },
                    "complete_cycle_quantiles": _quantiles(complete_hist, len(resolved)),
                }
            )
    payload = {
        "generated_at_utc": _now(),
        "revision": PINNED_REVISION,
        "sample": "first parquet shard of each pinned config/split",
        "groups": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parse_sample.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    review_dir = ROOT / "artifacts" / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    sample_path = review_dir / "tool_observation_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(chosen_observations):
            row["sample_id"] = f"s{index:03d}"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload["review_sample_path"] = str(sample_path)
    payload["review_sample_size"] = len(chosen_observations)
    return payload


def _quantiles(hist: Counter, n: int) -> dict:
    if n == 0:
        return {}
    ordered = []
    for value, count in sorted(hist.items()):
        ordered.extend([value] * count)
    def pick(q: float) -> int:
        return ordered[min(n - 1, int(q * (n - 1)))]
    return {"min": ordered[0], "p50": pick(0.50), "p90": pick(0.90), "max": ordered[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("census", "parse-sample", "both"), default="both")
    args = parser.parse_args()
    out_dir = ROOT / "artifacts" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_parquet_files()
    (out_dir / "pinned_parquet_files.json").write_text(
        json.dumps({"revision": PINNED_REVISION, "files": files}, indent=2),
        encoding="utf-8",
    )
    if args.mode in ("census", "both"):
        payload = census(files)
        (out_dir / "label_census.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print("WROTE label_census.json rows", payload["rows"])
    if args.mode in ("parse-sample", "both"):
        payload = parse_sample(files, out_dir)
        print("WROTE parse_sample.json groups", len(payload["groups"]))


if __name__ == "__main__":
    main()
