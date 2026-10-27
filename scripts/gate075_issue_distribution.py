"""Metadata-only trajectory counts per issue.

Reads trajectory_id, instance_id, repo, and resolved from the pinned parquet
files. Does not read trajectory bodies. Does not overwrite the Gate 0 census.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from p8audit.issue_distribution import concentration, summarize_issue_counts
from p8audit.schema import DATASET_ID, PINNED_REVISION, group_for_shard
from p8audit.splits import assign_group_splits, connected_components, crossing_groups

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["trajectory_id", "instance_id", "repo", "resolved"]
SPLIT_SEED = 20261024
CENSUS_PATH = ROOT / "artifacts" / "manifests" / "label_census.json"
PINNED_PATH = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
DISTRIBUTION_PATH = ROOT / "artifacts" / "manifests" / "issue_trajectory_distribution.json"
SPLIT_PATH = ROOT / "artifacts" / "manifests" / "split_assignment_seed_20261024.json"
REPORT_PATH = ROOT / "artifacts" / "reports" / "ISSUE_TRAJECTORY_DISTRIBUTION_AUDIT.md"
PROTECTED = {
    CENSUS_PATH.resolve(),
    (ROOT / "artifacts" / "manifests" / "parse_sample.json").resolve(),
    (ROOT / "artifacts" / "manual_review" / "marker_review_queue.csv").resolve(),
    (ROOT / "artifacts" / "reports" / "GATE0_REPORT.md").resolve(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_new(path: Path) -> None:
    if path.resolve() in PROTECTED:
        raise SystemExit(f"refusing to write a protected historical file: {path}")


def _load_expected() -> dict:
    payload = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    if payload.get("revision") != PINNED_REVISION:
        raise SystemExit("label_census.json is not the pinned revision")
    return payload


def _load_files() -> list[dict]:
    payload = json.loads(PINNED_PATH.read_text(encoding="utf-8"))
    if payload.get("revision") != PINNED_REVISION:
        raise SystemExit("pinned_parquet_files.json is not the pinned revision")
    files = []
    for item in payload["files"]:
        config, dataset_split = group_for_shard(item["path"])
        files.append(
            {
                "path": item["path"],
                "bytes": int(item["bytes"]),
                "config": config,
                "dataset_split": dataset_split,
            }
        )
    return files


def _assert_revision() -> str:
    info = HfApi().dataset_info(DATASET_ID, revision=PINNED_REVISION)
    sha = getattr(info, "sha", None)
    if sha != PINNED_REVISION:
        raise SystemExit(f"pinned revision check failed: api sha={sha}")
    return sha


def _read_table(fs: HfFileSystem, path: str, manifest_bytes: int):
    remote = f"datasets/{DATASET_ID}@{PINNED_REVISION}/{path}"
    info = fs.info(remote)
    remote_bytes = int(info.get("size") or 0)
    if manifest_bytes and remote_bytes and remote_bytes != manifest_bytes:
        raise SystemExit(
            f"shard size mismatch for {path}: manifest={manifest_bytes} remote={remote_bytes}"
        )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with fs.open(remote, "rb") as handle:
                return pq.ParquetFile(handle).read(columns=COLUMNS)
        except Exception as exc:
            last_error = exc
            print(f"RETRY {attempt + 1} {path} {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not read {path}") from last_error


def _scope_block(counts: Counter) -> dict:
    values = list(counts.values())
    return {
        "summary": summarize_issue_counts(values),
        "concentration": concentration(values),
    }


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _summary_row(name: str, block: dict) -> str:
    summary = block["summary"]
    return (
        f"| {name} | {summary['issues']} | {summary['trajectories']} | "
        f"{summary['min']} | {summary['q1']} | {summary['median']} | {summary['q3']} | "
        f"{summary['p90']} | {summary['p95']} | {summary['max']} | "
        f"{summary['exactly_1']} ({_fmt_pct(summary['pct_exactly_1'])}) | "
        f"{summary['exactly_2']} ({_fmt_pct(summary['pct_exactly_2'])}) | "
        f"{summary['at_least_3']} ({_fmt_pct(summary['pct_at_least_3'])}) |"
    )


def _concentration_lines(block: dict) -> list[str]:
    lines = []
    for item in block["concentration"]["top_fractions"]:
        lines.append(
            f"- Top {_fmt_pct(item['issue_fraction'])} of issues "
            f"({item['issues']} issues) hold {item['trajectories']} trajectories "
            f"({_fmt_pct(item['trajectory_share'])})."
        )
    half = block["concentration"]["issues_covering_half_of_trajectories"]
    lines.append(
        f"- Issues needed to cover half of the trajectories in this scope: {half} "
        f"({_fmt_pct(block['concentration']['issues_covering_half_share_of_issues'])} of issues)."
    )
    lines.append(
        f"- Largest issue in this scope: {block['concentration']['largest_issue_trajectories']} "
        f"trajectories ({_fmt_pct(block['concentration']['largest_issue_share'])})."
    )
    return lines


def _write_report(payload: dict) -> None:
    recon = payload["reconciliation_with_gate0_census"]
    matched = recon["all_compared_fields_match"]
    mixed = payload["mixed_resolved_issues"]
    crossing = payload["boundary_checks"]
    lines = [
        "# Issue trajectory-count and concentration audit",
        "",
        "**Status:** metadata re-read of the pinned revision. No model was fit. No trajectory body was read.",
        "",
        f"**Generated:** {payload['generated_at_utc']}",
        "",
        "Research question, unchanged:",
        "",
        "> How well do published trajectory-reliability signals transfer to repository-level coding-agent traces, and do observable execution-feedback features add incremental predictive value beyond simple structural baselines under repository-held-out evaluation?",
        "",
        "## Source and reconciliation",
        "",
        f"- Revision: `{payload['revision']}`.",
        f"- Columns read: {', '.join(payload['columns_read'])}. The `trajectory` column was not requested.",
        f"- Files read: {payload['files_read']}. Bytes recorded from the existing pin manifest: {payload['bytes_recorded']}.",
        f"- Gate 0 census file was compared and not overwritten. Match: `{matched}`.",
        "",
        "| Field | Gate 0 census | This re-read | Match |",
        "|---|---:|---:|---|",
    ]
    for field, item in recon["fields"].items():
        lines.append(f"| `{field}` | {item['gate0']} | {item['reread']} | {item['match']} |")
    lines.extend(
        [
            "",
            "Quantiles are the smallest trajectory count whose empirical CDF reaches the stated probability. They are not linearly interpolated.",
            "",
            "## Corpus-wide trajectories per issue",
            "",
            "An issue is one `instance_id` across every scaffold and model configuration. The same issue often has trajectories in more than one configuration, so a corpus-wide count is larger than the count inside one configuration.",
            "",
            "| Scope | Issues | Trajectories | Min | Q1 | Median | Q3 | P90 | P95 | Max | Exactly 1 | Exactly 2 | At least 3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            _summary_row("All issues", payload["corpus"]),
            _summary_row("Mixed `resolved`", payload["mixed_resolved_issues"]["counts"]),
        ]
    )
    for name, block in payload["by_scaffold"].items():
        lines.append(_summary_row(f"Scaffold {name}", block))
    for name, block in payload["by_configuration"].items():
        lines.append(_summary_row(name, block))
    for name, block in payload["by_holdout_partition"].items():
        lines.append(_summary_row(f"Holdout {name}", block))
    lines.extend(["", "## Concentration of trajectories", ""])
    lines.append("Corpus-wide:")
    lines.extend(_concentration_lines(payload["corpus"]))
    lines.append("")
    lines.append("Mixed-label issues:")
    lines.extend(_concentration_lines(payload["mixed_resolved_issues"]["counts"]))
    lines.extend(
        [
            "",
            "## Issues whose trajectories do not share one `resolved` value",
            "",
            f"Count: {mixed['issues']}. Gate 0 reported 7,161. This re-read matches that count: `{mixed['matches_gate0_7161']}`.",
            "",
            "These issues are not treated as labeling errors. Each trajectory keeps its own `resolved` value.",
            "",
            "| `resolved` values present | Issues |",
            "|---|---:|",
        ]
    )
    for pattern, count in mixed["value_patterns"].items():
        lines.append(f"| {pattern} | {count} |")
    lines.extend(
        [
            "",
            "## Repository holdout, seed 20261024",
            "",
            "The split was recomputed from repository and instance identifiers with the existing `connected_components` and `assign_group_splits` functions. `resolved` was not an input to the assignment. No model prediction was read.",
            "",
            f"- Components: {crossing['components']}.",
            f"- Components equal the number of repositories: `{crossing['components_equal_repositories']}`.",
            f"- `instance_id` values stored under more than one repository: {crossing['instances_in_multiple_repos']}.",
            f"- `instance_id` values whose trajectories fall in more than one holdout partition: {crossing['instances_crossing_partitions']}.",
            f"- Repositories whose trajectories fall in more than one holdout partition: {crossing['repos_crossing_partitions']}.",
            "",
            "Partition row totals are in the reconciliation table under `holdout_*`. They are a feasibility description of the same seed used in Gate 0. They are not a test-set score.",
            "",
            "## What this file does not estimate",
            "",
            "No AUPRC, no confidence interval, and no `Delta_AUPRC` were calculated. The paired repository-cluster procedure for a later contrast is specified in `docs/protocol.md`: both AUPRCs are computed on indexes from one call to `repository_cluster_bootstrap_indices`, and the replicate stores their difference. Issue-equal sensitivity uses weight `1/n_j` on the trajectories that enter that analysis set and does not change trajectory-level outcomes.",
            "",
        ]
    )
    _assert_new(REPORT_PATH)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    expected = _load_expected()
    files = _load_files()
    sha = _assert_revision()
    print(f"REVISION_OK {sha} files={len(files)}", flush=True)
    fs = HfFileSystem()
    repos: list[str] = []
    instances: list[str] = []
    resolved_values: list[int] = []
    groups: list[str] = []
    scaffolds: list[str] = []
    seen_trajectories: set[str] = set()
    duplicate_trajectory_ids = 0
    bytes_recorded = 0
    for file_info in files:
        table = _read_table(fs, file_info["path"], file_info["bytes"])
        bytes_recorded += file_info["bytes"]
        group_name = f"{file_info['config']}/{file_info['dataset_split']}"
        trajectory_ids = table.column("trajectory_id").to_pylist()
        instance_values = table.column("instance_id").to_pylist()
        repo_values = table.column("repo").to_pylist()
        labels = table.column("resolved").to_pylist()
        for trajectory_id, instance_id, repo, resolved in zip(
            trajectory_ids, instance_values, repo_values, labels, strict=True
        ):
            if trajectory_id in seen_trajectories:
                duplicate_trajectory_ids += 1
            seen_trajectories.add(str(trajectory_id))
            repos.append(str(repo))
            instances.append(str(instance_id))
            resolved_values.append(int(resolved))
            groups.append(group_name)
            scaffolds.append(file_info["config"])
        print(f"META {group_name} rows={table.num_rows} file={file_info['path']}", flush=True)

    corpus_counts: Counter[str] = Counter(instances)
    outcomes: dict[str, set[int]] = defaultdict(set)
    repos_of: dict[str, set[str]] = defaultdict(set)
    for instance_id, repo, resolved in zip(instances, repos, resolved_values, strict=True):
        outcomes[instance_id].add(resolved)
        repos_of[instance_id].add(repo)
    components = connected_components(repos, instances)
    assignment = assign_group_splits(components, seed=SPLIT_SEED)
    split_names = [assignment[component] for component in components]
    instance_crossing = crossing_groups(instances, split_names)
    repo_crossing = crossing_groups(repos, split_names)
    multi_repo = {key: sorted(value) for key, value in repos_of.items() if len(value) > 1}

    by_configuration = {}
    for name in sorted(set(groups)):
        counts: Counter[str] = Counter()
        for instance_id, group_name in zip(instances, groups, strict=True):
            if group_name == name:
                counts[instance_id] += 1
        by_configuration[name] = _scope_block(counts)
    by_scaffold = {}
    for name in sorted(set(scaffolds)):
        counts = Counter(
            instance_id
            for instance_id, scaffold in zip(instances, scaffolds, strict=True)
            if scaffold == name
        )
        by_scaffold[name] = _scope_block(counts)
    by_holdout = {}
    holdout_rows: Counter[str] = Counter()
    holdout_resolved: dict[str, Counter[int]] = defaultdict(Counter)
    for split_name, resolved in zip(split_names, resolved_values, strict=True):
        holdout_rows[split_name] += 1
        holdout_resolved[split_name][resolved] += 1
    for name in ("train", "validation", "test"):
        counts = Counter(
            instance_id
            for instance_id, split_name in zip(instances, split_names, strict=True)
            if split_name == name
        )
        by_holdout[name] = _scope_block(counts)

    mixed_ids = [key for key, values in outcomes.items() if len(values) > 1]
    mixed_counts: Counter[str] = Counter({key: corpus_counts[key] for key in mixed_ids})
    patterns = Counter(
        ",".join(str(value) for value in sorted(outcomes[key])) for key in mixed_ids
    )
    resolved_counter = Counter(str(value) for value in resolved_values)
    group_rows = Counter(groups)
    compared = {
        "rows": {"gate0": expected["rows"], "reread": len(instances)},
        "unique_repos": {"gate0": expected["unique_repos"], "reread": len(set(repos))},
        "unique_instances": {"gate0": expected["unique_instances"], "reread": len(corpus_counts)},
        "unique_trajectory_ids": {
            "gate0": expected["unique_trajectory_ids"],
            "reread": len(seen_trajectories),
        },
        "duplicate_trajectory_id_rows": {
            "gate0": expected["duplicate_trajectory_id_rows"],
            "reread": duplicate_trajectory_ids,
        },
        "instances_in_multiple_repos": {
            "gate0": expected["instances_in_multiple_repos"],
            "reread": len(multi_repo),
        },
        "instances_with_conflicting_resolved": {
            "gate0": expected["instances_with_conflicting_resolved"],
            "reread": len(mixed_ids),
        },
        "resolved_0": {"gate0": expected["resolved"]["0"], "reread": resolved_counter["0"]},
        "resolved_1": {"gate0": expected["resolved"]["1"], "reread": resolved_counter["1"]},
        "resolved_-1": {"gate0": expected["resolved"]["-1"], "reread": resolved_counter["-1"]},
    }
    for group in expected["groups"]:
        key = f"rows_{group['config']}/{group['split']}"
        compared[key] = {"gate0": group["rows"], "reread": group_rows[f"{group['config']}/{group['split']}"]}
    gate0_split = expected["feasibility_split_seed_20261024"]
    for name in ("train", "validation", "test"):
        compared[f"holdout_rows_{name}"] = {
            "gate0": gate0_split[name]["rows"],
            "reread": holdout_rows[name],
        }
        compared[f"holdout_repos_{name}"] = {
            "gate0": gate0_split[name]["unique_repos"],
            "reread": len({repo for repo, split_name in zip(repos, split_names, strict=True) if split_name == name}),
        }
    for field, item in compared.items():
        item["match"] = item["gate0"] == item["reread"]
    all_match = all(item["match"] for item in compared.values())

    repo_to_split = {}
    for repo, split_name in zip(repos, split_names, strict=True):
        previous = repo_to_split.get(repo)
        if previous is not None and previous != split_name:
            repo_to_split[repo] = "CROSSING"
        else:
            repo_to_split[repo] = split_name

    payload = {
        "generated_at_utc": _now(),
        "revision": PINNED_REVISION,
        "api_sha": sha,
        "columns_read": COLUMNS,
        "trajectory_bodies_read": False,
        "files_read": len(files),
        "bytes_recorded": bytes_recorded,
        "elapsed_seconds": round(time.time() - started, 1),
        "models_fit": False,
        "auprc_calculated": False,
        "gate0_census_overwritten": False,
        "reconciliation_with_gate0_census": {
            "all_compared_fields_match": all_match,
            "fields": compared,
        },
        "corpus": _scope_block(corpus_counts),
        "by_scaffold": by_scaffold,
        "by_configuration": by_configuration,
        "by_holdout_partition": by_holdout,
        "mixed_resolved_issues": {
            "issues": len(mixed_ids),
            "matches_gate0_7161": len(mixed_ids) == 7161,
            "counts": _scope_block(mixed_counts),
            "value_patterns": {key: patterns[key] for key in sorted(patterns)},
        },
        "boundary_checks": {
            "components": len(set(components)),
            "repositories": len(set(repos)),
            "components_equal_repositories": len(set(components)) == len(set(repos)),
            "instances_in_multiple_repos": len(multi_repo),
            "instances_crossing_partitions": len(instance_crossing),
            "repos_crossing_partitions": len(repo_crossing),
            "instance_crossing_examples": dict(list(instance_crossing.items())[:20]),
            "repo_crossing_examples": dict(list(repo_crossing.items())[:20]),
            "multi_repo_examples": {key: multi_repo[key] for key in list(multi_repo)[:20]},
        },
        "holdout_resolved_counts": {
            name: {str(key): int(value) for key, value in sorted(holdout_resolved[name].items())}
            for name in ("train", "validation", "test")
        },
        "quantile_definition": "smallest count with empirical CDF at least the stated probability",
        "split_seed": SPLIT_SEED,
    }
    _assert_new(DISTRIBUTION_PATH)
    DISTRIBUTION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    split_payload = {
        "generated_at_utc": payload["generated_at_utc"],
        "revision": PINNED_REVISION,
        "seed": SPLIT_SEED,
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "method": "connected_components of repo and instance_id, then assign_group_splits",
        "outcomes_used_to_assign_split": False,
        "predictions_used": False,
        "n_components": payload["boundary_checks"]["components"],
        "n_repos": payload["boundary_checks"]["repositories"],
        "instances_crossing_partitions": len(instance_crossing),
        "repos_crossing_partitions": len(repo_crossing),
        "instances_in_multiple_repos": len(multi_repo),
        "row_counts_match_gate0_census": all(
            compared[f"holdout_rows_{name}"]["match"] for name in ("train", "validation", "test")
        ),
        "repo_to_split": dict(sorted(repo_to_split.items())),
    }
    _assert_new(SPLIT_PATH)
    SPLIT_PATH.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    _write_report(payload)
    print(
        f"ISSUE_DISTRIBUTION_DONE match={all_match} issues={len(corpus_counts)} "
        f"elapsed={payload['elapsed_seconds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
