"""Gate 0.75 review queue.

The historical 548-row queue is not opened for writing. This script streams
eight additional pinned shards, chosen without using outcomes, plus no other
trajectory bodies. Precision rows are a simple random sample of parser
positives. Diagnostic rows are stratified and are not a precision sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from p8audit.exit_formats import describe_exit_formats
from p8audit.review_queue import (
    Reservoir,
    context_excerpt,
    is_hard_negative,
    signal_families,
    stratify,
)
from p8audit.review_stats import clopper_pearson_lower, minimum_n_for_perfect_precision
from p8audit.sampling import (
    OPENHANDS_ONLY_PRECISION_FAMILIES,
    PRECISION_FAMILIES,
    PRECISION_SAMPLE_SIZE,
    SUPPLEMENTAL_SHARD_SEED,
    SUPPLEMENTAL_SHARDS_PER_GROUP,
    inclusion_probability,
    select_supplemental_shards,
)
from p8audit.schema import DATASET_ID, PINNED_REVISION, group_for_shard
from p8audit.trajectory_parser import parse_trajectory

ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_SEED = 20261076
QUEUE_VERSION = "gate075-v2"
DIAGNOSTIC_RESERVOIR = 24
DIAGNOSTIC_PER_GROUP = 6
FIRST_SHARD_PATHS = {
    "data/minimax_m25_openhands_trajectories/train-00000-of-00020.parquet",
    "data/qwen35_openhands_trajectories/train-00000-of-00023.parquet",
    "data/minimax_m25_sweagent_trajectories/train-00000-of-00023.parquet",
    "data/qwen35_sweagent_trajectories/train-00000-of-00018.parquet",
}
DIAGNOSTIC_FAMILIES = (
    "explicit_exception",
    "explicit_test_failure",
    "explicit_test_pass",
    "tool_exit_nonzero",
    "interrupt_or_timeout",
    "missed_test_format",
    "discordant_exit0_test_failure",
    "hard_negative",
    "random_observation",
    "execution_status_unknown",
)
SELECTION_METHOD = (
    f"numpy Generator seed {SUPPLEMENTAL_SHARD_SEED}; "
    f"{SUPPLEMENTAL_SHARDS_PER_GROUP} shards per configuration after excluding "
    "the four Gate 0 first shards; choice uses sorted file paths only"
)
CSV_FIELDS = [
    "sample_id",
    "dataset_revision",
    "config",
    "parquet_shard",
    "trajectory_id",
    "instance_id",
    "repo",
    "language",
    "observation_index",
    "signal_family",
    "parser_predicted_label",
    "matched_pattern",
    "context_excerpt",
    "reviewer_label",
    "reviewer_notes",
    "student_verified",
    "split",
    "row_index",
    "content_sha1_16",
    "n_chars",
    "queue_seed",
    "observation_key",
    "queue_version",
    "sampling_stratum",
    "sampling_probability",
    "review_purpose",
    "source_shard_selection_method",
    "duplicate_signal_families",
    "excerpt_truncated",
]
PROTECTED_NAMES = {
    "marker_review_queue.csv",
    "parse_sample.json",
    "label_census.json",
    "GATE0_REPORT.md",
    "manual_review_queue_manifest.json",
    "marker_v2_sample_counts.json",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refuse(path: Path) -> None:
    if path.name in PROTECTED_NAMES:
        raise SystemExit(f"refusing to write historical file {path}")


def _pinned_records() -> list[dict]:
    payload = json.loads(
        (ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json").read_text(encoding="utf-8")
    )
    if payload.get("revision") != PINNED_REVISION:
        raise SystemExit("pinned manifest revision mismatch")
    records = []
    for item in payload["files"]:
        config, dataset_split = group_for_shard(item["path"])
        records.append(
            {
                "path": item["path"],
                "bytes": int(item["bytes"]),
                "config": config,
                "dataset_split": dataset_split,
            }
        )
    return records


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
                table = pq.ParquetFile(handle).read(
                    columns=["instance_id", "repo", "language", "trajectory_id", "trajectory"]
                )
            return table, remote_bytes or manifest_bytes
        except Exception as exc:
            last_error = exc
            print(f"RETRY {attempt + 1} {path} {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not read {path}") from last_error


def _record(base: dict, message_index: int, content: str, family: str, signals, families: list[str]) -> dict:
    excerpt = context_excerpt(content, signals.reasons)
    others = [name for name in families if name != family]
    return {
        "config": base["config"],
        "split": base["dataset_split"],
        "parquet_shard": base["shard"],
        "trajectory_id": base["trajectory_id"],
        "instance_id": base["instance_id"],
        "repo": base["repo"],
        "language": base["language"],
        "row_index": base["row_index"],
        "observation_index": message_index,
        "signal_family": family,
        "parser_predicted_label": signals.predicted_label(),
        "matched_pattern": ",".join(signals.reasons),
        "context_excerpt": excerpt,
        "content_sha1_16": hashlib.sha1((content or "").encode("utf-8")).hexdigest()[:16],
        "n_chars": len(content or ""),
        "observation_key": f"{base['shard']}|{base['trajectory_id']}|{message_index}",
        "duplicate_signal_families": "|".join(others),
        "excerpt_truncated": "true" if excerpt.endswith("[TRUNCATED]") else "false",
    }


def _probability_label(seen: int, kept: int) -> str:
    if seen < 1:
        return "0/0"
    if kept != min(PRECISION_SAMPLE_SIZE, seen):
        raise SystemExit(f"precision sample size {kept} is inconsistent with population {seen}")
    probability = inclusion_probability(kept, seen)
    return f"{kept}/{seen} ({probability:.10f})"


def main() -> None:
    started = time.time()
    sha = _assert_revision()
    selected = select_supplemental_shards(
        _pinned_records(),
        seed=SUPPLEMENTAL_SHARD_SEED,
        per_group=SUPPLEMENTAL_SHARDS_PER_GROUP,
        excluded_paths=set(FIRST_SHARD_PATHS),
    )
    if len(selected) != 8:
        raise SystemExit(f"expected 8 supplemental shards, found {len(selected)}")
    if any(item["path"] in FIRST_SHARD_PATHS for item in selected):
        raise SystemExit("supplemental draw included a Gate 0 first shard")
    download_bytes = sum(item["bytes"] for item in selected)
    print(
        "SHARDS "
        + ",".join(item["path"] for item in selected)
        + f" bytes={download_bytes}",
        flush=True,
    )

    precision: dict[str, Reservoir] = {}
    for index, family in enumerate(PRECISION_FAMILIES + OPENHANDS_ONLY_PRECISION_FAMILIES):
        precision[family] = Reservoir(PRECISION_SAMPLE_SIZE, random.Random(OBSERVATION_SEED + index))
    diagnostic: dict[tuple[str, str, str], Reservoir] = {}
    population: Counter[str] = Counter()
    population_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    exit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    exit_examples: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    observations_by_scaffold: Counter[str] = Counter()
    fs = HfFileSystem()
    shard_stats = []

    def diagnostic_bucket(family: str, config: str, dataset_split: str) -> Reservoir:
        key = (family, config, dataset_split)
        if key not in diagnostic:
            diagnostic[key] = Reservoir(
                DIAGNOSTIC_RESERVOIR,
                random.Random(OBSERVATION_SEED + 100 + len(diagnostic)),
            )
        return diagnostic[key]

    for shard in selected:
        table, remote_bytes = _read_table(fs, shard["path"], shard["bytes"])
        config = shard["config"]
        dataset_split = shard["dataset_split"]
        group_name = f"{config}/{dataset_split}"
        n_observations = 0
        trajectories = table.column("trajectory")
        instances = table.column("instance_id").to_pylist()
        repos = table.column("repo").to_pylist()
        languages = table.column("language").to_pylist()
        trajectory_ids = table.column("trajectory_id").to_pylist()
        for row_index in range(table.num_rows):
            parsed = parse_trajectory(trajectories[row_index].as_py())
            base = {
                "config": config,
                "dataset_split": dataset_split,
                "shard": shard["path"],
                "trajectory_id": str(trajectory_ids[row_index]),
                "instance_id": str(instances[row_index]),
                "repo": str(repos[row_index]),
                "language": str(languages[row_index]),
                "row_index": row_index,
            }
            for cycle in parsed.cycles:
                for message_index, signals in zip(
                    cycle.observation_indexes,
                    cycle.execution,
                    strict=True,
                ):
                    content = parsed.messages[message_index].content or ""
                    n_observations += 1
                    observations_by_scaffold[config] += 1
                    formats = describe_exit_formats(content)
                    for name, codes in formats.items():
                        if codes:
                            exit_counts[config][name] += 1
                            bucket = exit_examples[config][name]
                            if len(bucket) < 8:
                                line = next(
                                    (
                                        item.strip()
                                        for item in content.splitlines()
                                        if item.strip()
                                    ),
                                    "",
                                )
                                bucket.append(
                                    {
                                        "parquet_shard": shard["path"],
                                        "trajectory_id": base["trajectory_id"],
                                        "observation_index": message_index,
                                        "codes": codes[:6],
                                        "first_nonempty_line": line[:180],
                                    }
                                )
                    families = signal_families(signals)
                    if is_hard_negative(content, signals):
                        families.append("hard_negative")
                    if signals.execution_status_unknown:
                        families.append("execution_status_unknown")
                    families.append("random_observation")
                    for family in families:
                        population[family] += 1
                        population_by_group[group_name][family] += 1
                        record = _record(base, message_index, content, family, signals, families)
                        if family in PRECISION_FAMILIES:
                            precision[family].consider(record)
                        elif family in OPENHANDS_ONLY_PRECISION_FAMILIES and config == "openhands":
                            precision[family].consider(record)
                        if family in DIAGNOSTIC_FAMILIES:
                            diagnostic_bucket(family, config, dataset_split).consider(record)
        shard_stats.append(
            {
                "config": config,
                "dataset_split": dataset_split,
                "path": shard["path"],
                "manifest_bytes": shard["bytes"],
                "remote_bytes": remote_bytes,
                "rows": table.num_rows,
                "tool_observations": n_observations,
                "columns": [
                    "instance_id",
                    "repo",
                    "language",
                    "trajectory_id",
                    "trajectory",
                ],
                "resolved_column_read": False,
            }
        )
        print(
            f"QUEUE_V2_SHARD_DONE {group_name} rows={table.num_rows} observations={n_observations}",
            flush=True,
        )
        del table

    rows: list[dict] = []
    precision_meta = {}
    for family, reservoir in precision.items():
        label = _probability_label(reservoir.seen, len(reservoir.items))
        stratum = (
            f"precision/{family}/openhands_shards_only"
            if family in OPENHANDS_ONLY_PRECISION_FAMILIES
            else f"precision/{family}/eight_selected_shards"
        )
        precision_meta[family] = {
            "population": reservoir.seen,
            "sample": len(reservoir.items),
            "sampling_probability": label,
            "design": "simple random sample without replacement",
            "clopper_pearson_describes": (
                "precision among parser-predicted positives in the OpenHands "
                "selected shards only"
                if family in OPENHANDS_ONLY_PRECISION_FAMILIES
                else "precision among parser-predicted positives in the eight selected shards only"
            ),
            "full_snapshot_interval": False,
        }
        for item in reservoir.items:
            rows.append(
                {
                    **item,
                    "review_purpose": "precision_validation",
                    "sampling_stratum": stratum,
                    "sampling_probability": label,
                }
            )

    diagnostic_meta = {}
    grouped_diagnostic: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for (family, config, dataset_split), reservoir in diagnostic.items():
        grouped_diagnostic[family][f"{config}/{dataset_split}"].extend(reservoir.items)
        diagnostic_meta.setdefault(family, {"population": 0, "reservoir_kept": 0, "queued": 0})
        diagnostic_meta[family]["population"] += reservoir.seen
        diagnostic_meta[family]["reservoir_kept"] += len(reservoir.items)
    for family, by_group in grouped_diagnostic.items():
        for group_name, pool in by_group.items():
            rng = random.Random(OBSERVATION_SEED + 500 + sum(ord(char) for char in family + group_name))
            chosen = stratify(pool, DIAGNOSTIC_PER_GROUP, rng)
            diagnostic_meta[family]["queued"] += len(chosen)
            for item in chosen:
                rows.append(
                    {
                        **item,
                        "review_purpose": "diagnostic_coverage",
                        "sampling_stratum": f"diagnostic/{family}/{group_name}",
                        "sampling_probability": "not_applicable_stratified_diagnostic",
                    }
                )

    rows.sort(
        key=lambda item: (
            item["review_purpose"],
            item["signal_family"],
            item["config"],
            item["split"],
            item["language"],
            item["trajectory_id"],
            item["observation_index"],
        )
    )
    for index, item in enumerate(rows, start=1):
        item["sample_id"] = f"g075-{index:05d}"
        item["dataset_revision"] = PINNED_REVISION
        item["queue_version"] = QUEUE_VERSION
        item["queue_seed"] = OBSERVATION_SEED
        item["source_shard_selection_method"] = SELECTION_METHOD
        item["reviewer_label"] = ""
        item["reviewer_notes"] = ""
        item["student_verified"] = "false"
        if "resolved" in item:
            raise SystemExit("resolved leaked into a review row")

    destination = ROOT / "artifacts" / "manual_review" / "marker_review_queue_v2.csv"
    _refuse(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for item in rows:
            writer.writerow({field: item.get(field, "") for field in CSV_FIELDS})

    unique_observations = len({item["observation_key"] for item in rows})
    purpose_counts = Counter(item["review_purpose"] for item in rows)
    family_counts = Counter((item["review_purpose"], item["signal_family"]) for item in rows)
    minimum_n = minimum_n_for_perfect_precision(0.90, 0.05)
    manifest = {
        "generated_at_utc": _now(),
        "revision": sha,
        "queue_version": QUEUE_VERSION,
        "queue_seed": OBSERVATION_SEED,
        "shard_selection_seed": SUPPLEMENTAL_SHARD_SEED,
        "source_shard_selection_method": SELECTION_METHOD,
        "historical_queue_overwritten": False,
        "rows": len(rows),
        "unique_observations": unique_observations,
        "rows_by_purpose": dict(purpose_counts),
        "rows_by_purpose_and_family": {
            f"{purpose}|{family}": count for (purpose, family), count in sorted(family_counts.items())
        },
        "student_verified": False,
        "reviewer_labels_filled": 0,
        "resolved_column_in_queue": False,
        "feature_freeze": "NOT FROZEN",
        "models_fit": False,
        "full_snapshot_precision_claim": False,
        "diagnostic_clopper_pearson_valid": False,
        "diagnostic_interval_reason": (
            "Diagnostic rows are a language-stratified subset of a reservoir. "
            "An ordinary binomial or Clopper-Pearson interval does not describe that design."
        ),
        "precision_clopper_pearson_valid_for_selected_shard_population": True,
        "precision_clopper_pearson_valid_for_full_84_shard_snapshot": False,
        "precision_interval_scope": (
            "A later one-sided 95% Clopper-Pearson interval is valid only for "
            "the finite population of parser-predicted positives in the eight "
            "named shards, and only for rows whose review_purpose is "
            "precision_validation. It does not by itself estimate the other 76 shards."
        ),
        "minimum_true_positives_if_no_false_positives_for_0.90": minimum_n,
        "lower_bound_if_50_of_50_true": clopper_pearson_lower(50, 50),
        "precision": precision_meta,
        "diagnostic": diagnostic_meta,
        "population_candidates_on_selected_shards": dict(population),
        "population_candidates_by_group": {
            key: dict(value) for key, value in sorted(population_by_group.items())
        },
        "shards": shard_stats,
        "download_bytes_manifest": download_bytes,
        "additional_shards": 8,
        "elapsed_seconds": round(time.time() - started, 1),
        "manual_validation": "MANUAL VALIDATION PENDING",
    }
    manifest_path = ROOT / "artifacts" / "manifests" / "manual_review_queue_v2_manifest.json"
    _refuse(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    exit_path = ROOT / "logs" / "gate075" / "swe_exit_format_counts.json"
    exit_path.parent.mkdir(parents=True, exist_ok=True)
    exit_payload = {
        "generated_at_utc": _now(),
        "revision": sha,
        "scope": "eight supplemental shards only; not the other 76 shards",
        "observations_by_scaffold": dict(observations_by_scaffold),
        "observations_with_pattern": {key: dict(value) for key, value in exit_counts.items()},
        "examples": exit_examples,
        "tool_exit_nonzero_not_inferred_from_these_patterns": True,
    }
    exit_path.write_text(json.dumps(exit_payload, indent=2), encoding="utf-8")
    print(
        f"QUEUE_V2_DONE rows={len(rows)} unique={unique_observations} "
        f"elapsed={manifest['elapsed_seconds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
