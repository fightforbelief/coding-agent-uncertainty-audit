"""Build the Gate 0.5 marker-review queue from the four already audited shards.

Re-streams only the first parquet of each pinned config/split. Those shards
were read in Gate 0, and the tool observations were not saved. This script
does not write ``parse_sample.json``, does not read ``resolved``, and does
not open the other 80 shards.
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

from p8audit.execution_signals import classify_execution
from p8audit.markers import mark_observation
from p8audit.review_queue import (
    ALL_FAMILIES,
    Reservoir,
    context_excerpt,
    is_hard_negative,
    per_group_caps,
    signal_families,
    stratify,
)
from p8audit.review_stats import clopper_pearson_lower, minimum_n_for_perfect_precision
from p8audit.schema import DATASET_ID, PINNED_REVISION
from p8audit.trajectory_parser import parse_trajectory

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261026
PER_GROUP = 16
PER_LANGUAGE = 40
FIRST_SHARDS = (
    ("openhands", "minimax_m25", "data/minimax_m25_openhands_trajectories/train-00000-of-00020.parquet"),
    ("openhands", "qwen35_122b", "data/qwen35_openhands_trajectories/train-00000-of-00023.parquet"),
    ("sweagent", "minimax_m25", "data/minimax_m25_sweagent_trajectories/train-00000-of-00023.parquet"),
    ("sweagent", "qwen35_122b", "data/qwen35_sweagent_trajectories/train-00000-of-00018.parquet"),
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
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_revision() -> str:
    info = HfApi().dataset_info(DATASET_ID, revision=PINNED_REVISION)
    sha = getattr(info, "sha", None)
    if sha != PINNED_REVISION:
        raise SystemExit(f"pinned revision check failed: api sha={sha}")
    return sha


def _manifest_bytes() -> dict[str, int]:
    path = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("revision") != PINNED_REVISION:
        raise SystemExit("pinned_parquet_files.json is not the pinned revision")
    return {item["path"]: int(item["bytes"]) for item in payload["files"]}


def _historical_counts() -> dict[str, dict]:
    path = ROOT / "artifacts" / "manifests" / "parse_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("revision") != PINNED_REVISION:
        raise SystemExit("parse_sample.json revision does not match the pin")
    return {group["path"]: group["marker_counts"] for group in payload["groups"]}


def _record(row: dict, message_index: int, content: str, family: str, signals) -> dict:
    return {
        "config": row["config"],
        "split": row["split"],
        "parquet_shard": row["shard"],
        "trajectory_id": row["trajectory_id"],
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "language": row["language"],
        "row_index": row["row_index"],
        "observation_index": message_index,
        "signal_family": family,
        "parser_predicted_label": signals.predicted_label(),
        "matched_pattern": ",".join(signals.reasons),
        "context_excerpt": context_excerpt(content, signals.reasons),
        "content_sha1_16": hashlib.sha1((content or "").encode("utf-8")).hexdigest()[:16],
        "n_chars": len(content or ""),
    }


def _scan_shard(fs: HfFileSystem, config: str, split: str, path: str, manifest_bytes: int) -> dict:
    remote = f"datasets/{DATASET_ID}@{PINNED_REVISION}/{path}"
    info = fs.info(remote)
    remote_bytes = int(info.get("size") or 0)
    if manifest_bytes and remote_bytes and remote_bytes != manifest_bytes:
        raise SystemExit(
            f"shard size mismatch for {path}: manifest={manifest_bytes} remote={remote_bytes}"
        )
    print(f"QUEUE_SHARD {config}/{split} bytes={remote_bytes or manifest_bytes}", flush=True)
    last_error: Exception | None = None
    table = None
    for attempt in range(3):
        try:
            with fs.open(remote, "rb") as handle:
                table = pq.ParquetFile(handle).read(
                    columns=["instance_id", "repo", "language", "trajectory_id", "trajectory"]
                )
            last_error = None
            break
        except Exception as exc:  # Hugging Face drops long streams.
            last_error = exc
            print(f"RETRY {attempt + 1} {config}/{split} {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    if table is None:
        raise RuntimeError(f"could not read {path}") from last_error

    salt = {"openhands": 0, "sweagent": 2}[config] + {"minimax_m25": 0, "qwen35_122b": 1}[split]
    rng = random.Random(SEED + salt)
    reservoirs: dict[tuple[str, str], Reservoir] = {}
    seen: Counter[str] = Counter()
    v1_counts: Counter[str] = Counter()
    v2_counts: Counter[str] = Counter()
    language_seen: dict[str, Counter[str]] = defaultdict(Counter)
    n_observations = 0
    trajectories = table.column("trajectory")
    instances = table.column("instance_id").to_pylist()
    repos = table.column("repo").to_pylist()
    languages = table.column("language").to_pylist()
    trajectory_ids = table.column("trajectory_id").to_pylist()

    def reservoir(family: str, language: str) -> Reservoir:
        key = (family, language)
        if key not in reservoirs:
            reservoirs[key] = Reservoir(PER_LANGUAGE, rng)
        return reservoirs[key]

    for row_index in range(table.num_rows):
        parsed = parse_trajectory(trajectories[row_index].as_py())
        language = str(languages[row_index])
        base = {
            "config": config,
            "split": split,
            "shard": path,
            "trajectory_id": str(trajectory_ids[row_index]),
            "instance_id": str(instances[row_index]),
            "repo": str(repos[row_index]),
            "language": language,
            "row_index": row_index,
        }
        for cycle in parsed.cycles:
            for message_index, historical, signals in zip(
                cycle.observation_indexes,
                cycle.markers,
                cycle.execution,
                strict=True,
            ):
                content = parsed.messages[message_index].content
                n_observations += 1
                if historical.explicit_error:
                    v1_counts["explicit_error"] += 1
                if historical.explicit_test_failure:
                    v1_counts["explicit_test_failure"] += 1
                if historical.explicit_test_pass:
                    v1_counts["explicit_test_pass"] += 1
                if signals.tool_exit_nonzero:
                    v2_counts["tool_exit_nonzero"] += 1
                if signals.explicit_exception:
                    v2_counts["explicit_exception"] += 1
                if signals.explicit_test_failure:
                    v2_counts["explicit_test_failure"] += 1
                if signals.explicit_test_pass:
                    v2_counts["explicit_test_pass"] += 1
                if signals.interrupt_or_timeout:
                    v2_counts["interrupt_or_timeout"] += 1
                if signals.execution_status_unknown:
                    v2_counts["execution_status_unknown"] += 1
                if signals.exploratory_test_format:
                    v2_counts["exploratory_test_format"] += 1
                families = signal_families(signals)
                if is_hard_negative(content, signals):
                    families.append("hard_negative")
                families.append("random_observation")
                for family in families:
                    seen[family] += 1
                    language_seen[family][language] += 1
                    reservoir(family, language).consider(
                        _record(base, message_index, content, family, signals)
                    )
    print(
        f"QUEUE_SHARD_DONE {config}/{split} rows={table.num_rows} observations={n_observations}",
        flush=True,
    )
    del table
    stored: dict[str, list[dict]] = defaultdict(list)
    for (family, _language), bucket in reservoirs.items():
        stored[family].extend(bucket.items)
    return {
        "config": config,
        "split": split,
        "path": path,
        "bytes": remote_bytes or manifest_bytes,
        "rows": len(instances),
        "tool_observations": n_observations,
        "v1_marker_counts": dict(v1_counts),
        "v2_signal_counts": dict(v2_counts),
        "family_seen": dict(seen),
        "family_language_seen": {family: dict(counts) for family, counts in language_seen.items()},
        "stored": stored,
    }


def _select(scans: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    for family_index, family in enumerate(ALL_FAMILIES):
        seen = [scan["family_seen"].get(family, 0) for scan in scans]
        caps = per_group_caps(seen, minimum_total=50, base_per_group=PER_GROUP)
        for group_index, scan in enumerate(scans):
            pool = scan["stored"].get(family, [])
            rng = random.Random(SEED + 1000 + family_index * 10 + group_index)
            chosen.extend(stratify(pool, caps[group_index], rng))
    chosen.sort(
        key=lambda item: (
            item["signal_family"],
            item["config"],
            item["split"],
            item["language"],
            item["trajectory_id"],
            item["observation_index"],
        )
    )
    rows = []
    for index, item in enumerate(chosen, start=1):
        rows.append(
            {
                "sample_id": f"g05-{index:05d}",
                "dataset_revision": PINNED_REVISION,
                "reviewer_label": "",
                "reviewer_notes": "",
                "student_verified": "false",
                "queue_seed": SEED,
                **item,
            }
        )
    return rows


def _diff(scans: list[dict], historical: dict[str, dict]) -> dict:
    report = {}
    for scan in scans:
        old = historical[scan["path"]]
        new = scan["v1_marker_counts"]
        report[scan["path"]] = {
            "historical_gate0": old,
            "v1_rescan": new,
            "delta_explicit_error": new.get("explicit_error", 0) - old.get("explicit_error", 0),
            "delta_explicit_test_failure": new.get("explicit_test_failure", 0) - old.get("explicit_test_failure", 0),
            "delta_explicit_test_pass": new.get("explicit_test_pass", 0) - old.get("explicit_test_pass", 0),
        }
    return report


def _write_diff_markdown(diff: dict, destination: Path) -> None:
    lines = [
        "# Marker count difference",
        "",
        "Historical counts in `artifacts/manifests/parse_sample.json` were not overwritten.",
        "The v1 rescan uses the current `mark_observation` function, including the exit-130 regex fix.",
        "v2 counts are separated signals and are not expected to equal v1 `explicit_error`.",
        "",
        "| Shard | Historical error | v1 rescan error | Delta error | Delta test failure | Delta test pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for path, item in diff.items():
        old = item["historical_gate0"]
        new = item["v1_rescan"]
        lines.append(
            f"| `{path}` | {old.get('explicit_error', 0)} | {new.get('explicit_error', 0)} "
            f"| {item['delta_explicit_error']} | {item['delta_explicit_test_failure']} "
            f"| {item['delta_explicit_test_pass']} |"
        )
    lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    sha = _assert_revision()
    sizes = _manifest_bytes()
    historical = _historical_counts()
    fs = HfFileSystem()
    scans = []
    for config, split, path in FIRST_SHARDS:
        if path not in sizes:
            raise SystemExit(f"first shard is absent from the pinned manifest: {path}")
        scans.append(_scan_shard(fs, config, split, path, sizes[path]))
    rows = _select(scans)
    review_dir = ROOT / "artifacts" / "manual_review"
    manifest_dir = ROOT / "artifacts" / "manifests"
    report_dir = ROOT / "artifacts" / "reports"
    review_dir.mkdir(parents=True, exist_ok=True)
    csv_path = review_dir / "marker_review_queue.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    family_seen: dict[str, int] = Counter()
    family_queued: dict[str, int] = Counter()
    for scan in scans:
        family_seen.update(scan["family_seen"])
    for row in rows:
        family_queued[row["signal_family"]] += 1
    minimum_n = minimum_n_for_perfect_precision(0.90, 0.05)
    diff = _diff(scans, historical)
    _write_diff_markdown(diff, report_dir / "MARKER_V1_V2_COUNT_DIFF.md")
    counts_payload = {
        "generated_at_utc": _now(),
        "revision": sha,
        "status": "PROVISIONAL",
        "manual_validation": "MANUAL VALIDATION PENDING",
        "sample": "re-stream of the four Gate 0 first shards only",
        "additional_shards": 0,
        "resolved_column_read": False,
        "models_fit": False,
        "groups": [
            {
                "config": scan["config"],
                "split": scan["split"],
                "path": scan["path"],
                "bytes": scan["bytes"],
                "rows": scan["rows"],
                "tool_observations": scan["tool_observations"],
                "v1_marker_counts": scan["v1_marker_counts"],
                "v2_signal_counts": scan["v2_signal_counts"],
                "family_seen": scan["family_seen"],
            }
            for scan in scans
        ],
        "v1_rescan_vs_historical": diff,
    }
    (manifest_dir / "marker_v2_sample_counts.json").write_text(
        json.dumps(counts_payload, indent=2),
        encoding="utf-8",
    )
    keys = [
        (row["content_sha1_16"], row["parquet_shard"], row["observation_index"])
        for row in rows
    ]
    overlap = len(keys) - len(set(keys))
    manifest = {
        "generated_at_utc": _now(),
        "revision": sha,
        "queue_seed": SEED,
        "per_group_cap": PER_GROUP,
        "per_language_reservoir": PER_LANGUAGE,
        "rows_queued": len(rows),
        "student_verified": False,
        "reviewer_labels_filled": 0,
        "manual_validation": "MANUAL VALIDATION PENDING",
        "prevalence_estimable_from_queue": False,
        "population_recall_estimable_from_queue": False,
        "freeze_threshold_one_sided_95_lower_bound": 0.90,
        "minimum_true_positives_if_no_false_positives": minimum_n,
        "lower_bound_if_7_of_7_true": clopper_pearson_lower(7, 7),
        "lower_bound_if_3_of_3_true": clopper_pearson_lower(3, 3),
        "lower_bound_if_50_of_50_true": clopper_pearson_lower(50, 50),
        "feature_freeze": "NOT FROZEN",
        "full_corpus_downloaded": False,
        "additional_shards_beyond_gate0_first_shards": 0,
        "models_fit": False,
        "historical_parse_sample_untouched": True,
        "duplicate_observation_slots_across_families": overlap,
        "family_seen_on_10k_trajectories": dict(family_seen),
        "family_queued": dict(family_queued),
        "shards": [
            {
                "config": scan["config"],
                "split": scan["split"],
                "path": scan["path"],
                "bytes": scan["bytes"],
                "rows": scan["rows"],
                "purpose": "Re-stream of the Gate 0 first shard. Tool text was not retained after Gate 0.",
            }
            for scan in scans
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (manifest_dir / "manual_review_queue_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"QUEUE_DONE rows={len(rows)} elapsed={manifest['elapsed_seconds']}", flush=True)


if __name__ == "__main__":
    main()
