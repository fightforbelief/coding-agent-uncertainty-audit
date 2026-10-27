"""Parser v2 validation on eight previously unread pinned shards.

Does not read ``resolved``. Does not open the other 76 trajectory bodies.
Does not write historical queues.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from p8audit.execution_signals_v2 import PARSER_VERSION, classify_execution_v2
from p8audit.fresh_shards import (
    DIAGNOSTIC_SAMPLE_SIZE,
    FRESH_SHARD_SEED,
    GATE075_SUPPLEMENTAL_SHARDS,
    OBSERVATION_SEED,
    PRECISION_SAMPLE_SIZE,
    RANDOM_SAMPLE_SIZE,
    select_fresh_shards,
)
from p8audit.review_queue import Reservoir, context_excerpt
from p8audit.sampling import SUPPLEMENTAL_SHARD_SEED, select_supplemental_shards
from p8audit.schema import DATASET_ID, PINNED_REVISION
from p8audit.trajectory_parser import parse_trajectory
from p8audit.validation_package import write_validation_package

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
DESTINATION = ROOT / "artifacts" / "manual_review" / "gate09_parser_v2_validation"
FIRST_SHARDS = {
    "data/minimax_m25_openhands_trajectories/train-00000-of-00020.parquet",
    "data/qwen35_openhands_trajectories/train-00000-of-00023.parquet",
    "data/minimax_m25_sweagent_trajectories/train-00000-of-00023.parquet",
    "data/qwen35_sweagent_trajectories/train-00000-of-00018.parquet",
}
PRECISION = ("explicit_exception", "explicit_test_failure", "tool_exit_nonzero")
DIAGNOSTIC = (
    "build_or_compile_failure",
    "test_setup_or_collection_failure",
    "package_level_failure_unresolved",
    "missed_test_format",
    "interrupt_mention_not_event",
    "discordant_exit_status_and_test_result",
    "execution_status_unknown",
)
READ_COLUMNS = ["instance_id", "repo", "language", "trajectory_id", "trajectory"]


def _records() -> list[dict]:
    payload = json.loads(PINNED.read_text(encoding="utf-8"))
    files = payload["files"] if isinstance(payload, dict) and "files" in payload else payload
    records = []
    for item in files:
        records.append(
            {
                "path": item["path"],
                "config": item["config"],
                "dataset_split": item["split"],
                "bytes": int(item["bytes"]),
            }
        )
    return records


def _families(signals) -> list[str]:
    found = []
    if signals.explicit_exception:
        found.append("explicit_exception")
    if signals.explicit_test_failure:
        found.append("explicit_test_failure")
    if signals.tool_exit_nonzero:
        found.append("tool_exit_nonzero")
    if signals.build_or_compile_failure:
        found.append("build_or_compile_failure")
    if signals.test_setup_or_collection_failure:
        found.append("test_setup_or_collection_failure")
    if signals.package_level_failure_unresolved:
        found.append("package_level_failure_unresolved")
    if signals.exploratory_test_format and not signals.explicit_test_failure and not signals.explicit_test_pass:
        found.append("missed_test_format")
    if signals.interrupt_mention_not_event:
        found.append("interrupt_mention_not_event")
    if signals.discordant_exit_status_and_test_result:
        found.append("discordant_exit_status_and_test_result")
    if signals.execution_status_unknown:
        found.append("execution_status_unknown")
    found.append("random_observation")
    return found


def _assert_revision() -> None:
    info = HfApi().dataset_info(DATASET_ID, revision=PINNED_REVISION)
    sha = getattr(info, "sha", None)
    if sha != PINNED_REVISION:
        raise SystemExit(f"pinned revision check failed: api sha={sha}")


def _read_table(fs: HfFileSystem, path: str, manifest_bytes: int):
    remote = f"datasets/{DATASET_ID}@{PINNED_REVISION}/{path}"
    info = fs.info(remote)
    remote_bytes = int(info.get("size") or 0)
    if manifest_bytes and remote_bytes and remote_bytes != manifest_bytes:
        raise SystemExit(f"shard size mismatch for {path}: manifest={manifest_bytes} remote={remote_bytes}")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with fs.open(remote, "rb") as handle:
                table = pq.ParquetFile(handle).read(columns=READ_COLUMNS)
            if "resolved" in table.column_names:
                raise SystemExit("resolved column was read")
            return table, remote_bytes or manifest_bytes
        except Exception as exc:
            last_error = exc
            print(f"RETRY {attempt + 1} {path} {type(exc).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not read {path}") from last_error


def _blank_row(base: dict, message_index: int, content: str, family: str, signals, families: list[str]) -> dict:
    others = [name for name in families if name not in {family, "random_observation"}]
    excerpt = context_excerpt(content, signals.reasons)
    return {
        "sample_id": "",
        "parser_version": PARSER_VERSION,
        "dataset_revision": PINNED_REVISION,
        "config": base["config"],
        "split": base["dataset_split"],
        "parquet_shard": base["shard"],
        "trajectory_id": base["trajectory_id"],
        "instance_id": base["instance_id"],
        "repo": base["repo"],
        "language": base["language"],
        "observation_index": message_index,
        "observation_key": f"{base['shard']}|{base['trajectory_id']}|{message_index}",
        "signal_family": family,
        "parser_predicted_label": signals.predicted_label(),
        "matched_pattern": ",".join(signals.reasons),
        "context_excerpt": excerpt,
        "excerpt_truncated": "true" if excerpt.endswith("[TRUNCATED]") else "false",
        "n_chars": len(content or ""),
        "review_purpose": "",
        "sampling_probability": "",
        "source_shard_selection_method": base["selection_method"],
        "selection_seed": FRESH_SHARD_SEED,
        "duplicate_signal_families": "|".join(others),
        "reviewer_label": "",
        "reviewer_notes": "",
        "student_verified": "false",
        "adjudication": "pending",
    }


def _number(rows: list[dict], start: int) -> int:
    rows.sort(key=lambda row: (row["parquet_shard"], row["trajectory_id"], row["observation_index"], row["signal_family"]))
    for offset, row in enumerate(rows):
        row["sample_id"] = f"g09-{start + offset:05d}"
    return start + len(rows)


def main() -> None:
    started = time.time()
    records = _records()
    supplemental = select_supplemental_shards(
        records,
        seed=SUPPLEMENTAL_SHARD_SEED,
        per_group=2,
        excluded_paths=FIRST_SHARDS,
    )
    supplemental_paths = tuple(item["path"] for item in supplemental)
    if supplemental_paths != GATE075_SUPPLEMENTAL_SHARDS:
        raise SystemExit("Gate 0.75 shard recomputation does not match the recorded eight paths")
    selected = select_fresh_shards(records, seed=FRESH_SHARD_SEED)
    if len(selected) != 8:
        raise SystemExit(f"expected 8 fresh shards, found {len(selected)}")
    fresh_paths = {item["path"] for item in selected}
    if fresh_paths & FIRST_SHARDS or fresh_paths & set(GATE075_SUPPLEMENTAL_SHARDS):
        raise SystemExit("fresh draw included an already read shard")
    _assert_revision()
    print("FRESH_SHARDS " + ",".join(item["path"] for item in selected), flush=True)

    precision = {
        family: Reservoir(PRECISION_SAMPLE_SIZE, random.Random(OBSERVATION_SEED + index))
        for index, family in enumerate(PRECISION)
    }
    diagnostic = {
        family: Reservoir(DIAGNOSTIC_SAMPLE_SIZE, random.Random(OBSERVATION_SEED + 100 + index))
        for index, family in enumerate(DIAGNOSTIC)
    }
    random_rows = Reservoir(RANDOM_SAMPLE_SIZE, random.Random(OBSERVATION_SEED + 200))
    population: Counter[str] = Counter()
    fs = HfFileSystem()
    shard_stats = []
    for shard in selected:
        table, remote_bytes = _read_table(fs, shard["path"], shard["bytes"])
        observations = 0
        trajectories = table.column("trajectory")
        instances = table.column("instance_id").to_pylist()
        repos = table.column("repo").to_pylist()
        languages = table.column("language").to_pylist()
        trajectory_ids = table.column("trajectory_id").to_pylist()
        for row_index in range(table.num_rows):
            parsed = parse_trajectory(trajectories[row_index].as_py())
            base = {
                "config": shard["config"],
                "dataset_split": shard["dataset_split"],
                "shard": shard["path"],
                "trajectory_id": str(trajectory_ids[row_index]),
                "instance_id": str(instances[row_index]),
                "repo": str(repos[row_index]),
                "language": str(languages[row_index]),
                "selection_method": shard["selection_method"],
            }
            for cycle in parsed.cycles:
                for message_index in cycle.observation_indexes:
                    content = parsed.messages[message_index].content or ""
                    observations += 1
                    signals = classify_execution_v2(content)
                    families = _families(signals)
                    for family in families:
                        population[family] += 1
                        record = _blank_row(base, message_index, content, family, signals, families)
                        if family in PRECISION and (family != "tool_exit_nonzero" or shard["config"] == "openhands"):
                            precision[family].consider(record)
                        elif family in DIAGNOSTIC:
                            diagnostic[family].consider(record)
                        elif family == "random_observation":
                            random_rows.consider(record)
        shard_stats.append(
            {
                "path": shard["path"],
                "config": shard["config"],
                "dataset_split": shard["dataset_split"],
                "manifest_bytes": shard["bytes"],
                "remote_bytes": remote_bytes,
                "rows": table.num_rows,
                "tool_observations": observations,
                "resolved_column_read": False,
            }
        )
        print(f"FRESH_SHARD_DONE {shard['path']} obs={observations}", flush=True)

    def _finish(bucket: Reservoir, purpose: str) -> list[dict]:
        rows = []
        seen = bucket.seen
        kept = len(bucket.items)
        probability = "0/0" if seen < 1 else f"{kept}/{seen} ({kept / seen:.10f})"
        for item in bucket.items:
            item["review_purpose"] = purpose
            item["sampling_probability"] = probability
            item["sampling_frame_observations"] = seen
            rows.append(item)
        return rows

    sheets_rows = {
        "explicit_exception_validation.csv": _finish(precision["explicit_exception"], "precision_validation"),
        "explicit_test_failure_validation.csv": _finish(precision["explicit_test_failure"], "precision_validation"),
        "openhands_exit_status_validation.csv": _finish(precision["tool_exit_nonzero"], "precision_validation"),
        "diagnostic_validation.csv": [],
        "random_observation_validation.csv": _finish(random_rows, "exploratory_error_analysis"),
    }
    for family in DIAGNOSTIC:
        sheets_rows["diagnostic_validation.csv"].extend(_finish(diagnostic[family], "diagnostic_coverage"))
    next_id = 1
    for name in (
        "explicit_exception_validation.csv",
        "explicit_test_failure_validation.csv",
        "openhands_exit_status_validation.csv",
        "diagnostic_validation.csv",
        "random_observation_validation.csv",
    ):
        next_id = _number(sheets_rows[name], next_id)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parser_version": PARSER_VERSION,
        "dataset_revision": PINNED_REVISION,
        "fresh_shard_seed": FRESH_SHARD_SEED,
        "observation_seed": OBSERVATION_SEED,
        "excluded_previously_read_shards": sorted(FIRST_SHARDS | set(GATE075_SUPPLEMENTAL_SHARDS)),
        "shards": shard_stats,
        "download_bytes_manifest": sum(item["manifest_bytes"] for item in shard_stats),
        "population_on_fresh_shards": dict(population),
        "sheet_counts": {name: len(rows) for name, rows in sheets_rows.items()},
        "precision_shortfall": {
            family: precision[family].seen < PRECISION_SAMPLE_SIZE
            for family in PRECISION
        },
        "labels_filled": False,
        "student_verified": False,
        "confidence_bounds_calculated": False,
        "models_fit": False,
        "delta_auprc_calculated": False,
        "full_snapshot_precision_claim": False,
        "inference_note": (
            "Each precision sheet is a simple random sample of parser-v2 predicted "
            "positives on these eight shards only. A one-sided 95% lower bound of "
            "0.90 remains the operational gate after adjudication. Ambiguous and "
            "insufficient-evidence rows are not confirmed detections. The binomial "
            "interval is not an exact design-based interval for the shard stage, "
            "and these eight shards do not prove precision on all 84 files. "
            "Bounds are not calculated before adjudication."
        ),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    write_validation_package(DESTINATION, sheets_rows, manifest)
    digest = hashlib.sha256(DESTINATION.joinpath("package_manifest.json").read_bytes()).hexdigest()
    print(
        "FRESH_VALIDATION_DONE"
        f" sheets={manifest['sheet_counts']}"
        f" bytes={manifest['download_bytes_manifest']}"
        f" manifest_sha256={digest}"
        f" elapsed={manifest['elapsed_seconds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
