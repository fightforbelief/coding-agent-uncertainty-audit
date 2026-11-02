"""Full-corpus v3 census. Writes a new run directory and does not touch L2 outputs.

This module is the runner. Importing it does not scan a dataset.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis_v2")]

import pandas as pd
import pyarrow.parquet as pq

from census_integrity import (
    BLOCKED,
    compare_trajectory_ids,
    ids_match,
    parser_v2_failure_trigger,
    render_report,
    render_trigger,
    require_data_root,
    stratified_sample,
    v3_failure_trigger,
    verify_dataset_revision,
)
from census_metrics import census_flags
from p8audit.schema import SHARD_GROUPS
from traj_adapter import iter_calls

OUT = ROOT / "analysis_v2"
RUNS = OUT / "census_runs"
MANIFEST = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
META_COLUMNS = [
    "trajectory_id",
    "issue_id",
    "repo",
    "scaffold",
    "model",
    "resolved",
    "split",
    "n_rounds_total",
    "parser_v2_strict_fail_r5",
]
FLAG_KEYS = list(census_flags([]).keys())


def _shard_group(path: str) -> tuple[str, str] | None:
    text = path.replace("\\", "/")
    for prefix, group in SHARD_GROUPS.items():
        if text.startswith(prefix + "/") or f"/{prefix}/" in f"/{text}":
            return group
    return None


def _census_shard(path_text: str, config: str) -> tuple[list[dict], list[dict]]:
    table = pq.ParquetFile(path_text).read(columns=["trajectory_id", "trajectory"])
    rows: list[dict] = []
    errors: list[dict] = []
    for trajectory_id, trajectory in zip(
        table.column("trajectory_id").to_pylist(),
        table.column("trajectory").to_pylist(),
        strict=True,
    ):
        try:
            flags = census_flags(iter_calls({"config": config, "trajectory": trajectory}))
            flags["census_parse_ok"] = True
        except Exception as exc:
            flags = {key: None for key in FLAG_KEYS}
            flags["census_parse_ok"] = False
            errors.append(
                {
                    "trajectory_id": str(trajectory_id),
                    "shard": path_text,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            )
        flags["trajectory_id"] = str(trajectory_id)
        flags["shard"] = path_text
        rows.append(flags)
    return rows, errors


def _open_run() -> tuple[Path, Path]:
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final = RUNS / f"v3_census_{stamp}"
    if final.exists():
        raise SystemExit(f"refusing to overwrite {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=RUNS))
    return staging, final


def _publish(staging: Path, final: Path) -> None:
    if final.exists():
        raise SystemExit(f"refusing to overwrite {final}")
    staging.rename(final)


def _write_status(directory: Path, status: str, detail: dict) -> None:
    payload = {"status": status, **detail}
    (directory / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_sampled_calls(chosen: pd.DataFrame) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for (shard, scaffold), group in chosen.groupby(["shard", "scaffold"], sort=True):
        wanted = set(group["trajectory_id"].astype(str))
        table = pq.ParquetFile(str(shard)).read(columns=["trajectory_id", "trajectory"])
        for trajectory_id, trajectory in zip(
            table.column("trajectory_id").to_pylist(),
            table.column("trajectory").to_pylist(),
            strict=True,
        ):
            if str(trajectory_id) not in wanted:
                continue
            found[str(trajectory_id)] = iter_calls({"config": str(scaffold), "trajectory": trajectory})
            found[str(trajectory_id) + "::trajectory"] = trajectory
    return found


def _disagreement_markdown(frame: pd.DataFrame) -> str:
    sections = [
        "# Disagreement sample",
        "",
        "This is a diagnostic sample, not a recall estimate. Seed `20261102`.",
        "Parser-positive rows use the frozen five-completed-round prefix and `classify_execution_v2`.",
        "V3-positive rows use the first high-confidence failed test call at round <= 5.",
        "An unresolved trigger is not replaced with another observation.",
        "",
    ]
    specs = (
        ("v3_test_fail_r5=1 and parser_v2=0", (frame["v3_test_fail_r5"] == 1) & (frame["parser_v2_strict_fail_r5"] == 0), 30, "v3"),
        ("parser_v2=1 and v3_test_fail_r5=0", (frame["parser_v2_strict_fail_r5"] == 1) & (frame["v3_test_fail_r5"] == 0), 10, "parser"),
    )
    for title, mask, limit, kind in specs:
        pool = frame.loc[mask]
        chosen, strata = stratified_sample(pool, limit, 20261102)
        sections.append(f"## {title}")
        sections.append("")
        sections.append(f"Available rows: {len(pool)}. Sampled rows: {len(chosen)}.")
        sections.append("")
        sections.append("| scaffold | model | population | sampled |")
        sections.append("|---|---|---:|---:|")
        for item in strata:
            sections.append(
                f"| {item['scaffold']} | {item['model']} | {item['population']} | {item['sampled']} |"
            )
        sections.append("")
        cached = _load_sampled_calls(chosen) if len(chosen) else {}
        for row in chosen.itertuples(index=False):
            trajectory_id = str(row.trajectory_id)
            calls = cached.get(trajectory_id, [])
            if kind == "v3":
                trigger = v3_failure_trigger(calls)
            else:
                trigger = parser_v2_failure_trigger(cached.get(trajectory_id + "::trajectory"), str(row.scaffold))
            sections.append(f"### {trajectory_id}")
            sections.append("")
            sections.append(f"shard: {row.shard}")
            sections.append(f"scaffold: {row.scaffold}")
            sections.append(f"model: {row.model}")
            sections.extend(render_trigger(trigger))
    return "\n".join(sections)


def main() -> None:
    data_root = require_data_root(os.environ.get("P8_DATA_ROOT"), repository_root=ROOT)
    try:
        inventory = verify_dataset_revision(MANIFEST, data_root)
    except SystemExit as exc:
        if str(exc) == BLOCKED or (exc.args and exc.args[0] == BLOCKED):
            raise SystemExit(BLOCKED) from exc
        raise
    meta_path = OUT / "meta.parquet"
    if not meta_path.is_file():
        raise SystemExit("analysis_v2/meta.parquet is missing. Census was not run.")
    staging, final = _open_run()
    try:
        rows: list[dict] = []
        errors: list[dict] = []
        workers = max(1, min(8, os.cpu_count() or 1))
        jobs = []
        for item in inventory:
            group = _shard_group(item["path"])
            if group is None:
                raise SystemExit(f"manifest path is outside the pinned layout: {item['path']}")
            jobs.append((str(item["local"]), group[0]))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_census_shard, path, config) for path, config in jobs]
            for future in futures:
                shard_rows, shard_errors = future.result()
                rows.extend(shard_rows)
                errors.extend(shard_errors)
        census = pd.DataFrame(rows)
        meta = pd.read_parquet(meta_path)
        missing_columns = [column for column in META_COLUMNS if column not in meta.columns]
        if missing_columns:
            _write_status(staging, "FAILED", {"reason": "meta is missing columns", "columns": missing_columns})
            _publish(staging, final)
            raise SystemExit(f"meta is missing columns: {missing_columns}")
        comparison = compare_trajectory_ids(meta, census)
        (staging / "id_comparison.json").write_text(
            json.dumps(
                {
                    "meta_rows": comparison["meta_rows"],
                    "census_rows": comparison["census_rows"],
                    "meta_duplicate_ids": comparison["meta_duplicate_ids"],
                    "census_duplicate_ids": comparison["census_duplicate_ids"],
                    "n_missing_from_census": len(comparison["missing_from_census"]),
                    "n_extra_in_census": len(comparison["extra_in_census"]),
                    "missing_from_census": comparison["missing_from_census"][:50],
                    "extra_in_census": comparison["extra_in_census"][:50],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not ids_match(comparison):
            _write_status(staging, "FAILED", {"reason": "trajectory id sets do not match"})
            _publish(staging, final)
            raise SystemExit("trajectory id sets do not match. Status FAILED. Census parquet was not published.")
        merged = census.merge(meta[META_COLUMNS], on="trajectory_id", how="inner", validate="one_to_one")
        report, measurement_status = render_report(merged, len(errors))
        status = measurement_status
        (staging / "census_errors.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in errors),
            encoding="utf-8",
        )
        (staging / "01_census.md").write_text(report, encoding="utf-8")
        (staging / "census_disagreements.md").write_text(_disagreement_markdown(merged), encoding="utf-8")
        merged.to_parquet(staging / "census.parquet", index=False)
        _write_status(
            staging,
            status,
            {
                "census_parse_errors": len(errors),
                "rows": int(len(merged)),
                "full_corpus_measurement": status == "COMPLETE",
            },
        )
        _publish(staging, final)
        print(f"CENSUS_STATUS {status} ROWS {len(merged)} ERRORS {len(errors)} DIR {final}")
    except Exception:
        if staging.exists() and not final.exists():
            _write_status(staging, "FAILED", {"reason": "runner exception"})
            try:
                _publish(staging, final)
            except OSError:
                pass
        raise


if __name__ == "__main__":
    main()
