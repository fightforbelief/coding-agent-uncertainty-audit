"""Run the V3.3 pilot on hanqing-win. Do not import this during local unit tests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

from v3_3_pilot import (  # noqa: E402
    DATASET_ID,
    PINNED_REVISION,
    columns_to_read,
    inspect_trajectory,
    sha256_file,
    verify_file,
)

HISTORICAL = {
    "src/p8audit/trajectory_parser.py": "af9f34a06fd0fe938506f65cec636001c55bef51030521258186facd999d9718",
    "src/p8audit/execution_signals_v2.py": "51761afbaf48ba02de3ea4fee0abb5bcc3cfecc79fe34d2f4ebe79557cd826bd",
    "src/p8audit/execution_signals.py": "2baa412bae170d6353f60feb7004db45dae99fe99d3425cd2e500b183b68c4cf",
    "src/p8audit/prefix_features.py": "981cbd8c74a2e34c87f9f1071d7f4abb4df88f542f50a2f8e353de67e479da09",
    "src/p8audit/schema.py": "62dd3a5273df627264da1ab6095971bd9c0a7d22daf8eba9a6385d1930deea10",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_modules(project: Path, code_manifest: dict) -> list[dict]:
    import p8audit.execution_signals as execution_signals
    import p8audit.execution_signals_v2 as execution_signals_v2
    import p8audit.prefix_features as prefix_features
    import p8audit.schema as schema
    import p8audit.trajectory_parser as trajectory_parser
    import traj_adapter
    import v3_3_pilot

    imported = {
        "src/p8audit/trajectory_parser.py": Path(trajectory_parser.__file__),
        "src/p8audit/execution_signals_v2.py": Path(execution_signals_v2.__file__),
        "src/p8audit/execution_signals.py": Path(execution_signals.__file__),
        "src/p8audit/prefix_features.py": Path(prefix_features.__file__),
        "src/p8audit/schema.py": Path(schema.__file__),
        "traj_adapter.py": Path(traj_adapter.__file__),
        "v3_3_pilot.py": Path(v3_3_pilot.__file__),
        "v3_3_run.py": Path(__file__),
    }
    expected = dict(HISTORICAL)
    expected.update(code_manifest["v3_files"])
    rows = []
    for name, path in imported.items():
        digest = sha256_file(path)
        project_resolved = project.resolve()
        inside_project = name.startswith("src/") and project_resolved in path.resolve().parents
        rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": digest,
                "expected_sha256": expected.get(name),
                "match": digest == expected.get(name),
                "historical_module_inside_original_project": inside_project if name.startswith("src/") else None,
            }
        )
    return rows


def confirm_revision() -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_ID, revision=PINNED_REVISION)
    observed = getattr(info, "sha", None)
    if observed != PINNED_REVISION:
        raise SystemExit(f"revision check failed: {observed}")
    return str(observed)


def remote_sizes(paths: list[str]) -> dict[str, int | None]:
    from huggingface_hub import HfApi

    api = HfApi()
    found: dict[str, int | None] = {}
    infos = api.get_paths_info(DATASET_ID, paths, repo_type="dataset", revision=PINNED_REVISION)
    for item in infos:
        found[str(item.path)] = getattr(item, "size", None)
    return found


def download_one(filename: str, data_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=DATASET_ID,
        filename=filename,
        repo_type="dataset",
        revision=PINNED_REVISION,
        local_dir=str(data_dir),
    )
    return Path(downloaded)


def _add(counter: Counter, values: dict[str, int] | None) -> None:
    if values:
        counter.update(values)


def parse_shards(selection: dict, data_dir: Path, run_dir: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    metas: list[dict] = []
    categories: Counter = Counter()
    json_keys: Counter = Counter()
    signals: Counter = Counter()
    reasons: Counter = Counter()
    tools: Counter = Counter()
    cycles: Counter = Counter()
    statuses: Counter = Counter()
    prefix_status: Counter = Counter()
    edges: list[dict] = []
    excerpts: list[dict] = []
    by_shard: list[dict] = []
    n_total = 0
    n_exception = 0
    n_valid = 0
    assistant = 0
    tool_calls = 0
    observations = 0
    malformed = 0
    for item in selection["selected"]:
        path = data_dir / item["path"]
        parquet = pq.ParquetFile(path)
        chosen = columns_to_read(parquet.schema_arrow.names)
        shard_rows = 0
        shard_signals: Counter = Counter()
        shard_reasons: Counter = Counter()
        for group_index in range(parquet.num_row_groups):
            print(f"ROW_GROUP {item['path']} {group_index}/{parquet.num_row_groups}", flush=True)
            table = parquet.read_row_group(group_index, columns=chosen)
            if any(name in table.column_names for name in ("resolved", "metadata", "reference_patch", "model_patch")):
                raise SystemExit("prohibited column was read")
            rows = table.to_pylist()
            del table
            for row in rows:
                shard_rows += 1
                n_total += 1
                try:
                    measured = inspect_trajectory(
                        scaffold=item["scaffold"],
                        model=item["model"],
                        shard_path=item["path"],
                        trajectory_id=row["trajectory_id"],
                        instance_id=row["instance_id"],
                        repo=row["repo"],
                        language=row["language"],
                        trajectory=row["trajectory"],
                    )
                except Exception:
                    n_exception += 1
                    statuses["exception"] += 1
                    prefix_status["not_checked"] += 1
                    metas.append(
                        {
                            "trajectory_id": str(row.get("trajectory_id")),
                            "instance_id": str(row.get("instance_id")),
                            "repo": str(row.get("repo")),
                            "language": None if row.get("language") is None else str(row.get("language")),
                            "scaffold": item["scaffold"],
                            "model": item["model"],
                            "shard_path": item["path"],
                            "n_rounds_total": None,
                            "parser_status": "exception",
                            "prefix_completed_rounds": None,
                            "prefix_explicit_test_failure_any": None,
                            "command_extraction_status": None,
                        }
                    )
                    if len(edges) < 500:
                        edges.append(
                            {
                                "kind": "exception",
                                "shard_path": item["path"],
                                "trajectory_id": str(row.get("trajectory_id")),
                                "detail": traceback.format_exc(limit=2)[-500:],
                            }
                        )
                    continue
                meta = measured["meta"]
                metas.append({key: meta.get(key) for key in (
                    "trajectory_id",
                    "instance_id",
                    "repo",
                    "language",
                    "scaffold",
                    "model",
                    "shard_path",
                    "n_rounds_total",
                    "parser_status",
                    "prefix_completed_rounds",
                    "prefix_explicit_test_failure_any",
                    "command_extraction_status",
                )})
                statuses[meta["parser_status"]] += 1
                if meta["parser_status"] == "exception":
                    n_exception += 1
                    prefix_status["not_checked"] += 1
                elif meta["parser_status"] in {"ok", "no_cycles"}:
                    n_valid += 1
                if not measured["error"] or meta["parser_status"] != "exception":
                    assistant += measured.get("n_assistant", 0)
                    tool_calls += measured.get("n_tool_calls", 0)
                    observations += measured.get("n_observations", 0)
                    malformed += measured.get("n_malformed", 0)
                    _add(categories, measured.get("categories"))
                    _add(json_keys, measured.get("json_keys"))
                    _add(signals, measured.get("signal_counts"))
                    _add(reasons, measured.get("reason_counts"))
                    _add(shard_signals, measured.get("signal_counts"))
                    _add(shard_reasons, measured.get("reason_counts"))
                    _add(cycles, measured.get("cycle_counts"))
                    tools.update(name for name in measured.get("tool_names", []) if name)
                    comparison = measured.get("comparison") or {}
                    prefix_status[comparison.get("status", "not_checked")] += 1
                    if comparison.get("status") == "mismatch" and sum(1 for edge in edges if edge["kind"] == "prefix_mismatch") < 5:
                        edges.append(
                            {
                                "kind": "prefix_mismatch",
                                "shard_path": item["path"],
                                "trajectory_id": meta["trajectory_id"],
                                "detail": json.dumps(
                                    {
                                        "round_indexes": (comparison.get("round_indexes") or [])[:20],
                                        "prefix_indexes": (comparison.get("prefix_indexes") or [])[:20],
                                        "end": comparison.get("end"),
                                    }
                                ),
                            }
                        )
                    for edge in measured.get("edges", []):
                        if len(edges) >= 500:
                            break
                        edges.append(
                            {
                                "kind": edge["kind"],
                                "shard_path": item["path"],
                                "trajectory_id": meta["trajectory_id"],
                                "detail": edge["detail"],
                            }
                        )
                    for excerpt in measured.get("excerpts", []):
                        if len(excerpts) < 100:
                            excerpts.append(excerpt)
            del rows
            print(f"PARSED {item['path']} group={group_index} rows={shard_rows}", flush=True)
        expected_rows = parquet.metadata.num_rows
        if shard_rows != expected_rows:
            raise SystemExit(f"row accounting failed for {item['path']}: {shard_rows} != {expected_rows}")
        by_shard.append(
            {
                "path": item["path"],
                "scaffold": item["scaffold"],
                "model": item["model"],
                "rows": shard_rows,
                "signals": dict(shard_signals),
                "failure_reasons": dict(shard_reasons),
            }
        )
    table = pa.Table.from_pylist(metas)
    pq.write_table(table, run_dir / "pilot_meta.parquet")
    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_error": n_total - n_valid,
        "n_exception": n_exception,
        "parser_status": dict(statuses),
        "n_assistant_messages": assistant,
        "n_tool_calls": tool_calls,
        "n_tool_observations": observations,
        "n_malformed_messages": malformed,
        "cycle_status": dict(cycles),
        "prefix_comparison": dict(prefix_status),
        "command_categories": dict(categories),
        "execution_json_key_sets": json_keys.most_common(40),
        "tool_names": tools.most_common(40),
        "strict_signals": dict(signals),
        "strict_failure_reasons": dict(reasons),
        "excerpt_rows_stored": len(excerpts),
        "edge_rows_stored": len(edges),
        "shards": by_shard,
        "edges": edges,
        "excerpts": excerpts,
        "meta_rows": table.num_rows,
        "meta_columns": table.column_names,
        "duplicate_trajectory_ids": len(metas) - len({row["trajectory_id"] for row in metas}),
    }


def write_tables(run_dir: Path, summary: dict, selection: dict) -> None:
    with (run_dir / "command_extraction_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "n_calls"])
        writer.writeheader()
        for category, count in sorted(summary["command_categories"].items()):
            writer.writerow({"category": category, "n_calls": count})
    with (run_dir / "parser_edge_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "shard_path", "trajectory_id", "detail"])
        writer.writeheader()
        writer.writerows(summary["edges"])
    with (run_dir / "strict_failure_excerpts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trajectory_id", "shard_path", "scaffold", "model", "message_index", "reasons", "excerpt"],
        )
        writer.writeheader()
        writer.writerows(summary["excerpts"])
    stored = dict(summary)
    stored.pop("edges", None)
    stored.pop("excerpts", None)
    stored["selection"] = selection["selected"]
    _write(run_dir / "pilot_summary.json", stored)


def decide(verification: list[dict], summary: dict | None) -> str:
    if not verification or not all(item["pass"] for item in verification):
        return "V3.3 PILOT BLOCKED — SHARD PROVENANCE OR DOWNLOAD FAILURE"
    if summary is None:
        return "V3.3 PILOT BLOCKED — SHARD PROVENANCE OR DOWNLOAD FAILURE"
    mismatches = int(summary["prefix_comparison"].get("mismatch", 0))
    if mismatches or summary["n_exception"] or summary.get("duplicate_trajectory_ids"):
        return "V3.3 PILOT PARTIAL — PARSER OR DATA ISSUES"
    if summary["n_total"] != summary["meta_rows"]:
        return "V3.3 PILOT PARTIAL — PARSER OR DATA ISSUES"
    return "V3.3 PILOT PASS — LIMITED TO 8 VERIFIED SHARDS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging
    data_dir = staging / "data"
    run_dir = staging / "runs" / "v3_3_pilot_01"
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    selection = _load(args.selection)
    if selection.get("revision") != PINNED_REVISION or len(selection.get("selected", [])) != 8:
        raise SystemExit("selection manifest is not the pinned eight-shard sample")
    modules = verify_modules(args.project, _load(args.code_manifest))
    if not all(row["match"] for row in modules):
        _write(run_dir / "module_hashes.json", {"modules": modules})
        raise SystemExit("imported module hash mismatch")
    if any(row["historical_module_inside_original_project"] is False for row in modules):
        raise SystemExit("historical module was not imported from the original project")
    print("MODULES_OK", flush=True)
    observed_revision = confirm_revision()
    print(f"REVISION_OK {observed_revision}", flush=True)
    sizes = remote_sizes([item["path"] for item in selection["selected"]])
    for item in selection["selected"]:
        remote_size = sizes.get(item["path"])
        if remote_size is not None and int(remote_size) != int(item["bytes"]):
            _write(
                run_dir / "verification.json",
                {"revision": observed_revision, "failure": "remote size mismatch before download", "path": item["path"], "remote_size": remote_size, "manifest_bytes": item["bytes"]},
            )
            raise SystemExit("remote size mismatch before download")
    verification = []
    for item in selection["selected"]:
        destination = data_dir / item["path"]
        if not destination.is_file():
            print(f"DOWNLOAD {item['path']}", flush=True)
            download_one(item["path"], data_dir)
        checked = verify_file(destination, item["bytes"], item["sha256"])
        checked["manifest_path"] = item["path"]
        checked["requested_revision"] = PINNED_REVISION
        verification.append(checked)
        print(f"VERIFY {item['path']} pass={checked['pass']}", flush=True)
        if not checked["pass"]:
            _write(run_dir / "verification.json", {"revision": observed_revision, "files": verification})
            _write(run_dir / "run_manifest.json", {"status": decide(verification, None), "verified_at_utc": _now(), "modules": modules})
            raise SystemExit("shard hash mismatch; parse was not started")
    _write(run_dir / "verification.json", {"revision": observed_revision, "files": verification})
    summary = parse_shards(selection, data_dir, run_dir)
    write_tables(run_dir, summary, selection)
    status = decide(verification, summary)
    _write(
        run_dir / "run_manifest.json",
        {
            "status": status,
            "verified_at_utc": _now(),
            "dataset": DATASET_ID,
            "revision": observed_revision,
            "python": sys.version.split()[0],
            "modules": modules,
            "n_total": summary["n_total"],
            "n_valid": summary["n_valid"],
            "n_error": summary["n_error"],
            "prefix_comparison": summary["prefix_comparison"],
            "prohibited_columns_read": False,
        },
    )
    print(status, flush=True)


if __name__ == "__main__":
    main()
