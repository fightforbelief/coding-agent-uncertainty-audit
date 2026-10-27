"""Extract prefix features without reading ``resolved``."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from p8audit.prefix_features import prefix_feature_row
from p8audit.schema import DATASET_ID, PINNED_REVISION, SHARD_GROUPS
from p8audit.trajectory_parser import parse_trajectory

ROOT = Path(r"C:\Users\hanqi\Desktop\projects\p8-coding-agent-uncertainty-audit")
PINNED = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
RUN = ROOT / "artifacts" / "l2_run_20261027_01"
PARTS = RUN / "feature_parts"


def shard_records() -> list[dict]:
    payload = json.loads(PINNED.read_text(encoding="utf-8"))
    files = payload["files"] if isinstance(payload, dict) and "files" in payload else payload
    rows = []
    for item in files:
        config, dataset_split = SHARD_GROUPS[str(Path(item["path"]).parent).replace("\\", "/")]
        rows.append(
            {
                "path": item["path"],
                "config": config,
                "dataset_split": dataset_split,
            }
        )
    return rows


def main() -> None:
    if not (RUN / "L2_DATASET_CENSUS.json").is_file():
        raise SystemExit("census reconciliation has not been written")
    census = json.loads((RUN / "L2_DATASET_CENSUS.json").read_text(encoding="utf-8"))
    if not census.get("match"):
        raise SystemExit("census did not match")
    PARTS.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    for shard_index, shard in enumerate(shard_records()):
        part = PARTS / f"{shard_index:02d}.parquet"
        if part.is_file():
            print(f"FEATURE_SKIP {shard['path']}", flush=True)
            continue
        remote = f"datasets/{DATASET_ID}@{PINNED_REVISION}/{shard['path']}"
        with fs.open(remote, "rb") as handle:
            table = pq.ParquetFile(handle).read(
                columns=["instance_id", "repo", "language", "trajectory_id", "trajectory"]
            )
        if "resolved" in table.column_names:
            raise SystemExit("resolved was read during feature extraction")
        rows = []
        trajectories = table.column("trajectory")
        ids = table.column("trajectory_id").to_pylist()
        instances = table.column("instance_id").to_pylist()
        repos = table.column("repo").to_pylist()
        languages = table.column("language").to_pylist()
        for row_index in range(table.num_rows):
            parsed = parse_trajectory(trajectories[row_index].as_py())
            features = prefix_feature_row(parsed, config=shard["config"])
            record = {
                "shard": shard["path"],
                "row_index": row_index,
                "trajectory_id": str(ids[row_index]),
                "instance_id": str(instances[row_index]),
                "repo": str(repos[row_index]),
                "language": str(languages[row_index]),
                "config": shard["config"],
                "dataset_split": shard["dataset_split"],
            }
            record.update(features)
            rows.append(record)
        pq.write_table(pa.Table.from_pylist(rows), part)
        print(f"FEATURE_DONE {shard['path']} rows={table.num_rows}", flush=True)
    tables = [pq.read_table(path) for path in sorted(PARTS.glob("*.parquet"))]
    combined = pa.concat_tables(tables)
    if "resolved" in combined.column_names:
        raise SystemExit("feature table contains resolved")
    pq.write_table(combined, RUN / "prefix_features.parquet")
    schema = {
        "columns": combined.column_names,
        "rows": combined.num_rows,
        "resolved_present": False,
        "predictor_note": "n_complete_cycles, parse_status, repo, instance_id, and trajectory_id are not predictors",
    }
    (RUN / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"FEATURES_WRITTEN rows={combined.num_rows}", flush=True)


if __name__ == "__main__":
    main()
