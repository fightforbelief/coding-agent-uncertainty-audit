"""Count present-invocation evidence in Go package summaries, without outcome labels."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

# The project root is also the invocation directory when staged on the research host.
ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis_v2")]
from p8audit.schema import PINNED_REVISION, group_for_shard
from p8audit.trajectory_parser import coerce_trajectory, parse_trajectory
from v3_4_lib import build_calls

# Single shell invocation only. Ordinary flags, package selectors, and an optional cd.
COMMAND = re.compile(r"(?:cd[ \t]+[\w./@~-]+[ \t]*&&[ \t]*)?go[ \t]+test(?:[ \t]+[\w./@:=,+%~-]+)*", re.ASCII)
SUMMARY = re.compile(r"^ok(?P<sep>[ \t]+)(?P<pkg>[\w./~+-]+)[ \t]+(?P<time>\(cached\)|\d+(?:\.\d+)?s)(?:[ \t]+\[(?P<note>no tests to run)\])?[ \t]*$", re.ASCII)


def digest(path):
    with Path(path).open("rb") as source:
        return sha256_stream(source)


def sha256_stream(stream):
    h = sha256()
    for chunk in iter(lambda: stream.read(4 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def package_counts(command, output):
    if not COMMAND.fullmatch((command or "").strip()):
        return None
    counts = Counter()
    for line in (output or "").splitlines():
        match = SUMMARY.fullmatch(line)
        if match:
            kind = "cached" if match["time"] == "(cached)" else "no_match" if match["note"] else "timed"
            counts[kind] += 1
            if match["sep"].startswith("\t"):
                counts["tab_" + kind] += 1
    return counts


def prefix_calls(raw, scaffold):
    records, _ = coerce_trajectory(raw)
    if records is None:
        raise ValueError("Unparseable trajectory")
    limit = min(len(records), 80)
    while True:
        parsed = parse_trajectory(records[:limit])
        if parsed.n_complete_cycles >= 30 or limit == len(records):
            break
        limit = min(len(records), limit + 80)
    if parsed.status not in {"ok", "no_cycles"}:
        raise ValueError(parsed.status)
    calls = build_calls(scaffold, parsed, records[:limit])
    return [c for c in calls if c["observation_index"] is not None and c["round"] <= min(30, parsed.n_complete_cycles)]


def worker(task):
    import pyarrow.parquet as pq
    root, item = task
    path = Path(root) / item["path"]
    if digest(path) != item["sha256"]:
        raise ValueError("Unexpected source hash: " + item["path"])
    group = group_for_shard(item["path"])
    cells = {k: Counter() for k in (15, 30)}
    repos = {k: set() for k in cells}
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=12, columns=["language", "repo", "trajectory"]):
        n += batch.num_rows
        for i, lang in enumerate(batch.column("language").to_pylist()):
            if str(lang).lower() != "go":
                continue
            events = []
            for call in prefix_calls(batch.column("trajectory")[i].as_py(), group[0]):
                counts = package_counts(call["command"], call["output"])
                if counts is not None:
                    events.append((call["round"], counts))
            for k, cell in cells.items():
                cell["go_trajectories"] += 1
                seen = Counter()
                for rnd, counts in events:
                    if rnd <= k:
                        cell["commands"] += 1
                        seen.update(counts)
                        cell["success_observations"] += bool(counts)
                        cell["cached_observations"] += counts["cached"] > 0
                        cell["mixed_observations"] += counts["cached"] > 0 and counts["timed"] + counts["no_match"] > 0
                cell.update(seen)
                if seen:
                    repos[k].add(batch.column("repo")[i].as_py())
                cell["success_trajectories"] += bool(seen)
                cell["cached_trajectories"] += seen["cached"] > 0
    return group, cells, repos, n


def run(data, manifest_file, workers):
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    assert manifest["revision"] == PINNED_REVISION and len(set(x["path"] for x in manifest["files"])) == 84
    totals = defaultdict(Counter)
    repos = defaultdict(set)
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (group, cells, observed, rows) in enumerate(pool.map(worker, [(data,x) for x in manifest["files"]]), 1):
            for k, cell in cells.items():
                for key in (("/".join(group), k), ("all", k)):
                    totals[key].update(cell)
                    repos[key].update(observed[k])
            n += rows
            print(f"{index}/84 shards; {n} trajectories", file=sys.stderr, flush=True)
    assert n == 207489
    assert totals["all",15]["go_trajectories"] == sum(totals[g,15]["go_trajectories"] for g,k in totals if g != "all" and k == 15)
    return {"revision": PINNED_REVISION, "manifest_sha256": digest(manifest_file),
            "source_sha256": digest(__file__), "trajectories_scanned": n, "shards_verified": 84,
            "cells": [dict(group=g, checkpoint=k, repositories=len(repos[g,k]), **totals[g,k])
                      for g,k in sorted(totals)]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    result = run(args.data, args.manifest, args.workers)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
