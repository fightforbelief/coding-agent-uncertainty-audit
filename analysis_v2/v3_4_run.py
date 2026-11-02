"""Run the V3.4 descriptive census on this Windows machine.

Measurement code lives in ``v3_4_lib``. This script only checks provenance,
builds metadata, and writes the run directory. It does not train a model.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis_v2")]

from p8audit.schema import DATASET_ID, PINNED_REVISION
from v3_4_lib import (
    EXPECTED_MANIFEST_SHA256,
    FEATURE_SHA256,
    LABEL_SHA256,
    PARAMS,
    STATUS_BLOCKED,
    V33_SHARD_PATHS,
    assemble_metadata,
    code_token,
    decide_status,
    group_for_relative,
    manifest_problems,
    params_token,
    process_shard,
    safe_shard_name,
    sha256_file,
    verify_local_file,
)
from v3_4_report import assessment, disagreement_markdown, hash_outputs, render_report, short_review, write_tables

STAGING = ROOT.parent / "_p8-v3_4-census-staging"
OLD_STAGING = Path(r"C:\Users\hanqi\Desktop\projects\_p8-v3_3-pilot-staging")
MANIFEST_PATH = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
LABELS_PATH = ROOT / "artifacts" / "l2_run_20261027_01" / "labels_for_join.parquet"
FEATURES_PATH = ROOT / "artifacts" / "l2_run_20261027_01" / "prefix_features.parquet"
SPLIT_PATH = ROOT / "artifacts" / "manifests" / "split_assignment_seed_20261024.json"
MAX_IN_FLIGHT_SHARDS = 2
MEASUREMENT_FILES = [
    ROOT / "analysis_v2" / "v3_4_lib.py",
    ROOT / "analysis_v2" / "traj_adapter.py",
    ROOT / "analysis_v2" / "census_metrics.py",
    ROOT / "analysis_v2" / "p8v3" / "testcmd.py",
    ROOT / "analysis_v2" / "v3_3_pilot.py",
    ROOT / "src" / "p8audit" / "trajectory_parser.py",
    ROOT / "src" / "p8audit" / "execution_signals_v2.py",
    ROOT / "src" / "p8audit" / "prefix_features.py",
    ROOT / "src" / "p8audit" / "schema.py",
]


def _log(path: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _writable(path: Path) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


def _load_manifest() -> tuple[dict, str]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), sha256_file(MANIFEST_PATH)


def _confirm_revision() -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_ID, revision=PINNED_REVISION)
    observed = getattr(info, "sha", None)
    if observed != PINNED_REVISION:
        raise RuntimeError(f"Hugging Face revision is {observed}")
    return str(observed)


def _old_candidates(relative: str) -> list[Path]:
    rel = Path(relative)
    candidates = [
        OLD_STAGING / "data" / rel,
        OLD_STAGING / "data" / "data" / rel,
    ]
    if rel.parts and rel.parts[0] == "data":
        tail = Path(*rel.parts[1:])
        candidates.extend([OLD_STAGING / "data" / tail, OLD_STAGING / "data" / "data" / tail])
    found = []
    seen = set()
    for candidate in candidates:
        if candidate.is_file():
            key = str(candidate.resolve()).lower()
            if key not in seen:
                seen.add(key)
                found.append(candidate)
    return found


def _place_verified_copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        _writable(dest)
        dest.unlink()
    shutil.copy2(source, dest)
    _readonly(dest)


def _download(relative: str, dest_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=DATASET_ID,
        filename=relative,
        repo_type="dataset",
        revision=PINNED_REVISION,
        local_dir=str(dest_root),
    )
    return Path(downloaded)


def run_provenance(log_path: Path, on_verified=None) -> tuple[bool, list[dict], Path]:
    data_root = STAGING / "data"
    manifest_dir = STAGING / "manifests"
    for folder in (data_root, manifest_dir, STAGING / "metadata", STAGING / "runs", STAGING / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    payload, digest = _load_manifest()
    problems = manifest_problems(payload, digest=digest)
    if digest != EXPECTED_MANIFEST_SHA256:
        problems.append("pinned manifest hash does not match the known V3.2 hash")
    if problems:
        text = "\n".join(problems)
        (manifest_dir / "DATA_PROVENANCE_REPORT.md").write_text(
            "# Data provenance\n\nManifest check failed.\n\n" + text + "\n",
            encoding="utf-8",
        )
        _log(log_path, "PROVENANCE_BLOCKED manifest")
        return False, [], data_root
    try:
        observed = _confirm_revision()
    except Exception as exc:
        (manifest_dir / "DATA_PROVENANCE_REPORT.md").write_text(
            f"# Data provenance\n\nRevision check failed: {exc}\n",
            encoding="utf-8",
        )
        _log(log_path, f"PROVENANCE_BLOCKED revision {exc}")
        return False, [], data_root
    rows = []
    for item in payload["files"]:
        relative = str(item["path"]).replace("\\", "/")
        dest = data_root / relative
        source_kind = None
        if dest.is_file():
            current = verify_local_file(dest, int(item["bytes"]), item["sha256"])
            if current["pass"]:
                source_kind = "already_verified"
        if source_kind is None:
            copied = False
            for candidate in _old_candidates(relative):
                check = verify_local_file(candidate, int(item["bytes"]), item["sha256"])
                if check["pass"]:
                    _place_verified_copy(candidate, dest)
                    source_kind = "reused"
                    copied = True
                    _log(log_path, f"REUSED {relative}")
                    break
            if not copied:
                last_error = "not downloaded"
                for attempt in (1, 2):
                    try:
                        if dest.exists():
                            _writable(dest)
                            dest.unlink()
                        returned = _download(relative, data_root)
                        if returned.resolve() != dest.resolve() and returned.is_file():
                            _place_verified_copy(returned, dest)
                        source_kind = "downloaded"
                        _log(log_path, f"DOWNLOADED {relative} attempt={attempt}")
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        _log(log_path, f"DOWNLOAD_ERROR {relative} attempt={attempt} {exc}")
                if source_kind is None:
                    source_kind = f"failed: {last_error}"
        verified = verify_local_file(dest, int(item["bytes"]), item["sha256"]) if dest.is_file() else {
            "actual_bytes": None,
            "actual_sha256": None,
            "pass": False,
        }
        if verified["pass"]:
            _readonly(dest)
        rows.append(
            {
                "Relative path": relative,
                "Actual path": str(dest if dest.is_file() else ""),
                "Expected bytes": int(item["bytes"]),
                "Actual bytes": verified["actual_bytes"],
                "Expected SHA-256": item["sha256"],
                "Actual SHA-256": verified["actual_sha256"],
                "Verification status": "pass" if verified["pass"] else "fail",
                "Source revision": observed,
                "Whether reused or downloaded": source_kind,
            }
        )
        if rows[-1]["Verification status"] == "pass" and on_verified is not None:
            on_verified(rows[-1])
    frame = pd.DataFrame(rows)
    frame.to_csv(manifest_dir / "SHARD_PROVENANCE.csv", index=False)
    frame.to_csv(manifest_dir / "SHARD_DOWNLOAD_STATUS.csv", index=False)
    passed = int((frame["Verification status"] == "pass").sum())
    report = [
        "# Data provenance",
        "",
        f"Dataset: `{DATASET_ID}`",
        f"Revision: `{observed}`",
        f"Manifest SHA-256: `{digest}`",
        f"Verified shards: {passed} / {len(frame)}",
        "",
        "每个分片都按 manifest 的相对路径、字节数和 SHA-256 核验。",
        "V3.3 的八个分片只在字节数和 SHA-256 都一致时复制进本次 staging。",
        "同一个相对路径只保留一条记录。",
        "",
    ]
    (manifest_dir / "DATA_PROVENANCE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    ok = passed == 84 and digest == EXPECTED_MANIFEST_SHA256
    _log(log_path, f"PROVENANCE {'PASS' if ok else 'FAIL'} {passed}/84")
    return ok, rows, data_root


def _scan_ids(rows: list[dict], data_root: Path, log_path: Path) -> pd.DataFrame:
    pieces = []
    for item in rows:
        if item["Verification status"] != "pass":
            continue
        path = Path(item["Actual path"])
        relative = item["Relative path"]
        reader = pq.ParquetFile(path)
        tables = []
        for batch in reader.iter_batches(batch_size=5000, columns=["trajectory_id", "instance_id", "repo"]):
            frame = pa_to_frame(batch, relative)
            tables.append(frame)
        pieces.append(pd.concat(tables, ignore_index=True))
        _log(log_path, f"IDS {relative} rows={sum(len(table) for table in tables)}")
    return pd.concat(pieces, ignore_index=True)


def pa_to_frame(batch, relative: str) -> pd.DataFrame:
    frame = batch.to_pandas()
    frame["trajectory_id"] = frame["trajectory_id"].astype(str)
    frame["instance_id"] = frame["instance_id"].astype(str)
    frame["repo"] = frame["repo"].astype(str)
    frame["shard"] = relative
    return frame


def run_metadata(provenance_rows: list[dict], log_path: Path) -> tuple[bool, Path | None, dict]:
    if sha256_file(LABELS_PATH) != LABEL_SHA256 or sha256_file(FEATURES_PATH) != FEATURE_SHA256:
        _log(log_path, "METADATA_BLOCKED historical hash mismatch")
        return False, None, {"problems": ["historical label or feature hash mismatch"]}
    split_payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if split_payload.get("revision") != PINNED_REVISION:
        return False, None, {"problems": ["saved split file revision differs"]}
    repo_to_split = split_payload.get("repo_to_split")
    if not isinstance(repo_to_split, dict):
        return False, None, {"problems": ["repo_to_split is missing"]}
    shard_ids = _scan_ids(provenance_rows, STAGING / "data", log_path)
    labels = pq.read_table(
        LABELS_PATH,
        columns=["trajectory_id", "instance_id", "repo", "config", "resolved", "split", "shard"],
    ).to_pandas()
    features = pq.read_table(
        FEATURES_PATH,
        columns=[
            "trajectory_id",
            "instance_id",
            "repo",
            "config",
            "dataset_split",
            "n_complete_cycles",
            "k5_prefix_explicit_test_failure_any",
            "shard",
        ],
    ).to_pandas()
    meta, detail = assemble_metadata(shard_ids, labels, features, repo_to_split)
    out = STAGING / "metadata" / "meta.parquet"
    sources = {
        "trajectory_id": "trajectory_id column read from the verified shard",
        "issue_id": "labels_for_join.instance_id, the dataset instance_id and the historical L2 issue unit; not synthesized",
        "repo": "shard repo column, required to match labels_for_join.repo and prefix_features.repo",
        "scaffold": "SHARD_GROUPS config for the shard directory, required to match labels.config",
        "model": "SHARD_GROUPS dataset split for the shard directory, required to match prefix_features.dataset_split",
        "resolved": "labels_for_join.resolved; the shard resolved column is not read",
        "split": "labels_for_join.split, checked row by row against split_assignment_seed_20261024.json repo_to_split",
        "n_rounds_total": "prefix_features.n_complete_cycles from the frozen parser feature table",
        "parser_v2_strict_fail_r5": "prefix_features.k5_prefix_explicit_test_failure_any; null stays null",
        "label_sha256": LABEL_SHA256,
        "feature_sha256": FEATURE_SHA256,
        "split_sha256": sha256_file(SPLIT_PATH),
    }
    (STAGING / "metadata" / "METADATA_SOURCES.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    (STAGING / "metadata" / "metadata_integrity.json").write_text(json.dumps(detail, indent=2, default=str), encoding="utf-8")
    if meta is None:
        _log(log_path, "METADATA_BLOCKED " + "; ".join(detail.get("problems", [])))
        return False, None, detail
    partial = out.with_suffix(".parquet.partial")
    meta.to_parquet(partial, index=False)
    partial.replace(out)
    _log(log_path, f"METADATA_PASS rows={len(meta)}")
    return True, out, detail


def _measurement_tokens() -> tuple[str, str]:
    return code_token(MEASUREMENT_FILES), params_token(PARAMS)


def stage_shard(job_path: Path) -> None:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    process_shard(
        parquet_path=Path(job["parquet_path"]),
        relative_path=job["relative_path"],
        scaffold=job["scaffold"],
        output_parquet=Path(job["output_parquet"]),
        output_errors=Path(job["output_errors"]),
        output_checkpoint=Path(job["output_checkpoint"]),
        code_token_value=job["code_token"],
        params_token_value=job["params_token"],
        input_sha256=job["input_sha256"],
        input_bytes=int(job["input_bytes"]),
        log_path=Path(job["log_path"]),
    )


def _write_job(run_dir: Path, item: dict, code_value: str, param_value: str) -> Path:
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    name = safe_shard_name(item["Relative path"])
    scaffold, _model = group_for_relative(item["Relative path"])
    job = {
        "parquet_path": item["Actual path"],
        "relative_path": item["Relative path"],
        "scaffold": scaffold,
        "output_parquet": str(shard_dir / f"{name}.parquet"),
        "output_errors": str(shard_dir / f"{name}.jsonl"),
        "output_checkpoint": str(shard_dir / f"{name}.checkpoint.json"),
        "code_token": code_value,
        "params_token": param_value,
        "input_sha256": item["Actual SHA-256"],
        "input_bytes": int(item["Actual bytes"]),
        "log_path": str(run_dir / "logs" / f"{name}.log"),
    }
    job_path = run_dir / "jobs" / f"{name}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    Path(job["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job), encoding="utf-8")
    return job_path


def _reap(active: list[tuple[subprocess.Popen, Path]], failures: list[str], log_path: Path) -> list[tuple[subprocess.Popen, Path]]:
    still = []
    for process, job_path in active:
        code = process.poll()
        if code is None:
            still.append((process, job_path))
        elif code != 0:
            failures.append(job_path.name)
            _log(log_path, f"SHARD_FAIL {job_path.name} exit={code}")
        else:
            _log(log_path, f"SHARD_EXIT0 {job_path.name}")
    return still


def consume_census(verified: queue.Queue, run_dir: Path, log_path: Path) -> bool:
    """Measure shards as soon as their files have been verified."""
    code_value, param_value = _measurement_tokens()
    script = str(Path(__file__).resolve())
    active: list[tuple[subprocess.Popen, Path]] = []
    failures: list[str] = []
    saw_sentinel = False
    while True:
        active = _reap(active, failures, log_path)
        if failures:
            for process, _job_path in active:
                process.terminate()
            _log(log_path, f"CENSUS_STOPPED failures={failures}")
            return False
        while len(active) < MAX_IN_FLIGHT_SHARDS and not saw_sentinel:
            try:
                item = verified.get(timeout=1)
            except queue.Empty:
                break
            if item is None:
                saw_sentinel = True
                break
            job_path = _write_job(run_dir, item, code_value, param_value)
            process = subprocess.Popen([sys.executable, script, "--stage", "shard", "--job", str(job_path)])
            active.append((process, job_path))
            _log(log_path, f"SHARD_LAUNCH {job_path.name}")
        if saw_sentinel and not active:
            break
        time.sleep(0.5)
    _log(log_path, "CENSUS_SHARDS_DONE")
    return True


def _load_census(shard_dir: Path) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    frames = []
    categories: dict[str, int] = {}
    problems = []
    checkpoints = sorted(shard_dir.glob("*.checkpoint.json"))
    if len(checkpoints) != 84:
        problems.append(f"checkpoint count is {len(checkpoints)}")
    seen = set()
    for checkpoint_path in checkpoints:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        relative = payload["relative_path"]
        if relative in seen:
            problems.append(f"duplicate shard output {relative}")
        seen.add(relative)
        output = shard_dir / f"{safe_shard_name(relative)}.parquet"
        if sha256_file(output) != payload["output_sha256"]:
            problems.append(f"shard output hash changed after checkpoint: {relative}")
            continue
        frame = pd.read_parquet(output)
        if len(frame) != int(payload["rows"]):
            problems.append(f"shard row count changed: {relative}")
        if frame["trajectory_id"].astype(str).duplicated().any():
            problems.append(f"duplicate ids inside {relative}")
        frames.append(frame)
        for key, value in payload.get("command_categories", {}).items():
            categories[key] = categories.get(key, 0) + int(value)
    if not frames:
        return pd.DataFrame(), categories, problems or ["no shard frames"]
    combined = pd.concat(frames, ignore_index=True)
    duplicate = int(combined["trajectory_id"].astype(str).duplicated().sum())
    if duplicate:
        problems.append(f"census has {duplicate} duplicate trajectory ids")
    return combined, categories, problems


def _join(census: pd.DataFrame, meta_path: Path) -> tuple[pd.DataFrame | None, int, list[str]]:
    meta = pd.read_parquet(meta_path)
    problems = []
    left = census[["trajectory_id", "instance_id", "repo", "shard"]].copy()
    right = meta[["trajectory_id", "instance_id", "repo", "shard"]].copy()
    for frame in (left, right):
        for column in frame.columns:
            frame[column] = frame[column].astype(str)
    if left["trajectory_id"].duplicated().any() or right["trajectory_id"].duplicated().any():
        return None, 0, ["join inputs have duplicate trajectory ids"]
    if set(left["trajectory_id"]) != set(right["trajectory_id"]):
        return None, 0, ["census trajectory ids do not match metadata"]
    compared = left.merge(right, on="trajectory_id", suffixes=("_census", "_meta"), validate="one_to_one")
    for column in ("instance_id", "repo", "shard"):
        mismatch = int((compared[f"{column}_census"] != compared[f"{column}_meta"]).sum())
        if mismatch:
            problems.append(f"joined {column} disagrees on {mismatch} rows")
    if problems:
        return None, 0, problems
    measure_columns = [column for column in census.columns if column not in {"instance_id", "repo"}]
    merged = census[measure_columns].merge(meta.drop(columns=["shard"]), on="trajectory_id", how="inner", validate="one_to_one")
    both = merged["n_rounds_total"].notna() & merged["n_rounds_observed"].notna()
    mismatches = int((merged.loc[both, "n_rounds_total"] != merged.loc[both, "n_rounds_observed"]).sum())
    return merged, mismatches, []


def run_pytest(log_path: Path, run_dir: Path) -> bool:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    tmp = run_dir / "pytest_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(tmp / "base")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    text = completed.stdout + "\n" + completed.stderr
    (run_dir / "pytest.txt").write_text(text, encoding="utf-8")
    _log(log_path, f"PYTEST exit={completed.returncode}")
    return completed.returncode == 0


def write_stopped(run_dir: Path, log_path: Path, status: str, reason: str) -> None:
    report = "\n".join(
        [
            "# V3.4 全库描述性普查",
            "",
            f"状态：`{status}`",
            "",
            reason,
            "",
            "已完成的核验文件保留在本次 staging。历史 L2 工件没有被改写。",
            "",
        ]
    )
    (run_dir / "V3_4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (run_dir / "V3_4_RUN_MANIFEST.json").write_text(
        json.dumps({"status": status, "reason": reason}, indent=2),
        encoding="utf-8",
    )
    _log(log_path, reason)
    _publish_review(run_dir, status, report)


def _publish_review(run_dir: Path, status: str, report: str) -> None:
    text = short_review(status, run_dir, report)
    dest = ROOT / "analysis_v2" / "V3_4_FULL_CENSUS_REVIEW.md"
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = ROOT / "analysis_v2" / f"V3_4_FULL_CENSUS_REVIEW_{stamp}.md"
    dest.write_text(text, encoding="utf-8")
    uploads = ROOT / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    for old in uploads.glob("*.md"):
        old.unlink()
    (uploads / "V3_4_FULL_CENSUS_FOR_REVIEW.md").write_text(text, encoding="utf-8")


def finish_reports(run_dir: Path, log_path: Path, *, provenance_ok: bool, metadata_ok: bool, tests_passed: bool, meta_path: Path, provenance_rows: list[dict]) -> str:
    census, categories, problems = _load_census(run_dir / "shards")
    joined = None
    round_mismatches = 0
    if problems:
        pass
    elif census.empty:
        problems.append("census frame is empty")
    else:
        joined, round_mismatches, join_problems = _join(census, meta_path)
        problems.extend(join_problems)
    if joined is None:
        status = decide_status(provenance_ok=provenance_ok, metadata_ok=metadata_ok, census_finished=False, problems=problems or ["join failed"])
        write_stopped(run_dir, log_path, status, "\n".join(problems))
        return status
    error_parts = sorted((run_dir / "shards").glob("*.jsonl"))
    with (run_dir / "census_errors.jsonl").open("w", encoding="utf-8") as handle:
        for part in error_parts:
            handle.write(part.read_text(encoding="utf-8"))
    partial = run_dir / "census.parquet.partial"
    joined.to_parquet(partial, index=False)
    partial.replace(run_dir / "census.parquet")
    tables = write_tables(run_dir, joined, categories)
    problems.extend(assessment(joined, tests_passed=tests_passed, round_mismatches=round_mismatches))
    status = decide_status(
        provenance_ok=provenance_ok,
        metadata_ok=metadata_ok,
        census_finished=len(problems) == 0,
        problems=problems,
    )
    metadata_detail = json.loads((STAGING / "metadata" / "metadata_integrity.json").read_text(encoding="utf-8"))
    provenance = {
        "revision": PINNED_REVISION,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "verified": sum(item["Verification status"] == "pass" for item in provenance_rows),
        "expected_shards": 84,
    }
    report = render_report(
        status=status,
        frame=joined,
        support=tables["support"],
        v33=tables["v33"],
        problems=problems,
        provenance=provenance,
        metadata=metadata_detail,
        categories=categories,
        round_mismatches=round_mismatches,
    )
    (run_dir / "01_census.md").write_text(report, encoding="utf-8")
    (run_dir / "V3_4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    shard_paths = {item["Relative path"]: Path(item["Actual path"]) for item in provenance_rows}
    (run_dir / "census_disagreements.md").write_text(disagreement_markdown(joined, shard_paths), encoding="utf-8")
    shutil.copy2(log_path, run_dir / "V3_4_RUN_LOG.txt")
    manifest = {
        "status": status,
        "revision": PINNED_REVISION,
        "rows": int(len(joined)),
        "problems": problems,
        "tests_passed": tests_passed,
        "max_in_flight_shards": MAX_IN_FLIGHT_SHARDS,
        "resolved_passed_to_detector": False,
        "model_fitted": False,
        "code_token": _measurement_tokens()[0],
        "params": PARAMS,
    }
    (run_dir / "V3_4_RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    hash_outputs(run_dir)
    _publish_review(run_dir, status, report)
    _log(log_path, f"STATUS {status}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--job", default="")
    args = parser.parse_args()
    if args.stage == "shard":
        try:
            stage_shard(Path(args.job))
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        return
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            raise SystemExit(f"run directory does not exist: {run_dir}")
    else:
        STAGING.joinpath("runs").mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = STAGING / "runs" / f"v3_4_census_{stamp}"
        run_dir.mkdir(parents=True)
    log_path = STAGING / "logs" / f"{run_dir.name}.log"
    _log(log_path, f"RUN_DIR {run_dir}")
    tests_passed = run_pytest(log_path, run_dir)
    if not tests_passed:
        write_stopped(run_dir, log_path, "V3.4 PARTIAL — DATA OR PARSER ISSUES", "pytest failed; full census was not started")
        raise SystemExit("V3.4 PARTIAL — DATA OR PARSER ISSUES")
    verified: queue.Queue = queue.Queue()
    holder: dict = {}

    def _produce() -> None:
        try:
            holder["result"] = run_provenance(log_path, on_verified=verified.put)
        except Exception as exc:
            holder["error"] = exc
            _log(log_path, f"PROVENANCE_EXCEPTION {exc}")
        finally:
            verified.put(None)

    producer = threading.Thread(target=_produce, name="v3-4-provenance")
    producer.start()
    census_ok = consume_census(verified, run_dir, log_path)
    producer.join()
    if "error" in holder:
        write_stopped(run_dir, log_path, STATUS_BLOCKED, f"provenance failed: {holder['error']}")
        raise SystemExit(STATUS_BLOCKED)
    provenance_ok, rows, _data_root = holder["result"]
    manifest_dir = STAGING / "manifests"
    for name in ("SHARD_PROVENANCE.csv", "SHARD_DOWNLOAD_STATUS.csv", "DATA_PROVENANCE_REPORT.md"):
        source = manifest_dir / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)
    if not provenance_ok:
        write_stopped(run_dir, log_path, STATUS_BLOCKED, "84/84 shard verification did not pass")
        raise SystemExit(STATUS_BLOCKED)
    metadata_ok, meta_path, _detail = run_metadata(rows, log_path)
    if not metadata_ok or meta_path is None:
        write_stopped(run_dir, log_path, STATUS_BLOCKED, "metadata integrity failed")
        raise SystemExit(STATUS_BLOCKED)
    if not census_ok:
        write_stopped(run_dir, log_path, "V3.4 PARTIAL — DATA OR PARSER ISSUES", "one or more shards failed during measurement")
        raise SystemExit("V3.4 PARTIAL — DATA OR PARSER ISSUES")
    status = finish_reports(
        run_dir,
        log_path,
        provenance_ok=provenance_ok,
        metadata_ok=metadata_ok,
        tests_passed=tests_passed,
        meta_path=meta_path,
        provenance_rows=rows,
    )
    print(status)


if __name__ == "__main__":
    main()
