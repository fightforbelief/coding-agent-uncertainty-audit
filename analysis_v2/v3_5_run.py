"""Build the unlabeled V3.5 measurement-validation package.

Reads the saved census and the already verified shard cache. It does not
download data, rerun the census, or fill reviewer labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from p8audit.schema import PINNED_REVISION
from v3_4_lib import locate_shard, sha256_file
from v3_5_design import (
    DIAGNOSTIC_STRATA,
    PRIMARY_STRATA,
    SEED,
    blank_review_queue,
    design_sample,
    sealed_records,
)
from v3_5_evidence import extract_prefix_evidence

ROOT = Path(__file__).resolve().parents[1]
STAGING_V34 = ROOT.parent / "_p8-v3_4-census-staging"
STAGING = ROOT.parent / "_p8-v3_5-validation-staging"
CENSUS_PATH = STAGING_V34 / "runs" / "v3_4_census_20261102T153940Z" / "census.parquet"
META_PATH = STAGING_V34 / "metadata" / "meta.parquet"
CENSUS_SHA_MANIFEST = STAGING_V34 / "runs" / "v3_4_census_20261102T153940Z" / "V3_4_SHA256_MANIFEST.csv"
SUPPORT_MANIFEST = STAGING_V34 / "runs" / "v3_4_1_issue_support_20261102T183235Z" / "V3_4_1_RUN_MANIFEST.json"
PINNED_MANIFEST = ROOT / "artifacts" / "manifests" / "pinned_parquet_files.json"
DATA_ROOT = STAGING_V34 / "data"
STATUS_PREPARED = "V3.5 VALIDATION PACKAGE PREPARED — ADJUDICATION PENDING"
STATUS_PARTIAL = "V3.5 PARTIAL — VALIDATION FRAME INCOMPLETE"
STATUS_BLOCKED = "V3.5 BLOCKED — SOURCE EVIDENCE UNAVAILABLE"
REVIEWER_MEMBERS = (
    "review_queue.csv",
    "review_evidence.jsonl",
    "validation_instructions.md",
    "adjudication_rules.md",
)
EXPECTED_D = 495


def _bytes_for(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, pd.DataFrame):
        return value.to_csv(index=False).encode("utf-8")
    if isinstance(value, Path):
        return value.read_bytes()
    return str(value).encode("utf-8")


def write_reviewer_zip(path: Path, members: dict[str, Any]) -> Path:
    """Write the reviewer archive. The sealed key is not an accepted member."""
    unexpected = set(members).difference(REVIEWER_MEMBERS)
    if unexpected:
        raise ValueError(f"reviewer zip refuses: {sorted(unexpected)}")
    payload = {name: _bytes_for(members[name]) for name in REVIEWER_MEMBERS}
    listing = [
        {"path": name, "bytes": len(payload[name]), "sha256": hashlib.sha256(payload[name]).hexdigest()}
        for name in REVIEWER_MEMBERS
    ]
    manifest = json.dumps({"members": listing}, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REVIEWER_MEMBERS:
            archive.writestr(name, payload[name])
        archive.writestr("review_package_manifest.json", manifest)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("reviewer zip failed integrity check")
        for item in listing:
            body = archive.read(item["path"])
            if hashlib.sha256(body).hexdigest() != item["sha256"] or len(body) != item["bytes"]:
                raise RuntimeError(f"zip member hash mismatch: {item['path']}")
        if "sealed_sampling_key.csv" in archive.namelist():
            raise RuntimeError("sealed key entered the reviewer zip")
    return path


def _pinned_files() -> dict[str, dict[str, Any]]:
    payload = json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))
    return {str(item["path"]).replace("\\", "/"): item for item in payload["files"]}


def _recorded_sha(manifest_csv: Path, name: str) -> str:
    table = pd.read_csv(manifest_csv)
    matched = table.loc[table["path"] == name, "sha256"]
    if matched.empty:
        raise RuntimeError(f"{name} is absent from {manifest_csv}")
    return str(matched.iloc[0])


def _check_sources() -> list[str]:
    problems = []
    if not CENSUS_PATH.is_file() or not META_PATH.is_file():
        return ["census or metadata file is absent"]
    census_sha = sha256_file(CENSUS_PATH)
    recorded = _recorded_sha(CENSUS_SHA_MANIFEST, "census.parquet")
    if census_sha != recorded:
        problems.append(f"census sha256 is {census_sha}")
    support = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
    if support.get("input_sha256", {}).get("census") != census_sha:
        problems.append("V3.4.1 manifest census hash does not match the file")
    meta_sha = sha256_file(META_PATH)
    if support.get("input_sha256", {}).get("metadata") != meta_sha:
        problems.append("metadata sha256 does not match the V3.4.1 manifest")
    if support.get("revision") != PINNED_REVISION:
        problems.append("V3.4.1 revision differs from the pinned revision")
    return problems


def _stream_selected(path: Path, wanted: set[str]) -> dict[str, tuple[int, Any]]:
    reader = pq.ParquetFile(path)
    names = set(reader.schema_arrow.names)
    if "trajectory_id" not in names or "trajectory" not in names:
        raise RuntimeError(f"shard is missing trajectory columns: {path}")
    found: dict[str, tuple[int, Any]] = {}
    row_index = 0
    for batch in reader.iter_batches(batch_size=4, columns=["trajectory_id", "trajectory"]):
        identifiers = [str(value) for value in batch.column("trajectory_id").to_pylist()]
        hits = [index for index, identifier in enumerate(identifiers) if identifier in wanted]
        trajectories = batch.column("trajectory").to_pylist() if hits else None
        for index in hits:
            found[identifiers[index]] = (row_index + index, trajectories[index])
        row_index += len(identifiers)
        del batch
    return found


def _instructions() -> str:
    return "\n".join(
        [
            "# V3.5 测量核对说明",
            "",
            "请只根据每条轨迹附带的工具调用、原始参数、抽取出的命令和工具观察来判断。",
            "",
            "四个问题要分开回答，不能互相代替：",
            "",
            "1. 是否发出了测试命令。",
            "2. 该命令是否有证据表明已经执行。",
            "3. 是否观察到测试输出。",
            "4. 输出是否明确显示通过或失败。",
            "",
            "前 5 个完成轮次和前 15 个完成轮次分别填写。完成轮次少于 15 时，证据里只有实际完成的轮次；`prefix15_boundary = no_prefix_boundary` 表示没有第 15 轮边界。缺掉的轮次没有观察，不能把它填成明确没有测试命令。",
            "",
            "裁决栏是空的。请按 `adjudication_rules.md` 里的固定用词填写。本包没有给出检测器预测、最终结局或抽样分层。",
            "",
        ]
    )


def _rules() -> str:
    return "\n".join(
        [
            "# 裁决用词",
            "",
            "每个窗口分别填写。没有足够证据时使用规定的缺失类别，不要留成检测器的默认阴性。",
            "",
            "## test_command_status",
            "",
            "- `definite_test_command`",
            "- `definite_no_test_command`",
            "- `ambiguous`",
            "- `insufficient_evidence`",
            "- `command_unrecoverable`",
            "",
            "## execution_status",
            "",
            "- `command_issued`",
            "- `execution_confirmed`",
            "- `execution_not_confirmed`",
            "- `unknown`",
            "",
            "## test_output_status",
            "",
            "- `explicit_pass`",
            "- `explicit_failure`",
            "- `unknown_output`",
            "- `no_output_observed`",
            "- `not_applicable`",
            "",
            "另外记录支持结论的 message index、runner family、简短原文摘录、reviewer confidence 和备注。",
            "",
            "命令抽取状态描述的是参数里能否抽出命令字符串。它不是测试命令的裁决，也不是测试是否通过的裁决。",
            "",
        ]
    )


def _summary(status: str, run_dir: Path, design: dict[str, Any], sealed: pd.DataFrame, problems: list[str], tests_ok: bool) -> str:
    summary = design["summary"]
    lines = [
        "# V3.5 Independent Measurement Validation",
        "",
        f"状态：`{status}`",
        "",
        "这次交付的是未裁决的独立测量核对包。检测器没有被人工核对成已验证，也没有估计精确率或漏检率。",
        "",
        f"- Census：`{CENSUS_PATH}`",
        f"- 分片缓存：`{DATA_ROOT}`",
        f"- 本次运行：`{run_dir}`",
        f"- Revision：`{PINNED_REVISION}`",
        f"- 抽样种子：`{SEED}`",
        f"- 审阅包轨迹数：{sealed['review_id'].nunique() if len(sealed) else 0:,}",
        f"- 测试是否通过：`{tests_ok}`",
        "",
        "## 抽样",
        "",
        "主窗口是前 15 个完成轮次，因为四个 scaffold/model 组都有可观察的共同支持。前 5 轮保留为次级窗口。这是测量核对的设计选择，不是预测价值判断。",
        "",
        "缺失命令的轨迹全部进入诊断层，不进入已知阴性。含糊测试命令单独抽样，不并入主精度估计。同一条轨迹可以同时属于主层和诊断层；审阅包里只出现一次，密封键保留每一层的纳入概率。",
        "",
        "| Stratum | Group | Population | Sample | Inclusion probability |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        probability = "" if row.inclusion_probability is None or pd.isna(row.inclusion_probability) else f"{row.inclusion_probability:.6f}"
        lines.append(
            f"| {row.stratum} | {row.scaffold} / {row.model} | {int(row.population_size):,} | {int(row.sample_size):,} | {probability} |"
        )
    overlap = design.get("masks_overlap", {})
    lines.extend(
        [
            "",
            f"分层重叠计数：`{json.dumps(overlap, ensure_ascii=False)}`",
            "",
            f"R15 缺失命令的核实总数是 {int(summary.loc[summary['stratum']=='D_r15_missing_command', 'population_size'].sum()) if len(summary) else 0:,}。历史记录是 {EXPECTED_D:,}。",
            "",
            "## 估计",
            "",
            "现在没有人工裁决，所以 R5 阳性预测值、R15 阳性预测值、已知阴性中的可能漏检、缺失命令诊断、含糊命令诊断和输出状态一致性都还没有数值。",
            "",
            "抽样设计按 scaffold/model 保留了总体和样本量。裁决完成之后，主样本可以用分层有限总体权重汇总。同一 issue 中的多条轨迹不能当成独立观测，因此这次不计算合并的二项区间。0.90 的旧阈值也不用于这个测量工具。",
            "",
        ]
    )
    if len(sealed) and "issue_id" in sealed.columns:
        counts = sealed.groupby("issue_id").size()
        lines.append(f"审阅包涉及 {sealed['issue_id'].nunique():,} 个 issue，其中 {int((counts >= 2).sum()):,} 个 issue 有至少两条被抽中的轨迹。")
        lines.append("")
    lines.extend(
        [
            "## 判断",
            "",
            "1. 原始轨迹证据已经按冻结的前缀规则抽出，裁决栏仍是空的。检测器还没有经过独立人工核对。",
            "2. 各 scaffold/model 组的阳性和已知阴性样本量见上面的抽样表。",
            "3. 缺失命令和含糊命令都有单独的层，没有并入已知阴性或主精度样本。",
            "4. R15 的精确率和已知阴性中的漏检可以在裁决之后按分层权重估计。当前没有裁决，区间也没有估计。",
            "5. 测量质量还没有建立，不能据此进入探索性的 issue 分层关联分析。",
            "6. 前 5 轮在 SWE-agent / Qwen 只有很少的阳性轨迹；前 15 轮缺失命令仍需单独阅读；同一 issue 的轨迹相关。这些限制在裁决前保持不变。",
            "",
            "## 没有做的事",
            "",
            "- 没有裁决样本，没有把模型输出当成人工核对。",
            "- 没有回归、预测模型、AUPRC、ROC-AUC 或 Delta AP。",
            "- 没有修改检测器、parser v2、L2、R5/R15 定义或历史结果。",
            "- 没有重新下载分片，没有重跑全库普查。",
            "- 没有开始 V3.6 或 L3。",
            "",
        ]
    )
    if problems:
        lines.extend(["## 未解决事项", ""])
        lines.extend(f"- {item}" for item in problems)
        lines.append("")
    return "\n".join(lines)


def _pytest(run_dir: Path) -> tuple[str, bool]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    plain = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    text = "$ python -m pytest -q\n" + (plain.stdout or "") + (plain.stderr or "")
    if plain.returncode == 0:
        return text, True
    tmp = STAGING / "_pytest_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)
    rerun = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(tmp / "base")],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    text += "\n\n$ python -m pytest -q -p no:cacheprovider --basetemp <staging>/_pytest_tmp/base\n"
    text += (rerun.stdout or "") + (rerun.stderr or "")
    shutil.rmtree(tmp, ignore_errors=True)
    _ = run_dir
    return text, rerun.returncode == 0


def _load_census() -> pd.DataFrame:
    columns = [
        "trajectory_id",
        "issue_id",
        "scaffold",
        "model",
        "shard",
        "resolved",
        "census_parse_ok",
        "v3_ran_tests_r5",
        "v3_ran_tests_r15",
        "v3_ran_tests_amb_r5",
        "v3_ran_tests_amb_r15",
        "exec_missing_r5",
        "exec_missing_r15",
        "n_rounds_observed",
        "prefix5_status",
        "prefix15_status",
    ]
    return pd.read_parquet(CENSUS_PATH, columns=columns)


def _extract_all(run_dir: Path, sealed: pd.DataFrame, pinned: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    problems: list[str] = []
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    wanted: dict[str, dict[str, dict[str, str]]] = {}
    for row in sealed.itertuples(index=False):
        wanted.setdefault(str(row.shard), {})[str(row.trajectory_id)] = {
            "review_id": str(row.review_id),
            "scaffold": str(row.scaffold),
        }
    evidence_rows = []
    for shard_index, (relative, items) in enumerate(sorted(wanted.items()), start=1):
        checkpoint = work / (hashlib.sha256(relative.encode("utf-8")).hexdigest() + ".jsonl")
        if checkpoint.is_file():
            evidence_rows.extend(json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines() if line)
            print(f"SHARD_REUSE {shard_index}/{len(wanted)} {relative}", flush=True)
            continue
        spec = pinned.get(relative)
        path = locate_shard(DATA_ROOT, relative)
        if spec is None or path is None:
            problems.append(f"shard missing: {relative}")
            continue
        if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != str(spec["sha256"]):
            problems.append(f"shard hash mismatch: {relative}")
            continue
        found = _stream_selected(path, set(items))
        missing_ids = sorted(set(items).difference(found))
        if missing_ids:
            problems.append(f"{relative} did not contain {len(missing_ids)} sampled ids")
            continue
        shard_rows = []
        for trajectory_id, (row_index, trajectory) in found.items():
            evidence = extract_prefix_evidence(trajectory, items[trajectory_id]["scaffold"])
            evidence["review_id"] = items[trajectory_id]["review_id"]
            evidence["source_row_index"] = int(row_index)
            shard_rows.append(evidence)
        checkpoint.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard_rows), encoding="utf-8")
        evidence_rows.extend(shard_rows)
        print(f"SHARD_DONE {shard_index}/{len(wanted)} {relative} extracted={len(shard_rows)}", flush=True)
    evidence = pd.DataFrame(evidence_rows)
    return evidence, problems


def _hash_outputs(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "V3_5_SHA256_MANIFEST.csv":
            rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(rows).to_csv(run_dir / "V3_5_SHA256_MANIFEST.csv", index=False)


def run(run_dir: Path, *, tests_text: str, tests_ok: bool) -> str:
    problems = _check_sources()
    (run_dir / "V3_5_TEST_RESULTS.txt").write_text(tests_text, encoding="utf-8")
    if problems:
        status = STATUS_BLOCKED
        summary = _summary(status, run_dir, {"summary": pd.DataFrame(), "masks_overlap": {}}, pd.DataFrame(columns=["review_id"]), problems, tests_ok)
        (run_dir / "V3_5_SUMMARY.md").write_text(summary, encoding="utf-8")
        (run_dir / "V3_5_RUN_MANIFEST.json").write_text(json.dumps({"status": status, "problems": problems}, indent=2), encoding="utf-8")
        _hash_outputs(run_dir)
        return status
    census = _load_census()
    design = design_sample(census)
    attributes = census[
        [
            "trajectory_id",
            "issue_id",
            "scaffold",
            "model",
            "shard",
            "resolved",
            "n_rounds_observed",
            "prefix5_status",
            "prefix15_status",
        ]
    ].copy()
    attributes["trajectory_id"] = attributes["trajectory_id"].astype(str)
    sealed = sealed_records(design["selected"], attributes, design["review_ids"])
    evidence, extract_problems = _extract_all(run_dir, sealed, _pinned_files())
    problems.extend(extract_problems)
    if evidence.empty or problems:
        status = STATUS_BLOCKED if any("missing" in item or "hash" in item for item in problems) else STATUS_PARTIAL
        summary = _summary(status, run_dir, design, sealed, problems, tests_ok)
        (run_dir / "V3_5_SUMMARY.md").write_text(summary, encoding="utf-8")
        (run_dir / "V3_5_RUN_MANIFEST.json").write_text(json.dumps({"status": status, "problems": problems}, indent=2, ensure_ascii=False), encoding="utf-8")
        design["frame"].to_csv(run_dir / "sampling_frame.csv", index=False)
        design["summary"].to_csv(run_dir / "sampling_design_summary.csv", index=False)
        _hash_outputs(run_dir)
        return status
    evidence["review_id"] = evidence["review_id"].astype(str)
    merged = sealed.merge(evidence[["review_id", "completed_rounds", "prefix5_boundary", "prefix15_boundary", "prefix_parse_status", "source_row_index"]], on="review_id", how="left", validate="one_to_one")
    round_mismatch = int((merged["n_rounds_observed"].astype("Int64") != pd.to_numeric(merged["completed_rounds"], errors="coerce").astype("Int64")).sum())
    boundary_mismatch = int((merged["prefix15_status"].astype(str) != merged["prefix15_boundary"].astype(str)).sum())
    if round_mismatch or boundary_mismatch:
        problems.append(f"prefix mismatch rounds={round_mismatch} boundary={boundary_mismatch}")
    d_population = int(design["summary"].loc[design["summary"]["stratum"] == "D_r15_missing_command", "population_size"].sum())
    if d_population != EXPECTED_D:
        problems.append(f"R15 missing-command population is {d_population}, historical record is {EXPECTED_D}")
    queue = blank_review_queue(merged)
    evidence_path = run_dir / "review_evidence.jsonl"
    ordered = evidence.sort_values("review_id")
    public_columns = ["review_id", "completed_rounds", "prefix_parse_status", "prefix5_boundary", "prefix15_boundary", "r5", "r15"]
    with evidence_path.open("w", encoding="utf-8") as handle:
        for record in ordered[public_columns].to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    instructions = run_dir / "validation_instructions.md"
    rules = run_dir / "adjudication_rules.md"
    instructions.write_text(_instructions(), encoding="utf-8")
    rules.write_text(_rules(), encoding="utf-8")
    queue_path = run_dir / "review_queue.csv"
    queue.to_csv(queue_path, index=False)
    design["frame"].to_csv(run_dir / "sampling_frame.csv", index=False)
    design["summary"].to_csv(run_dir / "sampling_design_summary.csv", index=False)
    sealed_out = merged.drop(columns=[column for column in ("r5", "r15") if column in merged.columns])
    sealed_out.to_csv(run_dir / "sealed_sampling_key.csv", index=False)
    archive = write_reviewer_zip(
        run_dir / "validation_bundle.zip",
        {
            "review_queue.csv": queue_path,
            "review_evidence.jsonl": evidence_path,
            "validation_instructions.md": instructions,
            "adjudication_rules.md": rules,
        },
    )
    status = STATUS_PREPARED if tests_ok and not problems else STATUS_PARTIAL
    summary = _summary(status, run_dir, design, merged, problems, tests_ok)
    (run_dir / "V3_5_SUMMARY.md").write_text(summary, encoding="utf-8")
    manifest = {
        "status": status,
        "revision": PINNED_REVISION,
        "seed": SEED,
        "census_sha256": sha256_file(CENSUS_PATH),
        "metadata_sha256": sha256_file(META_PATH),
        "problems": problems,
        "tests_passed": tests_ok,
        "review_trajectories": int(merged["review_id"].nunique()),
        "primary_strata": list(PRIMARY_STRATA),
        "diagnostic_strata": list(DIAGNOSTIC_STRATA),
        "zip_sha256": sha256_file(archive),
        "zip_bytes": archive.stat().st_size,
        "models_fit": False,
        "adjudication_completed": False,
    }
    (run_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "V3_5_RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _hash_outputs(run_dir)
    review = ROOT / "analysis_v2" / "V3_5_VALIDATION_REVIEW.md"
    if review.exists():
        review = review.with_name(review.stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + review.suffix)
    review.write_text(summary, encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            raise SystemExit(f"resume directory is absent: {run_dir}")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = STAGING / "runs" / f"v3_5_validation_{stamp}"
        run_dir.mkdir(parents=True)
    tests_text, tests_ok = _pytest(run_dir)
    status = run(run_dir, tests_text=tests_text, tests_ok=tests_ok)
    print(status)
    print(run_dir)


if __name__ == "__main__":
    main()
