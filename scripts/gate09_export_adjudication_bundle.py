"""Export the existing Gate 0.9 validation rows for AI adjudication.

Reads the saved package only. Does not modify it, the parser, or the
feature dictionary. Does not read parquet or calculate a precision bound.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(r"C:\Users\hanqi\Desktop\projects\p8-coding-agent-uncertainty-audit")
SOURCE = ROOT / "artifacts" / "manual_review" / "gate09_parser_v2_validation"
DEST = ROOT / "artifacts" / "manual_review" / "gate09_ai_adjudication_export"
ZIP_PATH = ROOT / "artifacts" / "manual_review" / "P8_GATE09_AI_ADJUDICATION_BUNDLE.zip"
PINNED = "c144d9d160dcd9f5ebcd9520d2d54111dda8b681"

FILES = (
    "explicit_exception_validation.csv",
    "explicit_test_failure_validation.csv",
    "openhands_exit_status_validation.csv",
    "diagnostic_validation.csv",
    "random_observation_validation.csv",
)
EXPECTED_ROWS = {
    "explicit_exception_validation.csv": 50,
    "explicit_test_failure_validation.csv": 50,
    "openhands_exit_status_validation.csv": 50,
    "diagnostic_validation.csv": 84,
    "random_observation_validation.csv": 24,
}
BUNDLE_NAMES = FILES + (
    "README.md",
    "ADJUDICATION_RULES.md",
    "SAMPLE_SCOPE.md",
    "CONTEXT_LIMITATIONS.md",
    "package_manifest.json",
)
FORBIDDEN = {
    "resolved",
    "reference_patch",
    "model_patch",
    "reasoning_content",
    "execution_failure",
    "ai_label",
    "reviewer_label_proposed",
}
UNTOUCHED = (
    SOURCE / "README.md",
    SOURCE / "VALIDATION_INSTRUCTIONS.md",
    SOURCE / "package_manifest.json",
    *(SOURCE / name for name in FILES),
    ROOT / "src" / "p8audit" / "execution_signals.py",
    ROOT / "src" / "p8audit" / "execution_signals_v2.py",
    ROOT / "configs" / "feature_spec.yaml",
    ROOT / "configs" / "feature_spec_v2.yaml",
    ROOT / "artifacts" / "manual_review" / "marker_review_queue.csv",
    ROOT / "artifacts" / "manual_review" / "marker_review_queue_v2.csv",
    ROOT / "artifacts" / "manual_review" / "gate08_review_package" / "package_manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def snapshot() -> dict[str, str]:
    found = {}
    for path in UNTOUCHED:
        if not path.is_file():
            raise SystemExit(f"required source missing: {path}")
        found[str(path)] = sha256(path)
    return found


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def validate_table(name: str, fields: list[str], rows: list[dict[str, str]]) -> None:
    leaked = sorted(FORBIDDEN.intersection(fields))
    if leaked:
        raise SystemExit(f"{name} contains forbidden field {leaked[0]}")
    if len(rows) != EXPECTED_ROWS[name]:
        raise SystemExit(f"{name} has {len(rows)} rows, expected {EXPECTED_ROWS[name]}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{name} has a duplicate sample_id")
    for row in rows:
        if str(row.get("reviewer_label") or "") != "":
            raise SystemExit(f"{name} has a reviewer_label")
        if str(row.get("student_verified") or "") != "false":
            raise SystemExit(f"{name} student_verified is not false")
        if str(row.get("dataset_revision") or "") != PINNED:
            raise SystemExit(f"{name} revision is not the pin")
        if str(row.get("parser_version") or "") != "v2":
            raise SystemExit(f"{name} parser version is not v2")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    before = snapshot()
    source_manifest = json.loads((SOURCE / "package_manifest.json").read_text(encoding="utf-8"))
    dictionary = yaml.safe_load((ROOT / "configs" / "feature_spec_v2.yaml").read_text(encoding="utf-8"))
    if dictionary.get("dictionary_version") != "v2" or dictionary.get("parser_version") != "v2":
        raise SystemExit("feature dictionary version is not v2")

    tables = {}
    source_hashes = {}
    for name in FILES:
        path = SOURCE / name
        source_hashes[name] = sha256(path)
        fields, rows = read_rows(path)
        validate_table(name, fields, rows)
        tables[name] = (fields, rows)

    all_ids = [row["sample_id"] for _, rows in tables.values() for row in rows]
    if len(all_ids) != 258 or len(set(all_ids)) != 258:
        raise SystemExit(f"expected 258 unique sample ids, found {len(all_ids)} rows and {len(set(all_ids))} ids")

    by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    truncated = []
    empty_context = []
    for name, (_, rows) in tables.items():
        for row in rows:
            by_key[row["observation_key"]].append(row)
            if str(row.get("excerpt_truncated") or "").lower() == "true" or str(row.get("context_excerpt") or "").endswith("[TRUNCATED]"):
                truncated.append(row)
            if not str(row.get("context_excerpt") or "").strip():
                empty_context.append(row)

    repeated = {key: items for key, items in by_key.items() if len(items) > 1}
    shards = source_manifest["shards"]
    if len(shards) != 8:
        raise SystemExit(f"expected 8 shards in the source manifest, found {len(shards)}")

    if DEST.exists():
        for child in sorted(DEST.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        payload = (SOURCE / name).read_bytes()
        if sha256_bytes(payload) != source_hashes[name]:
            raise SystemExit(f"source changed while reading {name}")
        (DEST / name).write_bytes(payload)

    frame_lines = []
    for name in FILES:
        _, rows = tables[name]
        probabilities = sorted({row["sampling_probability"] for row in rows})
        families = Counter(row["signal_family"] for row in rows)
        purposes = sorted({row["review_purpose"] for row in rows})
        keys = {row["observation_key"] for row in rows}
        frame_lines.append(
            {
                "file": name,
                "rows": len(rows),
                "unique_observation_keys": len(keys),
                "review_purpose": purposes,
                "signal_families": dict(families),
                "sampling_probabilities": probabilities,
            }
        )

    population = source_manifest["population_on_fresh_shards"]
    precision_frames = {
        "explicit_exception_validation.csv": {
            "population": "parser-v2 predicted explicit_exception on the eight fresh shards",
            "predicted_positives": population["explicit_exception"],
            "sample_size": 50,
            "inclusion_probability": frame_lines[0]["sampling_probabilities"][0],
            "use": "precision sample only",
        },
        "explicit_test_failure_validation.csv": {
            "population": "parser-v2 predicted strict explicit_test_failure on the eight fresh shards",
            "predicted_positives": population["explicit_test_failure"],
            "sample_size": 50,
            "inclusion_probability": frame_lines[1]["sampling_probabilities"][0],
            "use": "precision sample only",
        },
        "openhands_exit_status_validation.csv": {
            "population": "parser-v2 predicted tool_exit_nonzero on the OpenHands fresh shards",
            "predicted_positives": population["tool_exit_nonzero"],
            "sample_size": 50,
            "inclusion_probability": frame_lines[2]["sampling_probabilities"][0],
            "use": "OpenHands precision sample only",
        },
        "diagnostic_validation.csv": {
            "population": "stratified diagnostic hits on the eight fresh shards",
            "predicted_positives": {
                family: population[family]
                for family in frame_lines[3]["signal_families"]
            },
            "sample_size": 84,
            "inclusion_probability": frame_lines[3]["sampling_probabilities"],
            "use": "not a precision or recall sample",
        },
        "random_observation_validation.csv": {
            "population": "tool observations on the eight fresh shards",
            "predicted_positives": population["random_observation"],
            "sample_size": 24,
            "inclusion_probability": frame_lines[4]["sampling_probabilities"][0],
            "use": "exploratory error analysis only",
        },
    }

    limitation_rows = []
    for row in truncated:
        limitation_rows.append(
            f"| `{row['sample_id']}` | `{row['signal_family']}` | {row['parquet_shard']} | {row['n_chars']} | saved excerpt ends at the original cap |"
        )
    for row in empty_context:
        limitation_rows.append(
            f"| `{row['sample_id']}` | `{row['signal_family']}` | {row['parquet_shard']} | {row['n_chars']} | saved excerpt is empty |"
        )
    if not limitation_rows:
        limitation_body = "No saved excerpt is truncated or empty.\n"
    else:
        limitation_body = "\n".join(
            [
                "| sample_id | signal_family | shard | saved characters | limitation |",
                "|---|---|---|---:|---|",
                *limitation_rows,
                "",
            ]
        )

    repeated_lines = []
    for key in sorted(repeated):
        items = repeated[key]
        families = ", ".join(f"{item['sample_id']}:{item['signal_family']}" for item in items)
        repeated_lines.append(f"- `{key}` — {families}")
    if not repeated_lines:
        repeated_body = "No observation key is repeated in this export.\n"
    else:
        repeated_body = "\n".join(repeated_lines) + "\n"

    shard_lines = [
        "| Group | Path | Bytes | Rows | Tool observations |",
        "|---|---|---:|---:|---:|",
    ]
    for shard in shards:
        shard_lines.append(
            f"| {shard['config']} / {shard['dataset_split']} | `{shard['path']}` | {shard['manifest_bytes']} | {shard['rows']} | {shard['tool_observations']} |"
        )

    readme = f"""# P8 Gate 0.9 AI adjudication bundle

This package contains 258 observations from the eight fresh validation shards and is intended for AI-assisted adjudication.

The rows are exact copies of the unlabeled parser-v2 validation package. No reviewer label has been filled. `student_verified` is false. `adjudication` remains `pending` and is not a judgment.

Dataset revision: `{PINNED}`.
Parser version: `v2`.
Feature dictionary version: `v2`.

Do not treat a parser field as a human or AI label. Do not calculate a confidence bound from this export. Diagnostic rows and the random observation rows are not precision or recall samples. This bundle does not include Gate 0.8 judgments.

Return any adjudication in a separate file. Do not overwrite these CSVs.
"""

    rules = """# Adjudication rules

These rules are the Gate 0.9 rules. This export does not change them.

## Explicit exception

A positive requires evidence of an actual exception event, such as a traceback, runtime exception, or explicit execution-time exception event.

A test name, property name, source-code fragment, configuration key, or explanatory mention of `Error` or `Exception` is not sufficient by itself.

## Strict explicit test failure

A positive requires an explicitly observed failed test case or test summary.

Keep these separate:

- Build or compilation failure.
- Test setup or collection failure.
- Unresolved package-level `FAIL`.
- Confirmed test failure.

Do not infer a failed assertion from a generic package failure.

## OpenHands nonzero exit status

Require an explicit OpenHands wrapper exit code that meets the existing parser v2 rule.

Do not infer missing SWE-agent exit codes or encode missingness as zero.

## Diagnostics

Keep interruption/timeout, unknown status, discordant exit status and test result, and textual mentions of interruption separate.

Do not combine them into `execution_failure`.

## Sampling and statistical interpretation

The three precision samples are simple random samples of predicted-positive observations within the eight selected fresh shards.

The diagnostic and random-observation samples are not substitutes for the precision samples and must not be used to estimate overall precision or recall.

Do not calculate confidence bounds in this export task.

## Labels

Leave `reviewer_label` blank in this bundle. If a later adjudication is recorded, mark it as AI-assisted. Do not set `student_verified` to true. Ambiguous or insufficient saved context stays unresolved; do not drop those rows from a later operational denominator.
"""

    scope = f"""# Sample scope

The population for every file is the eight fresh shards listed below. It is not all 84 pinned shards. Selection seed: `{source_manifest['fresh_shard_seed']}`. Observation seed: `{source_manifest['observation_seed']}`. Outcomes, marker hit counts, and earlier feature performance were not used to choose the shards. `resolved` was not read.

{chr(10).join(shard_lines)}

Downloaded body bytes recorded by the source package: {source_manifest['download_bytes_manifest']}.

| File | What the sample represents | Frame | Sample | Inclusion probability |
|---|---|---:|---:|---|
| `explicit_exception_validation.csv` | Predicted `explicit_exception` | {population['explicit_exception']} | 50 | {frame_lines[0]['sampling_probabilities'][0]} |
| `explicit_test_failure_validation.csv` | Predicted strict `explicit_test_failure` | {population['explicit_test_failure']} | 50 | {frame_lines[1]['sampling_probabilities'][0]} |
| `openhands_exit_status_validation.csv` | Predicted OpenHands `tool_exit_nonzero` | {population['tool_exit_nonzero']} | 50 | {frame_lines[2]['sampling_probabilities'][0]} |
| `diagnostic_validation.csv` | Diagnostic coverage by signal family | separate family frames | 84 | one probability within each family |
| `random_observation_validation.csv` | Exploratory tool observations | {population['random_observation']} | 24 | {frame_lines[4]['sampling_probabilities'][0]} |

Diagnostic family frames:

| Family | Hits on the eight shards | Rows in this export | Inclusion probability |
|---|---:|---:|---|
| `build_or_compile_failure` | {population['build_or_compile_failure']} | 12 | 12/{population['build_or_compile_failure']} |
| `test_setup_or_collection_failure` | {population['test_setup_or_collection_failure']} | 12 | 12/{population['test_setup_or_collection_failure']} |
| `package_level_failure_unresolved` | {population['package_level_failure_unresolved']} | 12 | 12/{population['package_level_failure_unresolved']} |
| `missed_test_format` | {population['missed_test_format']} | 12 | 12/{population['missed_test_format']} |
| `interrupt_mention_not_event` | {population['interrupt_mention_not_event']} | 12 | 12/{population['interrupt_mention_not_event']} |
| `discordant_exit_status_and_test_result` | {population['discordant_exit_status_and_test_result']} | 12 | 12/{population['discordant_exit_status_and_test_result']} |
| `execution_status_unknown` | {population['execution_status_unknown']} | 12 | 12/{population['execution_status_unknown']} |

SWE-agent predicted `tool_exit_nonzero` on these shards is 0. A missing SWE-agent exit code is not zero.

Repeated observation keys are separate rows. They are not merged:

{repeated_body}
"""

    limitations = f"""# Context limitations

Excerpts are the saved validation-package text. This export does not reread trajectory bodies and does not fill missing context.

Truncated saved excerpts: {len(truncated)}.
Empty saved excerpts: {len(empty_context)}.

{limitation_body}
A truncated or empty excerpt remains exactly as saved. If that text cannot show whether the event occurred, a later adjudication should record insufficient evidence. This file does not make that row-level decision.
"""

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "AI-assisted adjudication of the existing Gate 0.9 fresh-shard validation sample",
        "dataset_revision": PINNED,
        "parser_version": "v2",
        "feature_dictionary_version": dictionary["dictionary_version"],
        "feature_dictionary_path": "configs/feature_spec_v2.yaml",
        "fresh_shard_seed": source_manifest["fresh_shard_seed"],
        "observation_seed": source_manifest["observation_seed"],
        "selected_shard_paths": [shard["path"] for shard in shards],
        "source_directory": "artifacts/manual_review/gate09_parser_v2_validation",
        "source_files": [
            {"name": name, "sha256": source_hashes[name], "rows": EXPECTED_ROWS[name]}
            for name in FILES
        ],
        "row_counts": EXPECTED_ROWS,
        "total_rows": 258,
        "unique_sample_ids": 258,
        "unique_observation_counts": {
            item["file"]: item["unique_observation_keys"] for item in frame_lines
        },
        "unique_observation_keys_across_files": len(by_key),
        "repeated_observation_keys": {
            key: [
                {"sample_id": item["sample_id"], "signal_family": item["signal_family"], "file_signal": item["signal_family"]}
                for item in items
            ]
            for key, items in sorted(repeated.items())
        },
        "sampling_frames": precision_frames,
        "source_excerpt_truncated": len(truncated) > 0,
        "truncated_sample_ids": [row["sample_id"] for row in truncated],
        "empty_excerpt_sample_ids": [row["sample_id"] for row in empty_context],
        "labels_calculated": False,
        "confidence_bounds_calculated": False,
        "models_fit": False,
        "delta_auprc_calculated": False,
        "gate08_labels_included": False,
        "rows_relabelled": False,
    }

    write_text(DEST / "README.md", readme)
    write_text(DEST / "ADJUDICATION_RULES.md", rules)
    write_text(DEST / "SAMPLE_SCOPE.md", scope)
    write_text(DEST / "CONTEXT_LIMITATIONS.md", limitations)
    write_text(DEST / "package_manifest.json", json.dumps(manifest, indent=2) + "\n")

    export_hashes = {name: sha256(DEST / name) for name in BUNDLE_NAMES}
    for name in FILES:
        if export_hashes[name] != source_hashes[name]:
            raise SystemExit(f"export copy hash differs for {name}")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_NAMES:
            archive.write(DEST / name, arcname=name)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        if names != list(BUNDLE_NAMES):
            raise SystemExit(f"zip members differ: {names}")
        for name in BUNDLE_NAMES:
            payload = archive.read(name)
            if sha256_bytes(payload) != export_hashes[name]:
                raise SystemExit(f"zip hash differs for {name}")
            if name.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
                count = sum(1 for _ in reader)
                if count != EXPECTED_ROWS[name]:
                    raise SystemExit(f"zip row count differs for {name}: {count}")

    after = snapshot()
    for path, digest in before.items():
        if after[path] != digest:
            raise SystemExit(f"source changed: {path}")

    print(
        json.dumps(
            {
                "zip": str(ZIP_PATH),
                "zip_sha256": sha256(ZIP_PATH),
                "zip_bytes": ZIP_PATH.stat().st_size,
                "rows": EXPECTED_ROWS,
                "total_rows": 258,
                "unique_observation_keys": len(by_key),
                "repeated_observation_keys": len(repeated),
                "truncated": [row["sample_id"] for row in truncated],
                "empty": [row["sample_id"] for row in empty_context],
                "source_hashes_unchanged": True,
                "copy_hashes_match_source": True,
                "zip_hashes_match_manifest_files": True,
                "labels_calculated": False,
                "confidence_bounds_calculated": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
