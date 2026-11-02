"""V3.5 sampling design.

Selection uses scaffold/model strata and the saved test-command flags.
Outcome labels are dropped before sampling and never decide inclusion.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

SEED = 202611031
GROUPS = (
    ("openhands", "minimax_m25"),
    ("openhands", "qwen35_122b"),
    ("sweagent", "minimax_m25"),
    ("sweagent", "qwen35_122b"),
)
PRIMARY_STRATA = (
    "A_r5_positive",
    "B_r15_positive_r5_known_negative",
    "C_r15_known_negative",
    "D_r15_missing_command",
)
DIAGNOSTIC_STRATA = ("AMB_r5", "AMB_r15")
STRATA = PRIMARY_STRATA + DIAGNOSTIC_STRATA
TARGET = {
    "A_r5_positive": 50,
    "B_r15_positive_r5_known_negative": 50,
    "C_r15_known_negative": 50,
    "D_r15_missing_command": None,
    "AMB_r5": 25,
    "AMB_r15": 25,
}
METHOD = {
    "A_r5_positive": "simple_random_sample_without_replacement",
    "B_r15_positive_r5_known_negative": "simple_random_sample_without_replacement",
    "C_r15_known_negative": "simple_random_sample_without_replacement",
    "D_r15_missing_command": "include_all",
    "AMB_r5": "simple_random_sample_without_replacement",
    "AMB_r15": "simple_random_sample_without_replacement",
}
IGNORED_COLUMNS = ("resolved", "reference_patch", "model_patch", "reasoning_content")
REVIEWER_LABEL_COLUMNS = (
    "test_command_status_r5",
    "execution_status_r5",
    "test_output_status_r5",
    "supporting_message_indexes_r5",
    "runner_family_r5",
    "evidence_quotation_r5",
    "reviewer_confidence_r5",
    "adjudication_notes_r5",
    "test_command_status_r15",
    "execution_status_r15",
    "test_output_status_r15",
    "supporting_message_indexes_r15",
    "runner_family_r15",
    "evidence_quotation_r15",
    "reviewer_confidence_r15",
    "adjudication_notes_r15",
)
FORBIDDEN_REVIEWER_COLUMNS = {
    "resolved",
    "trajectory_id",
    "issue_id",
    "scaffold",
    "model",
    "shard",
    "stratum",
    "sampling_stratum",
    "inclusion_probability",
    "reference_patch",
    "model_patch",
    "reasoning_content",
    "v3_ran_tests_r5",
    "v3_ran_tests_r15",
    "v3_ran_tests_amb_r5",
    "v3_ran_tests_amb_r15",
    "parser_v2_strict_fail_r5",
}


def _seed(stratum: str, scaffold: str, model: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}|{stratum}|{scaffold}|{model}".encode("utf-8")).digest()
    return np.random.default_rng(np.random.SeedSequence(int.from_bytes(digest[:8], "little")))


def _eligible(frame: pd.DataFrame) -> dict[str, pd.Series]:
    ok = frame["census_parse_ok"].map(lambda value: value is True or value == 1).astype(bool)
    ran5 = pd.to_numeric(frame["v3_ran_tests_r5"], errors="coerce")
    ran15 = pd.to_numeric(frame["v3_ran_tests_r15"], errors="coerce")
    miss5 = pd.to_numeric(frame["exec_missing_r5"], errors="coerce")
    miss15 = pd.to_numeric(frame["exec_missing_r15"], errors="coerce")
    amb5 = pd.to_numeric(frame["v3_ran_tests_amb_r5"], errors="coerce")
    amb15 = pd.to_numeric(frame["v3_ran_tests_amb_r15"], errors="coerce")
    return {
        "A_r5_positive": ok & ran5.eq(1),
        "B_r15_positive_r5_known_negative": ok & ran15.eq(1) & ran5.eq(0) & miss5.eq(0),
        "C_r15_known_negative": ok & ran15.eq(0) & miss15.eq(0),
        "D_r15_missing_command": ok & ran15.eq(0) & miss15.gt(0),
        "AMB_r5": ok & amb5.eq(1),
        "AMB_r15": ok & amb15.eq(1),
    }


def _choose(ids: list[str], target: int | None, stratum: str, scaffold: str, model: str) -> list[str]:
    ordered = sorted(set(ids))
    if target is None or len(ordered) <= target:
        return ordered
    rng = _seed(stratum, scaffold, model)
    picked = rng.choice(len(ordered), size=target, replace=False)
    return [ordered[int(index)] for index in picked]


def assign_review_ids(trajectory_ids: list[str]) -> dict[str, str]:
    """Shuffle package order so it does not follow stratum or group."""
    ordered = sorted(set(trajectory_ids))
    rng = _seed("review_order", "all", "all")
    permutation = rng.permutation(len(ordered))
    return {ordered[int(index)]: f"V35-{rank + 1:06d}" for rank, index in enumerate(permutation)}


def design_sample(frame: pd.DataFrame) -> dict[str, Any]:
    """Sample every stratum inside each scaffold/model group."""
    work = frame.drop(columns=[column for column in IGNORED_COLUMNS if column in frame.columns], errors="ignore").copy()
    missing = [column for column in ("trajectory_id", "scaffold", "model", "shard") if column not in work.columns]
    if missing:
        raise ValueError(f"required columns missing: {missing}")
    masks = _eligible(work)
    frame_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for stratum in STRATA:
        mask = masks[stratum]
        for scaffold, model in GROUPS:
            group_mask = mask & work["scaffold"].eq(scaffold) & work["model"].eq(model)
            population_ids = work.loc[group_mask, "trajectory_id"].astype(str).tolist()
            chosen = _choose(population_ids, TARGET[stratum], stratum, scaffold, model)
            chosen_set = set(chosen)
            population = len(set(population_ids))
            sample_size = len(chosen_set)
            probability = None if population == 0 else sample_size / population
            summary_rows.append(
                {
                    "stratum": stratum,
                    "scaffold": scaffold,
                    "model": model,
                    "population_size": population,
                    "sample_size": sample_size,
                    "sampling_seed": SEED,
                    "inclusion_probability": probability,
                    "sampling_method": METHOD[stratum],
                    "role": "diagnostic" if stratum in DIAGNOSTIC_STRATA else "primary",
                }
            )
            lookup = work.loc[group_mask, ["trajectory_id", "shard"]].drop_duplicates("trajectory_id")
            shards = dict(zip(lookup["trajectory_id"].astype(str), lookup["shard"].astype(str), strict=True))
            for trajectory_id in sorted(set(population_ids)):
                frame_rows.append(
                    {
                        "stratum": stratum,
                        "scaffold": scaffold,
                        "model": model,
                        "trajectory_id": trajectory_id,
                        "shard": shards[trajectory_id],
                        "population_size": population,
                        "sample_size": sample_size,
                        "inclusion_probability": probability,
                        "selected": int(trajectory_id in chosen_set),
                        "sampling_seed": SEED,
                        "sampling_method": METHOD[stratum],
                        "role": "diagnostic" if stratum in DIAGNOSTIC_STRATA else "primary",
                    }
                )
            selected_ids.update(chosen_set)
    membership = pd.DataFrame(frame_rows)
    selected = membership.loc[membership["selected"] == 1].copy()
    review_ids = assign_review_ids(sorted(selected_ids))
    return {
        "frame": membership,
        "summary": pd.DataFrame(summary_rows),
        "selected": selected,
        "review_ids": review_ids,
        "masks_overlap": _overlap_counts(masks),
    }


def _overlap_counts(masks: dict[str, pd.Series]) -> dict[str, int]:
    return {
        "A_and_B": int((masks["A_r5_positive"] & masks["B_r15_positive_r5_known_negative"]).sum()),
        "A_and_C": int((masks["A_r5_positive"] & masks["C_r15_known_negative"]).sum()),
        "A_and_D": int((masks["A_r5_positive"] & masks["D_r15_missing_command"]).sum()),
        "B_and_C": int((masks["B_r15_positive_r5_known_negative"] & masks["C_r15_known_negative"]).sum()),
        "C_and_D": int((masks["C_r15_known_negative"] & masks["D_r15_missing_command"]).sum()),
        "A_and_AMB_r5": int((masks["A_r5_positive"] & masks["AMB_r5"]).sum()),
        "B_and_AMB_r15": int((masks["B_r15_positive_r5_known_negative"] & masks["AMB_r15"]).sum()),
        "C_and_AMB_r15": int((masks["C_r15_known_negative"] & masks["AMB_r15"]).sum()),
        "D_and_AMB_r15": int((masks["D_r15_missing_command"] & masks["AMB_r15"]).sum()),
        "AMB_r5_and_AMB_r15": int((masks["AMB_r5"] & masks["AMB_r15"]).sum()),
    }


def sealed_records(selected: pd.DataFrame, attributes: pd.DataFrame, review_ids: dict[str, str]) -> pd.DataFrame:
    """One reviewer-blind row per sampled trajectory, with every stratum probability kept."""
    base = attributes.drop_duplicates("trajectory_id").copy()
    base["trajectory_id"] = base["trajectory_id"].astype(str)
    rows = []
    for trajectory_id, group in selected.groupby("trajectory_id", sort=True):
        record: dict[str, Any] = {"trajectory_id": str(trajectory_id), "review_id": review_ids[str(trajectory_id)]}
        for stratum in STRATA:
            part = group.loc[group["stratum"] == stratum]
            record[f"member_{stratum}"] = int(len(part) > 0)
            record[f"pi_{stratum}"] = None if part.empty else float(part["inclusion_probability"].iloc[0])
        rows.append(record)
    sealed = pd.DataFrame(rows).merge(base, on="trajectory_id", how="left", validate="one_to_one")
    return sealed.sort_values("review_id").reset_index(drop=True)


def blank_review_queue(sealed: pd.DataFrame) -> pd.DataFrame:
    """Reviewer table. Membership, predictions, and outcomes stay out."""
    columns = ["review_id", "completed_rounds", "prefix5_boundary", "prefix15_boundary", "prefix_parse_status"]
    queue = sealed[columns].copy() if set(columns).issubset(sealed.columns) else sealed[["review_id"]].copy()
    for column in REVIEWER_LABEL_COLUMNS:
        queue[column] = ""
    unexpected = FORBIDDEN_REVIEWER_COLUMNS.intersection(queue.columns)
    if unexpected:
        raise ValueError(f"reviewer queue contains sealed fields: {sorted(unexpected)}")
    if queue[list(REVIEWER_LABEL_COLUMNS)].ne("").any().any():
        raise ValueError("reviewer labels were prefilled")
    return queue.sort_values("review_id").reset_index(drop=True)
