"""Fixed L2 logistic models, paired bootstrap, and issue-weighted scores.

The candidate grid and the positive class are fixed. This module does not
read a dataset and does not select features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from p8audit.dependence import issue_equal_trajectory_weights, repository_cluster_bootstrap_indices

CANDIDATE_C = (0.01, 0.1, 1.0, 10.0)
MAX_ITER = 5000
BOOTSTRAP_SEED = 20261092
BOOTSTRAP_REPLICATES = 1000


def make_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    max_iter=MAX_ITER,
                    class_weight=None,
                ),
            ),
        ]
    )


def select_c(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    best_c = CANDIDATE_C[0]
    best_score = -1.0
    for c_value in CANDIDATE_C:
        pipeline = make_pipeline(c_value)
        pipeline.fit(x_train, y_train)
        score = float(average_precision_score(y_valid, pipeline.predict_proba(x_valid)[:, 1]))
        scores[str(c_value)] = score
        if score > best_score + 1e-15 or (abs(score - best_score) <= 1e-15 and c_value < best_c):
            best_score = score
            best_c = c_value
    return best_c, scores


def fit_selected(x_fit: np.ndarray, y_fit: np.ndarray, c_value: float) -> Pipeline:
    pipeline = make_pipeline(c_value)
    pipeline.fit(x_fit, y_fit)
    return pipeline


@dataclass(frozen=True)
class BootstrapResult:
    deltas: list[float]
    structural_ap: list[float]
    augmented_ap: list[float]
    skipped_replicates: list[int]
    n_requested: int

    @property
    def n_valid(self) -> int:
        return len(self.deltas)

    def percentile_interval(self) -> tuple[float, float]:
        if not self.deltas:
            raise ValueError("no valid bootstrap replicates")
        values = np.asarray(self.deltas, dtype=float)
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)


def paired_bootstrap(
    y: np.ndarray,
    structural_proba: np.ndarray,
    augmented_proba: np.ndarray,
    repos: list[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    n_replicates: int = BOOTSTRAP_REPLICATES,
) -> BootstrapResult:
    """One repository-cluster resample, then both scores on those same rows."""
    if len(y) != len(structural_proba) or len(y) != len(augmented_proba) or len(y) != len(repos):
        raise ValueError("bootstrap inputs must describe the same rows")
    rng = np.random.default_rng(seed)
    replicates = repository_cluster_bootstrap_indices(repos, rng, n_replicates)
    deltas: list[float] = []
    structural: list[float] = []
    augmented: list[float] = []
    skipped: list[int] = []
    y_array = np.asarray(y)
    structural_array = np.asarray(structural_proba)
    augmented_array = np.asarray(augmented_proba)
    for replicate_index, indexes in enumerate(replicates):
        if not indexes:
            skipped.append(replicate_index)
            continue
        labels = y_array[indexes]
        if np.unique(labels).size < 2:
            skipped.append(replicate_index)
            continue
        structural_score = float(average_precision_score(labels, structural_array[indexes]))
        augmented_score = float(average_precision_score(labels, augmented_array[indexes]))
        structural.append(structural_score)
        augmented.append(augmented_score)
        deltas.append(augmented_score - structural_score)
    return BootstrapResult(deltas, structural, augmented, skipped, n_replicates)


def issue_weighted_average_precision(y: np.ndarray, proba: np.ndarray, instance_ids: list[str]) -> float:
    weights = issue_equal_trajectory_weights(instance_ids)
    return float(average_precision_score(y, proba, sample_weight=np.asarray(weights, dtype=float)))
