"""Finite-population and binomial bounds for marker precision.

These functions do not read review labels and do not declare a feature frozen.
"""

from __future__ import annotations

from scipy.stats import hypergeom

from p8audit.review_stats import clopper_pearson_lower

__all__ = [
    "adjudicable_precision",
    "clopper_pearson_lower",
    "conservative_lower_bounds",
    "hypergeometric_lower_count",
    "hypergeometric_lower_proportion",
]


def hypergeometric_lower_count(
    observed_positives: int,
    sample_size: int,
    population_size: int,
    alpha: float = 0.05,
) -> int:
    """Smallest population positive-count K with P(X >= observed) >= alpha.

    X is Hypergeometric(N=population_size, K=positives, n=sample_size).
    This is a one-sided exact lower bound for sampling without replacement.
    It is not a binomial Clopper-Pearson bound.
    """
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")
    if sample_size < 0 or sample_size > population_size:
        raise ValueError("sample_size must lie between 0 and population_size")
    if observed_positives < 0 or observed_positives > sample_size:
        raise ValueError("observed_positives must lie between 0 and sample_size")
    if observed_positives == 0:
        return 0
    lo = observed_positives
    hi = population_size
    while lo < hi:
        mid = (lo + hi) // 2
        tail = float(hypergeom.sf(observed_positives - 1, population_size, mid, sample_size))
        if tail >= alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


def hypergeometric_lower_proportion(
    observed_positives: int,
    sample_size: int,
    population_size: int,
    alpha: float = 0.05,
) -> float:
    """`hypergeometric_lower_count` divided by the finite population size."""
    count = hypergeometric_lower_count(
        observed_positives, sample_size, population_size, alpha
    )
    return count / population_size


def adjudicable_precision(
    true_positives: int,
    false_positives: int,
    excluded_unresolved: int,
) -> dict[str, float | int | None]:
    """Precision among definitive true-positive and false-positive labels only.

    Unresolved rows are not in the denominator. Callers must report
    `excluded_unresolved` beside the ratio. This function does not look up labels.
    """
    if min(true_positives, false_positives, excluded_unresolved) < 0:
        raise ValueError("counts must be non-negative")
    denominator = true_positives + false_positives
    precision = None if denominator == 0 else true_positives / denominator
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "denominator": denominator,
        "excluded_unresolved": excluded_unresolved,
        "precision": precision,
    }


def conservative_lower_bounds(
    confirmed_true_positives: int,
    sample_size: int,
    population_size: int,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Bounds that treat every non-confirmed row as not a true positive.

    `sample_size` stays in the denominator. Ambiguous and insufficient-evidence
    rows therefore cannot raise the bound by being dropped.
    """
    if confirmed_true_positives > sample_size:
        raise ValueError("confirmed true positives cannot exceed the sample")
    cp = clopper_pearson_lower(confirmed_true_positives, sample_size, alpha)
    hyper_count = hypergeometric_lower_count(
        confirmed_true_positives, sample_size, population_size, alpha
    )
    return {
        "confirmed_true_positives": confirmed_true_positives,
        "sample_size": sample_size,
        "population_size": population_size,
        "clopper_pearson_lower": cp,
        "hypergeometric_K_lower": hyper_count,
        "hypergeometric_proportion_lower": hyper_count / population_size,
        "threshold": 0.90,
        "clopper_pearson_clears_0_90": cp >= 0.90,
        "hypergeometric_clears_0_90": (hyper_count / population_size) >= 0.90,
    }
