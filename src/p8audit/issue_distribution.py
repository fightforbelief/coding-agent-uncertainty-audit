"""Summaries of how many trajectories share an issue.

These functions describe counts. They do not change labels, splits, or weights.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def cdf_quantile(sorted_counts: Sequence[int], probability: float) -> int:
    """Smallest value whose empirical CDF is at least ``probability``.

    ``sorted_counts`` must be non-empty and sorted in non-decreasing order.
    For probability 0.90 this is the smallest trajectory count such that at
    least 90% of issues have that many trajectories or fewer.
    """
    if not sorted_counts:
        raise ValueError("quantile requires at least one issue")
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    if any(sorted_counts[index] > sorted_counts[index + 1] for index in range(len(sorted_counts) - 1)):
        raise ValueError("counts must be sorted")
    rank = math.ceil(probability * len(sorted_counts)) - 1
    return int(sorted_counts[rank])


def summarize_issue_counts(counts: Sequence[int]) -> dict:
    """Median and tail quantiles, plus the 1 / 2 / 3+ issue shares."""
    if not counts:
        raise ValueError("at least one issue is required")
    if any(int(count) < 1 for count in counts):
        raise ValueError("every issue count must be positive")
    values = sorted(int(count) for count in counts)
    issues = len(values)
    trajectories = sum(values)
    exactly_1 = sum(1 for value in values if value == 1)
    exactly_2 = sum(1 for value in values if value == 2)
    at_least_3 = sum(1 for value in values if value >= 3)
    histogram = {str(count): frequency for count, frequency in sorted(Counter(values).items())}
    return {
        "issues": issues,
        "trajectories": trajectories,
        "min": values[0],
        "q1": cdf_quantile(values, 0.25),
        "median": cdf_quantile(values, 0.50),
        "q3": cdf_quantile(values, 0.75),
        "p90": cdf_quantile(values, 0.90),
        "p95": cdf_quantile(values, 0.95),
        "max": values[-1],
        "exactly_1": exactly_1,
        "exactly_2": exactly_2,
        "at_least_3": at_least_3,
        "pct_exactly_1": exactly_1 / issues,
        "pct_exactly_2": exactly_2 / issues,
        "pct_at_least_3": at_least_3 / issues,
        "mean_trajectories_per_issue": trajectories / issues,
        "histogram": histogram,
        "quantile_definition": (
            "Smallest count with empirical CDF at least the stated probability. "
            "Not a linearly interpolated quantile."
        ),
    }


def concentration(counts: Sequence[int], fractions: Sequence[float] = (0.01, 0.05, 0.10)) -> dict:
    """Share of trajectories held by the highest-volume issues."""
    if not counts:
        raise ValueError("at least one issue is required")
    ordered = sorted((int(count) for count in counts), reverse=True)
    total = sum(ordered)
    issues = len(ordered)
    if total <= 0:
        raise ValueError("trajectory total must be positive")
    by_fraction = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("fractions must be in (0, 1]")
        kept = max(1, math.ceil(fraction * issues))
        held = sum(ordered[:kept])
        by_fraction.append(
            {
                "issue_fraction": fraction,
                "issues": kept,
                "trajectories": held,
                "trajectory_share": held / total,
            }
        )
    cumulative = 0
    issues_for_half = issues
    for index, count in enumerate(ordered, start=1):
        cumulative += count
        if cumulative >= 0.5 * total:
            issues_for_half = index
            break
    return {
        "top_fractions": by_fraction,
        "largest_issue_trajectories": ordered[0],
        "largest_issue_share": ordered[0] / total,
        "issues_covering_half_of_trajectories": issues_for_half,
        "issues_covering_half_share_of_issues": issues_for_half / issues,
    }
