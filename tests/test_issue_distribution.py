"""Quantiles and concentration use issue counts, not model scores."""

import pytest

from p8audit.issue_distribution import cdf_quantile, concentration, summarize_issue_counts


def test_cdf_quantile_is_the_smallest_count_reaching_the_probability() -> None:
    # 10 issues: six with 1 trajectory, four with 5.
    counts = sorted([1, 1, 1, 1, 1, 1, 5, 5, 5, 5])
    assert cdf_quantile(counts, 0.50) == 1
    assert cdf_quantile(counts, 0.90) == 5
    assert cdf_quantile(counts, 1.0) == 5


def test_summary_counts_partition_the_issues() -> None:
    summary = summarize_issue_counts([1, 2, 2, 4])
    assert summary["issues"] == 4
    assert summary["trajectories"] == 9
    assert summary["exactly_1"] == 1
    assert summary["exactly_2"] == 2
    assert summary["at_least_3"] == 1
    assert summary["pct_exactly_1"] + summary["pct_exactly_2"] + summary["pct_at_least_3"] == 1
    assert summary["min"] == 1
    assert summary["max"] == 4
    assert summary["median"] == 2


def test_concentration_puts_the_largest_issues_first() -> None:
    result = concentration([1, 1, 1, 10], fractions=(0.25,))
    assert result["top_fractions"][0]["issues"] == 1
    assert result["top_fractions"][0]["trajectories"] == 10
    assert result["top_fractions"][0]["trajectory_share"] == pytest.approx(10 / 13)
    assert result["issues_covering_half_of_trajectories"] == 1


def test_empty_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_issue_counts([])
