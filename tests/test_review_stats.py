"""Clopper-Pearson bound used to decide whether a marker may be frozen."""

from p8audit.review_stats import clopper_pearson_lower, minimum_n_for_perfect_precision


def test_perfect_precision_needs_29_reviewed_positives() -> None:
    assert minimum_n_for_perfect_precision(0.90, 0.05) == 29
    assert clopper_pearson_lower(28, 28) < 0.90
    assert clopper_pearson_lower(29, 29) >= 0.90


def test_gate0_positive_counts_do_not_clear_the_bound() -> None:
    assert clopper_pearson_lower(7, 7) < 0.90
    assert clopper_pearson_lower(3, 3) < 0.90


def test_zero_successes_has_a_zero_lower_bound() -> None:
    assert clopper_pearson_lower(0, 50) == 0.0
