"""Deterministic bounds. These are not observed review results."""

from p8audit.precision_inference import (
    adjudicable_precision,
    conservative_lower_bounds,
    hypergeometric_lower_count,
    hypergeometric_lower_proportion,
)
from p8audit.review_stats import clopper_pearson_lower


def test_hypergeometric_lower_count_on_a_hand_checked_population() -> None:
    # C(10, 4) = 210. P(X=4) is 1/210, 5/210, 15/210 for K=4, 5, 6.
    assert hypergeometric_lower_count(4, 4, 10, 0.05) == 6
    assert hypergeometric_lower_proportion(4, 4, 10, 0.05) == 0.6


def test_zero_observed_positives_has_a_zero_population_lower_count() -> None:
    assert hypergeometric_lower_count(0, 50, 100) == 0


def test_fifty_of_fifty_binomial_bound_is_the_published_conditional_figure() -> None:
    bound = clopper_pearson_lower(50, 50)
    assert abs(bound - 0.9418449208830277) < 1e-12


def test_eight_shard_population_hypergeometric_nearly_matches_binomial() -> None:
    # Conditional illustration for the explicit_exception population. Not an observation.
    proportion = hypergeometric_lower_proportion(50, 50, 28680)
    assert abs(proportion - 0.9419107391910739) < 1e-12
    assert hypergeometric_lower_count(49, 50, 28680) == 26061


def test_dropping_two_unresolved_rows_inflates_the_adjudicable_bound() -> None:
    adjudicable = adjudicable_precision(48, 0, 2)
    assert adjudicable["precision"] == 1.0
    assert adjudicable["excluded_unresolved"] == 2
    assert clopper_pearson_lower(48, 48) >= 0.90
    conservative = conservative_lower_bounds(48, 50, 28680)
    assert conservative["clopper_pearson_lower"] < 0.90
    assert conservative["hypergeometric_clears_0_90"] is False


def test_one_nonconfirmed_row_still_clears_both_illustrative_bounds() -> None:
    bounds = conservative_lower_bounds(49, 50, 30339)
    assert bounds["clopper_pearson_clears_0_90"] is True
    assert bounds["hypergeometric_clears_0_90"] is True
