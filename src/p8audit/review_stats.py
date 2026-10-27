"""Sample-size calculations for marker review. Not a population estimate."""

from __future__ import annotations

from scipy.stats import beta


def clopper_pearson_lower(successes: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson lower bound for a binomial proportion.

    ``alpha`` is the tail probability. The default 0.05 is a one-sided 95%
    bound. When every reviewed positive is a true positive, ``successes == n``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must lie between 0 and n")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")
    if successes == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, n - successes + 1))


def minimum_n_for_perfect_precision(lower_bound: float = 0.90, alpha: float = 0.05) -> int:
    """Smallest n such that zero false positives clears ``lower_bound``."""
    if not 0 < lower_bound < 1:
        raise ValueError("lower_bound must lie between 0 and 1")
    n = 1
    while clopper_pearson_lower(n, n, alpha) < lower_bound:
        n += 1
        if n > 100000:
            raise RuntimeError("bound was not reached")
    return n
