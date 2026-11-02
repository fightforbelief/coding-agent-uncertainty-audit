import numpy as np
import pandas as pd
import pytest
from p8v3.audit_stats import (wilson, stratified_prevalence, enriched_confusion,
                              cohen_kappa, percent_agreement)


def test_wilson_known_value():
    p, lo, hi = wilson(50, 50)
    assert p == 1 and hi == 1 and abs(lo - 0.9287) < 1e-3
    p, lo, hi = wilson(0, 200)
    assert p == 0 and lo == 0 and abs(hi - 0.01884) < 1e-4


def test_prevalence_srs_equals_wilson():
    df = pd.DataFrame({"y": [1] * 30 + [0] * 70, "s": "a"})
    r = stratified_prevalence(df, "y", "s", {"a": 10**9})
    _, lo, hi = wilson(30, 100)
    assert abs(r["estimate"] - 0.3) < 1e-12
    assert abs(r["lo"] - lo) < 1e-3 and abs(r["hi"] - hi) < 1e-3


def test_prevalence_reweights():
    # stratum a: 10% of pop, all positive; b: 90% of pop, none positive; equal sample sizes
    df = pd.DataFrame({"y": [1] * 50 + [0] * 50, "s": ["a"] * 50 + ["b"] * 50})
    r = stratified_prevalence(df, "y", "s", {"a": 1000, "b": 9000})
    assert abs(r["estimate"] - 0.1) < 1e-12   # unweighted would say 0.5


def test_prevalence_errors():
    df = pd.DataFrame({"y": [1, 0], "s": ["a", "z"]})
    with pytest.raises(ValueError):
        stratified_prevalence(df, "y", "s", {"a": 10})


def _simulate(seed=0):
    """Population: 100k rows. True positives 1,000. Parser catches 100 (recall .1, precision 1).
    Screen catches 800 of the 900 misses plus 4,000 true negatives."""
    rng = np.random.default_rng(seed)
    N = 100_000
    truth = np.zeros(N, int); truth[:1000] = 1
    pred = np.zeros(N, int); pred[:100] = 1
    screen = np.zeros(N, int); screen[100:900] = 1; screen[1000:5000] = 1
    stratum = np.where(pred == 1, "P", np.where(screen == 1, "S", "R"))
    pop = pd.DataFrame(dict(truth=truth, pred=pred, s=stratum))
    sizes = pop["s"].value_counts().to_dict()
    samp = pd.concat([pop[pop.s == "P"].sample(20, random_state=seed),
                      pop[pop.s == "S"].sample(80, random_state=seed),
                      pop[pop.s == "R"].sample(100, random_state=seed)])
    return samp, sizes


def test_enriched_recall_close_to_truth():
    ests = []
    for seed in range(30):
        samp, sizes = _simulate(seed)
        r = enriched_confusion(samp, "truth", "pred", "s", sizes, n_draws=2000)
        ests.append(r["recall"]["estimate"])
        assert r["precision"]["estimate"] == 1.0
    # true recall 0.1; HT estimator should centre near it (R stratum adds noise)
    assert 0.05 < np.median(ests) < 0.2


def test_enriched_interval_contains_truth_mostly():
    cover = 0
    for seed in range(40):
        samp, sizes = _simulate(seed)
        r = enriched_confusion(samp, "truth", "pred", "s", sizes, n_draws=2000, seed=seed)
        cover += r["recall"]["lo"] <= 0.1 <= r["recall"]["hi"]
    assert cover >= 32


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert percent_agreement([1, 0, None], [1, 1, 0]) == 0.5
