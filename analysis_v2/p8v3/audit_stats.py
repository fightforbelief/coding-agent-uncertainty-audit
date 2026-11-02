"""
p8v3.audit_stats — estimators for the human recall audit.

The audit sample is drawn from DISJOINT population strata (e.g. parser-positive /
screen-positive-parser-negative / rest, crossed with config) with known
population sizes N_h and unequal sampling fractions. Every estimator here
therefore weights by N_h / n_h. Unweighted proportions from an enriched sample
are biased and must not be reported as prevalence or recall.

Public API
----------
wilson(k, n, alpha=0.05) -> (p, lo, hi)
stratified_prevalence(df, label, stratum, pop_sizes) -> dict
enriched_confusion(df, truth, pred, stratum, pop_sizes, n_draws, seed) -> dict
    Horvitz–Thompson point estimates of recall / precision / specificity,
    plus 95% intervals from per-stratum Dirichlet(1/2) posterior draws
    (Jeffreys-type; well-behaved with zero cells, unlike a stratified bootstrap).
percent_agreement / cohen_kappa — for the intra-rater re-label check.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


def wilson(k: float, n: float, alpha: float = 0.05):
    if n <= 0:
        return (float("nan"),) * 3
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k <= 0 else max(0.0, c - h)
    hi = 1.0 if k >= n else min(1.0, c + h)
    return p, lo, hi


def _check(df, stratum, pop_sizes):
    missing = set(df[stratum].unique()) - set(pop_sizes)
    if missing:
        raise ValueError(f"pop_sizes lacks strata present in sample: {missing}")
    for h, g in df.groupby(stratum):
        if len(g) > pop_sizes[h]:
            raise ValueError(f"stratum {h}: sample {len(g)} > population {pop_sizes[h]}")


def stratified_prevalence(df: pd.DataFrame, label: str, stratum: str,
                          pop_sizes: Mapping, alpha: float = 0.05) -> dict:
    """Population prevalence of a 0/1 label under stratified SRS.

    Rows with NaN label are dropped (reported as n_missing).
    Strata that exist in the population but have no labelled rows are
    reported in `unsampled_strata`; their population share is excluded
    from the estimate (the estimate is then for the covered population).
    CI: Wilson with Kish effective sample size (Korn–Graubard style), using
    the stratified variance with finite-population correction.
    """
    d = df[[label, stratum]].copy()
    n_missing = int(d[label].isna().sum())
    d = d.dropna(subset=[label])
    _check(d, stratum, pop_sizes)
    covered = {h: pop_sizes[h] for h in d[stratum].unique()}
    N = sum(covered.values())
    est, var = 0.0, 0.0
    for h, g in d.groupby(stratum):
        n_h, N_h = len(g), covered[h]
        p_h = g[label].astype(float).mean()
        W = N_h / N
        est += W * p_h
        if n_h > 1:
            fpc = 1 - n_h / N_h
            var += W * W * fpc * p_h * (1 - p_h) / (n_h - 1)
    n = len(d)
    if var > 0 and 0 < est < 1:
        n_eff = min(n, est * (1 - est) / var)
    else:
        n_eff = n
    _, lo, hi = wilson(est * n_eff, n_eff, alpha)
    return dict(estimate=est, lo=lo, hi=hi, n=n, n_eff=n_eff, n_missing=n_missing,
                pop_covered=N,
                unsampled_strata=sorted(set(pop_sizes) - set(covered), key=str))


def enriched_confusion(df: pd.DataFrame, truth: str, pred: str, stratum: str,
                       pop_sizes: Mapping, n_draws: int = 20000, seed: int = 20261101,
                       alpha: float = 0.05) -> dict:
    """Population confusion-matrix metrics from a stratified (enriched) sample.

    truth: human label (0/1); pred: parser/detector output (0/1).
    Rows with NaN truth are dropped.
    """
    d = df[[truth, pred, stratum]].dropna(subset=[truth]).copy()
    d[truth] = d[truth].astype(int)
    d[pred] = d[pred].astype(int)
    _check(d, stratum, pop_sizes)
    rng = np.random.default_rng(seed)
    cells = [(1, 1), (0, 1), (1, 0), (0, 0)]       # (truth, pred): TP FP FN TN
    point = np.zeros(4)
    draws = np.zeros((n_draws, 4))
    per_stratum = []
    for h, g in d.groupby(stratum):
        cnt = np.array([((g[truth] == t) & (g[pred] == p)).sum() for t, p in cells], float)
        N_h, n_h = pop_sizes[h], len(g)
        point += N_h * cnt / n_h
        # Dirichlet(1/2) only over cells that are possible in this stratum:
        # if the stratum is defined by pred value, the other pred cells are structurally 0.
        possible = np.array([True] * 4)
        if (g[pred] == 1).all():
            possible = np.array([True, True, False, False])
        elif (g[pred] == 0).all():
            possible = np.array([False, False, True, True])
        a = np.where(possible, cnt + 0.5, 0.0)
        sub = rng.dirichlet(a[possible], size=n_draws)
        full = np.zeros((n_draws, 4)); full[:, possible] = sub
        draws += N_h * full
        per_stratum.append(dict(stratum=h, N=N_h, n=n_h, TP=cnt[0], FP=cnt[1], FN=cnt[2], TN=cnt[3]))

    def metrics(x):
        tp, fp, fn, tn = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(recall=tp / (tp + fn), precision=tp / (tp + fp),
                        specificity=tn / (tn + fp), prevalence=(tp + fn) / x.sum(-1))
    pt = metrics(point)
    dr = metrics(draws)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    out = {}
    for k in pt:
        v = dr[k][np.isfinite(dr[k])]
        lo, hi = (np.percentile(v, q) if len(v) else (np.nan, np.nan))
        out[k] = dict(estimate=float(pt[k]), lo=float(lo), hi=float(hi))
    out["est_pop_counts"] = dict(zip(["TP", "FP", "FN", "TN"], map(float, point)))
    out["per_stratum"] = pd.DataFrame(per_stratum)
    return out


def percent_agreement(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    return float((a[m] == b[m]).mean())


def cohen_kappa(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    a, b = a[m].astype(int), b[m].astype(int)
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in set(a) | set(b))
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))
