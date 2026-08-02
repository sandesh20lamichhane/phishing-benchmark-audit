"""Inference for transfer experiments: DeLong, bootstrap CIs, corrections."""
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ DeLong
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _structural_components(scores, n_pos, n_neg):
    """V10 (positives) and V01 (negatives) components for each predictor."""
    k = scores.shape[0]
    tx = np.empty([k, n_pos]); ty = np.empty([k, n_neg]); tz = np.empty([k, n_pos + n_neg])
    for r in range(k):
        pos = scores[r, :n_pos]
        neg = scores[r, n_pos:]
        tx[r, :] = _midrank(pos)
        ty[r, :] = _midrank(neg)
        tz[r, :] = _midrank(scores[r, :])
    aucs = (tz[:, :n_pos].sum(axis=1) / n_pos - (n_pos + 1) / 2) / n_neg
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1 - (tz[:, n_pos:] - ty) / n_pos
    return aucs, v01, v10


def delong_test(y_true, scores_a, scores_b):
    """Paired DeLong test for two AUCs on the same sample.

    Returns (auc_a, auc_b, z, p). Two-sided.
    """
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="mergesort")   # positives first
    y = y[order]
    a = np.asarray(scores_a)[order]
    b = np.asarray(scores_b)[order]
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan, np.nan

    scores = np.vstack([a, b])
    aucs, v01, v10 = _structural_components(scores, n_pos, n_neg)
    s01 = np.cov(v01); s10 = np.cov(v10)
    s = s01 / n_pos + s10 / n_neg
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan, np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def delong_ci(y_true, scores, alpha=0.05):
    """DeLong analytic confidence interval for a single AUC."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="mergesort")
    y = y[order]; s = np.asarray(scores)[order]
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan
    aucs, v01, v10 = _structural_components(s.reshape(1, -1), n_pos, n_neg)
    var = np.var(v01, ddof=1) / n_pos + np.var(v10, ddof=1) / n_neg
    se = np.sqrt(var)
    zc = stats.norm.ppf(1 - alpha / 2)
    auc = float(aucs[0])
    return auc, max(0.0, auc - zc * se), min(1.0, auc + zc * se)


# --------------------------------------------------------------- bootstrap
def bootstrap_ci(y_true, scores, metric_fn, n_boot=2000, alpha=0.05, seed=42):
    """Stratified percentile bootstrap CI for any metric(y, s)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true); s = np.asarray(scores)
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return np.nan, np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            vals[i] = metric_fn(y[idx], s[idx])
        except Exception:
            vals[i] = np.nan
    vals = vals[~np.isnan(vals)]
    point = metric_fn(y, s)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ------------------------------------------------------- permutation tests
def permutation_gap_test(within_vals, cross_vals, n_perm=10000, seed=42):
    """Is the within-vs-cross gap larger than chance reassignment would give?"""
    rng = np.random.default_rng(seed)
    a = np.asarray(within_vals, dtype=float); b = np.asarray(cross_vals, dtype=float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); n = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if (pool[:n].mean() - pool[n:].mean()) >= obs:
            count += 1
    return float(obs), (count + 1) / (n_perm + 1)


def cliffs_delta(a, b):
    """Non-parametric effect size. |d|: .147 small, .33 medium, .474 large."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


# --------------------------------------------- multiple comparison control
def holm_bonferroni(pvals, alpha=0.05):
    """Holm step-down. Returns (adjusted p, reject) in the original order."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(len(p), np.nan); rej = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out, rej
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)     # enforce monotonicity
        out[i] = running
        rej[i] = running <= alpha
    return out, rej
