"""Statistics as required by the brief: one score per seed, bootstrap 95% CIs,
Mann-Whitney U / Welch t, Kruskal-Wallis for >= 3 groups, Holm-Bonferroni
correction, and effect sizes (Cliff's delta; Hedges' g where defined)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, stat=np.mean, n_boot: int = 10_000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of a statistic over seeds."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(stat(x)), float(lo), float(hi)


def bootstrap_curve_ci(curves: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0):
    """Mean curve and pointwise bootstrap CI; curves has shape (n_seeds, n_points)."""
    rng = np.random.default_rng(seed)
    n = curves.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = curves[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2], axis=0)
    return curves.mean(axis=0), lo, hi


def cliffs_delta(a, b) -> float:
    """P(A > B) - P(A < B); robust to ties and zero variance. Range [-1, 1]."""
    a = np.asarray(a, float)[:, None]; b = np.asarray(b, float)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def hedges_g(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = a.size, b.size
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return float("nan")
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def mann_whitney(a, b, alternative: str = "two-sided") -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        return 1.0
    return float(stats.mannwhitneyu(a, b, alternative=alternative, method="auto").pvalue)


def welch_t(a, b, alternative: str = "two-sided") -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.var() == 0 and b.var() == 0:
        return 0.0 if a.mean() != b.mean() else 1.0
    return float(stats.ttest_ind(a, b, equal_var=False, alternative=alternative).pvalue)


def kruskal(*groups) -> float:
    gs = [np.asarray(g, float) for g in groups]
    if all(np.all(g == gs[0][0]) for g in gs):
        return 1.0
    return float(stats.kruskal(*gs).pvalue)


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
