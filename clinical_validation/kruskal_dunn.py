"""
kruskal_dunn.py  –  Kruskal-Wallis test with Dunn's post-hoc and Holm correction
==================================================================================

Statistical Rationale (manuscript-ready)
-----------------------------------------
When outcome data violate the assumption of normality or arise from small,
unbalanced samples — as is common in clinical imaging studies — non-parametric
methods are preferred. The Kruskal-Wallis test is a non-parametric analogue of
the one-way analysis of variance that evaluates whether k independent groups
originate from the same underlying distribution (Kruskal & Wallis, 1952). Rather
than operating on raw values, the test ranks all N pooled observations and
computes an H statistic based on the between-group variability of those ranks.
Under the null hypothesis that all groups share the same distribution, H follows
an approximate chi-squared distribution with k − 1 degrees of freedom. A
statistically significant Kruskal-Wallis result indicates that at least one group
differs from the others but does not identify which specific pairs differ.

To localise pairwise differences, Dunn's post-hoc test was applied (Dunn, 1964).
Dunn's test uses the same pooled rank sums computed for the Kruskal-Wallis test,
avoiding the loss of information that would result from re-ranking data for each
individual pairwise comparison. For each pair of groups (i, j), a z-statistic is
calculated as the standardised difference in mean ranks:

    z_ij = (R̄_i − R̄_j) / sqrt( [N(N+1)/12 − T/(N−1)] × (1/n_i + 1/n_j) )

where N is the total sample size, n_i and n_j are the respective group sizes,
R̄_i and R̄_j are the mean ranks of each group, and T = Σ t_k(t_k² − 1) / 12
is a correction term for tied observations summed over all tie groups of size
t_k. Two-sided p-values are derived from the standard normal distribution.

Because multiple simultaneous comparisons inflate the family-wise Type I error
rate, the raw p-values from all tested pairs were adjusted using Holm's
sequential Bonferroni method (Holm, 1979). The Holm procedure sorts the m
p-values in ascending order and applies sequentially tightening critical
thresholds (α/m, α/(m−1), …, α/1), rejecting hypotheses until the first
non-rejection. Unlike the standard Bonferroni correction, the Holm method is
uniformly more powerful while still providing strong control of the family-wise
error rate, making it the preferred correction when the number of comparisons is
modest (Aickin & Gensler, 1996).

References
----------
Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance
    analysis. Journal of the American Statistical Association, 47(260), 583-621.
Dunn, O. J. (1964). Multiple comparisons using rank sums. Technometrics,
    6(3), 241-252.
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics, 6(2), 65-70.
Aickin, M., & Gensler, H. (1996). Adjusting for multiple testing when reporting
    research results: the Bonferroni vs Holm methods. American Journal of Public
    Health, 86(5), 726-728.

How to use
----------
    from kruskal_dunn import kruskal_dunn

    kw_stat, kw_p, results = kruskal_dunn(
        data   = df["Values"].values,
        groups = df["Score"].values,
        pairs  = [(0, 1), (1, 2), (2, 3)],   # None → all pairs
    )

    for pair, res in results.items():
        print(pair, res["p_holm"])

Written by: Jack Consolini
"""

import numpy as np
from scipy import stats
from itertools import combinations


def holm_correction(p_values, alpha=0.05):
    """Holm-Bonferroni adjustment for a family of p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    reject_sorted = np.zeros(m, dtype=bool)

    for i in range(m):
        adjusted_sorted[i] = min(1.0, sorted_p[i] * (m - i))
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    for i in range(m):
        if sorted_p[i] <= alpha / (m - i):
            reject_sorted[i] = True
        else:
            break

    adjusted = np.empty(m, dtype=float)
    reject = np.zeros(m, dtype=bool)
    adjusted[order] = adjusted_sorted
    reject[order] = reject_sorted
    return reject, adjusted


def kruskal_dunn(data, groups, pairs=None, alpha=0.05):
    """
    Kruskal-Wallis test followed by Dunn's post-hoc with Holm correction.

    Parameters
    ----------
    data   : array-like  – continuous outcome values (length N)
    groups : array-like  – integer or categorical group labels (length N)
    pairs  : list of (g1, g2) tuples to test in post-hoc;
             if None, all unique pairwise combinations are tested
    alpha  : float – significance threshold (default 0.05)

    Returns
    -------
    kw_stat  : float – Kruskal-Wallis H statistic
    kw_p     : float – Kruskal-Wallis p-value
    results  : dict of {(g1, g2): dict} where each inner dict contains:
                   'z'      : Dunn z-statistic
                   'p_raw'  : two-sided p-value before correction
                   'p_holm' : Holm-corrected p-value
                   'reject' : bool – significant at given alpha after correction
    """
    data   = np.asarray(data,   dtype=float)
    groups = np.asarray(groups, dtype=int)

    unique_groups = np.unique(groups)
    N = len(data)

    if pairs is None:
        pairs = list(combinations(unique_groups, 2))

    # ── 1) Kruskal-Wallis overall test ────────────────────────────────────────
    group_arrays = [data[groups == g] for g in unique_groups]
    kw_stat, kw_p = stats.kruskal(*group_arrays)

    print(f"\n  Kruskal-Wallis H = {kw_stat:.4f},  p = {kw_p:.4f}  "
          f"(df = {len(unique_groups) - 1})")

    # ── 2) Pooled ranks with tie handling ─────────────────────────────────────
    all_ranks = stats.rankdata(data)   # average ranks for ties

    # Mean rank per group
    mean_rank = {g: all_ranks[groups == g].mean() for g in unique_groups}
    n_group   = {g: np.sum(groups == g)           for g in unique_groups}

    # Tie correction:  T = Σ t_k(t_k² − 1) / 12
    _, tie_counts = np.unique(data, return_counts=True)
    T = np.sum(tie_counts * (tie_counts ** 2 - 1)) / 12.0

    # Shared variance base term
    var_base = N * (N + 1) / 12.0 - T / (N - 1)

    # ── 3) Dunn z-statistics and raw p-values ─────────────────────────────────
    raw_p  = []
    z_vals = []
    for (g1, g2) in pairs:
        se    = np.sqrt(var_base * (1.0 / n_group[g1] + 1.0 / n_group[g2]))
        z     = (mean_rank[g1] - mean_rank[g2]) / se
        p_raw = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        z_vals.append(z)
        raw_p.append(p_raw)

    # ── 4) Holm correction across the tested pairs ────────────────────────────
    reject, p_holm = holm_correction(raw_p, alpha=alpha)

    # ── 5) Pack results ───────────────────────────────────────────────────────
    results = {}
    print(f"\n  {'Pair':<20}  {'z':>7}  {'p_raw':>8}  {'p_holm':>8}  sig")
    print(f"  {'-'*20}  {'-'*7}  {'-'*8}  {'-'*8}  ---")
    for i, (g1, g2) in enumerate(pairs):
        results[(g1, g2)] = {
            "z":      z_vals[i],
            "p_raw":  raw_p[i],
            "p_holm": p_holm[i],
            "reject": bool(reject[i]),
        }
        sig_marker = "*" if reject[i] else ""
        print(f"  ({g1}, {g2}){'':<14}  {z_vals[i]:>7.3f}  "
              f"{raw_p[i]:>8.4f}  {p_holm[i]:>8.4f}  {sig_marker}")

    return kw_stat, kw_p, results
