"""Paired statistical tests: paired t-test, Wilcoxon, bootstrap CI.

Used by :mod:`cracks_yolo.pipeline.compare` to compare two model variants
across 5-fold CV (per-fold metric as the paired sample).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats as sp_stats

from cracks_yolo.metrics.schemas import StatisticalTest


def paired_t_test(a: list[float], b: list[float]) -> StatisticalTest:
    """Two-sided paired t-test on per-fold metric differences (a - b)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return StatisticalTest(
            test_name="paired_t",
            statistic=0.0,
            p_value=1.0,
            ci_low=None,
            ci_high=None,
            n_samples=len(a_arr),
        )
    t_stat, p_val = sp_stats.ttest_rel(a_arr, b_arr)
    if np.isnan(t_stat) or np.isnan(p_val):
        t_stat = 0.0
        p_val = 1.0
    return StatisticalTest(
        test_name="paired_t",
        statistic=float(t_stat),
        p_value=float(p_val),
        ci_low=None,
        ci_high=None,
        n_samples=len(a_arr),
    )


def wilcoxon(a: list[float], b: list[float]) -> StatisticalTest:
    """Two-sided Wilcoxon signed-rank test on paired samples."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return StatisticalTest(
            test_name="wilcoxon",
            statistic=0.0,
            p_value=1.0,
            ci_low=None,
            ci_high=None,
            n_samples=len(a_arr),
        )
    try:
        w_stat, p_val = sp_stats.wilcoxon(a_arr, b_arr)
    except ValueError:
        # All differences zero — Wilcoxon undefined.
        w_stat, p_val = 0.0, 1.0
    if np.isnan(w_stat) or np.isnan(p_val):
        w_stat = 0.0
        p_val = 1.0
    return StatisticalTest(
        test_name="wilcoxon",
        statistic=float(w_stat),
        p_value=float(p_val),
        ci_low=None,
        ci_high=None,
        n_samples=len(a_arr),
    )


def bootstrap_ci(
    a: list[float],
    b: list[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> StatisticalTest:
    """Bootstrap 95% CI on the mean of (a - b).

    Returns a :class:`StatisticalTest` with ``ci_low`` / ``ci_high`` set;
    ``statistic`` is the point-estimate mean difference, ``p_value`` is
    the fraction of bootstrap samples whose mean difference is ≤ 0 (a
    one-sided sign-test-style p-value).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n = min(len(a_arr), len(b_arr))
    if n < 2:
        return StatisticalTest(
            test_name="bootstrap_ci",
            statistic=0.0,
            p_value=1.0,
            ci_low=None,
            ci_high=None,
            n_samples=n,
        )
    diff = a_arr[:n] - b_arr[:n]
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(diff[idx].mean())
    ci_low = float(np.quantile(boot_means, alpha / 2))
    ci_high = float(np.quantile(boot_means, 1 - alpha / 2))
    p_value = float((boot_means <= 0).sum() / n_boot)
    return StatisticalTest(
        test_name="bootstrap_ci",
        statistic=float(diff.mean()),
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_samples=n,
    )


def run_statistical_test(
    a: list[float],
    b: list[float],
    test_name: Literal["paired_t", "bootstrap_ci", "wilcoxon"] = "paired_t",
) -> StatisticalTest:
    """Dispatch to the right test by name."""
    if test_name == "paired_t":
        return paired_t_test(a, b)
    if test_name == "wilcoxon":
        return wilcoxon(a, b)
    if test_name == "bootstrap_ci":
        return bootstrap_ci(a, b)
    raise ValueError(f"unknown test_name: {test_name!r}")


__all__ = [
    "bootstrap_ci",
    "paired_t_test",
    "run_statistical_test",
    "wilcoxon",
]
