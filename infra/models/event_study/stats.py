"""The event-study tests (the fit's statistics), per instrument. Pure, numpy / scipy.

Given the completed events' window moves ``x`` (and their dates) and the placebo windows'
moves ``y`` (same window shape on non-event days), ``event_stats`` returns:

* size: ``n``, ``mean``, ``median``, ``std``, ``t`` / ``p_t`` (one-sample t on the mean);
* robustness: ``hit`` (share moving the mean's way) / ``p_hit`` (two-sided binomial vs
  50%), ``p_wilcoxon`` (signed-rank: the median, so one large event can't carry it),
  ``trimmed_mean`` / ``trimmed_t`` (``trim`` events dropped at each tail);
* economic size: ``ev_abs`` = |mean| (source units, bp), ``ev_vol`` = mean / placebo sd
  (moves of the SAME window on non-event days: the event's move in units of that window's
  ordinary volatility);
* baseline: ``placebo_n``, ``placebo_mean``, ``excess`` = mean - placebo mean, ``t_excess``
  (Welch) - a 06:00-09:00 drift that every day has is not an event effect;
* stability: ``mean_first_half`` / ``mean_second_half`` / ``halves_same_sign``,
  ``year_share`` (share of years with >= 2 events whose mean has the overall sign).

``passes`` applies the spec's thresholds (a ``None`` threshold is off) and reports each test.
Multiple-testing control is the family's job (``family.benjamini_hochberg``), since it
needs every test of the family at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

STAT_NAMES = ("n", "mean", "median", "std", "t", "p_t", "hit", "p_hit", "p_wilcoxon", "trimmed_mean", "trimmed_t",
              "ev_abs", "ev_vol", "placebo_n", "placebo_mean", "placebo_std", "excess", "t_excess", "p_excess",
              "mean_first_half", "mean_second_half", "halves_same_sign", "year_share", "years")


def _t(x):
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    sd = np.std(x, ddof=1)
    if sd == 0:
        return np.nan, np.nan
    t = np.mean(x) / (sd / np.sqrt(n))
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def event_stats(x: np.ndarray, dates: pd.DatetimeIndex, y: np.ndarray | None = None, *, trim: int = 1) -> dict:
    x = np.asarray(x, dtype="float64")
    ok = np.isfinite(x)
    x, dates = x[ok], pd.DatetimeIndex(dates)[ok]
    out = dict.fromkeys(STAT_NAMES, np.nan)
    n = len(x)
    out["n"] = n
    if n == 0:
        return out
    m = float(np.mean(x))
    out.update(mean=m, median=float(np.median(x)), std=float(np.std(x, ddof=1)) if n > 1 else np.nan,
               ev_abs=abs(m))
    out["t"], out["p_t"] = _t(x)
    sign = np.sign(m) if m != 0 else 1.0
    wins = int(np.sum(np.sign(x) == sign))
    nonzero = int(np.sum(x != 0))
    out["hit"] = wins / n
    out["p_hit"] = float(stats.binomtest(wins, nonzero, 0.5).pvalue) if nonzero else np.nan
    if nonzero >= 2 and np.any(x != x[0]):
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(x[x != 0]).pvalue)
        except ValueError:
            pass
    if n > 2 * trim + 1:
        xt = np.sort(x)[trim:n - trim] if trim else x
        out["trimmed_mean"] = float(np.mean(xt))
        out["trimmed_t"] = _t(xt)[0]
    order = np.argsort(dates.to_numpy())
    half = n // 2
    if half >= 1:
        a, b = x[order[:half]], x[order[half:]]
        out.update(mean_first_half=float(np.mean(a)), mean_second_half=float(np.mean(b)),
                   halves_same_sign=float(np.sign(np.mean(a)) == sign and np.sign(np.mean(b)) == sign))
    yr = pd.Series(x, index=dates.year).groupby(level=0).agg(["mean", "size"])
    yr = yr[yr["size"] >= 2]
    if len(yr):
        out["year_share"] = float(np.mean(np.sign(yr["mean"]) == sign))
        out["years"] = len(yr)
    if y is not None:
        y = np.asarray(y, dtype="float64")
        y = y[np.isfinite(y)]
        out["placebo_n"] = len(y)
        if len(y) > 1:
            out.update(placebo_mean=float(np.mean(y)), placebo_std=float(np.std(y, ddof=1)))
            if out["placebo_std"] > 0:
                out["ev_vol"] = m / out["placebo_std"]
            out["excess"] = m - out["placebo_mean"]
            if n > 1:
                w = stats.ttest_ind(x, y, equal_var=False)
                out["t_excess"], out["p_excess"] = float(w.statistic), float(w.pvalue)
    return out


def passes(st: dict, spec) -> dict:
    """Each enabled test -> 1.0 / 0.0, plus ``passed`` (all of them). Too few events fails."""
    def ge(v, lo):
        return float(np.isfinite(v) and v >= lo)
    tests = {"test_min_obs": float(st["n"] >= spec.min_obs)}
    if spec.t_min is not None:
        tests["test_t"] = ge(abs(st["t"]) if np.isfinite(st["t"]) else np.nan, spec.t_min)
    if spec.ev_abs_min is not None:
        tests["test_ev_abs"] = ge(st["ev_abs"], spec.ev_abs_min)
    if spec.ev_vol_min is not None:
        tests["test_ev_vol"] = ge(abs(st["ev_vol"]) if np.isfinite(st["ev_vol"]) else np.nan, spec.ev_vol_min)
    if spec.hit_min is not None:
        tests["test_hit"] = ge(st["hit"], spec.hit_min)
    if spec.placebo_t_min is not None:
        tests["test_placebo"] = ge(abs(st["t_excess"]) if np.isfinite(st["t_excess"]) else np.nan,
                                   spec.placebo_t_min)
    if spec.stable_halves:
        tests["test_halves"] = float(st["halves_same_sign"] == 1.0)
    if spec.year_share_min is not None:
        tests["test_years"] = ge(st["year_share"], spec.year_share_min)
    tests["passed"] = float(all(v == 1.0 for v in tests.values()))
    return tests
