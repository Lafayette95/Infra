"""OIS discount curve (SOFR) bootstrapped from par swap closes - pure, no I/O.

Instruments (root CLAUDE.md 16): spot-starting par OIS swaps, ANNUAL fixed vs compounded
overnight, both legs ACT/360, effective = trade + ``spot_lag`` business days, annual dates
from the effective date, modified following. An OIS's floating leg is worth
DF(effective) - DF(maturity), so a par rate r satisfies

    DF(eff) - DF(T_N) = r * sum_i alpha_i DF(t_i)

Bootstrap (``bootstrap_ois``): pillars in tenor order, each solved exactly (one root per
pillar) with the discount curve LOG-LINEAR in time between nodes (= piecewise-FLAT
instantaneous forwards, the market's plain-vanilla choice: every input reprices exactly,
nothing is smoothed away, and a bad input shows as a kinked forward, never spreads). Beyond
the last node the last forward continues. Optional SHORT-END nodes below the first pillar
(e.g. the SR1-fitted overnight path) are fixed before the swaps are solved.

Simplifications (v1, stated): payment dates = accrual end dates (SOFR OIS pays 2 business
days later - a few hundredths of a bp); no end-of-month roll rule; times in ACT/365.25 from
the trade day (interpolation only - accruals use ACT/360 on real dates).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from infra.analytics.sofr_curve import business_days

YEAR = 365.25
_BDAYS: set | None = None


def _is_bday(d: pd.Timestamp) -> bool:
    global _BDAYS
    if _BDAYS is None:
        _BDAYS = set(business_days("1990-01-01", "2075-12-31"))
    return d in _BDAYS


def add_business_days(day, n: int) -> pd.Timestamp:
    d = pd.Timestamp(day).normalize()
    while n > 0:
        d += pd.Timedelta(days=1)
        if _is_bday(d):
            n -= 1
    return d


def modified_following(day) -> pd.Timestamp:
    d0 = pd.Timestamp(day).normalize()
    d = d0
    while not _is_bday(d):
        d += pd.Timedelta(days=1)
    if d.month != d0.month:  # rolled into the next month: go back instead
        d = d0
        while not _is_bday(d):
            d -= pd.Timedelta(days=1)
    return d


@dataclass(frozen=True)
class SwapSchedule:
    effective: pd.Timestamp
    dates: tuple  # adjusted accrual end (= payment) dates, the last one is maturity
    accruals: np.ndarray  # ACT/360 year fractions


def swap_schedule(trade_day, tenor_years: int, spot_lag: int = 2) -> SwapSchedule:
    eff = add_business_days(trade_day, spot_lag)
    ends = [modified_following(eff + pd.DateOffset(years=k)) for k in range(1, tenor_years + 1)]
    starts = [eff] + ends[:-1]
    acc = np.array([(e - s).days / 360.0 for s, e in zip(starts, ends)])
    return SwapSchedule(eff, tuple(ends), acc)


def year_fraction(trade_day, dates) -> np.ndarray:
    t0 = pd.Timestamp(trade_day).normalize()
    return np.array([(pd.Timestamp(d) - t0).days / YEAR for d in np.atleast_1d(dates)], dtype="float64")


@dataclass
class OisCurve:
    """Nodes (t years from the trade day, discount factor), log-linear in between."""
    t: np.ndarray
    df: np.ndarray

    def discount(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype="float64"))
        tn = np.concatenate([[0.0], self.t])
        ln = np.concatenate([[0.0], np.log(self.df)])
        out = np.interp(t, tn, ln)
        if len(tn) >= 2:  # flat last forward beyond the last node
            f = (ln[-1] - ln[-2]) / (tn[-1] - tn[-2])
            beyond = t > tn[-1]
            out[beyond] = ln[-1] + f * (t[beyond] - tn[-1])
        return np.exp(out)

    def zero(self, t) -> np.ndarray:
        """Continuously compounded zero rate, % (ACT/365.25)."""
        t = np.atleast_1d(np.asarray(t, dtype="float64"))
        return -np.log(self.discount(t)) / t * 100.0

    def forward(self, t1, t2) -> np.ndarray:
        """Continuously compounded forward between two times, %."""
        t1, t2 = np.atleast_1d(t1).astype(float), np.atleast_1d(t2).astype(float)
        return np.log(self.discount(t1) / self.discount(t2)) / (t2 - t1) * 100.0


def par_rate(curve: OisCurve, trade_day, sched: SwapSchedule) -> float:
    """Par OIS rate (%) of a schedule on the curve."""
    d_eff = curve.discount(year_fraction(trade_day, [sched.effective]))[0]
    dfs = curve.discount(year_fraction(trade_day, sched.dates))
    return float((d_eff - dfs[-1]) / (sched.accruals @ dfs) * 100.0)


def bootstrap_ois(trade_day, quotes: dict[int, float], *, spot_lag: int = 2,
                  short_nodes: list[tuple[float, float]] | None = None) -> tuple[OisCurve, pd.DataFrame]:
    """Bootstrap from par rates ``{tenor_years: rate %}``. ``short_nodes``: fixed (t, DF)
    nodes, all before the first pillar's maturity. Returns the curve and one row per node
    (``node``, ``t_years``, ``df``, ``source``, ``input_rate``, ``reprice_bp``)."""
    nodes_t = [t for t, _ in (short_nodes or [])]
    nodes_df = [d for _, d in (short_nodes or [])]
    rows = [{"node": f"{round(t * 12)}M", "t_years": t, "df": d, "source": "short_end", "input_rate": np.nan}
            for t, d in (short_nodes or [])]
    for tenor in sorted(quotes):
        r = quotes[tenor] / 100.0
        sched = swap_schedule(trade_day, tenor, spot_lag)
        tm = year_fraction(trade_day, [sched.dates[-1]])[0]
        if nodes_t and tm <= nodes_t[-1]:
            raise ValueError(f"{tenor}y pillar ({tm:.3f}y) not after the last node ({nodes_t[-1]:.3f}y)")
        teff = year_fraction(trade_day, [sched.effective])[0]
        tp = year_fraction(trade_day, sched.dates)

        def gap(ln_df):
            c = OisCurve(np.array(nodes_t + [tm]), np.array(nodes_df + [np.exp(ln_df)]))
            return c.discount([teff])[0] - c.discount([tm])[0] - r * (sched.accruals @ c.discount(tp))

        ln = brentq(gap, -6.0, 1.0, xtol=1e-15, rtol=1e-15, maxiter=200)
        nodes_t.append(tm)
        nodes_df.append(float(np.exp(ln)))
        rows.append({"node": f"{tenor}Y", "t_years": tm, "df": nodes_df[-1], "source": "swap", "input_rate": quotes[tenor]})
    curve = OisCurve(np.array(nodes_t), np.array(nodes_df))
    out = pd.DataFrame(rows)
    out["zero_pct"] = curve.zero(out["t_years"].to_numpy())
    rep = [(par_rate(curve, trade_day, swap_schedule(trade_day, int(n[:-1]), spot_lag)) - q) * 100.0
           if s == "swap" else np.nan for n, s, q in zip(out["node"], out["source"], out["input_rate"])]
    out["reprice_bp"] = rep
    return curve, out


def curve_from_nodes(nodes: pd.DataFrame) -> OisCurve:
    """Rebuild a curve from stored node rows (``t_years``, ``df``)."""
    n = nodes.sort_values("t_years")
    return OisCurve(n["t_years"].to_numpy(dtype="float64"), n["df"].to_numpy(dtype="float64"))


def short_end_nodes(compounded, trade_day, months: int) -> list[tuple[float, float]]:
    """Monthly (t, DF) nodes from an overnight path: ``compounded(start, end)`` = the
    path's simple ACT/360 rate (%) of rolling overnight over [start, end)
    (``infra.analytics.sofr_curve.SofrPath.compounded``)."""
    t0 = pd.Timestamp(trade_day).normalize()
    out = []
    for k in range(1, months + 1):
        d = t0 + pd.DateOffset(months=k)
        days = (d - t0).days
        out.append(((d - t0).days / YEAR, 1.0 / (1.0 + compounded(t0, d) / 100.0 * days / 360.0)))
    return out


def fill_missing_tenors(rates: pd.DataFrame, *, max_age_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill an interior tenor missing on a day (no print in its close window) from the SAME
    day's nearest available tenors on each side plus that tenor's last observed FLY RESIDUAL
    against that same neighbour pair (rate minus the line through the pair), if observed in
    the last ``max_age_days`` calendar days - the curve's shape persists, its level moves.
    ``rates``: day x tenor (%), days ascending. Point in time: a day uses its own and earlier
    days only. End tenors are never filled. Returns (filled rates, bool frame of fills).

    Out of sample on observed tenors (USD NY1530 pure, 2024-09..2026-10): MAE 0.48-0.64bp,
    bias <= 0.05bp, against 0.7-7.1bp (biased up to 7bp: the curve's hump) for the bare line."""
    tenors = list(rates.columns)
    out, filled = rates.copy(), pd.DataFrame(False, index=rates.index, columns=tenors)
    last: dict = {}  # (tenor, lo, hi) -> (day, residual)

    def pair(row, t):
        lo = [x for x in tenors if x < t and pd.notna(row[x])]
        hi = [x for x in tenors if x > t and pd.notna(row[x])]
        return (lo[-1], hi[0]) if lo and hi else None

    def line(row, t, a, b):
        return row[a] + (row[b] - row[a]) * (t - a) / (b - a)

    for day, row in rates.iterrows():
        for t in tenors[1:-1]:  # learn today's residuals from observed tenors
            if pd.notna(row[t]) and (p := pair(row, t)):
                last[(t, *p)] = (day, row[t] - line(row, t, *p))
        for t in tenors[1:-1]:
            if pd.isna(row[t]) and (p := pair(row, t)):
                seen = last.get((t, *p))
                if seen and (day - seen[0]).days <= max_age_days:
                    out.loc[day, t] = line(row, t, *p) + seen[1]
                    filled.loc[day, t] = True
    return out, filled
