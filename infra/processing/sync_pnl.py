"""Synchronized benchmark P&L - pure (no I/O). Every series is marked at ONE instant (the
sync snap, e.g. 16:15 London) so that day D means the same 24 hours for every issuer
(root CLAUDE.md 12). Sign as every bmk: + = long duration, bp.

* Futures: the mid at the instant of the contract HELD over the interval (the one chosen on
  the previous marked day); P&L per bp = price change / that contract's previous-day DV01.
* Cash yields moved to the instant: ``y + ratio x (mid at the instant - mid at the source's
  own price time)`` on the same contract - ratio = bp of yield per point of the hedge
  future (the swap-hedge book's, known before the day); the P&L is then -(change) x 100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PNL_COLUMNS = ["timestamp", "ticker", "bmk", "currency", "yield", "prev_timestamp", "prev_yield", "cusip",
               "prev_cusip", "pnl", "pnl_per_dv01"]


def held_futures_pnl(marks: pd.DataFrame, held: pd.Series, dv01: pd.DataFrame, *, ticker: str, bmk: str,
                     currency: str, point_value: float, max_gap_days: int = 7) -> pd.DataFrame:
    """``marks``: day x contract mids at the instant; ``held``: day -> contract chosen that day
    (held over the NEXT interval); ``dv01``: day x contract, price points per bp (NaN where
    unknown). Day D's row: the move of the contract held from the previous marked day."""
    days = marks.index.sort_values()
    rows = []
    for prev, day in zip(days[:-1], days[1:]):
        if (day - prev).days > max_gap_days:
            continue
        c = held.get(prev)
        if c is None or pd.isna(c) or c not in marks:
            continue
        a, b = marks.at[prev, c], marks.at[day, c]
        if pd.isna(a) or pd.isna(b):
            continue
        d = dv01.at[prev, c] if (prev in dv01.index and c in dv01.columns) else np.nan
        rows.append({"timestamp": day, "ticker": ticker, "bmk": bmk, "currency": currency, "yield": b,
                     "prev_timestamp": prev, "prev_yield": a, "cusip": c, "prev_cusip": c,
                     "pnl": (b - a) * point_value, "pnl_per_dv01": (b - a) / d if d and np.isfinite(d) and d > 0 else np.nan})
    return pd.DataFrame(rows, columns=PNL_COLUMNS)


def moved_yield(y_pct: float, ratio_bp_per_point: float, mid_sync: float, mid_source: float) -> float:
    """A yield observed at the source's price time, moved to the sync instant by the hedge
    future's move between the two (bp per point x points, in %)."""
    if any(pd.isna(v) for v in (y_pct, ratio_bp_per_point, mid_sync, mid_source)):
        return float("nan")
    return float(y_pct + ratio_bp_per_point * (mid_sync - mid_source) / 100.0)


def rolling_beta(y: pd.Series, x: pd.Series, window: int = 60, min_obs: int = 40) -> pd.Series:
    """Day -> the OLS slope of ``y`` on ``x`` over the ``window`` observations BEFORE that day
    (point in time: day t's own move is never in its ratio); NaN with fewer than ``min_obs``."""
    df = pd.DataFrame({"y": y.astype(float), "x": x.astype(float)}).dropna().sort_index()
    out = {}
    for i in range(len(df)):
        w = df.iloc[max(0, i - window):i]
        if len(w) >= min_obs and w["x"].var() > 0:
            out[df.index[i]] = float(np.polyfit(w["x"], w["y"], 1)[0])
    return pd.Series(out, dtype=float)


def moved_price(price: float, beta: float, mid_sync: float, mid_source: float) -> float:
    """A price observed at the source's time moved to the sync instant: ``price + beta x (hedge
    mid at the instant - hedge mid at the source's time)`` (same units as the price)."""
    if any(pd.isna(v) for v in (price, beta, mid_sync, mid_source)):
        return float("nan")
    return float(price + beta * (mid_sync - mid_source))
