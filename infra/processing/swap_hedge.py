"""Hedge maths for futures-adjusted swap closes (CLAUDE.md 16). Pure functions, no I/O;
every time tz-naive UTC."""
from __future__ import annotations

import numpy as np
import pandas as pd


def same_contract_changes(settlements: pd.DataFrame, contract: pd.Series) -> pd.Series:
    """Daily settlement change of each day's hedge contract against ITS OWN prior
    settlement - never across a roll. ``settlements``: trading day x ticker;
    ``contract``: trading day -> ticker. Days without both settlements are left out."""
    out = {}
    days = settlements.index
    for prev, day in zip(days[:-1], days[1:]):
        t = contract.get(day)
        if t is None or pd.isna(t) or t not in settlements:
            continue
        a, b = settlements.at[prev, t], settlements.at[day, t]
        if pd.notna(a) and pd.notna(b):
            out[day] = b - a
    return pd.Series(out, dtype="float64")


def hedge_ratios(price_changes: pd.Series, yield_changes_bp: pd.Series, window: int) -> pd.Series:
    """Per day, the bp of yield per point of price - a regression through the origin of
    yield changes on price changes over the ``window`` days BEFORE it (point in time: day
    D's ratio never sees D). Needs a full window."""
    d = pd.concat([price_changes.rename("dp"), yield_changes_bp.rename("dy")], axis=1, join="inner").dropna()
    beta = (d["dp"] * d["dy"]).rolling(window).sum() / (d["dp"] ** 2).rolling(window).sum()
    return beta.shift(1).dropna()


def mid_at(quotes: pd.DataFrame, times) -> np.ndarray:
    """The quote mid KNOWN at each of ``times``: the latest ``bbo-1m`` sample stamped at
    or before it (a sample is the book at its own timestamp - CLAUDE.md 14). NaN before
    the first sample or where the book was one-sided."""
    times = pd.DatetimeIndex(times)
    q = quotes.dropna(subset=["mid"]).sort_values("timestamp")
    if q.empty:
        return np.full(len(times), np.nan)
    stamps = q["timestamp"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(stamps, times.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    mids = q["mid"].to_numpy(dtype=float)
    return np.where(idx >= 0, mids[np.clip(idx, 0, None)], np.nan)
