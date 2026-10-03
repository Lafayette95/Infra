"""Futures snaps (CLAUDE.md 23): quotes -> one row per (snap instant, contract). Pure.

A snap is the LAST two-sided quote at or before the instant, at most ``tolerance`` old (a
bbo-1m sample is the book at its own timestamp - CLAUDE.md 14 - so a sample stamped at the
instant counts). Known at the instant itself (point in time).
"""
from __future__ import annotations

import pandas as pd

SNAP_COLUMNS = ["timestamp", "snap", "ticker", "bid", "ask", "mid", "quote_time", "settlement"]
SNAP_KEYS = ["timestamp", "snap", "ticker"]
PRICE_COLUMNS = ("bid", "ask", "mid", "settlement")
_SCALE = 10_000


def select_snaps(quotes: pd.DataFrame, instants: pd.DataFrame, tolerance: pd.Timedelta) -> pd.DataFrame:
    """``quotes``: decoded bbo rows (``timestamp``, ``ticker``, ``bid``, ``ask``, ``mid``).
    ``instants``: ``snap`` (name), ``timestamp`` (UTC instant). Returns SNAP_COLUMNS
    without ``settlement`` (the caller adds it)."""
    q = quotes.dropna(subset=["bid", "ask"]).sort_values("timestamp")
    if q.empty or instants.empty:
        return pd.DataFrame(columns=SNAP_COLUMNS[:-1])
    parts = []
    for snap, t in zip(instants["snap"], instants["timestamp"]):
        w = q[(q["timestamp"] <= t) & (q["timestamp"] > t - tolerance)]
        if w.empty:
            continue
        last = w.groupby("ticker", observed=True).tail(1)
        parts.append(pd.DataFrame({"timestamp": t, "snap": snap, "ticker": last["ticker"].astype(str).to_numpy(),
                                   "bid": last["bid"].to_numpy(), "ask": last["ask"].to_numpy(),
                                   "mid": last["mid"].to_numpy(), "quote_time": last["timestamp"].to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=SNAP_COLUMNS[:-1])
    return pd.concat(parts, ignore_index=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df[SNAP_COLUMNS].copy()
    for c in ("timestamp", "quote_time"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    for c in PRICE_COLUMNS:
        out[c] = (out[c].astype("float64") * _SCALE).round().astype("Int32")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PRICE_COLUMNS:
        out[c] = out[c].astype("float64") / _SCALE
    for c in ("snap", "ticker"):
        out[c] = out[c].astype(str)
    return out
