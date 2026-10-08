"""Montréal Exchange futures - pure parsing of the historical-data CSV (no network, no files).

One row per (trading day, contract): ``settlement`` (the exchange's daily settlement price),
``open`` / ``high`` / ``low`` / ``last``, the closing ``bid`` / ``ask``, ``volume``,
``open_interest`` and the contract's ``expiry``. Prices are stored x10000 nullable Int32 (root
CLAUDE.md 6b); a 0.00 the source prints for "no trade / no quote" is NaN, never a price.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "ticker", "root", "expiry", "settlement", "open", "high", "low", "last", "bid", "ask",
           "volume", "open_interest"]
PRICES = ["settlement", "open", "high", "low", "last", "bid", "ask"]
_RENAME = {"Date": "timestamp", "Symbol": "ticker", "Root Symbol": "root", "Expiry Date": "expiry",
           "Settlement Price": "settlement", "Open Price": "open", "High Price": "high", "Low Price": "low",
           "Last Price": "last", "Bid Price": "bid", "Ask Price": "ask", "Volume": "volume", "Open Interest": "open_interest"}


def parse(text: str, root: str | None = None) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame(columns=COLUMNS)
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.rename(columns=_RENAME)
    if "root" not in df or df["root"].isna().all():
        df["root"] = root
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
    for c in PRICES:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace(0.0, np.nan)
    for c in ("volume", "open_interest"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "ticker"])
    if root is not None:
        df = df[df["root"].fillna(root).astype(str).str.strip() == root]
    return df[COLUMNS].drop_duplicates(["timestamp", "ticker"], keep="last").reset_index(drop=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PRICES:
        out[c] = (out[c] * 10000).round().astype("Int32")
    for c in ("volume", "open_interest"):
        out[c] = out[c].round().astype("Int64")
    for c in ("timestamp", "expiry"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    for c in ("ticker", "root"):
        out[c] = out[c].astype("string")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PRICES:
        out[c] = out[c].astype("float64") / 10000.0
    for c in ("ticker", "root"):
        out[c] = out[c].astype(str)
    return out


def front(df: pd.DataFrame) -> pd.Series:
    """Day -> the contract HELD that day: the one with the largest open interest on the
    PREVIOUS trading day (known before the day starts - point in time; open interest is
    published with the day's settlement). The first day of a history has none."""
    oi = df.pivot_table(index="timestamp", columns="ticker", values="open_interest").sort_index()
    prev = oi.shift(1)
    has = prev.notna().any(axis=1) & (prev.fillna(0) > 0).any(axis=1)
    return prev[has].fillna(-1).idxmax(axis=1)
