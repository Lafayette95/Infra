"""Timeframe aggregation for decoded OHLCV frames (pure pandas)."""
from __future__ import annotations

import pandas as pd

# label -> pandas offset alias
TIMEFRAMES: dict[str, str] = {
    "1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1D": "1D",
}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate ONE ticker's bars (columns incl. ``timestamp``) to ``timeframe``.

    Returns a frame indexed by bar start with open/high/low/close/volume.
    """
    rule = TIMEFRAMES[timeframe]
    if df.empty:
        return df.set_index("timestamp")[list(_AGG)]
    out = df.set_index("timestamp")[list(_AGG)].resample(rule).agg(_AGG)
    return out.dropna(subset=["open"])


def coarsen_to_fit(df: pd.DataFrame, timeframe: str, max_bars: int) -> tuple[pd.DataFrame, str]:
    """Resample, moving to coarser timeframes until at most ``max_bars`` bars remain."""
    labels = list(TIMEFRAMES)
    for label in labels[labels.index(timeframe):]:
        bars = resample_ohlcv(df, label)
        if len(bars) <= max_bars:
            return bars, label
    return bars, label
