"""Timeframe aggregation for decoded OHLCV frames (pure pandas)."""
from __future__ import annotations

import pandas as pd

# label -> pandas offset alias
TIMEFRAMES: dict[str, str] = {
    "1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1D": "1D",
}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
# Non-numeric columns carried through unchanged when present (see docstring for why
# "first" is exact here, not an approximation).
_PASSTHROUGH = ("contract",)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate ONE ticker's bars (columns incl. ``timestamp``) to ``timeframe``.

    Returns a frame indexed by bar start with open/high/low/close/volume, plus
    ``contract`` (the absolute ticker each bar came from - see
    ``infra.pipeline.series.load_series``) when the input has it, aggregated with
    "first". This is exact: ``infra.relative.series.apply_mapping`` assigns one
    contract per UTC calendar day, and every offered timeframe bins on UTC-midnight-
    aligned boundaries, so a bucket can never straddle a roll.
    """
    rule = TIMEFRAMES[timeframe]
    agg = {**_AGG, **{c: "first" for c in _PASSTHROUGH if c in df.columns}}
    columns = list(agg)
    if df.empty:
        return df.set_index("timestamp")[columns]
    out = df.set_index("timestamp")[columns].resample(rule).agg(agg)
    return out.dropna(subset=["open"])


def coarsen_to_fit(df: pd.DataFrame, timeframe: str, max_bars: int) -> tuple[pd.DataFrame, str]:
    """Resample, moving to coarser timeframes until at most ``max_bars`` bars remain."""
    labels = list(TIMEFRAMES)
    for label in labels[labels.index(timeframe):]:
        bars = resample_ohlcv(df, label)
        if len(bars) <= max_bars:
            return bars, label
    return bars, label
