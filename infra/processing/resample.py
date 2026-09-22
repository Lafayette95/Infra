"""Timeframe aggregation for decoded OHLCV frames (pure pandas).

The "1D" bucket is the exchange's TRADING day (infra.trading_calendar), not the naive
UTC calendar day - see CLAUDE.md section 6e. Intraday buckets (1m..4h) stay aligned to
UTC-clock boundaries (00:00, 04:00, ... UTC), unchanged: this project only redefines
what a "day" means, not intraday bucket alignment. One consequence worth knowing: since
a trading-day roll (CME/CBOT) does not fall on a UTC-clock boundary, an intraday bucket
that happens to contain the exact moment of a roll (rare - quarterly for a calendar-
ranked series, occasional for a volume-ranked one) can span two different absolute
contracts, "first"-aggregating the ``contract`` label and blending both contracts'
prices into one candle. The daily bucket and the relative-series roll-day assignment
itself are unaffected and always correct; only that one coarsened intraday candle, at
that one moment, is imprecise. See TOFIX.md.
"""
from __future__ import annotations

import pandas as pd

from infra.trading_calendar import trading_day

# label -> pandas offset alias (every timeframe except "1D", which uses the exchange
# trading day instead - see module docstring).
TIMEFRAMES: dict[str, str] = {
    "1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1D": "1D",
}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
# Non-numeric columns carried through unchanged when present (see docstring for why
# "first" is exact for the "1D" bucket, and the narrow intraday caveat above).
_PASSTHROUGH = ("contract",)


def resample_ohlcv(df: pd.DataFrame, timeframe: str, *, dataset: str | None = None) -> pd.DataFrame:
    """Aggregate ONE ticker's bars (columns incl. ``timestamp``) to ``timeframe``.

    ``dataset`` (a Databento dataset id, e.g. "GLBX.MDP3") is REQUIRED for
    ``timeframe == "1D"``: the trading-day boundary is exchange-specific
    (infra.trading_calendar, CLAUDE.md section 6e). Ignored for every other timeframe.

    Returns a frame indexed by bucket start with open/high/low/close/volume, plus
    ``contract`` (the absolute ticker each bar came from - see
    ``infra.pipeline.series.load_series``) when the input has it.
    """
    agg = {**_AGG, **{c: "first" for c in _PASSTHROUGH if c in df.columns}}
    columns = list(agg)
    if df.empty:
        return df.set_index("timestamp")[columns]
    if timeframe == "1D":
        if not dataset:
            raise ValueError(
                "resample_ohlcv(timeframe='1D') requires `dataset`: the trading-day "
                "boundary is exchange-specific (infra.trading_calendar, CLAUDE.md 6e)."
            )
        sorted_df = df.sort_values("timestamp")
        day = trading_day(pd.DatetimeIndex(sorted_df["timestamp"]), dataset)
        out = sorted_df[columns].groupby(day.values, sort=True).agg(agg)
        out.index.name = "timestamp"
        return out.dropna(subset=["open"])
    out = df.set_index("timestamp")[columns].resample(TIMEFRAMES[timeframe]).agg(agg)
    return out.dropna(subset=["open"])


def coarsen_to_fit(
    df: pd.DataFrame, timeframe: str, max_bars: int, *, dataset: str | None = None
) -> tuple[pd.DataFrame, str]:
    """Resample, moving to coarser timeframes until at most ``max_bars`` bars remain.

    ``dataset`` is forwarded to ``resample_ohlcv`` and is required if coarsening can
    reach "1D" - i.e. whenever "1D" is at or after ``timeframe`` in ``TIMEFRAMES``.
    """
    labels = list(TIMEFRAMES)
    for label in labels[labels.index(timeframe):]:
        bars = resample_ohlcv(df, label, dataset=dataset)
        if len(bars) <= max_bars:
            return bars, label
    return bars, label
