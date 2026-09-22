"""Turn raw Databento `statistics` rows into daily settlement price / open interest.

Long format in (one row per stat_type update), one row per (trading day, ticker) out.
See CLAUDE.md section 8 for the schema and the two quirks this module works around.
"""
from __future__ import annotations

import databento_dbn as dbn
import pandas as pd

from infra.trading_calendar import trading_day

_SETTLEMENT = int(dbn.StatType.SETTLEMENT_PRICE)
_OPEN_INTEREST = int(dbn.StatType.OPEN_INTEREST)

DAILY_COLUMNS = ["timestamp", "ticker", "settlement_price", "open_interest"]
DAILY_KEYS = ["timestamp", "ticker"]


def resolve_trading_day(raw: pd.DataFrame, dataset: str) -> pd.Series:
    """The trading day each statistics row belongs to.

    Prefers Databento's own ``ts_ref`` (the exchange's reference date for that
    statistic) when present - this is NOT always the same day as ``ts_recv``: open
    interest is commonly published early the next session but references the PRIOR
    trading day's close (``ts_ref`` correctly reflects that; naively bucketing by
    ``ts_recv`` would misclassify it - verified against real CME data 2026-09-21).
    Falls back to ``infra.trading_calendar.trading_day(ts_recv, dataset)`` only when
    ``ts_ref`` is null (observed for Eurex/ICE settlement, which never cross midnight
    so ``ts_recv``'s own trading day is unambiguous anyway).

    Requires ``ts_recv`` and ``ts_ref`` as plain COLUMNS (not the index) - callers
    reset the index first, since ``get_range().to_df()`` returns ``ts_recv`` as the
    DataFrame index.
    """
    ts_ref = pd.to_datetime(raw["ts_ref"], utc=True).dt.tz_localize(None).dt.normalize()
    ts_recv = pd.to_datetime(raw["ts_recv"], utc=True).dt.tz_localize(None)
    fallback = pd.Series(trading_day(pd.DatetimeIndex(ts_recv), dataset), index=raw.index)
    return ts_ref.fillna(fallback)


def clean_daily_statistics(raw: pd.DataFrame, ticker: str, dataset: str) -> pd.DataFrame:
    """Raw ``statistics`` rows for ONE absolute contract -> one row per trading day.

    Settlement price and open interest are each taken as the LAST update within their
    trading day (in ``ts_recv`` order) - the final value wins over any preliminary one.
    A day with only one of the two still produces a row (the other column is NaN/NA).
    """
    if raw.empty:
        return empty_daily()
    df = raw.reset_index() if raw.index.name == "ts_recv" else raw.copy()
    df = df[df["stat_type"].isin([_SETTLEMENT, _OPEN_INTEREST])]
    if df.empty:
        return empty_daily()
    df["_day"] = resolve_trading_day(df, dataset)
    df = df.sort_values("ts_recv")

    settle = (
        df[df["stat_type"] == _SETTLEMENT]
        .groupby("_day")["price"].last().rename("settlement_price")
    )
    oi = (
        df[df["stat_type"] == _OPEN_INTEREST]
        .groupby("_day")["quantity"].last().rename("open_interest")
    )
    out = pd.concat([settle, oi], axis=1, sort=True).reset_index(names="timestamp")
    out["ticker"] = ticker
    out["open_interest"] = out["open_interest"].astype("Int64")
    return out[DAILY_COLUMNS].sort_values(DAILY_KEYS).reset_index(drop=True)


def empty_daily() -> pd.DataFrame:
    """The canonical (decoded, float) empty frame - used by pipeline/daily.py when
    nothing is on disk yet, so callers always see the same dtypes either way."""
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ms]"),
        "ticker": pd.Series(dtype="str"),
        "settlement_price": pd.Series(dtype="float64"),
        "open_interest": pd.Series(dtype="Int64"),
    })[DAILY_COLUMNS]


def encode_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical float frame -> disk representation (fixed-point settlement price).

    ``settlement_price`` uses nullable ``Int32`` (like ``open_interest`` already does
    elsewhere) rather than plain ``int32``: a day can have one stat without the other
    (e.g. an OI update with no settlement yet), and ``scale_prices`` (CLAUDE.md 6b)
    assumes no NaN, so the ×10000 scaling is done directly here instead.
    """
    out = df[DAILY_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    scaled_price = (out["settlement_price"] * 10_000).round()
    out["settlement_price"] = scaled_price.astype("Int32")
    out["open_interest"] = out["open_interest"].astype("Int32")
    return out


def decode_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Disk representation -> float settlement price and a ``category`` ticker."""
    out = df.copy()
    out["settlement_price"] = out["settlement_price"].astype("float64") / 10_000.0
    out["ticker"] = out["ticker"].astype("category")
    return out
