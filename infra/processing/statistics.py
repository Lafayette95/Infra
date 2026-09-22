"""Turn raw Databento `statistics` rows into daily settlement price / open interest.

Long format in (one row per stat_type update), one row per (trading day, contract) out.
Futures functions key by ``ticker``; options functions key by
underlying/option_type/strike/expiry (matching infra.processing.transforms's options
schema) since one call can cover many option contracts at once. See CLAUDE.md section 8
for the schema and the quirks this module works around.
"""
from __future__ import annotations

import databento_dbn as dbn
import numpy as np
import pandas as pd

from infra.trading_calendar import trading_day

_SETTLEMENT = int(dbn.StatType.SETTLEMENT_PRICE)
_OPEN_INTEREST = int(dbn.StatType.OPEN_INTEREST)
_EPOCH = pd.Timestamp("1970-01-01")

DAILY_COLUMNS = ["timestamp", "ticker", "settlement_price", "open_interest"]
DAILY_KEYS = ["timestamp", "ticker"]

DAILY_OPTIONS_COLUMNS = [
    "timestamp", "underlying", "option_type", "strike", "expiry", "settlement_price", "open_interest",
]
DAILY_OPTIONS_KEYS = ["timestamp", "underlying", "option_type", "strike", "expiry"]


def resolve_trading_day(raw: pd.DataFrame, dataset: str) -> pd.Series:
    """The trading day each statistics row belongs to.

    Prefers Databento's own ``ts_ref`` (the exchange's reference date for that
    statistic) when present - this is NOT always the same day as ``ts_recv``: open
    interest is commonly published early the next session but references the PRIOR
    trading day's close (``ts_ref`` correctly reflects that; naively bucketing by
    ``ts_recv`` would misclassify it - verified against real CME futures AND option
    data 2026-09-21). Falls back to ``infra.trading_calendar.trading_day(ts_recv,
    dataset)`` only when ``ts_ref`` is null (observed for Eurex/ICE settlement, which
    never cross midnight so ``ts_recv``'s own trading day is unambiguous anyway).

    Requires ``ts_recv`` and ``ts_ref`` as plain COLUMNS (not the index) - callers
    reset the index first, since ``get_range().to_df()`` returns ``ts_recv`` as the
    DataFrame index.
    """
    ts_ref = pd.to_datetime(raw["ts_ref"], utc=True).dt.tz_localize(None).dt.normalize()
    ts_recv = pd.to_datetime(raw["ts_recv"], utc=True).dt.tz_localize(None)
    fallback = pd.Series(trading_day(pd.DatetimeIndex(ts_recv), dataset), index=raw.index)
    return ts_ref.fillna(fallback)


def _pivot_last_per_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Settlement price and open interest, each the LAST update within its group (in
    ``ts_recv`` order) - the final value wins over any preliminary one. A group with
    only one of the two still produces a row (the other column is NaN/NA)."""
    settle = df[df["stat_type"] == _SETTLEMENT].groupby(group_cols)["price"].last().rename("settlement_price")
    oi = df[df["stat_type"] == _OPEN_INTEREST].groupby(group_cols)["quantity"].last().rename("open_interest")
    return pd.concat([settle, oi], axis=1, sort=True)


def _prepare(raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Shared prep: reset index, keep only settlement/OI rows, tag each with its
    trading day, sort chronologically. Returns an empty (0-row, unfiltered-shape) frame
    when there's nothing to keep - callers check ``.empty`` themselves."""
    if raw.empty:
        return raw
    df = raw.reset_index() if raw.index.name == "ts_recv" else raw.copy()
    df = df[df["stat_type"].isin([_SETTLEMENT, _OPEN_INTEREST])]
    if df.empty:
        return df
    df["_day"] = resolve_trading_day(df, dataset)
    return df.sort_values("ts_recv")


def clean_daily_statistics(raw: pd.DataFrame, ticker: str, dataset: str) -> pd.DataFrame:
    """Raw ``statistics`` rows for ONE absolute FUTURES contract -> one row per trading day."""
    df = _prepare(raw, dataset)
    if df.empty:
        return empty_daily()
    out = _pivot_last_per_group(df, ["_day"]).reset_index(names="timestamp")
    out["ticker"] = ticker
    out["open_interest"] = out["open_interest"].astype("Int64")
    return out[DAILY_COLUMNS].sort_values(DAILY_KEYS).reset_index(drop=True)


def clean_daily_option_statistics(raw: pd.DataFrame, definitions: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Raw ``statistics`` rows for an array of option instrument ids -> one row per
    (trading day, contract). ``definitions`` supplies underlying/option_type/strike/expiry
    per ``instrument_id`` - the same normalised frame ``infra.pipeline.options`` already
    uses (Rule 2.3 step 1); joined here rather than trusting the statistics payload's own
    ``symbol`` column, which does not reliably resolve for options (verified 2026-09-21).
    """
    df = _prepare(raw, dataset)
    if df.empty:
        return empty_daily_options()
    pivoted = _pivot_last_per_group(df, ["_day", "instrument_id"]).reset_index(names=["timestamp", "instrument_id"])
    meta = definitions[["instrument_id", "underlying", "option_type", "strike", "expiry"]]
    out = pivoted.merge(meta, on="instrument_id", how="inner")
    out["open_interest"] = out["open_interest"].astype("Int64")
    return out[DAILY_OPTIONS_COLUMNS].sort_values(DAILY_OPTIONS_KEYS).reset_index(drop=True)


def empty_daily() -> pd.DataFrame:
    """The canonical (decoded, float) empty FUTURES frame - used by pipeline/daily.py
    when nothing is on disk yet, so callers always see the same dtypes either way."""
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ms]"),
        "ticker": pd.Series(dtype="str"),
        "settlement_price": pd.Series(dtype="float64"),
        "open_interest": pd.Series(dtype="Int64"),
    })[DAILY_COLUMNS]


def empty_daily_options() -> pd.DataFrame:
    """The canonical (decoded, float) empty OPTIONS frame."""
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ms]"),
        "underlying": pd.Series(dtype="str"),
        "option_type": pd.Series(dtype="str"),
        "strike": pd.Series(dtype="float64"),
        "expiry": pd.Series(dtype="datetime64[ms]"),
        "settlement_price": pd.Series(dtype="float64"),
        "open_interest": pd.Series(dtype="Int64"),
    })[DAILY_OPTIONS_COLUMNS]


def encode_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical float FUTURES frame -> disk representation (fixed-point settlement price).

    ``settlement_price`` uses nullable ``Int32`` (like ``open_interest`` already does
    elsewhere) rather than plain ``int32``: a day can have one stat without the other
    (e.g. an OI update with no settlement yet), and ``scale_prices`` (CLAUDE.md 6b)
    assumes no NaN, so the ×10000 scaling is done directly here instead.
    """
    out = df[DAILY_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    out["settlement_price"] = (out["settlement_price"] * 10_000).round().astype("Int32")
    out["open_interest"] = out["open_interest"].astype("Int32")
    return out


def decode_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Disk representation -> float settlement price and a ``category`` ticker."""
    out = df.copy()
    out["settlement_price"] = out["settlement_price"].astype("float64") / 10_000.0
    out["ticker"] = out["ticker"].astype("category")
    return out


def encode_daily_options(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical float OPTIONS frame -> disk representation.

    ``strike`` comes from ``definitions`` (never missing after the inner join in
    ``clean_daily_option_statistics``), so it uses plain ``int32`` like
    ``infra.processing.transforms``'s 1-minute options schema; ``settlement_price``
    stays nullable ``Int32`` for the same reason as the futures encoder.
    """
    out = df[DAILY_OPTIONS_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    out["settlement_price"] = (out["settlement_price"] * 10_000).round().astype("Int32")
    strike_scaled = np.rint(out["strike"].astype("float64") * 10_000)
    if ((out["strike"].abs() > 0) & (strike_scaled == 0)).any():
        raise ValueError(
            "strike has a non-zero value below 1/10000; the float32 fallback for "
            "sub-0.0001 premiums (CLAUDE.md 6c) is not implemented."
        )
    out["strike"] = strike_scaled.astype("int32")
    out["expiry"] = ((out["expiry"].dt.normalize() - _EPOCH) // pd.Timedelta(days=1)).astype("int32")
    out["open_interest"] = out["open_interest"].astype("Int32")
    return out


def decode_daily_options(df: pd.DataFrame) -> pd.DataFrame:
    """Disk representation -> float settlement price/strike, datetime expiry, categories."""
    out = df.copy()
    out["settlement_price"] = out["settlement_price"].astype("float64") / 10_000.0
    out["strike"] = out["strike"].astype("float64") / 10_000.0
    out["expiry"] = _EPOCH + pd.to_timedelta(out["expiry"].astype("int64"), unit="D")
    out["underlying"] = out["underlying"].astype("category")
    out["option_type"] = out["option_type"].astype("category")
    return out
