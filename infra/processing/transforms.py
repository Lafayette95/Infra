"""Cleaning and encoding of Databento payloads (pure pandas, no I/O).

Lifecycle:  raw API frame -> ``clean_*`` (float prices, canonical columns)
            -> ``encode_*`` (fixed-point int32, disk dtypes) -> parquet.
Reading:    parquet -> ``decode_*`` (float prices) -> optional ``to_multiindex``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import PARTITION_COLUMNS, PRICE_SCALE

FUTURES_COLUMNS = [
    "timestamp", "ticker", "open", "high", "low", "close", "volume", "open_interest",
]
FUTURES_PRICE_COLUMNS = ["open", "high", "low", "close"]
FUTURES_KEYS = ["timestamp", "ticker"]

OPTIONS_COLUMNS = [
    "timestamp", "underlying", "option_type", "strike", "expiry",
    "open", "high", "low", "close", "volume", "open_interest",
]
OPTIONS_PRICE_COLUMNS = ["open", "high", "low", "close", "strike"]
OPTIONS_KEYS = ["timestamp", "underlying", "option_type", "strike", "expiry"]

_INT32_MAX = np.iinfo("int32").max
_EPOCH = pd.Timestamp("1970-01-01")


# ------------------------------------------------------------- scaling primitives
def scale_prices(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Multiply price columns by PRICE_SCALE and cast to int32."""
    out = df.copy()
    for col in columns:
        scaled = np.rint(out[col].astype("float64") * PRICE_SCALE)
        if scaled.abs().max() > _INT32_MAX:
            raise OverflowError(f"Column {col!r} does not fit int32 at scale {PRICE_SCALE}.")
        if ((out[col].abs() > 0) & (scaled == 0)).any():
            raise ValueError(
                f"Column {col!r} has non-zero values below 1/{PRICE_SCALE}; the float32 "
                "fallback for sub-0.0001 premiums is not implemented."
            )
        out[col] = scaled.astype("int32")
    return out


def unscale_prices(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Divide fixed-point price columns by PRICE_SCALE (float64, in memory only)."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].astype("float64") / float(PRICE_SCALE)
    return out


def to_naive_utc_ms(values: pd.Series) -> pd.Series:
    """Convert a datetime series to tz-naive UTC ``datetime64[ms]``."""
    values = pd.to_datetime(values, utc=True)
    return values.dt.tz_localize(None).astype("datetime64[ms]")


def add_partition_columns(df: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Add hive partition columns (year, quarter) derived from the record timestamp."""
    out = df.copy()
    ts = out[timestamp_column]
    out["year"] = ts.dt.year.astype("int32")
    out["quarter"] = ts.dt.quarter.astype("int32")
    assert list(PARTITION_COLUMNS) == ["year", "quarter"]
    return out


# ---------------------------------------------------------------------- futures
def clean_futures_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw ``ohlcv-1m`` frame (index ``ts_event``) -> flat canonical futures frame.

    Prices stay float here. ``open_interest`` is not part of ``ohlcv-1m`` so it is
    stored as a nullable Int32 (null = not sourced), never a fake 0.
    """
    if raw.empty:
        return _empty_futures()
    df = raw.reset_index()
    df = df.rename(columns={"ts_event": "timestamp", "symbol": "ticker"})
    df["timestamp"] = to_naive_utc_ms(df["timestamp"])
    df["ticker"] = df["ticker"].astype(str)
    df = df.dropna(subset=FUTURES_PRICE_COLUMNS)
    df = df[(df[FUTURES_PRICE_COLUMNS] > 0).all(axis=1)]
    df["volume"] = df["volume"].astype("int32")
    df["open_interest"] = pd.array([pd.NA] * len(df), dtype="Int32")
    df = df.drop_duplicates(subset=FUTURES_KEYS, keep="last").sort_values(FUTURES_KEYS)
    return df[FUTURES_COLUMNS].reset_index(drop=True)


def _empty_futures() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ms]"),
        "ticker": pd.Series(dtype="str"),
        **{c: pd.Series(dtype="float64") for c in FUTURES_PRICE_COLUMNS},
        "volume": pd.Series(dtype="int32"),
        "open_interest": pd.Series(dtype="Int32"),
    })[FUTURES_COLUMNS]


def encode_futures(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical float frame -> disk representation (int32 fixed-point prices)."""
    out = scale_prices(df[FUTURES_COLUMNS], FUTURES_PRICE_COLUMNS)
    out["volume"] = out["volume"].astype("int32")
    out["open_interest"] = out["open_interest"].astype("Int32")
    return out


def decode_futures(df: pd.DataFrame) -> pd.DataFrame:
    """Disk representation -> float prices and a ``category`` ticker."""
    out = unscale_prices(df, FUTURES_PRICE_COLUMNS)
    out["ticker"] = out["ticker"].astype("category")
    return out


# ---------------------------------------------------------------------- options
def clean_options_ohlcv(raw_bars: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    """Join raw option bars to normalised definitions -> canonical options frame.

    ``definitions`` must be the output of ``processing.definitions.normalize_definitions``.
    """
    if raw_bars.empty:
        return _empty_options()
    bars = raw_bars.reset_index().rename(columns={"ts_event": "timestamp"})
    bars["timestamp"] = to_naive_utc_ms(bars["timestamp"])
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    meta = definitions[["instrument_id", "underlying", "option_type", "strike", "expiry"]]
    df = bars.merge(meta, on="instrument_id", how="inner")
    df["volume"] = df["volume"].astype("int32")
    df["open_interest"] = pd.array([pd.NA] * len(df), dtype="Int32")
    df = df.drop_duplicates(subset=OPTIONS_KEYS, keep="last").sort_values(OPTIONS_KEYS)
    return df[OPTIONS_COLUMNS].reset_index(drop=True)


def _empty_options() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ms]"),
        "underlying": pd.Series(dtype="str"),
        "option_type": pd.Series(dtype="str"),
        "strike": pd.Series(dtype="float64"),
        "expiry": pd.Series(dtype="datetime64[ms]"),
        **{c: pd.Series(dtype="float64") for c in ("open", "high", "low", "close")},
        "volume": pd.Series(dtype="int32"),
        "open_interest": pd.Series(dtype="Int32"),
    })[OPTIONS_COLUMNS]


def encode_options(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical options frame -> disk representation (expiry as int32 day epoch)."""
    out = scale_prices(df[OPTIONS_COLUMNS], OPTIONS_PRICE_COLUMNS)
    out["expiry"] = ((out["expiry"].dt.normalize() - _EPOCH) // pd.Timedelta(days=1)).astype("int32")
    out["volume"] = out["volume"].astype("int32")
    out["open_interest"] = out["open_interest"].astype("Int32")
    return out


def decode_options(df: pd.DataFrame) -> pd.DataFrame:
    """Disk representation -> float prices/strike, datetime expiry, category labels."""
    out = unscale_prices(df, OPTIONS_PRICE_COLUMNS)
    out["expiry"] = _EPOCH + pd.to_timedelta(out["expiry"].astype("int64"), unit="D")
    out["underlying"] = out["underlying"].astype("category")
    out["option_type"] = out["option_type"].astype("category")
    return out


# --------------------------------------------------------------------- in-memory
def to_multiindex(df: pd.DataFrame, level: str = "ticker") -> pd.DataFrame:
    """Build the (timestamp, ticker|underlying) MultiIndex in RAM only."""
    return df.set_index(["timestamp", level]).sort_index()
