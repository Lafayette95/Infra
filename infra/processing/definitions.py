"""Normalise and locally filter Databento ``definition`` frames (Rule 2.3 steps 1-2)."""
from __future__ import annotations

import pandas as pd

DEFINITION_COLUMNS = ["instrument_id", "underlying", "option_type", "strike", "expiry"]


def normalize_definitions(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw option ``definition`` rows -> one row per instrument with canonical columns."""
    if raw.empty:
        return pd.DataFrame({
            "instrument_id": pd.Series(dtype="int64"),
            "underlying": pd.Series(dtype="str"),
            "option_type": pd.Series(dtype="str"),
            "strike": pd.Series(dtype="float64"),
            "expiry": pd.Series(dtype="datetime64[ms]"),
        })
    df = raw.reset_index() if "ts_recv" in raw.index.names else raw.copy()
    df = df[df["instrument_class"].isin(["C", "P"])]
    out = pd.DataFrame({
        "instrument_id": df["instrument_id"].astype("int64"),
        "underlying": df["underlying"].astype(str),
        "option_type": df["instrument_class"].astype(str),
        "strike": df["strike_price"].astype("float64"),
        "expiry": pd.to_datetime(df["expiration"], utc=True)
        .dt.tz_localize(None).dt.normalize().astype("datetime64[ms]"),
    })
    return out.drop_duplicates(subset="instrument_id", keep="last").reset_index(drop=True)


def filter_definitions(
    definitions: pd.DataFrame,
    *,
    underlyings: list[str] | None = None,
    option_types: list[str] | None = None,
    strike_range: tuple[float, float] | None = None,
    expiry_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Filter contracts locally by underlying / type / strike / expiry (inclusive bounds)."""
    mask = pd.Series(True, index=definitions.index)
    if underlyings:
        mask &= definitions["underlying"].isin(underlyings)
    if option_types:
        mask &= definitions["option_type"].isin(option_types)
    if strike_range:
        mask &= definitions["strike"].between(*strike_range)
    if expiry_range:
        lo, hi = (pd.Timestamp(x) for x in expiry_range)
        mask &= definitions["expiry"].between(lo, hi)
    return definitions[mask].reset_index(drop=True)


CONTRACT_COLUMNS = ["root", "ticker", "instrument_id", "expiry", "activation"]


def normalize_futures_definitions(
    raw: pd.DataFrame, root: str, ticker_regex: str | None = None
) -> pd.DataFrame:
    """Raw futures ``definition`` rows -> one row per OUTRIGHT contract.

    Parent queries also return spreads (``instrument_class == 'S'``; e.g. 7,144 of
    7,190 rows for SR3), which are dropped here. ``ticker`` is the absolute raw symbol.
    ``ticker_regex`` further keeps only symbols that match (e.g. ICE's quarterly contracts).
    """
    empty = pd.DataFrame({
        "root": pd.Series(dtype="str"),
        "ticker": pd.Series(dtype="str"),
        "instrument_id": pd.Series(dtype="int64"),
        "expiry": pd.Series(dtype="datetime64[ms]"),
        "activation": pd.Series(dtype="datetime64[ms]"),
    })
    if raw.empty:
        return empty
    df = raw.reset_index() if "ts_recv" in raw.index.names else raw.copy()
    df = df[df["instrument_class"] == "F"]
    if ticker_regex:
        df = df[df["raw_symbol"].astype(str).str.contains(ticker_regex, regex=True)]
    if df.empty:
        return empty
    out = pd.DataFrame({
        "root": root,
        "ticker": df["raw_symbol"].astype(str),
        "instrument_id": df["instrument_id"].astype("int64"),
        "expiry": _to_day(df["expiration"]),
        "activation": _to_day(df["activation"]),
    })
    return out.drop_duplicates(subset=["ticker", "expiry"], keep="last").reset_index(drop=True)


def _to_day(values: pd.Series) -> pd.Series:
    """Datetimes -> tz-naive UTC midnight ``datetime64[ms]`` (NaT preserved)."""
    return pd.to_datetime(values, utc=True).dt.tz_localize(None).dt.normalize().astype("datetime64[ms]")
