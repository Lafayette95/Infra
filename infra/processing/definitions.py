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
