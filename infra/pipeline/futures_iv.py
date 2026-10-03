"""At-the-money implied vol of a root's futures options, from the STORED daily option
settlements (``Daily/Options``, fetched by ``infra.pipeline.futures_options_iv``) and
futures settlements (``Daily/Futures``) - reads disk only, never fetches.

    atm = atm_iv("ZN", "2019-01-02", "2026-10-01")          # per day x expiry
    h = vol_at_horizon(atm, horizon_dates)                   # per day, at a horizon

The math is ``infra.analytics.futures_iv`` (pure). Point in time: a day's row uses only
that day's settlements (known at its close).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.analytics.futures_iv import atm_vol, implied_vols, vol_at_horizon  # noqa: F401 (re-export)
from infra.config import DAILY_OPTIONS_DIR
from infra.pipeline import daily as dl
from infra.processing.statistics import decode_daily_options
from infra.storage import parquet_store


def option_settlements(root: str, start, end, *, store: Path = DAILY_OPTIONS_DIR) -> pd.DataFrame:
    """Stored option settlements whose underlying is one of ``root``'s futures."""
    raw = parquet_store.read_partitioned(store, start=pd.Timestamp(start), end=pd.Timestamp(end) + pd.Timedelta(days=1))
    if raw is None or raw.empty:
        return decode_daily_options(raw if raw is not None else pd.DataFrame(
            columns=["timestamp", "underlying", "option_type", "strike", "expiry", "settlement_price", "open_interest"]))
    df = decode_daily_options(raw)
    df["underlying"] = df["underlying"].astype(str)
    return df[df["underlying"].str.fullmatch(rf"{root}[FGHJKMNQUVXZ]\d")].reset_index(drop=True)


def atm_iv(root: str, start, end, *, store: Path = DAILY_OPTIONS_DIR, discount: float = 1.0) -> pd.DataFrame:
    """Per (day, expiry), the at-the-money implied vol (``infra.analytics.futures_iv.atm_vol``)."""
    opts = option_settlements(root, start, end, store=store)
    if opts.empty:
        return atm_vol(pd.DataFrame(columns=["timestamp", "expiry", "underlying", "option_type", "strike", "future", "T", "iv"]))
    fut = dl.read_daily_from_disk(sorted(opts["underlying"].unique()), pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))
    fut["ticker"] = fut["ticker"].astype(str)
    fut = fut.dropna(subset=["settlement_price"]).rename(columns={"settlement_price": "price"})[["timestamp", "ticker", "price"]]
    return atm_vol(implied_vols(opts, fut, discount=discount))
