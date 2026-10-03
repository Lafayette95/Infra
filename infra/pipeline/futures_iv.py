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

from infra.analytics.futures_iv import (atm_vol, ewma_points, front_changes, implied_vols,  # noqa: F401
                                        vol_at_horizon)
from infra.config import DAILY_FUTURES_DIR, DAILY_OPTIONS_DIR
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



def front_ewma_points(root: str, start, end, *, lam: float = 0.94, history_days: int = 400,
                      store: Path = DAILY_FUTURES_DIR) -> pd.Series:
    """EWMA vol (futures points/day) of ``root``'s front-contract settlement changes (most
    open interest the day before), by day - the realised side of the implied/realised
    ratio the basis models' IV add-on scales by. Reads ``Daily/Futures``."""
    first = pd.Timestamp(start) - pd.Timedelta(days=history_days)
    stop = pd.Timestamp(end) + pd.Timedelta(days=1)
    names = parquet_store.read_partitioned(store, start=first, end=stop, columns=["ticker"])
    tickers = pd.Series(names["ticker"].astype(str).unique()) if names is not None else pd.Series([], dtype=str)
    tickers = sorted(tickers[tickers.str.fullmatch(rf"{root}[FGHJKMNQUVXZ]\d")])
    fut = dl.read_daily_from_disk(tickers, first, stop)
    fut["ticker"] = fut["ticker"].astype(str)
    w = fut.pivot(index="timestamp", columns="ticker", values="settlement_price").sort_index()
    oi = fut.pivot(index="timestamp", columns="ticker", values="open_interest").sort_index()
    return ewma_points(front_changes(w, oi), lam)
