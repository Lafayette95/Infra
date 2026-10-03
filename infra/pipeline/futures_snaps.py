"""Futures snaps (CLAUDE.md 23): build / read ``Derived/FuturesSnaps`` from the stored
bbo-1m quotes - local computation only, never fetches.

One row per (snap instant UTC, snap name, contract): the last two-sided quote at or before
the instant, its time, and the contract's settlement on that CME trading day for
comparison. Every contract with bbo-1m quotes that day is snapped (the bond futures' front
contracts, the ZQ strip, ...). A rebuilt range REPLACES its days. Snap instants are
``infra.config.SWAP_CLOSES`` local times converted to UTC on each day
(``infra.trading_calendar.snap_instants``), so DST is handled at that one step.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.config import (
    BBO_FUTURES_DIR,
    DAILY_FUTURES_DIR,
    FUTURES_SNAP_TOLERANCE_MIN,
    FUTURES_SNAPS,
    FUTURES_SNAPS_DIR,
    SWAP_CLOSES,
)
from infra.pipeline.daily import read_daily_from_disk
from infra.processing import futures_snaps as fs
from infra.processing import transforms as tf
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants, trading_day

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)


def snap_times(days, snaps=FUTURES_SNAPS) -> pd.DataFrame:
    """``snap``, ``timestamp`` (UTC) for every weekday in ``days`` and configured snap."""
    days = pd.DatetimeIndex(days)
    days = days[days.weekday < 5]
    rows = [pd.DataFrame({"snap": name, "timestamp": snap_instants(days, SWAP_CLOSES[name].local_time,
                                                                     SWAP_CLOSES[name].timezone)}) for name in snaps]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["snap", "timestamp"])


def build_futures_snaps(start, end, *, snaps=FUTURES_SNAPS, bbo_root: Path = BBO_FUTURES_DIR,
                        daily_root: Path = DAILY_FUTURES_DIR, root: Path = FUTURES_SNAPS_DIR,
                        chunk_days: int = 31) -> int:
    """Compute and store the snaps for days ``[start, end]``, replacing those days.
    Reads quotes a month at a time. Returns the rows written."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    tolerance = pd.Timedelta(minutes=FUTURES_SNAP_TOLERANCE_MIN)
    total, d0 = 0, start
    while d0 <= end:
        d1 = min(d0 + pd.Timedelta(days=chunk_days - 1), end)
        instants = snap_times(pd.date_range(d0, d1), snaps)
        raw = parquet_store.read_partitioned(bbo_root, start=instants["timestamp"].min() - tolerance,
                                             end=instants["timestamp"].max() + pd.Timedelta(seconds=1)) \
            if len(instants) else None
        rows = pd.DataFrame(columns=fs.SNAP_COLUMNS)
        if raw is not None and len(raw):
            quotes = tf.decode_futures_bbo(raw[tf.BBO_COLUMNS])
            rows = fs.select_snaps(quotes, instants, tolerance)
            if len(rows):
                rows["settlement"] = _settlements(rows, daily_root)
        parquet_store.delete_where(root, lambda part, a=d0, b=d1: pd.to_datetime(part["timestamp"]).between(
            a, b + _ONE_DAY, inclusive="left"))
        if len(rows):
            parquet_store.write_partitioned(fs.encode(rows), root, fs.SNAP_KEYS)
            total += len(rows)
        log.info("futures snaps %s..%s: %d rows", d0.date(), d1.date(), len(rows))
        d0 = d1 + _ONE_DAY
    return total


def _settlements(rows: pd.DataFrame, daily_root: Path) -> pd.Series:
    """Each row's contract settlement on the CME trading day its snap falls in."""
    tday = trading_day(pd.DatetimeIndex(rows["timestamp"]), "GLBX.MDP3")
    s = read_daily_from_disk(sorted(rows["ticker"].unique()), tday.min(), tday.max() + _ONE_DAY,
                             root=daily_root, adjusted=False)
    if s.empty:
        return pd.Series(float("nan"), index=rows.index)
    m = s.assign(ticker=s["ticker"].astype(str)).set_index(["timestamp", "ticker"])["settlement_price"].astype(float)
    return pd.Series([m.get((d, t), float("nan")) for d, t in zip(tday, rows["ticker"])], index=rows.index)


def read_futures_snaps(start, end, *, snap: str | None = None, tickers=None,
                       root: Path = FUTURES_SNAPS_DIR) -> pd.DataFrame:
    """Snaps with instants in ``[start, end)`` (UTC), optionally one snap / some tickers."""
    eq = {}
    if snap is not None:
        eq["snap"] = [snap]
    if tickers is not None:
        eq["ticker"] = list(tickers)
    raw = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end), equals_in=eq or None)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=fs.SNAP_COLUMNS)
    return fs.decode(raw[fs.SNAP_COLUMNS]).sort_values(fs.SNAP_KEYS).reset_index(drop=True)
