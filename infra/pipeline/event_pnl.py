"""Step P&L on an event-study grid, per instrument: the "pnl" of root CLAUDE.md 26 (event studies). Disk only.

``grid_pnl(source, instruments, grid)`` -> wide frame, index = every grid point (UTC), one
column per instrument: the move over (previous grid point, point] - the previous point is
the previous trading day's last point for a day's first point (the overnight move).

Sources (``PNL_SOURCES``; a persisted 15-minute bmk store can be added later under the same
interface):

* ``FUTURE_BPS_BBO``: bbo-1m quote MIDS of the futures contract an instrument maps to
  (a relative ticker ``ZN.v.0`` -> its contract per CME trading day; an absolute ticker is
  itself), each step measured on ONE contract (the one mapped at the step's end point, at
  both ends - a roll is never a move), in bp: price change x point value / the contract's
  DV01 on the PRIOR trading day (the bmk risk store - the risk held into the day; STIR:
  x 100). A long contract's P&L in bp of yield.
* ``FUTURE_PTS_BBO``: the same mid changes in price points (no DV01).

A price at a point is the last two-sided quote at or before it, at most
``QUOTE_TOLERANCE`` old (a bbo-1m sample is the book AT its minute, root CLAUDE.md 14);
inside the venue's daily halt (CME 16:00-17:00 CT = 17:00-18:00 ET, inside the default
grid) the last quote before the halt; none -> NaN. bp steps also need the contract's DV01
in the bmk risk store (NaN where it has none). Point in time: a step is known at its point.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from infra.config import (BBO_FUTURES_DIR, BMK_ROOT, DAILY_FUTURES_DIR, FUTURES_CONTRACTS_FILE, FUTURES_ROOTS,
                          TRADING_HOURS)
from infra.pipeline.bbo import read_bbo_from_disk
from infra.relative.symbology import parse_relative
from infra.storage import parquet_store
from infra.trading_calendar import local_wallclock, snap_instants, trading_day

QUOTE_TOLERANCE = pd.Timedelta(minutes=5)
_ONE_DAY = pd.Timedelta(days=1)


def _root_of(instrument: str) -> str:
    spec = parse_relative(instrument)
    if spec is not None:
        return spec.root
    for r in sorted(FUTURES_ROOTS, key=len, reverse=True):
        if instrument.startswith(r):
            return r
    raise KeyError(f"no futures root for instrument {instrument!r}")


root_of = _root_of   # public: the futures root of a relative or absolute ticker


def contract_map(instrument: str, days: pd.DatetimeIndex, *, daily_root: Path = DAILY_FUTURES_DIR,
                 contracts_file: Path = FUTURES_CONTRACTS_FILE) -> pd.Series:
    """CME trading day -> the contract the instrument maps to that day (disk only)."""
    spec = parse_relative(instrument)
    if spec is None:
        return pd.Series(instrument, index=days)
    from infra.pipeline.relative_daily import load_relative_daily
    rel = load_relative_daily([spec], days.min(), days.max() + _ONE_DAY, fetch_missing=False,
                              daily_root=daily_root, contracts_file=contracts_file)
    m = rel.set_index(pd.to_datetime(rel["timestamp"]).dt.normalize())["contract"].astype(str)
    return m[~m.index.duplicated()].reindex(days)


def prices_at(quotes: pd.DataFrame, instants: pd.DatetimeIndex, contracts: pd.Series,
              tolerance: pd.Timedelta = QUOTE_TOLERANCE) -> np.ndarray:
    """Mid of ``contracts[i]`` at ``instants[i]``: last two-sided quote at or before it,
    within ``tolerance``; NaN otherwise."""
    q = quotes.dropna(subset=["bid", "ask"])
    q = q[["timestamp", "ticker", "mid"]].assign(ticker=q["ticker"].astype(str),
                                                 timestamp=pd.to_datetime(q["timestamp"]).astype("datetime64[ns]"))
    q = q.sort_values("timestamp")
    left = pd.DataFrame({"timestamp": pd.DatetimeIndex(instants).astype("datetime64[ns]"),
                         "ticker": contracts.astype(str).to_numpy(),
                         "i": np.arange(len(instants))}).dropna(subset=["timestamp"])
    left = left[left["ticker"] != "nan"].sort_values("timestamp")
    out = np.full(len(instants), np.nan)
    if left.empty or q.empty:
        return out
    m = pd.merge_asof(left, q, on="timestamp", by="ticker", tolerance=tolerance, direction="backward")
    out[m["i"].to_numpy()] = m["mid"].to_numpy()
    return out


def in_halt(instants: pd.DatetimeIndex, dataset: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """For a session crossing midnight (CME: close 16:00 CT, reopen 17:00 CT): which instants
    fall in the daily halt (close, reopen], and the halt's start (the close) for each - the
    price there is the last quote before the close (nothing trades)."""
    session = TRADING_HOURS[dataset]
    idx = pd.DatetimeIndex(instants)
    if not session.crosses_midnight:
        return np.zeros(len(idx), dtype=bool), pd.DatetimeIndex([pd.NaT] * len(idx))
    local = local_wallclock(idx, session.timezone)
    days = local.normalize()
    close = snap_instants(days, session.close_time, session.timezone)
    reopen = snap_instants(days, session.open_time, session.timezone)
    mask = np.asarray((idx > close) & (idx <= reopen))
    return mask, close


def dv01_prior(contracts: pd.Series, days: pd.Series, *, risk_root: Path) -> np.ndarray:
    """DV01 (currency per bp, per contract) of each contract on the trading day BEFORE each
    day (bmk risk store)."""
    tickers = sorted(set(contracts.dropna().astype(str)))
    if not tickers:
        return np.full(len(contracts), np.nan)
    raw = parquet_store.read_partitioned(risk_root, start=days.min() - pd.Timedelta(days=15), end=days.max() + _ONE_DAY,
                                         equals_in={"ticker": tickers})
    if raw is None or raw.empty:
        return np.full(len(contracts), np.nan)
    r = raw[raw["risk"] == "DV01"][["timestamp", "ticker", "value"]].copy()
    r["ticker"] = r["ticker"].astype(str)
    r["timestamp"] = pd.to_datetime(r["timestamp"]).astype("datetime64[ns]")
    left = pd.DataFrame({"timestamp": (pd.to_datetime(days.to_numpy()) - pd.Timedelta(seconds=1)).astype("datetime64[ns]"),
                         "ticker": contracts.astype(str).to_numpy(), "i": np.arange(len(contracts))})
    left = left.dropna(subset=["timestamp"]).sort_values("timestamp")
    m = pd.merge_asof(left, r.sort_values("timestamp"), on="timestamp", by="ticker", direction="backward",
                      tolerance=pd.Timedelta(days=10))
    out = np.full(len(contracts), np.nan)
    out[m["i"].to_numpy()] = m["value"].to_numpy()
    return out


def _futures_bbo(instruments, grid, *, bps: bool, bbo_root: Path = BBO_FUTURES_DIR,
                 daily_root: Path = DAILY_FUTURES_DIR, contracts_file: Path = FUTURES_CONTRACTS_FILE,
                 risk_root: Path | None = None) -> pd.DataFrame:
    points = grid.instants()
    prev = points[:-1].insert(0, pd.NaT)
    out = {}
    for inst in instruments:
        root = _root_of(inst)
        cfg = FUTURES_ROOTS[root]
        tday = trading_day(points, cfg.dataset)
        days = pd.DatetimeIndex(tday.unique())
        cmap = contract_map(inst, days, daily_root=daily_root, contracts_file=contracts_file)
        con = pd.Series(cmap.reindex(tday).to_numpy(), index=points)
        # after a close with no session following (Friday evening, a holiday eve) the point's
        # trading day has no contract: the market is shut until it reopens - hold the last
        # contract and its closing price
        shut = con.isna().to_numpy() & np.asarray(pd.Series(points).notna())
        con = con.ffill()
        contracts = sorted(set(con.dropna().astype(str)))
        quotes = read_bbo_from_disk(contracts, points.min() - _ONE_DAY, points.max() + _ONE_DAY, root=bbo_root) \
            if contracts else pd.DataFrame(columns=["timestamp", "ticker", "bid", "ask", "mid"])
        p1 = _price(quotes, points, con, cfg.dataset, shut)
        p0 = _price(quotes, prev, con, cfg.dataset, np.r_[False, shut[:-1]])  # the SAME contract, previous point
        step = p1 - p0
        if bps:
            dv01 = dv01_prior(con, pd.Series(tday, index=points), risk_root=risk_root or (BMK_ROOT / "Risk"))
            step = step * cfg.point_value / dv01
        out[inst] = step
    frame = pd.DataFrame(out, index=pd.DatetimeIndex(points, name="timestamp"))
    return frame


def _price(quotes, instants, contracts, dataset, shut=None) -> np.ndarray:
    """``prices_at``, except where the market is SHUT - inside the venue's daily halt, or
    after a close with no session following (``shut``: Friday evening, a holiday eve) - where
    the price is the last quote at or before that day's close. Found 2026-10-05: the
    06:00-19:00 New York grid includes CME's 17:00-18:00 ET halt and Friday's 17:00-19:00 ET,
    where no quote is fresh - 11% of ZN's grid points came out NaN."""
    idx = pd.DatetimeIndex(instants)
    p = prices_at(quotes, idx, contracts)
    mask, close = in_halt(idx, dataset)
    session = TRADING_HOURS[dataset]
    if shut is not None and session.crosses_midnight:
        local_days = local_wallclock(idx, session.timezone).normalize()
        day_close = snap_instants(local_days, session.close_time, session.timezone)
        after = np.asarray(idx >= day_close) & np.asarray(shut)
        close = pd.DatetimeIndex(np.where(after, day_close, close))
        mask = mask | after
    mask &= np.asarray(pd.Series(idx).notna())
    if mask.any():
        c = pd.Series(contracts.to_numpy() if hasattr(contracts, "to_numpy") else contracts)
        p[mask] = prices_at(quotes, close[mask], c[mask])
    return p


@dataclass(frozen=True)
class PnlSource:
    fn: Callable
    units: str
    description: str


PNL_SOURCES: dict[str, PnlSource] = {
    "FUTURE_BPS_BBO": PnlSource(lambda inst, grid, **kw: _futures_bbo(inst, grid, bps=True, **kw), "bp",
                                "bbo-1m mid changes of the mapped futures contract, bp (/ prior-day DV01)"),
    "FUTURE_PTS_BBO": PnlSource(lambda inst, grid, **kw: _futures_bbo(inst, grid, bps=False, **kw), "points",
                                "bbo-1m mid changes of the mapped futures contract, price points"),
}


def grid_pnl(source: str, instruments: list[str], grid, **kwargs) -> pd.DataFrame:
    """Step P&L of ``instruments`` on ``grid`` (``infra.processing.event_windows.Grid``)."""
    if source not in PNL_SOURCES:
        raise KeyError(f"unknown pnl source {source!r}; known: {sorted(PNL_SOURCES)}")
    out = PNL_SOURCES[source].fn(list(instruments), grid, **kwargs)
    out.attrs["source"], out.attrs["units"], out.attrs["cycle"] = source, PNL_SOURCES[source].units, grid.cycle.name
    return out
