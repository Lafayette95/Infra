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

from infra.config import (BBO_FUTURES_DIR, BMK_ROOT, BMK_YIELD_MAX_GAP_DAYS, DAILY_FUTURES_DIR, FUTURES_CONTRACTS_FILE, FUTURES_ROOTS,
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
    frequency: str = "intraday"     # the cycles it serves: intraday | daily


def _structures_bbo(structures, grid, *, structure_set: str = "ust_layers", state=None, **kw) -> pd.DataFrame:
    """Step P&L of curve STRUCTURES (``infra.reference.structures``): the legs' FUTURE_BPS_BBO
    steps x the structure's DV01 weights for the step's CME trading day (point in time: hedge
    betas fitted on settlements before that day). bp per unit of structure."""
    from infra.pipeline.structures import structure_state
    from infra.reference.structures import STRUCTURE_SETS
    sset = STRUCTURE_SETS[structure_set]
    legs = sset.legs()
    steps = _futures_bbo(legs, grid, bps=True, **kw)
    points = steps.index
    tday = pd.DatetimeIndex(trading_day(points, "GLBX.MDP3")).normalize()
    if state is None:
        state = structure_state(sset, tday.min(), tday.max(), with_dv01=False)
    out = {s: np.full(len(points), np.nan) for s in structures}
    for d in tday.unique():
        rows = np.flatnonzero(tday == d)
        if d not in state.weights:
            continue
        W = state.weights[d]
        for s in structures:
            out[s][rows] = steps.iloc[rows][legs].to_numpy() @ W.loc[s, legs].to_numpy()
    return pd.DataFrame(out, index=points)


def _on_daily_grid(by_day: pd.DataFrame, grid, *, carried_days: int | None = None) -> pd.DataFrame:
    """A trading-day x instrument frame -> the grid's points (one per day): day D's move at D's
    point. A grid day without a row is NaN - unless ``carried_days``: a source whose next row carries
    a skipped day's move (the yield bmks diff against the last PRICED day, so a bond-market holiday
    the market calendar trades - Columbus, Veterans Day - has no row and its move lands on the next
    one) gets 0 there when its next row is at most that many calendar days later (a longer gap is
    a real hole: the bmk stores no change across more than ``BMK_YIELD_MAX_GAP_DAYS``)."""
    points = grid.instants()
    days = pd.DatetimeIndex(grid.days)
    out = by_day.copy()
    out.index = pd.DatetimeIndex(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].reindex(days)
    if carried_days is not None:
        for c in out.columns:
            have = out[c].notna()
            nxt = pd.Series(np.where(have, days, pd.NaT), index=days, dtype="datetime64[ns]").bfill()
            fill = ~have & ((nxt - days.to_series(index=days)) <= pd.Timedelta(days=carried_days))
            out.loc[fill, c] = 0.0
    out.index = points
    return out


def _yield_settle(instruments, grid, *, source: str, bmk_root: Path | None = None, **kw) -> pd.DataFrame:
    """Daily yield P&L in bp of a long position (``-dy``), the persisted bmk ``yield_<source>``
    rows (infra.cycle.bmk_yields): instruments are ``US_BOND_<t>y`` tickers or yield STRUCTURES on
    them (``CURVE__US_BOND_5y__US_BOND_30y``: ``infra.reference.structures.yield_structure_weights``)."""
    from infra.reference.structures import yield_structure_weights
    weights = {i: (yield_structure_weights(i) if "__" in i else {i: 1.0}) for i in instruments}
    tickers = sorted({t for w in weights.values() for t in w})
    days = pd.DatetimeIndex(grid.days)
    raw = parquet_store.read_partitioned((bmk_root or BMK_ROOT) / "Pnl", start=days.min(), end=days.max() + _ONE_DAY,
                                         equals_in={"bmk": [f"yield_{source}"], "ticker": tickers})
    if raw is None or raw.empty:
        return pd.DataFrame(np.nan, index=grid.instants(), columns=list(instruments))
    raw = raw.assign(ticker=raw["ticker"].astype(str))
    legs = raw.pivot_table(index="timestamp", columns="ticker", values="pnl_per_dv01").reindex(columns=tickers)
    # a structure = its legs' bp x DV01 weights; NaN if any leg is missing that day
    wide = pd.DataFrame({i: legs[list(w)].mul(pd.Series(w)).sum(axis=1, min_count=len(w)) for i, w in weights.items()})
    return _on_daily_grid(wide, grid, carried_days=BMK_YIELD_MAX_GAP_DAYS)


def _series(instruments, grid, **kw) -> pd.DataFrame:
    """Daily P&L of ANY stored daily-MOVES series, by its ``infra.pipeline.series_panel`` id - e.g.
    ``bmk:yield_cmt@LDN1615:US_BOND_10y``, ``bmk:yield_boe@LDN1615:CURVE__UK_BOND_5y__UK_BOND_30y``,
    ``bmk:swsp_cmt:US_SWSP_10y``, ``bmk:fut@LDN1615:FUT_ZN``: one reader for every series the other
    models use, so a study names exactly where each instrument comes from. Level series are refused
    (an event window sums daily moves). A missing grid day counts 0 when the next row carries its
    move (the bmk P&L diffs against the last priced day), as for the yield sources."""
    from infra.pipeline.series_panel import kind_of, read_panel
    bad = [i for i in instruments if kind_of(i) != "moves"]
    if bad:
        raise ValueError(f"SERIES_BPS needs daily-moves series (bp P&L), not levels: {bad}")
    days = pd.DatetimeIndex(grid.days)
    wide = read_panel(list(instruments), days.min(), days.max())
    return _on_daily_grid(wide.reindex(columns=list(instruments)), grid, carried_days=BMK_YIELD_MAX_GAP_DAYS)


def _futures_settle(instruments, grid, **kw) -> pd.DataFrame:
    """Daily settlement-to-settlement bp of relative futures (the contract held that day, prior-day
    DV01: ``infra.pipeline.structures.daily_bp_moves``)."""
    from infra.pipeline.structures import daily_bp_moves
    days = pd.DatetimeIndex(grid.days)
    return _on_daily_grid(daily_bp_moves(list(instruments), days.min() - pd.Timedelta(days=10), days.max()), grid)


def _structures_settle(structures, grid, *, structure_set: str = "ust_layers", **kw) -> pd.DataFrame:
    """Daily bp of curve STRUCTURES: the legs' settlement bp x the day's point-in-time weights."""
    from infra.pipeline.structures import structure_state
    days = pd.DatetimeIndex(grid.days)
    st = structure_state(structure_set, days.min(), days.max(), with_dv01=False)
    return _on_daily_grid(st.moves.reindex(columns=list(structures)), grid)


PNL_SOURCES: dict[str, PnlSource] = {
    "FUTURE_BPS_BBO": PnlSource(lambda inst, grid, **kw: _futures_bbo(inst, grid, bps=True, **kw), "bp",
                                "bbo-1m mid changes of the mapped futures contract, bp (/ prior-day DV01)"),
    "FUTURE_PTS_BBO": PnlSource(lambda inst, grid, **kw: _futures_bbo(inst, grid, bps=False, **kw), "points",
                                "bbo-1m mid changes of the mapped futures contract, price points"),
    "STRUCT_BPS_BBO": PnlSource(lambda inst, grid, **kw: _structures_bbo(inst, grid, **kw), "bp",
                                "curve structures (infra.reference.structures): legs' bp steps x point-in-time "
                                "DV01 weights, bp per unit of structure"),
    # DAILY sources (cycle DAILY_SETTLE; root CLAUDE.md 29): bp, + = a long position made money
    "YIELD_BPS_CMT": PnlSource(lambda inst, grid, **kw: _yield_settle(inst, grid, source="cmt", **kw), "bp",
                               "daily -dy of the CMT par yield (bmk yield_cmt)", "daily"),
    "YIELD_BPS_OTR": PnlSource(lambda inst, grid, **kw: _yield_settle(inst, grid, source="otr", **kw), "bp",
                               "daily -dy of the on-the-run bond held the previous day (bmk yield_otr)", "daily"),
    "YIELD_BPS_CURVE": PnlSource(lambda inst, grid, **kw: _yield_settle(inst, grid, source="curve", **kw), "bp",
                                 "daily -dy of our fitted curve's par yield (bmk yield_curve)", "daily"),
    "SERIES_BPS": PnlSource(lambda inst, grid, **kw: _series(inst, grid, **kw), "bp",
                            "any stored daily-moves series by its series_panel id (bmk P&L of any bmk, incl. the "
                            "synchronized 16:15 London ones)", "daily"),
    "FUTURE_BPS_SETTLE": PnlSource(lambda inst, grid, **kw: _futures_settle(inst, grid, **kw), "bp",
                                   "settlement-to-settlement bp of the mapped futures contract", "daily"),
    "STRUCT_BPS_SETTLE": PnlSource(lambda inst, grid, **kw: _structures_settle(inst, grid, **kw), "bp",
                                   "settlement-to-settlement bp of curve structures", "daily"),
}


def grid_pnl(source: str, instruments: list[str], grid, **kwargs) -> pd.DataFrame:
    """Step P&L of ``instruments`` on ``grid`` (``infra.processing.event_windows.Grid``)."""
    if source not in PNL_SOURCES:
        raise KeyError(f"unknown pnl source {source!r}; known: {sorted(PNL_SOURCES)}")
    out = PNL_SOURCES[source].fn(list(instruments), grid, **kwargs)
    out.attrs["source"], out.attrs["units"], out.attrs["cycle"] = source, PNL_SOURCES[source].units, grid.cycle.name
    return out
