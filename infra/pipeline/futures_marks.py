"""Marks for ABSOLUTE futures contracts at given instants (disk only): what a position-keeping
layer needs to value and trade a contract at a decision time (root CLAUDE.md 27).

* ``quote_marks``: per (instant, contract) the LAST bbo-1m sample at or before the instant -
  ``bid``, ``ask`` (either may be missing: a one-sided book), sizes, ``quote_time`` (its age
  is the caller's freshness test) - plus ``halt`` (the instant is inside the venue's daily
  halt, CME 16:00-17:00 CT) and ``mark`` = the last TWO-SIDED mid at or before the instant
  within ``MARK_MAX_AGE`` (prices carry through a halt, a weekend, a holiday; never beyond).
* ``settlement_marks``: per (instant, contract) the execution TRADING DAY - the instant's own
  trading day if the instant is at or before that day's settlement, else the next day with a
  settlement (no look-ahead: a decision after 14:00 CT can't trade that day's settlement) -
  its ``settlement`` and the NY1500 snap's ``bid`` / ``ask`` that day (the settlement window's
  book, ``Derived/FuturesSnaps``), for the cost.
* ``contract_calendar``: point value, last trade and (CME Treasury roots) first notice day.

Long frames keyed ``timestamp, contract``; prices float in the contract's own units."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.config import BBO_FUTURES_DIR, DAILY_FUTURES_DIR, FUTURES_CONTRACTS_FILE, FUTURES_ROOTS
from infra.pipeline.bbo import read_bbo_from_disk
from infra.pipeline.event_pnl import in_halt, root_of
from infra.trading_calendar import snap_instants, trading_day

MARK_MAX_AGE = pd.Timedelta(days=5)
# The daily settlement instant per dataset (local time, zone). CME rates (Treasury and SOFR
# futures) settle on the 13:59-14:00 CT window (event_grid CME_TSY_SETTLE).
SETTLEMENT_TIMES = {"GLBX.MDP3": ("14:00", "America/Chicago")}
SETTLEMENT_SNAP = "NY1500"
CME_TREASURY_ROOTS = ("ZT", "ZF", "ZN", "TN", "ZB", "UB")
QUOTE_COLUMNS = ["timestamp", "contract", "bid", "ask", "bid_size", "ask_size", "quote_time", "mark", "halt"]
SETTLE_COLUMNS = ["timestamp", "contract", "exec_day", "settlement", "bid", "ask"]
_DAY = pd.Timedelta(days=1)


def _grid(instants: pd.DatetimeIndex, cons: list[str]) -> pd.DataFrame:
    inst = pd.DatetimeIndex(instants).astype("datetime64[ns]")
    return pd.DataFrame({"timestamp": np.repeat(inst.to_numpy(), len(cons)),
                         "contract": np.tile(np.array(cons, dtype=object), len(inst))})


def quote_marks(contracts, instants, *, bbo_root: Path = BBO_FUTURES_DIR, quotes: pd.DataFrame | None = None
                ) -> pd.DataFrame:
    """See the module docstring. ``quotes`` (decoded bbo-1m, ``ticker`` column) may be passed in."""
    instants = pd.DatetimeIndex(instants)
    cons = sorted(set(map(str, contracts)))
    left = _grid(instants, cons)
    if left.empty:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    if quotes is None:
        quotes = read_bbo_from_disk(cons, instants.min() - MARK_MAX_AGE, instants.max() + _DAY, root=bbo_root)
    q = quotes.rename(columns={"ticker": "contract"})
    q = q.assign(contract=q["contract"].astype(str),
                 timestamp=pd.to_datetime(q["timestamp"]).astype("datetime64[ns]")).sort_values("timestamp")
    last = q[["timestamp", "contract", "bid", "ask", "bid_size", "ask_size"]].assign(quote_time=q["timestamp"])
    two = q.dropna(subset=["bid", "ask"])
    two = pd.DataFrame({"timestamp": two["timestamp"], "contract": two["contract"],
                        "mark": (two["bid"] + two["ask"]) / 2})
    left = left.sort_values("timestamp")
    out = pd.merge_asof(left, last, on="timestamp", by="contract", direction="backward")
    out = pd.merge_asof(out, two, on="timestamp", by="contract", direction="backward", tolerance=MARK_MAX_AGE)
    out["halt"] = False
    for c in cons:
        m = (out["contract"] == c).to_numpy()
        out.loc[m, "halt"] = in_halt(pd.DatetimeIndex(out.loc[m, "timestamp"]), FUTURES_ROOTS[root_of(c)].dataset)[0]
    return out[QUOTE_COLUMNS].sort_values(["timestamp", "contract"]).reset_index(drop=True)


def execution_days(instants, dataset: str, settle_days) -> pd.DatetimeIndex:
    """The trading day whose settlement an order decided at each instant trades: the
    instant's trading day if at or before its settlement time and that day settles, else the
    next settled day (NaT if none stored yet)."""
    idx = pd.DatetimeIndex(instants)
    tday = pd.DatetimeIndex(trading_day(idx, dataset)).normalize()
    if dataset in SETTLEMENT_TIMES:
        local_time, tz = SETTLEMENT_TIMES[dataset]
        cutoff = snap_instants(tday, local_time, tz)
        tday = tday + pd.to_timedelta(np.where(np.asarray(idx > cutoff), 1, 0), unit="D")
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(settle_days).normalize())))
    pos = days.searchsorted(tday, side="left")
    ok = pos < len(days)
    out = np.full(len(idx), np.datetime64("NaT", "ns"))
    out[ok] = days.to_numpy()[pos[ok]]
    return pd.DatetimeIndex(out)


def settlement_marks(contracts, instants, *, daily_root: Path = DAILY_FUTURES_DIR, snaps: pd.DataFrame | None = None,
                     settlements: pd.DataFrame | None = None) -> pd.DataFrame:
    """See the module docstring. ``settlements`` (``timestamp, ticker, settlement_price``) and
    ``snaps`` (``timestamp, ticker, bid, ask`` at the settlement snap) may be passed in."""
    instants = pd.DatetimeIndex(instants)
    cons = sorted(set(map(str, contracts)))
    if not cons or instants.empty:
        return pd.DataFrame(columns=SETTLE_COLUMNS)
    lo, hi = instants.min() - 10 * _DAY, instants.max() + 10 * _DAY
    if settlements is None:
        from infra.pipeline.daily import read_daily_from_disk
        settlements = read_daily_from_disk(cons, lo, hi, root=daily_root)
    if snaps is None:
        from infra.pipeline.futures_snaps import read_futures_snaps
        snaps = read_futures_snaps(lo, hi, snap=SETTLEMENT_SNAP, tickers=cons)
    s = settlements.rename(columns={"ticker": "contract"}).dropna(subset=["settlement_price"])
    s = s.assign(contract=s["contract"].astype(str), day=pd.to_datetime(s["timestamp"]).dt.normalize())
    sn = snaps.rename(columns={"ticker": "contract"})
    sn = sn.assign(contract=sn["contract"].astype(str))
    parts = []
    for c in cons:
        ds = s[s["contract"] == c].drop_duplicates("day", keep="last").set_index("day")
        dataset = FUTURES_ROOTS[root_of(c)].dataset
        ex = execution_days(instants, dataset, ds.index)
        snc = sn[sn["contract"] == c]
        bid = ask = np.full(len(ex), np.nan)
        if len(snc):
            snc = snc.assign(day=pd.DatetimeIndex(trading_day(pd.DatetimeIndex(snc["timestamp"]), dataset)).normalize())
            snc = snc.drop_duplicates("day", keep="last").set_index("day")
            bid = snc["bid"].reindex(ex).to_numpy(dtype="float64")
            ask = snc["ask"].reindex(ex).to_numpy(dtype="float64")
        parts.append(pd.DataFrame({"timestamp": instants, "contract": c, "exec_day": ex,
                                   "settlement": ds["settlement_price"].reindex(ex).to_numpy(dtype="float64"),
                                   "bid": bid, "ask": ask}))
    return pd.concat(parts, ignore_index=True)[SETTLE_COLUMNS].sort_values(["timestamp", "contract"]).reset_index(
        drop=True)


def contract_calendar(contracts, *, contracts_file: Path = FUTURES_CONTRACTS_FILE,
                      table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per contract (index): ``root``, ``point_value``, ``last_trade`` (the contracts table's
    expiry) and ``first_notice`` (CME Treasury roots: the business day before the first
    delivery day, CME's rule on the market calendar; NaT elsewhere)."""
    from infra.analytics.futures_basis import delivery_window
    from infra.processing.schedule_rules import business_days
    cons = sorted(set(map(str, contracts)))
    if table is None:
        table = pd.read_parquet(contracts_file) if Path(contracts_file).exists() else pd.DataFrame(
            columns=["ticker", "expiry"])
    table = table.assign(ticker=table["ticker"].astype(str)).drop_duplicates("ticker", keep="last").set_index("ticker")
    rows, bd = [], None
    for c in cons:
        root = root_of(c)
        exp = pd.Timestamp(table.at[c, "expiry"]).normalize() if c in table.index and pd.notna(
            table.at[c, "expiry"]) else pd.NaT
        fn = pd.NaT
        if root in CME_TREASURY_ROOTS and pd.notna(exp):
            bd = business_days("2000-01-01", "2045-12-31", "market") if bd is None else bd
            fn = bd[bd < delivery_window(root, exp, bd)["first_delivery"]][-1]
        rows.append({"contract": c, "root": root, "point_value": FUTURES_ROOTS[root].point_value,
                     "last_trade": exp, "first_notice": fn})
    return pd.DataFrame(rows).set_index("contract")
