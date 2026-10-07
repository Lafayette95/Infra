"""German Federal securities: the reference data (the Finanzagentur's issuance history) and
daily per-ISIN prices / yields (the Bundesbank's BBSSY) - read / plan / fetch / store.
Pure parsing in ``infra.processing.bunds``; clients ``infra.api.finanzagentur_client`` and
``infra.api.bundesbank_client.fetch_bbssy_csv``.

* Auctions: ``RawData/DE_Auctions`` (keys ``timestamp`` = auction day, ``isin``), the whole
  file re-read each refresh (it is one ~350KB workbook) and upserted. ``read_securities`` is
  a VIEW of it (one row per ISIN, ``infra.processing.bunds.securities``), point in time with
  ``as_of`` (ISINs first auctioned by then).
* Prices: ``Daily/BundPrices`` (keys ``timestamp``, ``isin``; x10000 nullable Int32),
  coverage ``Daily/_coverage/bund_prices.parquet`` (key ``BBSSY``). A wildcard request returns
  every CURRENTLY listed security, so the daily update is one request; a matured bond is
  fetched by its ISIN (``backfill_isins``) - the Bundesbank drops it about four years after
  maturity, so the archive here is the record beyond that.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from infra.api import bundesbank_client, finanzagentur_client
from infra.config import (BUND_PRICES_SETTLE_DAYS, DAILY_BUND_PRICES_COVERAGE_FILE, DAILY_BUND_PRICES_DIR,
                          DE_AUCTIONS_DIR, DE_OTR_DEPTH, DE_OTR_TENORS)
from infra.coverage.intervals import find_missing_ranges
from infra.processing import bunds as pb
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)
KEY = "BBSSY"
AUCTION_KEYS = ["timestamp", "isin"]
PRICE_KEYS = ["timestamp", "isin"]
PRICES_START = "1994-01-03"
FETCH_ISSUANCE = finanzagentur_client.fetch_issuance_history   # network hooks; tests stub them
FETCH_PRICES = bundesbank_client.fetch_bbssy_csv
PAUSE_S = 0.2
_ONE_DAY = pd.Timedelta(days=1)


# ------------------------------------------------------------------ reference
def update_auctions(*, root: Path = DE_AUCTIONS_DIR) -> int:
    """Fetch the issuance history and upsert every row; returns the rows stored."""
    body = FETCH_ISSUANCE()
    if body is None:
        return 0
    df = pb.parse_issuance_history(body)
    if df.empty:
        return 0
    out = df.assign(timestamp=df["timestamp"].astype("datetime64[ms]"),
                    maturity_date=df["maturity_date"].astype("datetime64[ms]"))
    parquet_store.write_partitioned(out, root, AUCTION_KEYS)
    return len(out)


def read_auctions(start=None, end=None, *, root: Path = DE_AUCTIONS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end))
    if df is None or df.empty:
        return pd.DataFrame(columns=pb.AUCTION_COLUMNS)
    for c in ("isin", "type", "segment", "issue_kind", "process"):
        df[c] = df[c].astype(str)
    return df[pb.AUCTION_COLUMNS].sort_values(AUCTION_KEYS).reset_index(drop=True)


def read_securities(as_of=None, *, root: Path = DE_AUCTIONS_DIR) -> pd.DataFrame:
    """One row per ISIN first auctioned by ``as_of`` (all if None)."""
    a = read_auctions(None, None if as_of is None else pd.Timestamp(as_of).normalize() + _ONE_DAY, root=root)
    return pb.securities(a)


def otr_map(start, end, *, conventions=("issue",), depth: int = DE_OTR_DEPTH,
            root: Path = DE_AUCTIONS_DIR) -> pd.DataFrame:
    """The on/off-the-run map for business days in ``[start, end]``: per tenor
    (``DE_OTR_TENORS``) the conventional issues of that maturity segment ranked newest first
    (0 = on the run), dropped once matured - the US map's own function
    (``infra.processing.treasury_otr.otr_map``) on the Bund table, ISINs in its ``cusip``
    column. Computed, not stored (a view of the issuance history, ~a second). Point in time:
    only securities first auctioned by ``end``, and each day counts an issue from its issue
    day ("issue") or first auction ("auction")."""
    from infra.processing.treasury_otr import otr_map as _otr_map
    days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    sec = read_securities(pd.Timestamp(end), root=root)
    if days.empty or sec.empty:
        return pd.DataFrame()
    frames = []
    for tenor, (types, segment) in DE_OTR_TENORS.items():
        s = sec[sec["type"].isin(types) & (sec["segment"] == segment)]
        s = s.assign(security_type="X", original_term=tenor, cusip=s["isin"], auction_date=s["first_auction"])
        frames.append(_otr_map(s, days, {tenor: ("X", tenor)}, depth, conventions))
    return pd.concat([f for f in frames if len(f)], ignore_index=True).rename(columns={"cusip": "isin"})


def otr_yields(start, end, *, convention: str = "issue", prices_root: Path = DAILY_BUND_PRICES_DIR,
               root: Path = DE_AUCTIONS_DIR) -> pd.DataFrame:
    """``timestamp, ticker (DE_BOND_<t>y), isin, yield, price_clean`` - each day's on-the-run
    bond's Bundesbank yield (an 11:15 Frankfurt snapshot, ``BUND_PRICES_LOCAL_TIME``). A day
    whose on-the-run bond has no price gets no row."""
    m = otr_map(start, end, conventions=(convention,), depth=0, root=root)
    if m.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "isin", "yield", "price_clean"])
    px = read_bund_prices(start, pd.Timestamp(end) + _ONE_DAY, isins=sorted(set(m["isin"])), root=prices_root)
    out = m[["timestamp", "tenor", "isin"]].merge(px[["timestamp", "isin", "yield", "price_clean"]],
                                                    on=["timestamp", "isin"], how="inner").dropna(subset=["yield"])
    out = out.assign(ticker="DE_BOND_" + out["tenor"])
    return out[["timestamp", "ticker", "isin", "yield", "price_clean"]].sort_values(["timestamp", "ticker"]).reset_index(drop=True)


# ------------------------------------------------------------------ prices
def plan_prices(start, end, *, now=None, coverage_file: Path = DAILY_BUND_PRICES_COVERAGE_FILE,
                force_refetch: bool = False) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Uncovered ``[s, e)`` ranges in ``[start, end]``, never past the last settled day."""
    now = pd.Timestamp.now().normalize() if now is None else pd.Timestamp(now).normalize()
    lo = max(pd.Timestamp(start).normalize(), pd.Timestamp(PRICES_START))
    hi = min(pd.Timestamp(end).normalize() + _ONE_DAY, now - pd.Timedelta(days=BUND_PRICES_SETTLE_DAYS - 1))
    if hi <= lo:
        return []
    if force_refetch:
        return [(lo, hi)]
    return find_missing_ranges((lo, hi), coverage_store.read_covered(coverage_file, KEY))


def _store(df: pd.DataFrame, root: Path) -> int:
    if df.empty:
        return 0
    parquet_store.write_partitioned(pb.encode_prices(df), root, PRICE_KEYS)
    return len(df)


def fetch_and_store_prices(ranges, *, root: Path = DAILY_BUND_PRICES_DIR,
                           coverage_file: Path = DAILY_BUND_PRICES_COVERAGE_FILE) -> int:
    """One wildcard request per range (every currently listed security); coverage recorded
    for the range once stored."""
    n = 0
    for s, e in ranges:
        df = pb.parse_bbssy_csv(FETCH_PRICES(s, e - _ONE_DAY))
        n += _store(df, root)
        coverage_store.record_covered(coverage_file, KEY, [(s, e)])
    return n


def backfill_isins(isins, start=PRICES_START, end=None, *, root: Path = DAILY_BUND_PRICES_DIR, sleep=time.sleep) -> dict:
    """Whole history of each ISIN, one request each (matured bonds - absent from the wildcard
    answer). Coverage is not touched: it tracks the wildcard update. Returns per-ISIN rows."""
    end = pd.Timestamp.now().normalize() if end is None else pd.Timestamp(end)
    out = {}
    for isin in isins:
        out[isin] = _store(pb.parse_bbssy_csv(FETCH_PRICES(pd.Timestamp(start), end, isin=isin)), root)
        sleep(PAUSE_S)
    return out


def read_bund_prices(start, end, *, isins=None, root: Path = DAILY_BUND_PRICES_DIR) -> pd.DataFrame:
    """Rows with ``timestamp`` in ``[start, end)``."""
    df = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end),
                                        equals_in={"isin": list(isins)} if isins is not None else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=pb.PRICE_COLUMNS)
    return pb.decode_prices(df)[pb.PRICE_COLUMNS].sort_values(PRICE_KEYS).reset_index(drop=True)
