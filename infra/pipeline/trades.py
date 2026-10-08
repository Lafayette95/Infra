"""Tick trades (aggressor side) for ABSOLUTE futures contracts: plan, fetch, store, read;
and the 1-minute signed-volume bars built from them locally. Disk first (Rule 2.1).

Stores: ``~/Database/trades/Futures`` (every fill, ``infra.processing.trades``; keys
``timestamp`` (ns), ``ticker``, ``sequence``, ``fill``; coverage in UTC days per contract)
and ``~/Database/trades-1m/Futures`` (``signed_bars``, keys ``timestamp``, ``ticker``),
both hive year/quarter, flat, ZSTD 5. Fetched in calendar-month pieces, each priced
against ``max_cost_usd`` first (one contract-quarter of ZN is ~$8, over the $5 cap).
Never fetched by a cycle (yet): ``scripts/update_trades.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    MAX_COST_USD,
    SCHEMA_TRADES,
    SIGNED_1M_FUTURES_DIR,
    TRADES_FUTURES_COVERAGE_FILE,
    TRADES_FUTURES_DIR,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.pipeline.daily import bounded_by_availability
from infra.processing import trades as tr
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)

SIGNED_KEYS = ["timestamp", "ticker"]


def plan_trades_update(ticker: str, start, end, *,
                       coverage_file: Path = TRADES_FUTURES_COVERAGE_FILE) -> list[Interval]:
    """UTC days of ``[start, end)`` never queried for ``ticker``. No API."""
    requested = (to_utc_day(start), to_utc_day(end))
    return find_missing_ranges(requested, coverage_store.read_covered(coverage_file, ticker))


def month_pieces(gaps: list[Interval]) -> list[Interval]:
    """Each gap cut at calendar-month boundaries (one request per piece)."""
    out = []
    for a, b in gaps:
        cuts = [a] + [m for m in pd.date_range(a, b, freq="MS") if a < m < b] + [b]
        out += [(x, y) for x, y in zip(cuts[:-1], cuts[1:]) if x < y]
    return out


def fetch_trades_raw(ticker: str, pieces: list[Interval], *, dataset: str,
                     max_cost_usd: float = MAX_COST_USD, client=None) -> list[tuple[Interval, pd.Timestamp, pd.DataFrame]]:
    """NETWORK ONLY: one request per piece, each bounded by the schema's availability."""
    out = []
    for a, b in pieces:
        query_end, covered_end = bounded_by_availability(dataset, b, client, schema=SCHEMA_TRADES)
        if query_end <= a:
            continue
        try:
            raw = api.fetch_futures_trades(dataset, ticker, a, query_end, max_cost_usd=max_cost_usd, client=client)
        except Exception as e:
            if "could be resolved" not in str(e):
                raise
            raw = pd.DataFrame()    # the contract wasn't listed then: no trades, covered
        out.append(((a, b), covered_end, raw))
    return out


def store_trades_raw(ticker: str, fetched, *, root: Path = TRADES_FUTURES_DIR,
                     coverage_file: Path = TRADES_FUTURES_COVERAGE_FILE) -> int:
    """FILES ONLY: clean, save, record coverage (never past the last complete day)."""
    rows = 0
    for (a, b), covered_end, raw in fetched:
        clean = tr.clean_trades(raw, ticker)
        if not clean.empty:
            parquet_store.write_partitioned(tr.encode_trades(clean), root, tr.TRADE_KEYS)
        if covered_end > a:
            coverage_store.record_covered(coverage_file, ticker, [(a, covered_end)])
        rows += len(clean)
        log.info("%s %s->%s: %d trades saved", ticker, a.date(), b.date(), len(clean))
    return rows


def read_trades(tickers: list[str], start, end, *, root: Path = TRADES_FUTURES_DIR) -> pd.DataFrame:
    """Decoded trades of ``tickers`` with ``start <= timestamp < end``. No API."""
    raw = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end),
                                         equals_in={"ticker": list(tickers)})
    if raw is None or raw.empty:
        return tr.empty_trades()
    return tr.decode_trades(raw).sort_values(tr.TRADE_KEYS).reset_index(drop=True)


def load_trades(tickers: list[str], start, end, *, dataset: str, fetch_missing: bool = True,
                root: Path = TRADES_FUTURES_DIR, coverage_file: Path = TRADES_FUTURES_COVERAGE_FILE,
                max_cost_usd: float = MAX_COST_USD, client=None, read: bool = True) -> pd.DataFrame | None:
    """Parent: fetch what isn't covered (monthly pieces), then read (``read=False`` skips the
    read - a backfill of millions of rows needn't load them)."""
    if fetch_missing:
        for t in tickers:
            gaps = plan_trades_update(t, start, end, coverage_file=coverage_file)
            if gaps:
                fetched = fetch_trades_raw(t, month_pieces(gaps), dataset=dataset,
                                           max_cost_usd=max_cost_usd, client=client)
                store_trades_raw(t, fetched, root=root, coverage_file=coverage_file)
    return read_trades(tickers, start, end, root=root) if read else None


def build_signed_bars(tickers: list[str], start, end, *, freq: str = "1min", trades_root: Path = TRADES_FUTURES_DIR,
                      root: Path = SIGNED_1M_FUTURES_DIR) -> int:
    """Aggregate the stored trades into signed-volume bars and store them (upsert by bar),
    month by month to bound memory. Reads disk only. Returns bars written."""
    n = 0
    for a, b in month_pieces([(to_utc_day(start), to_utc_day(end))]):
        bars = tr.signed_bars(read_trades(tickers, a, b, root=trades_root), freq)
        if not bars.empty:
            parquet_store.write_partitioned(bars, root, SIGNED_KEYS)
            n += len(bars)
    return n


def read_signed_bars(tickers: list[str], start, end, *, root: Path = SIGNED_1M_FUTURES_DIR) -> pd.DataFrame:
    raw = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end),
                                         equals_in={"ticker": list(tickers)})
    if raw is None or raw.empty:
        return tr.signed_bars(tr.empty_trades())
    raw["ticker"] = raw["ticker"].astype(str)
    return raw.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
