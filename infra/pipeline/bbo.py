"""Top-of-book QUOTE pipeline for ABSOLUTE futures contracts (raw symbols), Databento
``bbo-1m`` (default) or ``bbo-1s``: disk first -> gaps -> API -> save -> read.

Same shape as infra.pipeline.futures (read / plan / fetch_and_store + a ``load`` parent),
the same storage primitives, a separate store per schema (``~/Database/bbo-1m``,
``~/Database/bbo-1s``) with its own coverage manifest. ``bbo-1m`` feeds intraday WIRP
(the intraday cycle); ``bbo-1s`` is ON DEMAND only, for small event windows - use
``load_bbo_1s``.

A sample is known AT its own timestamp (``ts_recv`` on the exact minute/second - the
book at that instant), unlike an OHLCV bar, stamped at its START and only complete one
bar later (see ``infra.pipeline.wirp`` for how each is read point-in-time).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    BBO_1S_FUTURES_COVERAGE_FILE,
    BBO_1S_FUTURES_DIR,
    BBO_FUTURES_COVERAGE_FILE,
    BBO_FUTURES_DIR,
    MAX_COST_USD,
    SCHEMA_BBO_1M,
    SCHEMA_BBO_1S,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.pipeline.daily import bounded_by_availability
from infra.processing import transforms as tf
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)


def read_bbo_from_disk(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path = BBO_FUTURES_DIR,
) -> pd.DataFrame:
    """Decoded quotes (float bid/ask + ``mid``) in ``[start, end)``. No API."""
    raw = parquet_store.read_partitioned(root, start=start, end=end, equals_in={"ticker": tickers})
    if raw is None or raw.empty:
        return tf.decode_futures_bbo(tf.encode_futures_bbo(tf.clean_futures_bbo(pd.DataFrame())))
    return tf.decode_futures_bbo(raw[tf.BBO_COLUMNS]).sort_values(tf.BBO_KEYS).reset_index(drop=True)


def plan_bbo_update(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    coverage_file: Path = BBO_FUTURES_COVERAGE_FILE,
    whole_days: bool = True,
) -> list[Interval]:
    """Ranges of ``[start, end)`` never queried before. Touches no API. ``whole_days``
    (default, the 1-minute store) plans in UTC days; ``False`` plans the EXACT window -
    the 1-second stores, where paying for a whole day to study one hour would be waste
    (coverage intervals work at any resolution)."""
    requested = (to_utc_day(start), to_utc_day(end)) if whole_days else (pd.Timestamp(start), pd.Timestamp(end))
    return find_missing_ranges(requested, coverage_store.read_covered(coverage_file, ticker))


# (range_start, query_end, covered_end, raw records) per queried range
Fetched = list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DataFrame]]


def fetch_bbo_raw(
    ticker: str,
    ranges: list[Interval],
    *,
    dataset: str,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    schema: str = SCHEMA_BBO_1M,
) -> Fetched:
    """NETWORK ONLY: the API's records for ``ranges`` (each clamped to the schema's
    queryable end); touches no files, so callers may run many concurrently."""
    out: Fetched = []
    for range_start, range_end in ranges:
        query_end, covered_end = bounded_by_availability(dataset, range_end, client, schema=schema)
        if query_end <= range_start:
            continue  # nothing queryable yet - not covered, retried later
        raw = api.fetch_futures_bbo(dataset, ticker, range_start, query_end, max_cost_usd=max_cost_usd,
                                    client=client, schema=schema)
        out.append((range_start, query_end, covered_end, raw))
    return out


def store_bbo_raw(
    ticker: str,
    fetched: Fetched,
    *,
    root: Path = BBO_FUTURES_DIR,
    coverage_file: Path = BBO_FUTURES_COVERAGE_FILE,
    schema: str = SCHEMA_BBO_1M,
) -> int:
    """FILES ONLY: clean, save and record coverage (only fully published days) for what
    ``fetch_bbo_raw`` returned. Never run two concurrently."""
    rows = 0
    for range_start, query_end, covered_end, raw in fetched:
        clean = tf.clean_futures_bbo(raw)
        parquet_store.write_partitioned(tf.encode_futures_bbo(clean), root, tf.BBO_KEYS)
        if covered_end > range_start:
            coverage_store.record_covered(coverage_file, ticker, [(range_start, covered_end)])
        rows += len(clean)
        log.info("%s %s %s->%s: %d rows saved", schema, ticker, range_start.date(), query_end, len(clean))
    return rows


def fetch_and_store_bbo(
    ticker: str,
    ranges: list[Interval],
    *,
    dataset: str,
    root: Path = BBO_FUTURES_DIR,
    coverage_file: Path = BBO_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    schema: str = SCHEMA_BBO_1M,
) -> int:
    """``store_bbo_raw(fetch_bbo_raw(...))`` - query, save, record coverage. Rows saved."""
    fetched = fetch_bbo_raw(ticker, ranges, dataset=dataset, max_cost_usd=max_cost_usd, client=client, schema=schema)
    return store_bbo_raw(ticker, fetched, root=root, coverage_file=coverage_file, schema=schema)


def load_bbo(
    tickers: list[str],
    start,
    end,
    *,
    dataset: str | None = None,
    fetch_missing: bool = True,
    root: Path = BBO_FUTURES_DIR,
    coverage_file: Path = BBO_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    schema: str = SCHEMA_BBO_1M,
    whole_days: bool = True,
) -> pd.DataFrame:
    """Parent: ensure ``[start, end)`` is on disk (API only for gaps), then read it.
    ``whole_days``: see ``plan_bbo_update`` (days for 1-minute, exact for 1-second)."""
    if fetch_missing and dataset is None:
        raise ValueError("dataset is required when fetch_missing=True")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if fetch_missing:
        if whole_days:
            plan_start = to_utc_day(start)
            plan_end = to_utc_day(end) + (pd.Timedelta(days=1) if end > to_utc_day(end) else pd.Timedelta(0))
        else:
            plan_start, plan_end = start, end
        for ticker in tickers:
            gaps = plan_bbo_update(ticker, plan_start, plan_end, coverage_file=coverage_file, whole_days=whole_days)
            if gaps:
                fetch_and_store_bbo(ticker, gaps, dataset=dataset, root=root, coverage_file=coverage_file,
                                    max_cost_usd=max_cost_usd, client=client, schema=schema)
    return read_bbo_from_disk(tickers, start, end, root=root)


def load_bbo_1s(tickers: list[str], start, end, *, dataset: str | None = None, fetch_missing: bool = True,
                root: Path = BBO_1S_FUTURES_DIR, coverage_file: Path = BBO_1S_FUTURES_COVERAGE_FILE,
                max_cost_usd: float = MAX_COST_USD, client=None) -> pd.DataFrame:
    """ON-DEMAND 1-second quotes for a small event window (e.g. an hour around an FOMC
    statement). Cached in ``~/Database/bbo-1s``; the EXACT window is fetched (not the
    whole day) and paid for once, priced by the ``max_cost_usd`` guardrail first."""
    return load_bbo(tickers, start, end, dataset=dataset, fetch_missing=fetch_missing, root=root,
                    coverage_file=coverage_file, max_cost_usd=max_cost_usd, client=client, schema=SCHEMA_BBO_1S,
                    whole_days=False)
