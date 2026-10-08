"""Daily settlement price / open interest for ABSOLUTE futures contracts.

Same 4-function shape as pipeline/futures.py, same reused storage/coverage primitives
(parquet_store, coverage_store) - just pointed at a separate root (Database/Daily) with
a separate coverage manifest, per CLAUDE.md section 8. Deliberately its own pipeline,
not layered under futures.py: the `statistics` schema, its per-stat_type pivoting, and
its trading-day resolution (infra.processing.statistics) are unrelated to `ohlcv-1m`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    ADJUSTMENTS_DIR,
    DAILY_FUTURES_COVERAGE_FILE,
    DAILY_FUTURES_DIR,
    MAX_COST_USD,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.processing import statistics as stats
from infra.storage import adjustment_store, coverage_store, parquet_store

log = logging.getLogger(__name__)

# This store's name in the adjustments log (infra.storage.adjustment_store).
STORE = "Daily/Futures"


def read_daily_from_disk(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path = DAILY_FUTURES_DIR,
    adjusted: bool = True,
    adjustments_dir: Path | None = None,
) -> pd.DataFrame:
    """Read decoded (float settlement price) daily rows with predicate pushdown. No API.

    ``adjusted=True`` (default) overlays the adjustments log - bad prints NA'd or rolled
    (CLAUDE.md 12) - so consumers get cleaned values; ``adjusted=False`` returns exactly
    what the vendor delivered (the px checks, which judge the raw data, use it)."""
    raw = parquet_store.read_partitioned(
        root, start=start, end=end, equals_in={"ticker": tickers}
    )
    if raw is None or raw.empty:
        return stats.decode_daily(stats.empty_daily())
    # reindex: a store written before a column existed (``volume``) reads it as all-null
    df = stats.decode_daily(raw.reindex(columns=stats.DAILY_COLUMNS)).sort_values(stats.DAILY_KEYS)
    if adjusted:
        # resolved at CALL time (not bound as a default), so tests can point the module
        # default at a temp dir (tests/conftest.py) and never read the real log
        adjustments_dir = ADJUSTMENTS_DIR if adjustments_dir is None else adjustments_dir
        adj = adjustment_store.read(adjustments_dir, store=STORE, start=start, end=end, keys=tickers)
        df = adjustment_store.apply(df, adj, key_column="ticker")
    return df.reset_index(drop=True)


def plan_daily_update(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    force_refetch: bool = False,
) -> list[Interval]:
    """Date ranges of ``[start, end)`` never queried before. Touches no API.

    ``force_refetch=True`` ignores the coverage manifest and plans the WHOLE requested
    range - the one deliberate exception to Rule 2.1, used only by the scheduled daily
    cycle's short trailing window (infra.cycle) to catch upstream revisions of data we
    already hold. Defaults to False everywhere.
    """
    requested = (to_utc_day(start), to_utc_day(end))
    if force_refetch:
        return [requested] if requested[0] < requested[1] else []
    return find_missing_ranges(requested, coverage_store.read_covered(coverage_file, ticker))


# (range_start, range_end, covered_end, raw statistics rows) per queried range
Fetched = list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DataFrame]]


def fetch_daily_raw(
    ticker: str,
    ranges: list[Interval],
    *,
    dataset: str,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> Fetched:
    """NETWORK ONLY: query the API for ``ranges`` (each bounded by availability); touches
    no files, so callers may run many of these concurrently (infra.cycle.px does)."""
    out: Fetched = []
    for range_start, range_end in ranges:
        query_end, covered_end = bounded_by_availability(dataset, range_end, client)
        if query_end <= range_start:
            continue  # nothing in this range is queryable yet; not covered, retried later
        raw = api.fetch_statistics(
            dataset, ticker, range_start, query_end, max_cost_usd=max_cost_usd, client=client
        )
        out.append((range_start, range_end, covered_end, raw))
    return out


def store_daily_raw(
    ticker: str,
    fetched: Fetched,
    *,
    dataset: str,
    root: Path = DAILY_FUTURES_DIR,
    coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    prune: bool = False,
) -> int:
    """FILES ONLY: clean, save and record coverage for what ``fetch_daily_raw`` returned.
    Read-modify-writes shared partition files - never run two of these concurrently.

    ``prune=True`` first deletes this ticker's entire stored history AND its coverage
    record, so afterwards both describe exactly the ranges fetched here (done once, up
    front - pruning per range would wipe the previous range's rows).
    """
    if prune:
        parquet_store.prune_rows(root, "ticker", [ticker])
        coverage_store.clear_key(coverage_file, ticker)
    rows = 0
    for range_start, range_end, covered_end, raw in fetched:
        clean = stats.clean_daily_statistics(raw, ticker, dataset)
        parquet_store.write_partitioned(stats.encode_daily(clean), root, stats.DAILY_KEYS,
        coalesce=True)  # settlement and OI are published in different sessions (CLAUDE.md 8)
        # Recorded even when empty (holidays, or a contract not yet publishing stats)
        # so we never pay to re-ask - but never past the last COMPLETE available day.
        if covered_end > range_start:
            coverage_store.record_covered(coverage_file, ticker, [(range_start, covered_end)])
        rows += len(clean)
        log.info("%s %s->%s: %d daily rows saved", ticker, range_start.date(), range_end.date(), len(clean))
    return rows


def fetch_and_store_daily(
    ticker: str,
    ranges: list[Interval],
    *,
    dataset: str,
    root: Path = DAILY_FUTURES_DIR,
    coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    prune: bool = False,
) -> int:
    """Query the API for ``ranges``, save to parquet, record coverage. Returns rows saved.
    = ``store_daily_raw(fetch_daily_raw(...))`` - the two halves are separate so a caller
    can parallelise the network part while keeping the file writes sequential."""
    fetched = fetch_daily_raw(ticker, ranges, dataset=dataset, max_cost_usd=max_cost_usd, client=client)
    return store_daily_raw(ticker, fetched, dataset=dataset, root=root, coverage_file=coverage_file, prune=prune)


# ------------------------------------------------------------------ bulk (many contracts at once)

def plan_daily_bulk(
    tickers: list[str], start, end, *, chunk_years: int = 1,
    coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    alive: dict[str, list[Interval]] | None = None,
) -> tuple[list[tuple[Interval, list[str]]], list[tuple[str, list[Interval]]]]:
    """Rule 2.1 for a many-contract backfill: ``[start, end)`` cut into ``chunk_years``
    pieces; per piece, the tickers with NO coverage in it go into one bulk request
    (``bulk``: [(piece, tickers)]), and tickers only PARTLY covered there keep the
    per-contract path for their exact gaps (``single``: [(ticker, gaps)]). Nothing already
    covered is asked again. ``alive``: each ticker's listed life(s) (activation to expiry; a
    one-digit-year CME symbol has one per decade) - a ticker goes only into pieces that
    overlap one, since a request whose symbols ALL fail to resolve is rejected (422).
    Touches no API."""
    start, end = to_utc_day(start), to_utc_day(end)
    bounds = [start] + [pd.Timestamp(year=y, month=1, day=1)
                        for y in range(start.year + 1, end.year + 1, chunk_years)] + [end]
    pieces = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if a < b]
    bulk, single = [], {}
    for piece in pieces:
        whole = []
        for t in tickers:
            if alive is not None and not any(a < piece[1] and b > piece[0] for a, b in alive.get(t, [])):
                continue
            gaps = plan_daily_update(t, *piece, coverage_file=coverage_file)
            if gaps == [piece]:
                whole.append(t)
            elif gaps:
                single.setdefault(t, []).extend(gaps)
        if whole:
            bulk.append((piece, whole))
    return bulk, list(single.items())


def fetch_daily_raw_bulk(
    tickers: list[str], piece: Interval, *, dataset: str, max_cost_usd: float = MAX_COST_USD, client=None,
) -> dict[str, Fetched]:
    """NETWORK ONLY: one bulk request (batched) for ``tickers`` over ``piece``, split back
    per ticker into the shape ``store_daily_raw`` takes. A ticker with no rows (not listed
    yet, or a symbol that resolves to nothing then) still gets its empty range, so it is
    recorded covered and never re-asked."""
    range_start, range_end = piece
    query_end, covered_end = bounded_by_availability(dataset, range_end, client)
    if query_end <= range_start:
        return {}
    raw = api.fetch_statistics_bulk(dataset, tickers, range_start, query_end,
                                    max_cost_usd=max_cost_usd, client=client)
    sym = raw["symbol"].astype(str) if "symbol" in raw else pd.Series(dtype=str)
    return {t: [(range_start, range_end, covered_end, raw[sym == t] if len(raw) else raw)] for t in tickers}


_RECENT = pd.Timedelta(days=7)


def bounded_by_availability(
    dataset: str, range_end: pd.Timestamp, client=None, *, now: pd.Timestamp | None = None,
    schema: str = api.SCHEMA_STATISTICS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``(query_end, covered_end)`` for a range ending at ``range_end``. Databento rejects
    any request ending after the dataset's advertised available end (e.g. "today,
    inclusive", as the daily cycle asks - GLBX runs ~8h behind now), so the query is
    CLAMPED to it; coverage is recorded only up to the start of that day, so a partial or
    not-yet-published day is never claimed as covered and a later normal run fills it in
    (see api.available_end for the verified behavior). Ranges ending well in the past
    skip the (free) metadata lookup entirely. ``schema``: whose availability to read
    (each schema has its own - shared by the 1-minute/1-second bar and quote pipelines)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    if range_end < now - _RECENT:
        return range_end, range_end
    available = api.available_end(dataset, schema, client)
    if range_end <= available:
        return range_end, range_end
    return available, available.normalize()


def load_daily(
    tickers: list[str],
    start,
    end,
    *,
    dataset: str | None = None,
    fetch_missing: bool = True,
    root: Path = DAILY_FUTURES_DIR,
    coverage_file: Path = DAILY_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    force_refetch: bool = False,
    prune: bool = False,
) -> pd.DataFrame:
    """Parent: ensure ``[start, end)`` is on disk (API only for gaps), then read it.

    ``tickers`` are absolute contracts; ``dataset`` is required only when fetching -
    matches infra.pipeline.futures.load_futures's convention exactly. ``force_refetch``
    and ``prune`` both default to False; see plan_daily_update / fetch_and_store_daily.
    """
    if fetch_missing and dataset is None:
        raise ValueError("dataset is required when fetch_missing=True")
    start, end = to_utc_day(start), to_utc_day(end)
    if fetch_missing:
        for ticker in tickers:
            gaps = plan_daily_update(
                ticker, start, end, coverage_file=coverage_file, force_refetch=force_refetch,
            )
            if gaps:
                fetch_and_store_daily(
                    ticker, gaps, dataset=dataset, root=root, coverage_file=coverage_file,
                    max_cost_usd=max_cost_usd, client=client, prune=prune,
                )
    return read_daily_from_disk(tickers, start, end, root=root)
