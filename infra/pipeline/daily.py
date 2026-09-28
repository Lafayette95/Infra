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
    DAILY_FUTURES_COVERAGE_FILE,
    DAILY_FUTURES_DIR,
    MAX_COST_USD,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.processing import statistics as stats
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)


def read_daily_from_disk(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path = DAILY_FUTURES_DIR,
) -> pd.DataFrame:
    """Read decoded (float settlement price) daily rows with predicate pushdown. No API."""
    raw = parquet_store.read_partitioned(
        root, start=start, end=end, equals_in={"ticker": tickers}
    )
    if raw is None or raw.empty:
        return stats.decode_daily(stats.empty_daily())
    df = stats.decode_daily(raw[stats.DAILY_COLUMNS]).sort_values(stats.DAILY_KEYS)
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

    ``prune=True`` first deletes this ticker's entire stored history AND its coverage
    record, so afterwards both describe exactly the ranges fetched here (done once, up
    front - pruning per range would wipe the previous range's rows).
    """
    if prune:
        parquet_store.prune_rows(root, "ticker", [ticker])
        coverage_store.clear_key(coverage_file, ticker)
    rows = 0
    for range_start, range_end in ranges:
        query_end, covered_end = bounded_by_availability(dataset, range_end, client)
        if query_end <= range_start:
            continue  # nothing in this range is queryable yet; not covered, retried later
        raw = api.fetch_statistics(
            dataset, ticker, range_start, query_end, max_cost_usd=max_cost_usd, client=client
        )
        clean = stats.clean_daily_statistics(raw, ticker, dataset)
        parquet_store.write_partitioned(stats.encode_daily(clean), root, stats.DAILY_KEYS)
        # Recorded even when empty (holidays, or a contract not yet publishing stats)
        # so we never pay to re-ask - but never past the last COMPLETE available day.
        if covered_end > range_start:
            coverage_store.record_covered(coverage_file, ticker, [(range_start, covered_end)])
        rows += len(clean)
        log.info("%s %s->%s: %d daily rows saved", ticker, range_start.date(), range_end.date(), len(clean))
    return rows


_RECENT = pd.Timedelta(days=7)


def bounded_by_availability(
    dataset: str, range_end: pd.Timestamp, client=None, *, now: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``(query_end, covered_end)`` for a range ending at ``range_end``. Databento rejects
    any request ending after the dataset's advertised available end (e.g. "today,
    inclusive", as the daily cycle asks - GLBX runs ~8h behind now), so the query is
    CLAMPED to it; coverage is recorded only up to the start of that day, so a partial or
    not-yet-published day is never claimed as covered and a later normal run fills it in
    (see api.available_end for the verified behavior). Ranges ending well in the past
    skip the (free) metadata lookup entirely."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    if range_end < now - _RECENT:
        return range_end, range_end
    available = api.available_end(dataset, api.SCHEMA_STATISTICS, client)
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
