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
) -> list[Interval]:
    """Date ranges of ``[start, end)`` never queried before. Touches no API."""
    requested = (to_utc_day(start), to_utc_day(end))
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
) -> int:
    """Query the API for ``ranges``, save to parquet, record coverage. Returns rows saved."""
    rows = 0
    for range_start, range_end in ranges:
        raw = api.fetch_statistics(
            dataset, ticker, range_start, range_end, max_cost_usd=max_cost_usd, client=client
        )
        clean = stats.clean_daily_statistics(raw, ticker, dataset)
        parquet_store.write_partitioned(stats.encode_daily(clean), root, stats.DAILY_KEYS)
        # Recorded even when empty (holidays, or a contract not yet publishing stats)
        # so we never pay to re-ask.
        coverage_store.record_covered(coverage_file, ticker, [(range_start, range_end)])
        rows += len(clean)
        log.info("%s %s->%s: %d daily rows saved", ticker, range_start.date(), range_end.date(), len(clean))
    return rows


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
) -> pd.DataFrame:
    """Parent: ensure ``[start, end)`` is on disk (API only for gaps), then read it.

    ``tickers`` are absolute contracts; ``dataset`` is required only when fetching -
    matches infra.pipeline.futures.load_futures's convention exactly.
    """
    if fetch_missing and dataset is None:
        raise ValueError("dataset is required when fetch_missing=True")
    start, end = to_utc_day(start), to_utc_day(end)
    if fetch_missing:
        for ticker in tickers:
            gaps = plan_daily_update(ticker, start, end, coverage_file=coverage_file)
            if gaps:
                fetch_and_store_daily(
                    ticker, gaps, dataset=dataset, root=root, coverage_file=coverage_file,
                    max_cost_usd=max_cost_usd, client=client,
                )
    return read_daily_from_disk(tickers, start, end, root=root)
