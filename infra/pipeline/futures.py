"""Futures pipeline for ABSOLUTE contracts (raw symbols): disk first -> gaps -> API -> save -> read.

Parent:   ``load_futures``            (what the dashboard and scripts call)
Children: ``read_futures_from_disk``, ``plan_futures_update``,
          ``fetch_and_store_futures`` - each usable on its own.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    FUTURES_COVERAGE_FILE,
    FUTURES_DIR,
    MAX_COST_USD,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.processing import transforms as tf
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)


def read_futures_from_disk(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path = FUTURES_DIR,
    multiindex: bool = False,
) -> pd.DataFrame:
    """Read decoded (float-price) bars from parquet with predicate pushdown. No API."""
    raw = parquet_store.read_partitioned(
        root, start=start, end=end, equals_in={"ticker": tickers}
    )
    if raw is None or raw.empty:
        empty = tf.decode_futures(tf.encode_futures(tf.clean_futures_ohlcv(pd.DataFrame())))
        return tf.to_multiindex(empty) if multiindex else empty
    df = tf.decode_futures(raw[tf.FUTURES_COLUMNS]).sort_values(tf.FUTURES_KEYS)
    df = df.reset_index(drop=True)
    return tf.to_multiindex(df) if multiindex else df


def plan_futures_update(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    coverage_file: Path = FUTURES_COVERAGE_FILE,
) -> list[Interval]:
    """Date ranges of ``[start, end)`` never queried before. Touches no API."""
    requested = (to_utc_day(start), to_utc_day(end))
    return find_missing_ranges(requested, coverage_store.read_covered(coverage_file, ticker))


def fetch_and_store_futures(
    ticker: str,
    ranges: list[Interval],
    *,
    dataset: str,
    root: Path = FUTURES_DIR,
    coverage_file: Path = FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> int:
    """Query the API for ``ranges``, save to parquet, record coverage. Returns rows saved."""
    rows = 0
    for range_start, range_end in ranges:
        raw = api.fetch_futures_ohlcv(
            dataset, ticker, range_start, range_end, max_cost_usd=max_cost_usd, client=client
        )
        clean = tf.clean_futures_ohlcv(raw)
        parquet_store.write_partitioned(tf.encode_futures(clean), root, tf.FUTURES_KEYS)
        # Recorded even when empty (holidays) so we never pay to re-ask.
        coverage_store.record_covered(coverage_file, ticker, [(range_start, range_end)])
        rows += len(clean)
        log.info("%s %s->%s: %d rows saved", ticker, range_start.date(), range_end.date(), len(clean))
    return rows


def load_futures(
    tickers: list[str],
    start,
    end,
    *,
    dataset: str | None = None,
    fetch_missing: bool = True,
    root: Path = FUTURES_DIR,
    coverage_file: Path = FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    multiindex: bool = False,
) -> pd.DataFrame:
    """Parent: ensure ``[start, end)`` is on disk (API only for gaps), then read it.

    ``tickers`` are absolute contracts; ``dataset`` is required only when fetching.
    """
    if fetch_missing and dataset is None:
        raise ValueError("dataset is required when fetch_missing=True")
    start, end = to_utc_day(start), to_utc_day(end)
    if fetch_missing:
        for ticker in tickers:
            gaps = plan_futures_update(ticker, start, end, coverage_file=coverage_file)
            if gaps:
                fetch_and_store_futures(
                    ticker, gaps, dataset=dataset, root=root, coverage_file=coverage_file,
                    max_cost_usd=max_cost_usd, client=client,
                )
    return read_futures_from_disk(tickers, start, end, root=root, multiindex=multiindex)
