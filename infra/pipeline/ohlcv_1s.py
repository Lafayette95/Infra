"""ON-DEMAND 1-second trade bars (Databento ``ohlcv-1s``) for ABSOLUTE futures contracts,
for small ad-hoc analyses (e.g. the seconds around an FOMC statement) - never fetched by
a cycle. Disk first -> gaps -> API -> save -> read, reusing infra.pipeline.futures
unchanged (same bar layout) pointed at its own store, ``~/Database/ohlcv-1s``.

Unlike the 1-minute store, the EXACT requested window is fetched (not whole UTC days):
paying for a whole day of 1-second bars to study one hour would be waste, and coverage
intervals work at any resolution. A bar is stamped at its START (complete 1s later).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.config import MAX_COST_USD, OHLCV_1S_FUTURES_COVERAGE_FILE, OHLCV_1S_FUTURES_DIR, SCHEMA_OHLCV_1S
from infra.coverage.intervals import Interval, find_missing_ranges
from infra.pipeline import futures as fut
from infra.storage import coverage_store


def plan_ohlcv_1s_update(ticker: str, start, end, *,
                         coverage_file: Path = OHLCV_1S_FUTURES_COVERAGE_FILE) -> list[Interval]:
    """Sub-ranges of the EXACT window ``[start, end)`` never queried before. No API."""
    requested = (pd.Timestamp(start), pd.Timestamp(end))
    return find_missing_ranges(requested, coverage_store.read_covered(coverage_file, ticker))


def load_ohlcv_1s(
    tickers: list[str],
    start,
    end,
    *,
    dataset: str | None = None,
    fetch_missing: bool = True,
    root: Path = OHLCV_1S_FUTURES_DIR,
    coverage_file: Path = OHLCV_1S_FUTURES_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
    multiindex: bool = False,
) -> pd.DataFrame:
    """Parent: ensure the window is on disk (API only for its uncovered parts, priced by
    the ``max_cost_usd`` guardrail first), then read exactly ``[start, end)``."""
    if fetch_missing and dataset is None:
        raise ValueError("dataset is required when fetch_missing=True")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if fetch_missing:
        for ticker in tickers:
            gaps = plan_ohlcv_1s_update(ticker, start, end, coverage_file=coverage_file)
            if gaps:
                fut.fetch_and_store_futures(ticker, gaps, dataset=dataset, root=root, coverage_file=coverage_file,
                                            max_cost_usd=max_cost_usd, client=client, schema=SCHEMA_OHLCV_1S)
    return fut.read_futures_from_disk(tickers, start, end, root=root, multiindex=multiindex)
