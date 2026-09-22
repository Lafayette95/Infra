"""Daily settlement price / open interest for options, implementing Rule 2.3:

1. pull the tiny ``definition`` schema (cached on disk, via infra.pipeline.options -
   reused as-is, no new definitions-fetching code needed here),
2. filter it locally by strike/expiry,
3. query ``statistics`` ONLY for the isolated instrument ids that are not yet on disk.

Mirrors infra/pipeline/options.py's shape exactly; the only difference is the schema
(``statistics``, not ``ohlcv-1m``) and cadence (once per trading day). See CLAUDE.md
section 8. Naming note: this pairs with infra/pipeline/options.py the way
infra/pipeline/daily.py (kept as-is, not renamed to daily_futures.py to avoid
disrupting its already-shipped call sites) pairs with infra/pipeline/futures.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    DAILY_OPTIONS_COVERAGE_FILE,
    DAILY_OPTIONS_DIR,
    DEFINITIONS_DIR,
    MAX_COST_USD,
    OPTIONS_UNIVERSE,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.pipeline.options import load_definitions
from infra.processing import definitions as defs
from infra.processing import statistics as stats
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)


def plan_daily_options_update(
    instrument_ids: list[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    coverage_file: Path = DAILY_OPTIONS_COVERAGE_FILE,
) -> dict[Interval, list[int]]:
    """Group instrument ids by the identical missing range they need. No API."""
    requested = (to_utc_day(start), to_utc_day(end))
    plan: dict[Interval, list[int]] = {}
    for instrument_id in instrument_ids:
        covered = coverage_store.read_covered(coverage_file, str(instrument_id))
        for gap in find_missing_ranges(requested, covered):
            plan.setdefault(gap, []).append(instrument_id)
    return plan


def fetch_and_store_daily_options(
    plan: dict[Interval, list[int]],
    definitions: pd.DataFrame,
    *,
    dataset: str,
    root: Path = DAILY_OPTIONS_DIR,
    coverage_file: Path = DAILY_OPTIONS_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> int:
    """Query settlement/OI for exactly the planned ids/ranges, save, record coverage."""
    rows = 0
    for (range_start, range_end), ids in plan.items():
        raw = api.fetch_statistics_by_instrument_ids(
            dataset, ids, range_start, range_end, max_cost_usd=max_cost_usd, client=client
        )
        clean = stats.clean_daily_option_statistics(raw, definitions, dataset)
        parquet_store.write_partitioned(stats.encode_daily_options(clean), root, stats.DAILY_OPTIONS_KEYS)
        for instrument_id in ids:
            coverage_store.record_covered(coverage_file, str(instrument_id), [(range_start, range_end)])
        rows += len(clean)
        log.info("%s->%s: %d instrument(s), %d daily rows saved",
                 range_start.date(), range_end.date(), len(ids), len(clean))
    return rows


def read_daily_options_from_disk(
    contracts: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    root: Path = DAILY_OPTIONS_DIR,
) -> pd.DataFrame:
    """Read decoded daily settlement/OI rows, restricted to the given (filtered) contracts."""
    raw = parquet_store.read_partitioned(
        root, start=start, end=end,
        equals_in={"underlying": contracts["underlying"].unique().tolist()},
    )
    if raw is None or raw.empty:
        return stats.decode_daily_options(stats.empty_daily_options())
    df = stats.decode_daily_options(raw[stats.DAILY_OPTIONS_COLUMNS])
    keys = contracts[["underlying", "option_type", "strike", "expiry"]].astype(
        {"underlying": "str", "option_type": "str"}
    )
    df = df.astype({"underlying": "str", "option_type": "str"}).merge(
        keys, on=["underlying", "option_type", "strike", "expiry"], how="inner"
    )
    return df.sort_values(stats.DAILY_OPTIONS_KEYS).reset_index(drop=True)


def load_daily_options(
    parent: str,
    definitions_day,
    start,
    end,
    *,
    underlyings: list[str] | None = None,
    option_types: list[str] | None = None,
    strike_range: tuple[float, float] | None = None,
    expiry_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    fetch_missing: bool = True,
    definitions_directory: Path = DEFINITIONS_DIR,
    root: Path = DAILY_OPTIONS_DIR,
    coverage_file: Path = DAILY_OPTIONS_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Parent: definitions -> local filter -> fetch only missing ids/ranges -> read.

    ``definitions_directory``/``root``/``coverage_file`` default to the real database
    paths but can be overridden - e.g. by tests, to avoid touching real disk state.
    """
    start, end = to_utc_day(start), to_utc_day(end)
    dataset = OPTIONS_UNIVERSE[parent]
    all_defs = load_definitions(
        parent, definitions_day, dataset=dataset, fetch_missing=fetch_missing,
        directory=definitions_directory, max_cost_usd=max_cost_usd, client=client,
    )
    contracts = defs.filter_definitions(
        all_defs, underlyings=underlyings, option_types=option_types,
        strike_range=strike_range, expiry_range=expiry_range,
    )
    if contracts.empty:
        return stats.decode_daily_options(stats.empty_daily_options())
    if fetch_missing:
        plan = plan_daily_options_update(contracts["instrument_id"].tolist(), start, end, coverage_file=coverage_file)
        fetch_and_store_daily_options(
            plan, contracts, dataset=dataset, root=root, coverage_file=coverage_file,
            max_cost_usd=max_cost_usd, client=client,
        )
    return read_daily_options_from_disk(contracts, start, end, root=root)
