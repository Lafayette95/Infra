"""Discover absolute contracts per root via periodic ``definition`` snapshots.

A single day's ``definition`` lists only contracts alive that day, so a multi-year window
is covered by sampling every ``DEFINITION_SNAPSHOT_DAYS``. Snapshots already pulled are
recorded in the coverage manifest, so none is ever fetched twice (Rule 2.1).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import databento_client as api
from infra.config import (
    DEFINITION_SNAPSHOT_DAYS,
    FUTURES_CONTRACTS_FILE,
    FUTURES_DEFS_COVERAGE_FILE,
    MAX_COST_USD,
    FuturesRoot,
)
from infra.coverage.intervals import Interval, find_missing_ranges, to_utc_day
from infra.processing.definitions import normalize_futures_definitions
from infra.storage import contract_store, coverage_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)


def snapshot_days(start, end, step_days: int = DEFINITION_SNAPSHOT_DAYS) -> list[pd.Timestamp]:
    """Weekday snapshot dates covering ``[start, end)``: the start, then every ``step_days``."""
    start, end = to_utc_day(start), to_utc_day(end)
    days = []
    for day in pd.date_range(start, end, freq=f"{step_days}D", inclusive="left"):
        day = day + pd.Timedelta(days=(7 - day.weekday()) % 7 if day.weekday() >= 5 else 0)
        days.append(day)
    return days


def missing_snapshots(
    cfg: FuturesRoot, start, end, *, coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE
) -> list[Interval]:
    """One-day intervals of the needed snapshots not yet pulled (past days only)."""
    covered = coverage_store.read_covered(coverage_file, f"defs:{cfg.root}")
    gaps: list[Interval] = []
    for day in snapshot_days(start, end):
        gaps += find_missing_ranges((day, day + _ONE_DAY), covered)
    return gaps


def ensure_contracts(
    cfg: FuturesRoot,
    start,
    end,
    *,
    fetch_missing: bool = True,
    contracts_file: Path = FUTURES_CONTRACTS_FILE,
    coverage_file: Path = FUTURES_DEFS_COVERAGE_FILE,
    max_cost_usd: float = MAX_COST_USD,
    client=None,
) -> pd.DataFrame:
    """Contracts table for ``cfg.root``; pulls only the missing definition snapshots."""
    if fetch_missing:
        for day, _ in missing_snapshots(cfg, start, end, coverage_file=coverage_file):
            raw = api.fetch_definitions(
                cfg.dataset, [cfg.parent], day, max_cost_usd=max_cost_usd, client=client
            )
            contract_store.write_contracts(contracts_file, normalize_futures_definitions(raw, cfg.root))
            coverage_store.record_covered(coverage_file, f"defs:{cfg.root}", [(day, day + _ONE_DAY)])
            log.info("%s definitions snapshot %s saved", cfg.root, day.date())
    return contract_store.read_contracts(contracts_file, cfg.root)
