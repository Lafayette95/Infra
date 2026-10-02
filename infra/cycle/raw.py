"""Step 1b - ``backfill_daily_raw_data``: non-price raw inputs (macro releases, EFFR,
calendars, ...). Registered: ``releases`` (every vintage of the nowcast's macro releases,
infra.cycle.raw_releases), ``bulk`` (full-granularity CPI / PPI / PCE snapshots,
infra.cycle.raw_bulk), ``cpi_weights`` (CPI relative importance, once a year,
infra.cycle.raw_cpi_weights) and ``dtcc`` (the daily DTCC swap-trade reports, archived
raw, infra.cycle.raw_dtcc), and the reference data - ``release_calendar``, ``tsy_auctions``,
``auction_tails`` (infra.cycle.raw_reference).

A raw source is a function ``(start, end, *, paths, force_refetch) -> dict`` added to
``RAW_SOURCES``; each should also contribute its own checks to ``RAW_CHECKS``.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from infra.cycle.core import Check, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw_bulk import BULK_CHECKS, backfill_daily_bulk
from infra.cycle.raw_cpi_weights import CPI_WEIGHTS_CHECKS, backfill_daily_cpi_weights
from infra.cycle.raw_dtcc import DTCC_CHECKS, backfill_daily_dtcc
from infra.cycle.raw_releases import RELEASE_CHECKS, backfill_daily_releases
from infra.cycle.raw_reference import (AUCTION_CHECKS, CALENDAR_CHECKS, TAILS_CHECKS, backfill_daily_auction_tails,
                                       backfill_daily_release_calendar, backfill_daily_tsy_auctions)

# Run in this order: "cpi_weights" dates each weight year from the ALFRED CPI vintages that
# "releases" stores, and matches item codes from the CPI catalog that "bulk" stores.
# "release_calendar" after "releases" (its releases_due check reads the fresh calendar);
# "auction_tails" after "tsy_auctions" (tails are matched to the stored auctions).
RAW_SOURCES: dict[str, Callable[..., dict]] = {"releases": backfill_daily_releases, "bulk": backfill_daily_bulk,
                                               "cpi_weights": backfill_daily_cpi_weights, "dtcc": backfill_daily_dtcc,
                                               "release_calendar": backfill_daily_release_calendar,
                                               "tsy_auctions": backfill_daily_tsy_auctions,
                                               "auction_tails": backfill_daily_auction_tails}
RAW_CHECKS: tuple[Check, ...] = (RELEASE_CHECKS + BULK_CHECKS + CPI_WEIGHTS_CHECKS + DTCC_CHECKS + CALENDAR_CHECKS
                                 + AUCTION_CHECKS + TAILS_CHECKS)


def backfill_daily_raw_data(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    sources: dict[str, Callable[..., dict]] | None = None,
) -> dict:
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    sources = RAW_SOURCES if sources is None else sources
    return {"sources": {name: fn(start, end, paths=paths, force_refetch=force_refetch)
                        for name, fn in sources.items()}}


def _run(ctx: StepContext) -> dict:
    return backfill_daily_raw_data(ctx.start, ctx.end, paths=ctx.paths, force_refetch=ctx.force_refetch,
                                   sources=ctx.options.get("raw_sources"))


RAW_STEP = Step("raw", _run, checks=RAW_CHECKS)
