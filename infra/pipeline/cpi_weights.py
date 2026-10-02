"""CPI relative importance (the item weights): read / plan / fetch / store + a ``load``
parent, over ``Database/RawData/CPIWeights``. Methodology: infra/models/inflation/CLAUDE.md.

* **One weight year Y = the weights as of December Y**, published with the January Y+1
  CPI and used to aggregate the index through Y+1.
* **Stored rows** (``infra.processing.cpi_weights.WEIGHT_COLUMNS`` plus ``timestamp``):
  ``timestamp`` is the PUBLICATION day - the January Y+1 CPI's first-print day, read from
  the ALFRED ``CPIAUCNS`` vintages already on disk (``infra.pipeline.releases``: exact for
  every year since 1987's), else the day it was first fetched. So an ``as_of`` read only
  sees weights published by then (CLAUDE.md 3, point-in-time).
* **Fetched once per year, never daily:** coverage is keyed by weight year (the interval
  ``[Dec 1 Y, Jan 1 Y+1)``). ``due_weight_year`` is the latest year that SHOULD be out; a
  run makes NO request at all while every year up to it is on disk. Only between February
  1 and the day BLS actually posts it does a run ask (one small request), and a year not
  yet published is simply not covered - asked again the next day.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import bls_client
from infra.config import (BULK_CATALOG_DIR, CPI_WEIGHTS_COVERAGE_FILE, CPI_WEIGHTS_DIR, CPI_WEIGHTS_FIRST_YEAR,
                          RELEASES_DIR)
from infra.coverage.intervals import merge_intervals, to_utc_day
from infra.pipeline import bulk_series, releases
from infra.processing import cpi_weights as cw
from infra.processing import releases as pr
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)

COVERAGE_KEY = "cpi_weights"
KEYS = ["weight_year", "line"]
COLUMNS = ["timestamp", *cw.WEIGHT_COLUMNS]
# fn(years) -> {year: (format, content)}: network only (infra.api.bls_client)
FETCH: Callable[..., dict[int, tuple[str, bytes]]] = bls_client.fetch_relative_importance
CONFIGURED: Callable[[], bool] = bls_client.has_contact
PARSERS = {"xlsx": cw.parse_xlsx, "txt": lambda content, y: cw.parse_txt(content.decode("latin-1"), y)}
# the January CPI - and so a new weight year - is released in mid-February; from this day
# on the year is due (asked for daily until published)
DUE_FROM = (2, 1)  # (month, day)


def _interval(year: int):
    return pd.Timestamp(year, 12, 1), pd.Timestamp(year + 1, 1, 1)


def due_weight_year(today) -> int:
    """The latest weight year that may already be published on ``today``."""
    today = pd.Timestamp(today)
    return today.year - 1 if (today.month, today.day) >= DUE_FROM else today.year - 2


def covered_years(coverage_file: Path = CPI_WEIGHTS_COVERAGE_FILE) -> set[int]:
    years = set()
    for s, e in merge_intervals(coverage_store.read_covered(coverage_file, COVERAGE_KEY)):
        years.update(range(s.year, e.year))  # [Dec 1 Y, Jan 1 Y+1) -> Y
    return years


def plan_weights_update(today=None, *, coverage_file: Path = CPI_WEIGHTS_COVERAGE_FILE,
                        first_year: int = CPI_WEIGHTS_FIRST_YEAR) -> list[int]:
    """Weight years to request: every due year not on disk (the whole history on the first
    run, afterwards at most the one new year - and nothing most days). No network."""
    today = pd.Timestamp.now() if today is None else pd.Timestamp(today)
    return sorted(set(range(first_year, due_weight_year(today) + 1)) - covered_years(coverage_file))


def publication_days(years, *, releases_root: Path = RELEASES_DIR, fetched_on=None) -> dict[int, pd.Timestamp]:
    """Each weight year's publication day: the first print of the January Y+1 CPI
    (ALFRED ``CPIAUCNS``, on disk), else ``fetched_on`` (default today)."""
    raw = releases.read_releases_from_disk(["CPIAUCNS"], root=releases_root)
    first = pr.estimate(raw, 0) if len(raw) else raw
    by_period = dict(zip(first["period"], first["timestamp"])) if len(first) else {}
    fallback = to_utc_day(pd.Timestamp.now() if fetched_on is None else fetched_on)
    return {y: pd.Timestamp(by_period.get(pd.Timestamp(y + 1, 1, 1), fallback)) for y in years}


def fetch_weights_raw(years, *, fetch=None) -> dict[int, tuple[str, bytes]]:
    """NETWORK ONLY."""
    return (FETCH if fetch is None else fetch)(years)


def store_weights_raw(
    fetched: dict[int, tuple[str, bytes]],
    *,
    root: Path = CPI_WEIGHTS_DIR,
    coverage_file: Path = CPI_WEIGHTS_COVERAGE_FILE,
    releases_root: Path = RELEASES_DIR,
    catalog_dir: Path = BULK_CATALOG_DIR,
) -> int:
    """FILES ONLY: parse each fetched year, fill item codes from the CPI catalog (when it's
    on disk), stamp the publication day, save, and record the year covered."""
    if not fetched:
        return 0
    items = bulk_series.read_catalog("cpi", catalog_dir=catalog_dir)
    items = items[["item_code", "item_name"]].dropna() if {"item_code", "item_name"} <= set(items.columns) else None
    published = publication_days(fetched, releases_root=releases_root)
    rows = 0
    for year, (fmt, content) in sorted(fetched.items()):
        df = PARSERS[fmt](content, year)
        if items is not None:
            df = cw.match_item_codes(df, items)
        df.insert(0, "timestamp", published[year])
        out = df.astype({"timestamp": "datetime64[ms]", "item_code": "string", "section": "string"})
        parquet_store.write_partitioned(out[COLUMNS], root, KEYS)
        coverage_store.record_covered(coverage_file, COVERAGE_KEY, [_interval(year)])
        rows += len(out)
        log.info("cpi weights %d (%s, published %s): %d rows", year, fmt, published[year].date(), len(out))
    return rows


def read_cpi_weights(as_of=None, weight_year: int | None = None, *, root: Path = CPI_WEIGHTS_DIR) -> pd.DataFrame:
    """The weights PUBLISHED by the end of day ``as_of`` (None = everything): one weight
    year - ``weight_year``, or the latest one published by ``as_of`` - or empty. No network."""
    end = None if as_of is None else to_utc_day(as_of) + pd.Timedelta(days=1)
    df = parquet_store.read_partitioned(root, end=end)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    year = int(df["weight_year"].max()) if weight_year is None else int(weight_year)
    return df[df["weight_year"] == year].sort_values("line")[COLUMNS].reset_index(drop=True)


def load_cpi_weights(as_of=None, weight_year: int | None = None, *, fetch_missing: bool = True,
                     root: Path = CPI_WEIGHTS_DIR, coverage_file: Path = CPI_WEIGHTS_COVERAGE_FILE,
                     releases_root: Path = RELEASES_DIR, catalog_dir: Path = BULK_CATALOG_DIR,
                     fetch=None) -> pd.DataFrame:
    """Parent: fetch any due weight year not on disk (nothing, most days), then read."""
    if fetch_missing:
        years = plan_weights_update(coverage_file=coverage_file)
        if years:
            store_weights_raw(fetch_weights_raw(years, fetch=fetch), root=root, coverage_file=coverage_file,
                              releases_root=releases_root, catalog_dir=catalog_dir)
    return read_cpi_weights(as_of, weight_year, root=root)
