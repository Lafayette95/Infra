"""US Treasury prices per CUSIP (CLAUDE.md 18): plan / fetch / store / read + a ``load``
parent over ``Daily/TreasuryPrices``, from FedInvest's daily pages.

* EVERY in-scope security on a day's page is stored - one free request returns them all,
  so filtering at fetch time would save nothing and risk missing a bond later (a new
  cheapest-to-deliver); consumers select through the reference views.
* Coverage (Rule 2.1) is per day, key ``"fedinvest"``, and claimed only for a COMPLETE
  day: its END OF DAY column posted (it appears hours after the page itself), or an empty
  page old enough to be a holiday (``TREASURY_PRICES_SETTLE_DAYS``). Anything else is
  simply asked again next run. Weekends are never asked.
* Yields need each security's coupon, dates and frequency: the reference table
  (infra.pipeline.treasury_ref), which must be built first.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import fedinvest_client
from infra.config import (
    DAILY_TREASURY_PRICES_COVERAGE_FILE,
    DAILY_TREASURY_PRICES_DIR,
    TREASURY_PRICES_SETTLE_DAYS,
    TREASURY_PRICES_START,
    TREASURY_SECURITIES_DIR,
)
from infra.coverage.intervals import find_missing_ranges, to_utc_day
from infra.pipeline.treasury_ref import read_securities
from infra.processing import treasury_prices as tp
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)
KEY = "fedinvest"
# fn(day) -> the page's price table (text); network only
FETCH: Callable[[pd.Timestamp], pd.DataFrame] = fedinvest_client.fetch_prices
PAUSE_S = 0.5  # between days: one site, two requests per day


def plan_prices_update(start, end, *, now=None, coverage_file: Path = DAILY_TREASURY_PRICES_COVERAGE_FILE,
                       force_refetch: bool = False) -> list[pd.Timestamp]:
    """Business days in ``[start, end]`` (inclusive, from TREASURY_PRICES_START, before
    today) not covered yet - or all of them with ``force_refetch``. No network."""
    today = to_utc_day(pd.Timestamp.now(tz="UTC") if now is None else now)
    start = max(pd.Timestamp(start).normalize(), pd.Timestamp(TREASURY_PRICES_START))
    end = min(pd.Timestamp(end).normalize(), today - pd.Timedelta(days=1))
    if end < start:
        return []
    days = pd.bdate_range(start, end)
    if force_refetch:
        return list(days)
    gaps = find_missing_ranges((start, end + pd.Timedelta(days=1)), coverage_store.read_covered(coverage_file, KEY))
    return [d for d in days if any(g0 <= d < g1 for g0, g1 in gaps)]


def store_prices_day(day, page: pd.DataFrame, securities: pd.DataFrame, *, now=None,
                     root: Path = DAILY_TREASURY_PRICES_DIR,
                     coverage_file: Path = DAILY_TREASURY_PRICES_COVERAGE_FILE) -> str:
    """FILES ONLY: store one fetched day if complete, cover it. Returns ``"stored"``,
    ``"holiday"`` (empty page, old enough: covered, nothing stored) or ``"pending"``
    (not covered: asked again next run)."""
    day = pd.Timestamp(day).normalize()
    today = to_utc_day(pd.Timestamp.now(tz="UTC") if now is None else now)
    rows = tp.prices_with_yields(page, day, securities)
    if rows.empty:
        if (today - day).days > TREASURY_PRICES_SETTLE_DAYS:
            coverage_store.record_covered(coverage_file, KEY, [(day, day + pd.Timedelta(days=1))])
            return "holiday"
        return "pending"
    if rows["price_eod"].notna().mean() < 0.5:  # END OF DAY not posted yet
        return "pending"
    parquet_store.write_partitioned(tp.encode(rows), root, tp.PRICE_KEYS)
    coverage_store.record_covered(coverage_file, KEY, [(day, day + pd.Timedelta(days=1))])
    return "stored"


def fetch_and_store_prices(days, *, root: Path = DAILY_TREASURY_PRICES_DIR,
                           coverage_file: Path = DAILY_TREASURY_PRICES_COVERAGE_FILE,
                           securities_root: Path = TREASURY_SECURITIES_DIR, fetch=None, sleep=None,
                           now=None) -> dict:
    """Fetch and store each day, one at a time; per-day failures collected, not raised.
    Returns ``{"stored": [...], "holiday": [...], "pending": [...], "errors": {day: msg}}``."""
    fetch = FETCH if fetch is None else fetch
    sleep = time.sleep if sleep is None else sleep
    securities = read_securities(root=securities_root)
    out = {"stored": [], "holiday": [], "pending": [], "errors": {}}
    for i, day in enumerate(days):
        if i:
            sleep(PAUSE_S)
        try:
            status = store_prices_day(day, fetch(day), securities, now=now, root=root, coverage_file=coverage_file)
            out[status].append(pd.Timestamp(day))
        except Exception as exc:
            out["errors"][pd.Timestamp(day)] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("treasury prices %s failed: %s", pd.Timestamp(day).date(), out["errors"][pd.Timestamp(day)])
    return out


def read_prices(start, end, *, cusips=None, root: Path = DAILY_TREASURY_PRICES_DIR) -> pd.DataFrame:
    """Decoded prices in ``[start, end)``, optionally only ``cusips``. No network."""
    raw = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end),
                                         equals_in={"cusip": list(cusips)} if cusips is not None else None)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tp.PRICE_COLUMNS)
    return tp.decode(raw[tp.PRICE_COLUMNS]).sort_values(tp.PRICE_KEYS).reset_index(drop=True)


def load_prices(start, end, *, cusips=None, fetch_missing: bool = True, root: Path = DAILY_TREASURY_PRICES_DIR,
                coverage_file: Path = DAILY_TREASURY_PRICES_COVERAGE_FILE,
                securities_root: Path = TREASURY_SECURITIES_DIR, fetch=None) -> pd.DataFrame:
    """Parent: fetch every uncovered business day in ``[start, end)``, then read."""
    if fetch_missing:
        days = plan_prices_update(start, pd.Timestamp(end) - pd.Timedelta(days=1), coverage_file=coverage_file)
        if days:
            fetch_and_store_prices(days, root=root, coverage_file=coverage_file, securities_root=securities_root,
                                   fetch=fetch)
    return read_prices(start, end, cusips=cusips, root=root)
