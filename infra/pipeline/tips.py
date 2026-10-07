"""TIPS prices, index ratios and real yields per CUSIP (root CLAUDE.md 18; pure maths
``infra.processing.tips``). Store ``Daily/TipsPrices`` (keys ``timestamp``, ``cusip``;
coverage ``Daily/_coverage/tips_prices.parquet``), from FedInvest's daily page - the SAME
page ``infra.pipeline.treasury_prices`` fetches for the nominal securities: the daily
cycle stores both from one request (``store_tips_day`` as its page hook); the history was
re-fetched once by ``backfill_tips``. Reference from the auctions store (original issues),
reference CPI from the stored CPI-U NSA (``release:CPIAUCNS``). A day is covered only once
END OF DAY is posted (as the nominal prices).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from infra.config import (DAILY_TIPS_PRICES_COVERAGE_FILE, DAILY_TIPS_PRICES_DIR, TREASURY_PRICES_SETTLE_DAYS,
                          TREASURY_PRICES_START)
from infra.coverage.intervals import find_missing_ranges, to_utc_day
from infra.processing import tips as tp
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)
KEY = "fedinvest_tips"
_ONE_DAY = pd.Timedelta(days=1)


def tips_reference(*, auctions_root=None) -> pd.DataFrame:
    from infra.pipeline.tsy_auctions import read_auctions
    kw = {} if auctions_root is None else {"root": auctions_root}
    return tp.reference(read_auctions(nominal_only=False, **kw))


def cpi_nsa(*, releases_root=None) -> pd.Series:
    from infra.pipeline.series_panel import read_panel
    kw = {} if releases_root is None else {"roots": {"release": releases_root}}
    p = read_panel(["release:CPIAUCNS"], "1990-01-01", pd.Timestamp.now().normalize() + pd.Timedelta(days=60), **kw)
    return p["release:CPIAUCNS"].dropna()


def plan_tips_update(start, end, *, now=None, coverage_file: Path = DAILY_TIPS_PRICES_COVERAGE_FILE) -> list[pd.Timestamp]:
    today = to_utc_day(pd.Timestamp.now(tz="UTC") if now is None else now)
    start = max(pd.Timestamp(start).normalize(), pd.Timestamp(TREASURY_PRICES_START))
    end = min(pd.Timestamp(end).normalize(), today - _ONE_DAY)
    if end < start:
        return []
    gaps = find_missing_ranges((start, end + _ONE_DAY), coverage_store.read_covered(coverage_file, KEY))
    return [d for d in pd.bdate_range(start, end) if any(g0 <= d < g1 for g0, g1 in gaps)]


def store_tips_day(day, page: pd.DataFrame, ref: pd.DataFrame, cpi: pd.Series, *, now=None,
                   root: Path = DAILY_TIPS_PRICES_DIR, coverage_file: Path = DAILY_TIPS_PRICES_COVERAGE_FILE) -> str:
    """FILES ONLY: one fetched page's TIPS rows -> stored and covered if END OF DAY is
    posted; ``"holiday"`` (an old empty page, covered) or ``"pending"`` otherwise."""
    day = pd.Timestamp(day).normalize()
    today = to_utc_day(pd.Timestamp.now(tz="UTC") if now is None else now)
    rows = tp.prices(page, day, ref, cpi) if len(page) else pd.DataFrame(columns=tp.PRICE_COLUMNS)
    if rows.empty:
        # a posted page without TIPS rows (or an old empty page - a holiday) has nothing to store
        eod = pd.to_numeric(page["end_of_day"], errors="coerce") if len(page) and "end_of_day" in page else pd.Series(dtype=float)
        if (len(eod) and (eod > 0).mean() >= 0.5) or (today - day).days > TREASURY_PRICES_SETTLE_DAYS:
            coverage_store.record_covered(coverage_file, KEY, [(day, day + _ONE_DAY)])
            return "holiday"
        return "pending"
    if rows["price_eod"].notna().mean() < 0.5:  # END OF DAY not posted yet
        return "pending"
    parquet_store.write_partitioned(tp.encode(rows), root, tp.PRICE_KEYS)
    coverage_store.record_covered(coverage_file, KEY, [(day, day + _ONE_DAY)])
    return "stored"


def backfill_tips(start, end, *, newest_first: bool = True, fetch=None, sleep=time.sleep, pause_s: float = 0.5,
                  root: Path = DAILY_TIPS_PRICES_DIR, coverage_file: Path = DAILY_TIPS_PRICES_COVERAGE_FILE) -> dict:
    """Fetch FedInvest's page for every uncovered business day in ``[start, end]`` and
    store its TIPS rows (free; ~3s a day). Per-day failures collected, not raised."""
    from infra.pipeline import treasury_prices
    fetch = treasury_prices.FETCH if fetch is None else fetch
    ref, cpi = tips_reference(), cpi_nsa()
    days = plan_tips_update(start, end, coverage_file=coverage_file)
    if newest_first:
        days = days[::-1]
    out = {"stored": 0, "holiday": 0, "pending": 0, "errors": {}}
    t0 = time.time()
    for i, day in enumerate(days):
        if i:
            sleep(pause_s)
        try:
            out[store_tips_day(day, fetch(day), ref, cpi, root=root, coverage_file=coverage_file)] += 1
        except Exception as exc:
            out["errors"][str(day.date())] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
        if i % 100 == 0:
            log.info("tips backfill %d/%d (%s) %.0fs", i, len(days), day.date(), time.time() - t0)
    return out


def read_tips(start=None, end=None, *, cusips=None, root: Path = DAILY_TIPS_PRICES_DIR) -> pd.DataFrame:
    raw = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                         end=None if end is None else pd.Timestamp(end),
                                         equals_in={"cusip": list(cusips)} if cusips is not None else None)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tp.PRICE_COLUMNS)
    return tp.decode(raw[tp.PRICE_COLUMNS]).sort_values(tp.PRICE_KEYS).reset_index(drop=True)
