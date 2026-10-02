"""US Treasury auctions: plan / fetch / store / read (Fiscal Data, ``auctions_query``) and
their release-calendar rows. Same 4-function shape as the other raw pipelines.

Coverage (``RawData/_coverage/tsy_auctions.parquet``, key ``"auctions"``) is on the
AUCTION-DATE axis and claimed only up to the earliest RECENT auction still without
results: an announced auction is re-fetched until it has been held and its results
published, and a held one is never asked again (Rule 2.1). One more than
UNHELD_AFTER_DAYS past with no results was never held (the 4-week bill of 2001-09-11) and
doesn't hold coverage back. NOT yet in the daily cycle.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.api import fiscaldata_client
from infra.config import TSY_AUCTIONS_COVERAGE_FILE, TSY_AUCTIONS_DIR
from infra.coverage.intervals import merge_intervals
from infra.pipeline import release_calendar as prc
from infra.processing import tsy_auctions as ta
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)
KEY = "auctions"
HISTORY_START = pd.Timestamp("1979-01-01")
UNHELD_AFTER_DAYS = 14


def plan_auctions_update(*, coverage_file: Path = TSY_AUCTIONS_COVERAGE_FILE) -> pd.Timestamp:
    """The first auction date to (re-)fetch: where the contiguous held-auction coverage ends."""
    for s, e in merge_intervals(coverage_store.read_covered(coverage_file, KEY)):
        if s <= HISTORY_START:
            return e
    return HISTORY_START


def store_auctions(records: pd.DataFrame, *, root: Path = TSY_AUCTIONS_DIR,
                   coverage_file: Path = TSY_AUCTIONS_COVERAGE_FILE, since: pd.Timestamp = HISTORY_START) -> int:
    """FILES ONLY: upsert parsed auctions; cover auction dates up to the first one not yet held."""
    df = ta.parse(records)
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, ta.KEYS)
    # an auction this far past with no results was never held (seen: the 4-week bill of
    # 2001-09-11) - it must not pin the coverage frontier and force daily re-fetches
    recent = df["auction_date"] >= pd.Timestamp.now().normalize() - pd.Timedelta(days=UNHELD_AFTER_DAYS)
    pending = df.loc[~ta.held(df) & recent, "auction_date"]
    covered_to = pending.min() if len(pending) else df["auction_date"].max() + pd.Timedelta(days=1)
    if covered_to > since:
        coverage_store.record_covered(coverage_file, KEY, [(since, covered_to)])
    return len(df)


def update_auctions(*, root: Path = TSY_AUCTIONS_DIR, coverage_file: Path = TSY_AUCTIONS_COVERAGE_FILE,
                    fetch=fiscaldata_client.fetch_auctions, calendar_root: Path | None = None, observed=None) -> int:
    """Parent: fetch every auction from the first not-yet-held one onward (history on the
    first run), store it, and fold the nominal-coupon auctions into the release calendar."""
    since = plan_auctions_update(coverage_file=coverage_file)
    records = fetch(since)
    n = store_auctions(records, root=root, coverage_file=coverage_file, since=since)
    observed = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() if observed is None else observed
    prc.store_observation(ta.calendar_rows(ta.parse(records), observed),
                          **({"root": calendar_root} if calendar_root else {}))
    log.info("treasury auctions since %s: %d stored", since.date(), n)
    return n


def read_auctions(as_of=None, *, nominal_only: bool = True, start=None, end=None,
                  root: Path = TSY_AUCTIONS_DIR) -> pd.DataFrame:
    """Stored auctions (close instant in ``[start, end)``), nominal 2y-30y coupons by
    default; with ``as_of`` (an instant, UTC) only what was known then (results blanked
    for auctions not yet closed). No network."""
    raw = parquet_store.read_partitioned(root, start=start, end=end)
    if raw is None or raw.empty:
        return ta.empty()
    df = raw[ta.COLUMNS].sort_values(ta.KEYS).reset_index(drop=True)
    df = ta.nominal_coupons(df) if nominal_only else df
    return ta.as_of(df, as_of) if as_of is not None else df
