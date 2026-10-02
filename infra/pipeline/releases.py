"""Macro-release vintages (``MACRO_RELEASES``): read / plan / fetch / store + a ``load``
parent - the same 4-function shape as infra/pipeline/bonds.py, on the same generic
primitives (parquet_store, coverage_store), pointed at ``Database/RawData/Releases``.

What differs from the price stores, all from storing VINTAGES (infra.processing.releases):

* coverage is on the PUBLICATION-day axis, keyed by source series id: "every value
  published on these days is on disk". It must stay CONTIGUOUS from the source's epoch:
  a request starting at day X also returns values published earlier and still current
  at X (dated X, or clipped to X), which only ``drop_unchanged`` can tell apart from a
  genuine publication on X - and only if everything published before X is already
  stored. So a series with no coverage is bootstrapped from the epoch (its whole
  history, every vintage), and later requests always start at the end of the contiguous
  coverage (or earlier, with ``force_refetch``), never leave a hole;
* sources are free (FRED needs a free key) - Rule 2.1 still applies: a covered day is
  never re-asked, except by ``force_refetch`` (the scheduled run's revision window).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import fred_client
from infra.config import (CALENDAR_PAGES, INFLATION_ALFRED_SERIES, MACRO_RELEASES, RELEASES_COVERAGE_FILE,
                          RELEASES_DIR, MacroRelease)
from infra.coverage.intervals import Interval, merge_intervals, to_utc_day
from infra.pipeline import econ_calendar
from infra.processing import releases as pr
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)

# source name -> fn(series_id, start, end) -> (frame realtime_start/date/value, covered)
def fred_with_calendar_prelims(series_id: str, start, end):
    """FRED's vintages (the finals, full history) PLUS the calendar's preliminary prints
    as earlier vintages of the same periods - for a release FRED only carries as finals
    (UMich). Coverage is FRED's; the prelims come from what the calendar harvest holds."""
    df, covered = fred_client.fetch_vintages(series_id, start, end)
    prelims = econ_calendar.calendar_prelims(series_id, end)
    return pd.concat([df, prelims], ignore_index=True).sort_values(["date", "realtime_start"]), covered


# "calendar" is LOCAL: vintages derived from the harvested economic-calendar store
# (infra.pipeline.econ_calendar), covered up to where that harvest has reached.
# "fred+prelims": FRED plus the calendar's preliminary prints (fred_with_calendar_prelims).
SOURCES: dict[str, Callable[..., tuple[pd.DataFrame, list[Interval]]]] = {
    "fred": fred_client.fetch_vintages,
    "calendar": econ_calendar.calendar_vintages,
    "fred+prelims": fred_with_calendar_prelims,
}
EPOCHS: dict[str, pd.Timestamp] = {
    "fred": fred_client.EPOCH,
    "calendar": pd.Timestamp(CALENDAR_PAGES["marketwatch"].history_start),
    "fred+prelims": fred_client.EPOCH,
}
# source name -> fn() -> bool: is the source usable here (its credentials configured)?
CONFIGURED: dict[str, Callable[[], bool]] = {"fred": fred_client.has_key, "calendar": lambda: True,
                                             "fred+prelims": fred_client.has_key}


def series_to_fetch(releases: dict[str, MacroRelease] = MACRO_RELEASES) -> dict[str, str]:
    """``{series_id: source}`` for every release with a free source (one series can back
    several releases, it is fetched once)."""
    return {r.series_id: r.source for r in releases.values() if r.available}


# Series fetched and stored the same way but NOT releases of the nowcast table: the ALFRED
# inflation components (INFLATION_ALFRED_SERIES), point-in-time history for inflation work.
EXTRA_SERIES: dict[str, str] = {sid: "fred" for sid in INFLATION_ALFRED_SERIES}


def all_series_to_fetch(releases: dict[str, MacroRelease] = MACRO_RELEASES,
                        extra: dict[str, str] | None = None) -> dict[str, str]:
    """What the fetch paths cover: the releases' series plus ``extra`` (default
    EXTRA_SERIES). Readers of the nowcast's own inputs use ``series_to_fetch``."""
    return {**series_to_fetch(releases), **(EXTRA_SERIES if extra is None else extra)}


def read_releases_from_disk(
    tickers: list[str] | None = None,
    as_of=None,
    *,
    start=None,
    root: Path = RELEASES_DIR,
) -> pd.DataFrame:
    """Decoded raw vintage rows PUBLISHED by the end of day ``as_of`` (inclusive; None =
    everything), optionally from publication day ``start``. No network. Point-in-time by
    construction: a later publication can never leak into an earlier ``as_of``."""
    end = None if as_of is None else to_utc_day(as_of) + pd.Timedelta(days=1)
    raw = parquet_store.read_partitioned(root, start=start, end=end,
                                         equals_in=None if tickers is None else {"ticker": list(tickers)})
    if raw is None or raw.empty:
        return pr.decode_raw(pr.empty_raw())
    return pr.decode_raw(raw)


def _contiguous_end(covered: list[Interval], epoch: pd.Timestamp) -> pd.Timestamp | None:
    """End of the coverage run that starts at the epoch (None = never bootstrapped)."""
    for s, e in merge_intervals(covered):
        if s <= epoch:
            return e
    return None


def plan_release_update(
    series_id: str,
    source: str,
    start,
    end,
    *,
    coverage_file: Path = RELEASES_COVERAGE_FILE,
    force_refetch: bool = False,
) -> list[Interval]:
    """The publication-day range to request so that ``[start, end)`` ends up covered,
    keeping coverage contiguous (module doc): at most ONE interval."""
    start, end = to_utc_day(start), to_utc_day(end)
    epoch = EPOCHS[source]
    covered_to = _contiguous_end(coverage_store.read_covered(coverage_file, series_id), epoch)
    if covered_to is None:
        begin = epoch  # bootstrap: the whole vintage history
    elif force_refetch:
        begin = min(start, covered_to)
    else:
        begin = covered_to
    return [(begin, end)] if begin < end else []


# (series id, range start, range end, vintages frame, covered intervals) per request
Fetched = list[tuple[str, pd.Timestamp, pd.Timestamp, pd.DataFrame, list[Interval]]]


def fetch_releases_raw(plan: dict[str, tuple[str, list[Interval]]], *, sources=None) -> Fetched:
    """NETWORK ONLY: ``plan`` is ``{series_id: (source, ranges)}``. Touches no files."""
    sources = SOURCES if sources is None else sources
    return [(sid, s, e, *sources[src](sid, s, e)) for sid, (src, ranges) in plan.items() for s, e in ranges]


def store_releases_raw(
    fetched: Fetched,
    *,
    root: Path = RELEASES_DIR,
    coverage_file: Path = RELEASES_COVERAGE_FILE,
) -> int:
    """FILES ONLY: keep what is new (``drop_unchanged`` vs what is stored), save it, and
    record only what the source said it covers."""
    rows = 0
    for sid, range_start, range_end, vintages, covered in fetched:
        stored = parquet_store.read_partitioned(root, equals_in={"ticker": [sid]})
        stored = pr.empty_raw() if stored is None else stored
        new = pr.drop_unchanged(pr.from_vintages(vintages, sid), stored)
        parquet_store.write_partitioned(pr.encode_raw(new), root, pr.RAW_KEYS)
        if covered:
            coverage_store.record_covered(coverage_file, sid, covered)
        rows += len(new)
        log.info("releases %s %s->%s: %d new of %d fetched", sid, range_start.date(), range_end.date(),
                 len(new), len(vintages))
    return rows


def load_releases(
    series_ids: list[str],
    as_of=None,
    *,
    fetch_missing: bool = True,
    root: Path = RELEASES_DIR,
    coverage_file: Path = RELEASES_COVERAGE_FILE,
    releases: dict[str, MacroRelease] = MACRO_RELEASES,
    sources=None,
) -> pd.DataFrame:
    """Parent: make sure every vintage published by ``as_of`` is on disk (network only
    for what isn't), then read them point-in-time."""
    if fetch_missing:
        source_of = all_series_to_fetch(releases)
        end = (to_utc_day(pd.Timestamp.now()) if as_of is None else to_utc_day(as_of)) + pd.Timedelta(days=1)
        plan = {}
        for sid in series_ids:
            ranges = plan_release_update(sid, source_of[sid], end - pd.Timedelta(days=1), end,
                                         coverage_file=coverage_file)
            if ranges:
                plan[sid] = (source_of[sid], ranges)
        if plan:
            store_releases_raw(fetch_releases_raw(plan, sources=sources), root=root, coverage_file=coverage_file)
    return read_releases_from_disk(series_ids, as_of, root=root)
