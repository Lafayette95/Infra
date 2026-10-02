"""Economic-calendar history from archived pages (``CALENDAR_PAGES``): plan / select /
harvest / store / read, plus ``calendar_vintages`` - the ``"calendar"`` source of
infra.pipeline.releases, so calendar-sourced releases (ISM, S&P Global PMIs, ...) land in
the SAME vintage store as the FRED ones and the nowcast reads them unchanged.

* **Store** ``~/Database/RawData/EconCalendar``: one row per (release day, report name,
  period label) - the calendar's own row, values parsed AND as shown (infra.processing.
  econ_calendar). Hive ``year=/quarter=`` on the release day.
* **Coverage** (``RawData/_coverage/econ_calendar.parquet``, keyed by page) is on the
  CAPTURE-day axis: "every archived capture of these days has been processed". Rule 2.1:
  a covered day is never re-fetched. Only SETTLED days (``SETTLE_DAYS`` old) are ever
  claimed: the archive can index a capture days late, so recent days are re-processed by
  the next run until they settle.
* **One capture per day** (``select_captures``): the day's LATEST capture of the
  highest-priority URL variant - late captures carry that day's actual values. A week
  shows on several days' pages, so a missed day costs little.
* **Merging** (``merge_rows``): the same calendar row appears on many captures, and its
  ``previous`` CHANGES at release (pre-release pages show the old prior value, post-
  release pages the revised one; seen 2026-10-01 on the 2020-04-28 consumer confidence
  row: 120.0 before, 118.8 after). So per key, a row captured AFTER its release
  (actual present) beats one captured before, then the later capture wins - a stale page
  archived years later never overwrites a post-release row.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from infra.api import wayback_client
from infra.config import CALENDAR_COVERAGE_FILE, CALENDAR_DIR, CALENDAR_PAGES, CalendarPage
from infra.coverage.intervals import Interval, find_missing_ranges, merge_intervals, to_utc_day
from infra.processing import econ_calendar as ec
from infra.reference.events import EVENTS, SERIES
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)
PAUSE_S = 3.0  # between page fetches - the archive is a shared, free service
# A capture day is only claimed covered once it is this many days old: the archive can
# index a capture days after it was taken, so a recent day processed today is processed
# AGAIN by the next run (re-merging is idempotent - merge_rows) until it is settled.
SETTLE_DAYS = 14
FLUSH_EVERY = 10  # pages between writes (and coverage records), so a harvest is resumable


def plan_harvest(page: CalendarPage, start, end, *, coverage_file: Path = CALENDAR_COVERAGE_FILE) -> list[Interval]:
    """Capture-day ranges of ``[start, end)`` not yet processed (past days only)."""
    start = max(to_utc_day(start), pd.Timestamp(page.history_start))
    end = to_utc_day(end)
    if start >= end:
        return []
    return find_missing_ranges((start, end), coverage_store.read_covered(coverage_file, page.key))


def list_page_captures(page: CalendarPage, start, end, *, list_fn=None) -> pd.DataFrame:
    """Every capture of every variant of ``page`` in ``[start, end)``, with its variant's
    priority (0 = best); captures of URLs matching no variant are dropped."""
    list_fn = list_fn or wayback_client.list_captures
    caps = pd.concat([list_fn(p, start, end) for p in page.cdx_prefixes], ignore_index=True)
    if caps.empty:
        return caps.assign(priority=pd.Series(dtype=int))

    def priority(url: str) -> int:
        return next((i for i, v in enumerate(page.variants) if re.search(v, url)), -1)

    caps["priority"] = caps["original"].map(priority)
    return caps[caps["priority"] >= 0].drop_duplicates(subset=["timestamp", "original"]).reset_index(drop=True)


def select_captures(caps: pd.DataFrame) -> pd.DataFrame:
    """One capture per UTC day: the best-priority variant, then the day's latest."""
    if caps.empty:
        return caps
    caps = caps.assign(day=caps["timestamp"].dt.normalize())
    caps = caps.sort_values(["day", "priority", "timestamp"], ascending=[True, True, False])
    return caps.groupby("day", sort=True).head(1).sort_values("timestamp").reset_index(drop=True)


def merge_rows(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Per key: a post-release row (actual present) beats a pre-release one, then the
    later capture wins (module doc)."""
    both = pd.concat([f for f in (existing, incoming) if not f.empty], ignore_index=True) \
        if not (existing.empty and incoming.empty) else ec.empty_rows()
    if both.empty:
        return both
    both = both.assign(_post=both["actual"].notna())
    both = both.sort_values(["_post", "capture"], kind="stable")
    return both.groupby(ec.ROW_KEYS, sort=True).tail(1).drop(columns="_post").reset_index(drop=True)


def store_rows(rows: pd.DataFrame, *, root: Path = CALENDAR_DIR) -> int:
    """FILES ONLY: merge ``rows`` with what is stored for the same keys, write."""
    if rows.empty:
        return 0
    lo, hi = rows["timestamp"].min(), rows["timestamp"].max() + _ONE_DAY
    existing = parquet_store.read_partitioned(root, start=lo, end=hi)
    existing = ec.empty_rows() if existing is None else existing[ec.ROW_COLUMNS]
    keys = rows[ec.ROW_KEYS].drop_duplicates()
    existing = existing.merge(keys, on=ec.ROW_KEYS, how="inner") if not existing.empty else existing
    merged = merge_rows(ec.encode_rows(existing) if not existing.empty else existing, ec.encode_rows(rows))
    parquet_store.write_partitioned(ec.encode_rows(merged), root, ec.ROW_KEYS)
    return len(merged)


def harvest(
    page_key: str,
    start,
    end,
    *,
    root: Path = CALENDAR_DIR,
    coverage_file: Path = CALENDAR_COVERAGE_FILE,
    list_fn=None,
    fetch_fn=None,
    pause_s: float = PAUSE_S,
    pages: dict[str, CalendarPage] = CALENDAR_PAGES,
) -> dict:
    """Process every not-yet-covered capture day in ``[start, end)``: list captures,
    pick one per day, fetch, parse, merge, store - writing and recording coverage every
    ``FLUSH_EVERY`` pages so an interrupted harvest resumes where it stopped. A page that
    fails to fetch or parse is logged and SKIPPED (its day stays covered - the week shows
    on neighbouring days' pages too); the count is returned."""
    page = pages[page_key]
    fetch_fn = fetch_fn or wayback_client.fetch_capture
    stats = {"days": 0, "pages": 0, "rows": 0, "failed": []}
    for g0, g1 in plan_harvest(page, start, end, coverage_file=coverage_file):
        chosen = select_captures(list_page_captures(page, g0, g1, list_fn=list_fn))
        buffer, cursor = [], g0
        for i, cap in enumerate(chosen.itertuples()):
            try:
                html = ec.decode_html(fetch_fn(cap.timestamp, cap.original))
                buffer.append(ec.parse_page(html, cap.timestamp, cap.original))
            except Exception as exc:  # one bad page never stops a multi-hour harvest
                stats["failed"].append((cap.timestamp, f"{type(exc).__name__}: {exc}"))
                log.warning("calendar capture %s failed: %s", cap.timestamp, exc)
            stats["pages"] += 1
            if (i + 1) % FLUSH_EVERY == 0:
                cursor = _flush(buffer, cap.day + _ONE_DAY, cursor, page, root, coverage_file, stats)
                buffer = []
                log.info("calendar %s: through %s (%d pages, %d rows)", page.key, cap.day.date(), stats["pages"],
                         stats["rows"])
            if pause_s:
                time.sleep(pause_s)
        _flush(buffer, g1, cursor, page, root, coverage_file, stats)
        stats["days"] += (g1 - g0).days
    return stats


def settled_end(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Exclusive end of the capture days old enough to claim covered (``SETTLE_DAYS``)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    return now.normalize() - pd.Timedelta(days=SETTLE_DAYS)


def _flush(buffer, through, since, page, root, coverage_file, stats) -> pd.Timestamp:
    frames = [b for b in buffer if not b.empty]
    if frames:
        stats["rows"] += store_rows(pd.concat(frames, ignore_index=True), root=root)
    through = min(through, settled_end())
    if through > since:
        coverage_store.record_covered(coverage_file, page.key, [(since, through)])
        return through
    return since


def read_calendar_from_disk(start=None, end=None, *, root: Path = CALENDAR_DIR) -> pd.DataFrame:
    """Stored calendar rows with release day in ``[start, end)``. No network."""
    raw = parquet_store.read_partitioned(root, start=start, end=end)
    if raw is None or raw.empty:
        return ec.empty_rows()
    return raw[ec.ROW_COLUMNS].sort_values(["timestamp", "report"]).reset_index(drop=True)


def harvested_through(page: CalendarPage, *, coverage_file: Path = CALENDAR_COVERAGE_FILE) -> pd.Timestamp | None:
    """End of the processed capture days that run contiguously from the page's start."""
    first = pd.Timestamp(page.history_start)
    for s, e in merge_intervals(coverage_store.read_covered(coverage_file, page.key)):
        if s <= first:
            return e
    return None


def _series_by_store_id(store_id: str, series: dict | None = None):
    series = SERIES if series is None else series
    return next(s_ for s_ in series.values() if s_.store_id == store_id)


def _staged(s_) -> bool:
    """Published as a preliminary (flash) and then a final estimate (the registry's event stages)."""
    ev = EVENTS.get(s_.event)
    return ev is not None and any(st in ("flash", "preliminary") for st in ev.stages)


def calendar_vintages(series_id: str, start, end, *, root: Path = CALENDAR_DIR,
                      coverage_file: Path = CALENDAR_COVERAGE_FILE, series=None, page_key: str = "marketwatch"):
    """The ``"calendar"`` source of infra.pipeline.releases (``fn(series_id, start, end)``):
    the release's vintages from the stored calendar rows - local, no network - and the
    publication days genuinely covered: up to where the harvest has reached."""
    s_ = _series_by_store_id(series_id, series)
    # an event published in flash/preliminary -> final stages (the S&P PMIs) takes actuals
    # only - its final's "previous" is the same period's flash (ec.release_vintages)
    df = ec.release_vintages(read_calendar_from_disk(root=root), s_.calendar_pattern,
                             use_previous=not _staged(s_), frequency=s_.frequency, scale=s_.calendar_scale)
    df = df[df["realtime_start"] < pd.Timestamp(end)].reset_index(drop=True)
    frontier = harvested_through(CALENDAR_PAGES[page_key], coverage_file=coverage_file)
    covered_end = min(pd.Timestamp(end), frontier) if frontier is not None else pd.Timestamp(start)
    return df, ([(pd.Timestamp(start), covered_end)] if covered_end > pd.Timestamp(start) else [])


PRELIM_LAST_DAY = 20  # an unlabelled row released by this day of its month is a preliminary


def calendar_prelims(series_id: str, end, *, root: Path = CALENDAR_DIR, series=None) -> pd.DataFrame:
    """The calendar's PRELIMINARY prints of a release FRED only carries as finals (UMich:
    FRED has the end-of-month final, the calendar also the mid-month preliminary;
    verified 2026-10-01, 75 calendar prints FRED never had). ``realtime_start, date,
    value`` vintages, typo-filtered. A row is preliminary when its name says so
    ("(preliminary)", "prelim"), never when it says final, and otherwise when released
    by PRELIM_LAST_DAY of its month (preliminaries mid-month, finals in the last week)."""
    s_ = _series_by_store_id(series_id, series)
    rows = read_calendar_from_disk(root=root)
    name = rows["report"].str.lower()
    prelim = ~name.str.contains("final") & (name.str.contains("prelim") | (rows["timestamp"].dt.day <= PRELIM_LAST_DAY))
    v = ec.release_vintages(rows[prelim], s_.calendar_pattern, use_previous=False, frequency=s_.frequency,
                            scale=s_.calendar_scale)
    return v[v["realtime_start"] < pd.Timestamp(end)].reset_index(drop=True)


def consensus(release, as_of=None, *, root: Path = CALENDAR_DIR) -> pd.DataFrame:
    """The calendar's CONSENSUS for one release, next to what printed: ``timestamp``
    (release day), ``period``, ``report``, ``consensus_mw``, ``actual``, ``surprise``
    (actual - consensus), all in the release's own units (``calendar_scale``).

    ``consensus_mw`` is MarketWatch's median forecast as shown on the calendar - NOT
    Bloomberg's survey (the table's ``BbgMedian``); usually close, never the same panel.
    Point in time: rows released by the end of ``as_of`` (a consensus is known BEFORE its
    release day, the actual on it). Typo-flagged actuals (ec.suspect_prints) are NaN."""
    rows = read_calendar_from_disk(root=root, end=None if as_of is None else to_utc_day(as_of) + _ONE_DAY)
    mine = rows[rows["report"].map(lambda r: ec.matches(r, release.calendar_pattern))].copy()
    mine["period_start"] = [ec.period_of(lbl, d, release.frequency) for lbl, d in zip(mine["period"], mine["timestamp"])]
    mine = mine.dropna(subset=["period_start"])
    sus = ec.suspect_prints(rows, release.calendar_pattern, frequency=release.frequency, staged=release.preliminary)
    bad = set(zip(sus["timestamp"], sus["period"]))
    actual = [np.nan if (d, p) in bad else a for d, p, a in zip(mine["timestamp"], mine["period_start"], mine["actual"])]
    out = pd.DataFrame({
        "timestamp": mine["timestamp"].to_numpy(), "period": mine["period_start"].to_numpy(),
        "report": mine["report"].to_numpy(),
        "consensus_mw": mine["forecast"].to_numpy() * release.calendar_scale,
        "actual": np.asarray(actual, dtype=float) * release.calendar_scale,
    })
    out["surprise"] = out["actual"] - out["consensus_mw"]
    return out.sort_values(["timestamp", "period"]).reset_index(drop=True)


# ------------------------------------------------------------------ cross-check
CROSSCHECK_SLACK = 0.011  # on top of the calendar's own rounding: FRED levels are rounded too


def crosscheck(release, *, since=None, root: Path = CALENDAR_DIR, releases_root: Path | None = None) -> pd.DataFrame:
    """Every calendar ACTUAL of a FRED-sourced release vs FRED's value for the same period
    AS PUBLISHED ON THE SAME DAY, in the release's units (infra.processing.releases).
    One row per calendar print, ``status``:

    * ``match`` - same value (within the calendar's displayed rounding) on the same day:
      the calendar's parsing, period, units AND release day are all right;
    * ``mismatch`` - FRED had that period on that day, with a different value;
    * ``not_on_fred`` - FRED did not have the period yet on that day (a wrong calendar
      day, or FRED publishing late);
    * ``before_vintages`` - the day precedes FRED's vintage archive: not comparable.

    For a ``fred+prelims`` release (UMich) only the FINALS are checked: its preliminaries
    are calendar data in the store already."""
    from infra.pipeline.releases import read_releases_from_disk  # releases imports this module
    from infra.processing import releases as pr
    from infra.config import RELEASES_DIR

    rows = read_calendar_from_disk(start=since, root=root)
    rows = rows[rows["report"].map(lambda r: ec.matches(r, release.calendar_pattern)) & rows["actual"].notna()]
    if release.source == "fred+prelims":  # its prelims ARE calendar data in the store: comparing is circular
        name = rows["report"].str.lower()
        rows = rows[name.str.contains("final") | (~name.str.contains("prelim") & (rows["timestamp"].dt.day > PRELIM_LAST_DAY))]
    raw = read_releases_from_disk([release.series_id], root=releases_root or RELEASES_DIR)
    first_vintage = raw["timestamp"].min() if not raw.empty else None
    out = []
    for r in rows.itertuples():
        period = ec.period_of(r.period, r.timestamp, release.frequency)
        if period is None:
            continue
        cal = r.actual * release.calendar_scale
        tol = ec.resolution(r.actual_raw) * release.calendar_scale + CROSSCHECK_SLACK
        if first_vintage is None or r.timestamp < first_vintage:
            out.append((r.timestamp, r.report, period, cal, np.nan, "before_vintages"))
            continue
        fred = pr.derive_units(pr.snapshot(raw, r.timestamp), release.units, release.frequency)
        hit = fred.loc[fred["period"] == period, "value"]
        if hit.empty:
            out.append((r.timestamp, r.report, period, cal, np.nan, "not_on_fred"))
        else:
            v = float(hit.iloc[0])
            out.append((r.timestamp, r.report, period, cal, v, "match" if abs(cal - v) <= tol else "mismatch"))
    return pd.DataFrame(out, columns=["timestamp", "report", "period", "calendar", "fred", "status"])


def upcoming_consensus(release, as_of, until, *, root: Path = CALENDAR_DIR) -> pd.DataFrame:
    """The consensus for prints of ``release`` due AFTER ``as_of`` (through ``until``), as
    the calendar showed it on a capture made by the end of ``as_of`` (the next-week table):
    ``timestamp`` (release day), ``period``, ``consensus_mw`` (release units)."""
    as_of = to_utc_day(as_of)
    rows = read_calendar_from_disk(start=as_of + _ONE_DAY, end=to_utc_day(until) + _ONE_DAY, root=root)
    rows = rows[(rows["capture"] < as_of + _ONE_DAY) & rows["forecast"].notna()
                & rows["report"].map(lambda r: ec.matches(r, release.calendar_pattern))]
    period = [ec.period_of(lbl, d, release.frequency) for lbl, d in zip(rows["period"], rows["timestamp"])]
    out = pd.DataFrame({"timestamp": rows["timestamp"].to_numpy(), "period": period,
                        "consensus_mw": rows["forecast"].to_numpy() * release.calendar_scale})
    return out.dropna(subset=["period"]).reset_index(drop=True)
