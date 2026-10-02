"""Step 1b, macro releases - ``backfill_daily_releases``: every published vintage of each
release in ``MACRO_RELEASES`` that has a free source, and of the ALFRED inflation components
(``INFLATION_ALFRED_SERIES``, monthly), through ``infra.pipeline.releases``
(no fetch/storage code here - only which series and days), plus the step's checks.

Registered in ``infra.cycle.raw.RAW_SOURCES`` under ``"releases"``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from infra.config import MACRO_RELEASES, MacroRelease
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import releases as prel

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)

# A series whose latest publication is older than this (calendar days, by observation
# frequency) is reported stale (warn): the longest normal gap between two publications
# of a series - GDP's advance/second/third estimates leave at most ~2 months; a government
# shutdown (Oct-Nov 2025) legitimately trips it.
STALE_DAYS = {"W": 14, "M": 45, "Q": 100}


def plan_daily_releases(
    start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
    releases: dict[str, MacroRelease] | None = None,
) -> dict[str, tuple[str, list]]:
    """``{series_id: (source, ranges)}`` the step WOULD fetch over ``[start, end]``
    (inclusive days). No network - also the dry run's view."""
    paths = paths or CyclePaths.default()
    releases = MACRO_RELEASES if releases is None else releases
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    plan = {}
    for sid, source in prel.all_series_to_fetch(releases).items():
        ranges = prel.plan_release_update(sid, source, start, end + _ONE_DAY, coverage_file=paths.releases_coverage,
                                          force_refetch=force_refetch)
        if ranges:
            plan[sid] = (source, ranges)
    return plan


def backfill_daily_releases(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    releases: dict[str, MacroRelease] | None = None,
    sources=None,
) -> dict:
    """Vintages published over ``[start, end]`` (inclusive) for every available release;
    a series never fetched before gets its whole history (``plan_release_update``).
    Per-series failures are collected, not raised."""
    paths = paths or CyclePaths.default()
    releases = MACRO_RELEASES if releases is None else releases
    plan = plan_daily_releases(start, end, paths=paths, force_refetch=force_refetch, releases=releases)
    # a source whose credentials aren't configured is SKIPPED with a warning (check
    # releases_configured), not failed: a missing key must not block the rest of the
    # cycle's vintage every day until it's added
    unconfigured = sorted({src for src, _ in plan.values() if not prel.CONFIGURED.get(src, lambda: True)()})
    plan = {sid: entry for sid, entry in plan.items() if entry[0] not in unconfigured}
    rows, errors = 0, {}
    for sid, entry in plan.items():
        try:
            fetched = prel.fetch_releases_raw({sid: entry}, sources=sources)
            rows += prel.store_releases_raw(fetched, root=paths.releases_dir, coverage_file=paths.releases_coverage)
        except Exception as exc:
            errors[sid] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            log.warning("release fetch failed for %s: %s", sid, errors[sid])
    series = sorted(prel.all_series_to_fetch(releases))
    return {"series": series, "releases": releases, "planned": sorted(plan), "rows": rows, "fetch_errors": errors,
            "unconfigured": unconfigured}


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext) -> dict:
    return ctx.output.get("sources", {}).get("releases", {})


def _series(ctx: StepContext) -> list[str]:
    return _out(ctx).get("series", [])


def _check_fetch_ok(ctx: StepContext):
    errors = _out(ctx).get("fetch_errors", {})
    if not errors:
        return True, f"{len(_out(ctx).get('planned', []))} series fetched, {_out(ctx).get('rows', 0)} new values", None
    details = pd.DataFrame({"series": list(errors), "error": list(errors.values())})
    return False, f"{len(errors)} series fetch error(s): {', '.join(errors)}", details


def _check_configured(ctx: StepContext):
    missing = _out(ctx).get("unconfigured", [])
    if not missing:
        return True, "every source configured", None
    return False, (f"source(s) not configured, NOT fetched: {', '.join(missing)} "
                   "(fred: set FRED_API_KEY in .env)"), None


def _check_sane(ctx: StepContext):
    """Every value published in the window is finite."""
    df = prel.read_releases_from_disk(_series(ctx), ctx.end, start=ctx.start, root=ctx.paths.releases_dir)
    bad = df[~np.isfinite(df["value"])]
    if bad.empty:
        return True, f"{len(df)} value(s) published in the window, all finite", None
    return False, f"{len(bad)} non-finite value(s)", bad


def _check_fresh(ctx: StepContext):
    """Warning: a series' latest publication is older than STALE_DAYS for its frequency."""
    releases: dict[str, MacroRelease] = _out(ctx).get("releases", {})
    freq = {r.series_id: r.frequency for r in releases.values() if r.available}
    lookback = pd.Timedelta(days=max(STALE_DAYS.values()) + 1)
    df = prel.read_releases_from_disk(_series(ctx), ctx.end, start=ctx.end - lookback, root=ctx.paths.releases_dir)
    latest = df.groupby("ticker", observed=True)["timestamp"].max()
    stale = []
    for sid in _series(ctx):
        last = latest.get(sid)
        if last is None or (ctx.end - last).days > STALE_DAYS[freq.get(sid, "M")]:
            stale.append((sid, freq.get(sid, "M"), last))
    if not stale:
        return True, "every series published within its expected gap", None
    return False, f"{len(stale)} stale series: {', '.join(s for s, _, _ in stale)}", \
        pd.DataFrame(stale, columns=["series", "frequency", "latest"])


# Release-calendar-driven presence: a FRED-dated release due in the window must have
# produced a new vintage of its series. Judged only once the release is DUE_GRACE old (FRED
# posts within hours) and only while the release calendar is fresh (refreshed within
# CALENDAR_STALE_DAYS - it is not in the daily cycle yet; a stale one would raise false
# alarms). Calendar-sourced series have no live feed yet and are not judged.
DUE_GRACE = pd.Timedelta(hours=12)
CALENDAR_STALE_DAYS = 3
# Only a series that historically updated on its FRED release's dates is judged: some
# release ids carry two publications a month (FRED release 27, New Residential
# Construction: HOUST updates on one of its two monthly dates - found 2026-10-01 as a
# false alarm). Measured over the past DUE_HISTORY_DAYS.
DUE_MIN_HISTORY_HIT = 0.9
DUE_HISTORY_DAYS = 730


def due_releases(start, end, *, paths: CyclePaths, releases: dict[str, MacroRelease] | None = None,
                 now: pd.Timestamp | None = None) -> tuple[pd.DataFrame | None, str]:
    """``(table, note)``: every FRED-dated release of a FRED-sourced series due in
    ``[start, end]`` (inclusive days) and already DUE_GRACE old, with ``arrived`` = a vintage
    of the series published on or after the release day; ``None`` + why when not judged."""
    from infra.pipeline.release_calendar import FRED_SOURCE, read_release_calendar
    from infra.reference.events import SERIES

    releases = MACRO_RELEASES if releases is None else releases
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    cal = read_release_calendar(root=paths.release_calendar_dir)
    fred = cal[cal["source"] == FRED_SOURCE]
    if fred.empty:
        return None, "no release calendar on disk"
    if fred["last_seen"].max() < now.normalize() - pd.Timedelta(days=CALENDAR_STALE_DAYS):
        return None, f"release calendar stale (last refreshed {fred['last_seen'].max().date()}), not judged"
    due = fred[(fred["timestamp"] >= start) & (fred["timestamp"] < end + _ONE_DAY) & (fred["timestamp"] + DUE_GRACE <= now)]
    sid_of = {r.ticker: r.series_id for r in releases.values() if (r.source or "").startswith("fred")}
    rows, tied = [], {}
    for d in due.itertuples():
        for s in SERIES.values():
            if s.event != d.event or s.bbg_ticker not in sid_of:
                continue
            sid = sid_of[s.bbg_ticker]
            if (d.event, sid) not in tied:
                past = fred[(fred["event"] == d.event) & (fred["timestamp"] < start)
                            & (fred["timestamp"] >= start - pd.Timedelta(days=DUE_HISTORY_DAYS))]
                pub = set(prel.read_releases_from_disk([sid], root=paths.releases_dir)["timestamp"].dt.normalize())
                days = set(past["timestamp"].dt.normalize())
                tied[(d.event, sid)] = bool(days) and len(days & pub) / len(days) >= DUE_MIN_HISTORY_HIT
            if not tied[(d.event, sid)]:
                continue
            got = prel.read_releases_from_disk([sid], start=d.timestamp.normalize(), root=paths.releases_dir)
            rows.append((d.timestamp, d.event, sid, not got.empty))
    return pd.DataFrame(rows, columns=["due", "event", "series", "arrived"]), ""


def _check_due(ctx: StepContext):
    table, note = due_releases(ctx.start, ctx.end, paths=ctx.paths, releases=_out(ctx).get("releases"))
    if table is None:
        return True, note, None
    missing = table[~table["arrived"]]
    if missing.empty:
        return True, f"all {len(table)} due release(s) in the window arrived", None
    return False, f"{len(missing)} due release(s) with no new vintage: {', '.join(missing['series'])}", missing


RELEASE_CHECKS = (
    Check("releases_configured", _check_configured, severity=Severity.WARN),
    Check("releases_fetch_ok", _check_fetch_ok),
    Check("releases_sane", _check_sane),
    Check("releases_fresh", _check_fresh, severity=Severity.WARN),
    Check("releases_due", _check_due, severity=Severity.WARN),
    # A stored row is one PUBLISHED vintage: it should never change or disappear later.
    revision_check(lambda p: p.releases_dir, ["timestamp", "ticker", "period"], name="releases_no_revisions",
                   equals_in=lambda ctx: {"ticker": _series(ctx)}),
)
