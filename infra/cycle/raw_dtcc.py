"""Step 1b, swap trades - ``backfill_daily_dtcc``: archive every DTCC cumulative report
(``DTCC_REPORTS``) for each day not on disk yet, through ``infra.pipeline.dtcc`` (no
fetch/storage code here), plus the step's checks.

Registered in ``infra.cycle.raw.RAW_SOURCES`` under ``"dtcc"``. DTCC keeps only a rolling
~2 years, so the archive must never fall behind for good: besides the run's own window,
every run re-plans the ``DTCC_RETENTION_DAYS`` before it - free on disk (a directory
listing), and a day missed while the Mac was off is picked up by the next run, long before
it ages out. ``force_refetch`` is ignored: DTCC publishes each day once, complete, and an
archived day is never requested again (Rule 2.1).

Checks are ``warn``-level on purpose: a missed day heals itself on a later run (above), so
it must not cost the day's vintage; only one still missing near the end of DTCC's window is
really at risk, and ``dtcc_complete`` lists every missing day each run until it's archived.
"""
from __future__ import annotations

import csv
import io
import zipfile

import pandas as pd

from infra.config import DTCC_FIRST_DAY, DTCC_REPORTS, DTCC_RETENTION_DAYS
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import dtcc as pdtcc

# DTCC posts day D shortly after midnight UTC; from this long after D ends it must be there
PUBLICATION_GRACE = pd.Timedelta(hours=3)
# Columns any extraction relies on: if DTCC renames one, the archive is still complete
# (raw bytes) but every reader must be updated - dtcc_sane says so the day it happens.
REQUIRED_COLUMNS = ("Dissemination Identifier", "Action type", "Event type", "Execution Timestamp",
                    "Effective Date", "Expiration Date", "Fixed rate-Leg 1", "Fixed rate-Leg 2",
                    "Notional amount-Leg 1", "Package indicator", "UPI FISN", "UPI Underlier Name")


def plan_daily_dtcc(start, end, *, paths: CyclePaths | None = None, now=None) -> dict[str, list[pd.Timestamp]]:
    """``{report kind: days to request}`` - the window plus the retention lookback. No network."""
    paths = paths or CyclePaths.default()
    end = pd.Timestamp(end).normalize()
    first = min(pd.Timestamp(start).normalize(), end - pd.Timedelta(days=DTCC_RETENTION_DAYS))
    return {kind: pdtcc.plan_dtcc_update(kind, first, end, now=now, root=paths.dtcc_dir) for kind in DTCC_REPORTS}


def backfill_daily_dtcc(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                        fetch=None, sleep=None, now=None) -> dict:
    """Archive every planned day (module doc). ``force_refetch`` is unused - kept for the
    raw-source signature."""
    paths = paths or CyclePaths.default()
    out = {}
    for kind, days in plan_daily_dtcc(start, end, paths=paths, now=now).items():
        out[kind] = {"planned": days, **pdtcc.fetch_and_store_dtcc(kind, days, root=paths.dtcc_dir,
                                                                    fetch=fetch, sleep=sleep)}
    return {"reports": out, "now": now}


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext) -> dict:
    return ctx.output.get("sources", {}).get("dtcc", {})


def expected_through(now=None) -> pd.Timestamp:
    """The latest day DTCC must have published by ``now``."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return pdtcc.last_complete_day(now - PUBLICATION_GRACE)


def _check_fetch_ok(ctx: StepContext):
    errors = [(kind, day, msg) for kind, r in _out(ctx).get("reports", {}).items()
              for day, msg in r.get("errors", {}).items()]
    if not errors:
        new = {kind: len(r.get("archived", [])) for kind, r in _out(ctx).get("reports", {}).items()}
        return True, f"days archived: {new}", None
    return False, f"{len(errors)} day(s) failed to download (retried next run)", \
        pd.DataFrame(errors, columns=["report", "day", "error"])


def _check_complete(ctx: StepContext):
    """Every day DTCC must have published by now is archived - over the window AND the
    retention lookback, so a day at risk of aging out is reported every run until it's in."""
    first = max(min(ctx.start, ctx.end - pd.Timedelta(days=DTCC_RETENTION_DAYS)), pd.Timestamp(DTCC_FIRST_DAY))
    last = min(ctx.end, expected_through(_out(ctx).get("now")))
    missing = []
    for kind in DTCC_REPORTS:
        have = pdtcc.archived_days(kind, root=ctx.paths.dtcc_dir)
        missing += [(kind, d) for d in pd.date_range(first, last) if d not in have] if last >= first else []
    if not missing:
        return True, "every published day in the window archived", None
    return False, f"{len(missing)} day(s) not archived", pd.DataFrame(missing, columns=["report", "day"])


def _header_and_first_row(path) -> tuple[list[str], list[str] | None]:
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace"))
        return next(reader, []), next(reader, None)


def _check_sane(ctx: StepContext):
    """Each file archived this run carries the columns readers rely on, and a weekday file
    holds at least one trade."""
    bad = []
    for kind, r in _out(ctx).get("reports", {}).items():
        for day in r.get("archived", []):
            header, row = _header_and_first_row(pdtcc.path_for(kind, day, root=ctx.paths.dtcc_dir))
            gone = [c for c in REQUIRED_COLUMNS if c not in header]
            if gone:
                bad.append((kind, day, f"missing columns {gone}"))
            elif row is None and pd.Timestamp(day).dayofweek < 5:
                bad.append((kind, day, "no rows on a weekday"))
    if not bad:
        return True, "every newly archived file has the expected layout", None
    return False, f"{len(bad)} file(s) look wrong", pd.DataFrame(bad, columns=["report", "day", "problem"])


DTCC_CHECKS = (
    Check("dtcc_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("dtcc_complete", _check_complete, severity=Severity.WARN),
    Check("dtcc_sane", _check_sane, severity=Severity.WARN),
)
