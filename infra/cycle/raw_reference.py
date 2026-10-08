"""Step 1b, reference data: the release calendar, US Treasury auctions and auction tails
- three ``raw`` sources over infra.pipeline.release_calendar / tsy_auctions /
auction_tails (no fetch or storage code here), with their checks.

All checks are WARN: this is reference data, and a publisher's page being down must not
fail the step and cost the day's vintage (the backup step needs every step ok).

Network hooks (``FRED_DATES_FETCH``, ``NAR_FETCH``, ``AUCTIONS_FETCH``, ``TAILS_LIST``,
``TAILS_FETCH``) are module-level so ``tests/conftest.py`` can stub them suite-wide.
"""
from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd

from infra.api import fiscaldata_client, fred_client, nar_client, treasury_schedule_client, wayback_client
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import auction_tails as pt
from infra.pipeline import release_calendar as prc
from infra.pipeline import tsy_auctions as pa
from infra.processing import tsy_auctions as ta
from infra.reference.events import MIN_RULE_HIT

log = logging.getLogger(__name__)
FRED_DATES_FETCH = fred_client.fetch_release_dates
NAR_FETCH = nar_client.fetch_page
TREASURY_SCHEDULE_FETCH = treasury_schedule_client.fetch_xml
AUCTIONS_FETCH = fiscaldata_client.fetch_auctions
# the FAST archive client: a scheduled run must not wait out the archive's bad spells
TAILS_LIST = functools.partial(wayback_client.list_captures, get=wayback_client.FAST_GET)
TAILS_FETCH = functools.partial(wayback_client.fetch_capture, get=wayback_client.FAST_GET)
TAILS_PER_RUN = 30  # new archived recaps processed per daily run (the archive is shared and slow)
TAILS_BUDGET_S = 300  # and never more than 5 minutes of the run
AUCTION_RESULTS_LAG = pd.Timedelta(days=1)
HIGH_YIELD_BOUNDS = (-1.0, 20.0)
BID_TO_COVER_BOUNDS = (0.5, 10.0)


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


# ------------------------------------------------------------------ sources
def backfill_daily_release_calendar(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False) -> dict:
    """Refresh the release calendar: FRED's dates (needs the FRED key), the harvested
    economic calendar (local), the validated rules' projections and NAR's next release.
    Failures are collected per source, never raised."""
    paths = paths or CyclePaths.default()
    errors: dict[str, str] = {}
    have_key = fred_client.has_key()
    try:
        n = prc.refresh_release_calendar(root=paths.release_calendar_dir, calendar_root=paths.econ_calendar_dir,
                                         fetch=FRED_DATES_FETCH, nar_fetch=NAR_FETCH,
                                         treasury_fetch=TREASURY_SCHEDULE_FETCH, with_fred=have_key,
                                         auctions_root=paths.tsy_auctions_dir, contracts_file=paths.contracts_file,
                                         daily_root=paths.daily_futures_dir, de_plan_root=paths.de_issuance_plan_dir,
                                         de_auctions_root=paths.de_auctions_dir, errors=errors)
    except Exception as exc:
        n, errors["calendar"] = 0, f"{type(exc).__name__}: {exc}"
    if not have_key:
        errors["fred"] = "FRED_API_KEY not set - FRED release dates not refreshed"
    return {"rows": n, "errors": errors}


def backfill_daily_tsy_auctions(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False) -> dict:
    """Auctions from the first one without results onward (held ones are never re-asked)."""
    paths = paths or CyclePaths.default()
    try:
        n = pa.update_auctions(root=paths.tsy_auctions_dir, coverage_file=paths.tsy_auctions_coverage,
                               fetch=AUCTIONS_FETCH, calendar_root=paths.release_calendar_dir)
        return {"rows": n, "errors": {}}
    except Exception as exc:
        log.warning("treasury auctions failed: %s", exc)
        return {"rows": 0, "errors": {"fiscal_data": f"{type(exc).__name__}: {exc}"}}


def backfill_daily_auction_tails(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False) -> dict:
    """Up to TAILS_PER_RUN new archived recaps per source (newest first)."""
    paths = paths or CyclePaths.default()
    try:
        auctions = pa.read_auctions(root=paths.tsy_auctions_dir)
        stats = pt.harvest_tails(root=paths.tsy_tails_dir, manifest=paths.tsy_tails_manifest, list_fn=TAILS_LIST,
                                 fetch_fn=TAILS_FETCH, auctions=auctions, limit=TAILS_PER_RUN,
                                 budget_s=TAILS_BUDGET_S)
        return {"stats": stats, "errors": {}}
    except Exception as exc:
        log.warning("auction tails failed: %s", exc)
        return {"stats": {}, "errors": {"wayback": f"{type(exc).__name__}: {exc}"}}


# ------------------------------------------------------------------ checks
def _out(ctx: StepContext, name: str) -> dict:
    return ctx.output.get("sources", {}).get(name, {})


def _errors_check(name: str, label: str):
    def fn(ctx: StepContext):
        errors = _out(ctx, name).get("errors", {})
        if not errors:
            return True, f"{label} refreshed", None
        return False, f"{label}: {len(errors)} source error(s): {', '.join(errors)}", \
            pd.DataFrame({"source": list(errors), "error": list(errors.values())})
    return fn


def _check_rules(ctx: StepContext):
    v = prc.validate_rules(root=ctx.paths.release_calendar_dir).dropna(subset=["hit"])
    bad = v[v["hit"] < MIN_RULE_HIT]
    if bad.empty:
        return True, f"{len(v)} date rule(s) still >= {MIN_RULE_HIT:.0%} on the observed history", None
    return False, f"{len(bad)} date rule(s) below {MIN_RULE_HIT:.0%}: {', '.join(bad['event'])}", bad


def _auctions(ctx: StepContext) -> pd.DataFrame:
    return pa.read_auctions(nominal_only=False, start=ctx.start - pd.Timedelta(days=30), end=ctx.end + pd.Timedelta(days=1),
                            root=ctx.paths.tsy_auctions_dir)


def _check_auctions_sane(ctx: StepContext):
    df = _auctions(ctx)
    held = df[ta.held(df) & df["high_yield"].notna()]
    lo, hi = HIGH_YIELD_BOUNDS
    blo, bhi = BID_TO_COVER_BOUNDS
    bad = held[~held["high_yield"].between(lo, hi)
               | (held["bid_to_cover_ratio"].notna() & ~held["bid_to_cover_ratio"].between(blo, bhi))]
    if bad.empty:
        return True, f"{len(held)} held auction(s) in range", None
    return False, f"{len(bad)} auction result(s) out of range", bad[["timestamp", "cusip", "high_yield", "bid_to_cover_ratio"]]


def _check_results_due(ctx: StepContext):
    df = _auctions(ctx)
    late = df[~ta.held(df) & (df["timestamp"] + AUCTION_RESULTS_LAG < pd.Timestamp.now())
              & (df["timestamp"] >= ctx.start)]
    if late.empty:
        return True, "every closed auction in the window has results", None
    return False, f"{len(late)} closed auction(s) without results yet", late[["timestamp", "cusip", "security_term"]]


def _check_tails(ctx: StepContext):
    stats = _out(ctx, "auction_tails").get("stats", {})
    n = sum(sum(v.values()) for v in stats.values())
    passed = sum(v.get("passed", 0) for v in stats.values())
    return True, f"{n} recap(s) processed this run, {passed} passed both checks", None


DE_PLAN_HORIZON_DAYS = 14   # German Federal auctions run nearly every week ...
DE_PLAN_QUIET = ((12, 10), (1, 4))   # ... except mid-December to early January


def _check_de_plan(ctx: StepContext):
    """The archived German issuance plan, as known now, has an auction in the next
    ``DE_PLAN_HORIZON_DAYS`` - else the plan stopped updating (a renamed file, a changed
    page)."""
    from infra.pipeline.de_issuance import plan_state
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    (m0, d0), (m1, d1) = DE_PLAN_QUIET
    if (now.month, now.day) >= (m0, d0) or (now.month, now.day) <= (m1, d1):
        return True, "year-end: no German auctions expected", None
    plan = plan_state(now, root=ctx.paths.de_issuance_plan_dir)
    soon = plan[plan["auction_date"] <= now.normalize() + pd.Timedelta(days=DE_PLAN_HORIZON_DAYS)]
    if len(soon):
        return True, f"{len(soon)} German auction(s) planned in the next {DE_PLAN_HORIZON_DAYS} days", None
    return False, f"no German auction planned in the next {DE_PLAN_HORIZON_DAYS} days (plan stale?)", None


CALENDAR_CHECKS = (
    Check("calendar_refreshed", _errors_check("release_calendar", "release calendar"), severity=Severity.WARN),
    Check("calendar_rules_valid", _check_rules, severity=Severity.WARN),
    Check("de_auction_plan", _check_de_plan, severity=Severity.WARN),
)
AUCTION_CHECKS = (
    Check("auctions_fetch_ok", _errors_check("tsy_auctions", "treasury auctions"), severity=Severity.WARN),
    Check("auctions_sane", _check_auctions_sane, severity=Severity.WARN),
    Check("auctions_results_due", _check_results_due, severity=Severity.WARN),
    revision_check(lambda p: p.tsy_auctions_dir, ta.KEYS, name="auctions_no_revisions"),
)
TAILS_CHECKS = (
    Check("tails_fetch_ok", _errors_check("auction_tails", "auction tails"), severity=Severity.WARN),
    Check("tails_progress", _check_tails, severity=Severity.INFO),
)
