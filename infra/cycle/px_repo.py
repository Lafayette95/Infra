"""Step 1a, repo rates and securities lending - ``backfill_daily_repo_px``: SOFR / TGCR /
BGCR, OFR's repo release and the DTCC GCF history (``infra.pipeline.repo``), and the NY
Fed's Treasury securities lending results (``infra.pipeline.sec_lending``), CLAUDE.md 19.
No fetch/storage code here, only which days, plus the checks.

Called by ``infra.cycle.px.backfill_daily_px_data``, which returns this under ``"repo"``.
Besides the window, each run re-plans never-covered days in a lookback before it: OFR's
data is only FINAL ~3 months after the trade date (``OFR_LOOKBACK_DAYS``), and a missed run
leaves NY Fed days uncovered (``GAP_LOOKBACK_DAYS``). All checks are warn: nothing
downstream consumes these yet, so a bad day must not cost the vintage.
"""
from __future__ import annotations

import pandas as pd

from infra.config import SEC_LENDING_MIN_FEE
from infra.cycle.bad_prints import last_weekday
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import repo as prepo
from infra.pipeline import sec_lending as psl
from infra.pipeline.treasury_ref import read_securities
from infra.pipeline.tsy_auctions import read_auctions
from infra.processing import repo as rp
from infra.processing.sec_lending import LENDING_KEYS, fee_floor

_ONE_DAY = pd.Timedelta(days=1)
GAP_LOOKBACK_DAYS = 30
OFR_LOOKBACK_DAYS = 150  # OFR finalises ~3 months after the trade date
STALE_BUSINESS_DAYS = {"SOFR": 2, rp.ofr_series("DVP", "OO"): 4}  # NY Fed posts D at ~08:00 on D+1
LENDING_STALE_BUSINESS_DAYS = 2
RATE_BOUNDS_PCT = (-1.0, 25.0)
MAX_FEE_PCT = 25.0
AVAILABLE_ROUNDING = 1e6  # the NY Fed reports availability in whole $m


def _merge(*plans: dict) -> dict:
    out = {}
    for plan in plans:
        for key, (a, b) in plan.items():
            out[key] = (min(a, out[key][0]), max(b, out[key][1])) if key in out else (a, b)
    return out


def plan_daily_repo_px(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False, now=None):
    """``(repo plan, lending ranges)`` for the window plus the lookbacks. No network - also
    the dry run's view."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    window = prepo.plan_repo_update(start, end, now=now, coverage_file=paths.repo_coverage,
                                    force_refetch=force_refetch)
    gaps = prepo.plan_repo_update(start - pd.Timedelta(days=GAP_LOOKBACK_DAYS), start - _ONE_DAY, now=now,
                                  coverage_file=paths.repo_coverage, sources=("nyfed", "dtcc_gcf"))
    ofr = prepo.plan_repo_update(start - pd.Timedelta(days=OFR_LOOKBACK_DAYS), start - _ONE_DAY, now=now,
                                 coverage_file=paths.repo_coverage, sources=("ofr",))
    lending = psl.plan_sec_lending_update(start, end, now=now, coverage_file=paths.sec_lending_coverage,
                                          force_refetch=force_refetch)
    lending += psl.plan_sec_lending_update(start - pd.Timedelta(days=GAP_LOOKBACK_DAYS), start - _ONE_DAY, now=now,
                                           coverage_file=paths.sec_lending_coverage)
    return _merge(window, gaps, ofr), sorted(lending)


def backfill_daily_repo_px(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                           fetch_missing: bool = True, now=None) -> dict:
    paths = paths or CyclePaths.default()
    plan, ranges = plan_daily_repo_px(start, end, paths=paths, force_refetch=force_refetch, now=now) \
        if fetch_missing else ({}, [])
    rates = prepo.fetch_and_store_repo(plan, now=now, root=paths.repo_dir, coverage_file=paths.repo_coverage) \
        if plan else {"rows": {}, "errors": {}}
    lending = psl.fetch_and_store_sec_lending(ranges, now=now, root=paths.sec_lending_dir,
                                              coverage_file=paths.sec_lending_coverage) \
        if ranges else {"rows": 0, "days": 0, "errors": {}}
    return {"plan": plan, "lending_ranges": ranges, "rates": rates, "lending": lending}


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext) -> dict:
    return ctx.output.get("repo", {})


def _check_fetch_ok(ctx: StepContext):
    o = _out(ctx)
    errors = {**o.get("rates", {}).get("errors", {}),
              **{f"lending {k}": v for k, v in o.get("lending", {}).get("errors", {}).items()}}
    if not errors:
        return True, (f"repo rows {sum(o.get('rates', {}).get('rows', {}).values())}, "
                      f"lending rows {o.get('lending', {}).get('rows', 0)}"), None
    return False, f"{len(errors)} repo/lending request(s) failed (retried next run)", \
        pd.DataFrame({"request": list(errors), "error": list(errors.values())})


def _check_fresh(ctx: StepContext):
    """Warning: the latest SOFR, OFR (DVP overnight, preliminary) and lending day lag the
    window's last weekday by more than their usual publication delay."""
    due_day = last_weekday(ctx.end)
    df = prepo.read_repo(ctx.end - pd.Timedelta(days=21), ctx.end + _ONE_DAY, series=list(STALE_BUSINESS_DAYS),
                         status="all", root=ctx.paths.repo_dir)
    stale = []
    for series, lag in STALE_BUSINESS_DAYS.items():
        latest = df.loc[df["series"] == series, "timestamp"].max() if len(df) else None
        if latest is None or pd.isna(latest) or latest < due_day - pd.offsets.BDay(lag):
            stale.append((series, latest))
    lend = psl.read_sec_lending(ctx.end - pd.Timedelta(days=21), ctx.end + _ONE_DAY, root=ctx.paths.sec_lending_dir)
    latest = lend["timestamp"].max() if len(lend) else None
    if latest is None or latest < due_day - pd.offsets.BDay(LENDING_STALE_BUSINESS_DAYS):
        stale.append(("securities lending", latest))
    if not stale:
        return True, "SOFR, OFR and securities lending all current", None
    return False, f"{len(stale)} stale: " + ", ".join(f"{s} (latest {None if t is None or pd.isna(t) else pd.Timestamp(t).date()})"
                                                      for s, t in stale), None


def _check_repo_sane(ctx: StepContext):
    """Rates within bounds; the NY Fed's percentiles ordered (p1 <= p25 <= rate <= p75 <= p99:
    each rate is the volume-weighted MEDIAN)."""
    df = prepo.read_repo(ctx.start, ctx.end + _ONE_DAY, status="all", root=ctx.paths.repo_dir)
    if df.empty:
        return True, "no repo rates in the window", None
    bad = df[~df["rate"].between(*RATE_BOUNDS_PCT)].assign(problem="rate out of bounds")
    ny = df[df["source"] == "nyfed"]
    order = ny[["p1", "p25", "rate", "p75", "p99"]]
    unordered = ny[(order.diff(axis=1).iloc[:, 1:] < -1e-9).any(axis=1)].assign(problem="percentiles out of order")
    bad = pd.concat([bad, unordered], ignore_index=True)
    if bad.empty:
        return True, f"{len(df)} repo rates sane", None
    return False, f"{len(bad)} repo rate problem(s)", bad[["timestamp", "series", "status", "rate", "problem"]]


def _check_lending_sane(ctx: StepContext):
    """Accepted lending: fee at or above the minimum fee in force (SEC_LENDING_MIN_FEE -
    a fee below it means the schedule is out of date) and below MAX_FEE_PCT; accepted <=
    submitted and <= actually available (to its $1m rounding); every lent Treasury CUSIP
    known (the auctions store, which also holds the TIPS/FRNs the reference table leaves
    out; agency debt is read but not judged). Only 3 bonds issued before the auctions
    store starts (1979-10-31) fail this in all of history, all lent before 2002."""
    df = psl.read_sec_lending(ctx.start, ctx.end + _ONE_DAY, root=ctx.paths.sec_lending_dir)
    df = df[df["par_accepted"] > 0]
    if df.empty:
        return True, "no securities lent in the window", None
    floor = fee_floor(df["timestamp"], SEC_LENDING_MIN_FEE).to_numpy()
    problems = [
        df[df["fee"].to_numpy() < floor - 1e-9].assign(problem="fee below the minimum fee"),
        df[df["fee"] > MAX_FEE_PCT].assign(problem="fee above MAX_FEE_PCT"),
        df[df["par_accepted"] > df["par_submitted"]].assign(problem="accepted > submitted"),
        df[df["par_accepted"] > df["actual_available"] + AVAILABLE_ROUNDING].assign(problem="accepted > available"),
    ]
    known = set(read_auctions(nominal_only=False, root=ctx.paths.tsy_auctions_dir)["cusip"].astype(str)) | \
        set(read_securities(root=ctx.paths.treasury_securities_dir)["cusip"].astype(str))
    problems.append(df[~df["cusip"].isin(known)].assign(problem="CUSIP not a known Treasury"))
    bad = pd.concat(problems, ignore_index=True)
    if bad.empty:
        return True, f"{len(df)} loans sane; every lent CUSIP a known Treasury", None
    return False, f"{len(bad)} lending problem(s)", bad[["timestamp", "cusip", "description", "fee", "par_accepted",
                                                         "problem"]]


REPO_CHECKS: tuple[Check, ...] = (
    Check("repo_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("repo_fresh", _check_fresh, severity=Severity.WARN),
    Check("repo_sane", _check_repo_sane, severity=Severity.WARN),
    Check("sec_lending_sane", _check_lending_sane, severity=Severity.WARN),
    revision_check(lambda p: p.repo_dir, rp.REPO_KEYS, name="repo_no_revisions"),
    revision_check(lambda p: p.sec_lending_dir, LENDING_KEYS, name="sec_lending_no_revisions"),
)
