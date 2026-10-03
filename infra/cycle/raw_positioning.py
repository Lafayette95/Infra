"""Step 1b, positioning and dealer balance sheets - raw sources ``"cftc_tff"`` (CFTC
Traders in Financial Futures) and ``"primary_dealer"`` (NY Fed Primary Dealer
Statistics), through ``infra.pipeline.cftc_tff`` / ``infra.pipeline.primary_dealer`` (no
fetch/storage code here), plus their checks (CLAUDE.md 24).

Both are weekly: each run asks only whether a new release exists (TFF: the source's
``Last-Modified``; dealers: whether the next Thursday release has passed), so an ordinary
day costs one tiny request or none. ``force_refetch`` re-reads both. Every check is
``warn``: a missed week heals on the next run, so it must never cost the day's vintage.
"""
from __future__ import annotations

import pandas as pd

from infra.config import (CFTC_TFF_DIR, CFTC_TFF_RELEASE, CFTC_TFF_STATE_FILE, PRIMARY_DEALER_DIR,
                          PRIMARY_DEALER_STATE_FILE, RAW_DATA_ROOT)
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import cftc_tff, primary_dealer
from infra.processing.cftc_tff import position_identity_gaps
from infra.trading_calendar import snap_instants

# a release this late (beyond its schedule) is flagged stale - a shutdown, or a broken feed
STALE_GRACE = pd.Timedelta(days=3)
# dealer series that must be in every new week (Treasury positions, repo, reverse repo)
PD_REQUIRED = ("PDPOSGST-TOT",)


def _under_raw(paths: CyclePaths, path):
    return paths.raw_data_root / path.relative_to(RAW_DATA_ROOT)


def backfill_daily_cftc_tff(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False, now=None) -> dict:
    paths = paths or CyclePaths.default()
    return {"reports": cftc_tff.update_tff(root=_under_raw(paths, CFTC_TFF_DIR),
                                           state_file=_under_raw(paths, CFTC_TFF_STATE_FILE),
                                           force=force_refetch, now=now), "now": now}


def backfill_daily_primary_dealer(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                                  now=None) -> dict:
    paths = paths or CyclePaths.default()
    return {**primary_dealer.update_primary_dealer(root=_under_raw(paths, PRIMARY_DEALER_DIR),
                                                   catalog_file=_under_raw(paths, primary_dealer.CATALOG_FILE),
                                                   state_file=_under_raw(paths, PRIMARY_DEALER_STATE_FILE),
                                                   force=force_refetch, now=now), "now": now}


# ------------------------------------------------------------------------ checks
def _src(ctx: StepContext, name: str) -> dict:
    return ctx.output.get("sources", {}).get(name, {})


def _now(out: dict) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None) if out.get("now") is None else pd.Timestamp(out["now"])


def latest_tff_due(now: pd.Timestamp) -> pd.Timestamp:
    """The latest Tuesday whose scheduled release (+ ``STALE_GRACE``) is before ``now``."""
    local_time, zone, lag = CFTC_TFF_RELEASE
    tuesdays = pd.date_range(now.normalize() - pd.Timedelta(days=28), now.normalize(), freq="W-TUE")
    released = snap_instants(tuesdays + pd.Timedelta(days=lag), local_time, zone) + STALE_GRACE
    return tuesdays[released <= now].max()


def _check_tff_fetch_ok(ctx: StepContext):
    rep = _src(ctx, "cftc_tff").get("reports", {})
    bad = {r: v["error"] for r, v in rep.items() if v.get("status") == "failed"}
    if not bad:
        return True, "TFF: " + ", ".join(f"{r} {v['status']} ({v['rows']} rows)" for r, v in rep.items()), None
    return False, f"TFF fetch failed for {sorted(bad)} (retried next run)", \
        pd.DataFrame(list(bad.items()), columns=["report", "error"])


def _check_tff_fresh(ctx: StepContext):
    out = _src(ctx, "cftc_tff")
    due = latest_tff_due(_now(out))
    root = _under_raw(ctx.paths, CFTC_TFF_DIR)
    stale = []
    for report in out.get("reports", {}):
        df = cftc_tff.read_tff(due - pd.Timedelta(days=14), None, report=report, root=root)
        latest = df["timestamp"].max() if not df.empty else None
        if latest is None or latest < due:
            stale.append((report, latest, due))
    if not stale:
        return True, f"TFF current through {due.date()}", None
    return False, f"TFF stale (a shutdown delays CFTC releases): {len(stale)} report(s)", \
        pd.DataFrame(stale, columns=["report", "latest", "due"])


def _check_tff_sane(ctx: StepContext):
    """Positions in the latest week add up to open interest (the report's identity)."""
    root = _under_raw(ctx.paths, CFTC_TFF_DIR)
    bad = []
    for report in _src(ctx, "cftc_tff").get("reports", {}):
        df = cftc_tff.read_tff(_now(_src(ctx, "cftc_tff")) - pd.Timedelta(days=21), None, report=report, root=root)
        if not df.empty:
            bad.append(position_identity_gaps(df[df["timestamp"] == df["timestamp"].max()]))
    bad = pd.concat(bad, ignore_index=True) if bad else pd.DataFrame()
    if bad.empty:
        return True, "latest TFF week: positions add up to open interest", None
    return False, f"{len(bad)} market(s) whose positions don't add up to open interest", bad


def _check_pd_fetch_ok(ctx: StepContext):
    out = _src(ctx, "primary_dealer")
    if out.get("status") != "failed":
        return True, f"primary dealers: {out.get('status')} ({out.get('rows', 0)} rows written, " \
                     f"{out.get('revised', 0)} revised)", None
    return False, f"primary dealer fetch failed (retried next run): {out.get('error')}", None


def _check_pd_fresh(ctx: StepContext):
    out = _src(ctx, "primary_dealer")
    root = _under_raw(ctx.paths, PRIMARY_DEALER_DIR)
    latest = primary_dealer.latest_stored(root)
    if latest is None:
        return False, "no primary dealer data stored", None
    overdue = primary_dealer.next_release(latest) + STALE_GRACE
    if _now(out) < overdue:
        return True, f"primary dealers current through {latest.date()}", None
    return False, f"primary dealers stale: latest {latest.date()}, next week overdue since {overdue}", None


def _check_pd_sane(ctx: StepContext):
    root = _under_raw(ctx.paths, PRIMARY_DEALER_DIR)
    latest = primary_dealer.latest_stored(root)
    if latest is None:
        return True, "no primary dealer data stored yet", None
    df = primary_dealer.read_primary_dealer(PD_REQUIRED, latest, latest, root=root)
    missing = sorted(set(PD_REQUIRED) - set(df.loc[df["value"].notna(), "series"]))
    if not missing:
        return True, f"key dealer series present for {latest.date()}", None
    return False, f"key dealer series missing for {latest.date()}: {missing}", None


POSITIONING_CHECKS = (
    Check("tff_fetch_ok", _check_tff_fetch_ok, severity=Severity.WARN),
    Check("tff_fresh", _check_tff_fresh, severity=Severity.WARN),
    Check("tff_sane", _check_tff_sane, severity=Severity.WARN),
    Check("primary_dealer_fetch_ok", _check_pd_fetch_ok, severity=Severity.WARN),
    Check("primary_dealer_fresh", _check_pd_fresh, severity=Severity.WARN),
    Check("primary_dealer_sane", _check_pd_sane, severity=Severity.WARN),
)
