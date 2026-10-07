"""Step 1a, Treasury prices - ``backfill_daily_treasury_px``: FedInvest END OF DAY prices,
accrued and yields per CUSIP (CLAUDE.md 18), through ``infra.pipeline.treasury_prices`` -
no fetch/storage code here, only which days, plus the checks.

Called by ``infra.cycle.px.backfill_daily_px_data``, which returns this under
``"treasuries"``. Yields use the reference table on disk, rebuilt minutes earlier by the
``ref`` step (which px runs after but does not depend on). A day is stored only once its
END OF DAY column is posted; a day not posted yet stays uncovered and is asked again by
the next run (the scheduled window, plus never-covered days in GAP_LOOKBACK_DAYS).
"""
from __future__ import annotations

import logging

import pandas as pd

from infra.cycle.bad_prints import last_weekday
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import treasury_otr as potr
from infra.pipeline import treasury_prices as ptp

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)
GAP_LOOKBACK_DAYS = 30
# A latest stored day more than this many business days behind the window's last weekday
# is reported stale (warn). FedInvest posts END OF DAY overnight-ish (not yet at 22:27 New
# York on the day; when exactly is still to be observed), so one day of lag is normal.
STALE_BUSINESS_DAYS = 2
YIELD_BOUNDS_PCT = (-5.0, 30.0)
PRICE_BOUNDS = (1.0, 200.0)


def plan_daily_treasury_px(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                           now=None) -> list[pd.Timestamp]:
    """Business days to request: the window (all of it with ``force_refetch``, else the
    uncovered ones) plus never-covered days in the GAP_LOOKBACK_DAYS before it. No network
    - also the dry run's view."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = set(ptp.plan_prices_update(start, end, now=now, coverage_file=paths.treasury_prices_coverage,
                                      force_refetch=force_refetch))
    days |= set(ptp.plan_prices_update(start - pd.Timedelta(days=GAP_LOOKBACK_DAYS), start - _ONE_DAY, now=now,
                                       coverage_file=paths.treasury_prices_coverage))
    # TIPS come from the same page: their own uncovered days are fetched too (the nominal
    # rows of such a day are stored again - an idempotent upsert)
    from infra.pipeline import tips
    days |= set(tips.plan_tips_update(start - pd.Timedelta(days=GAP_LOOKBACK_DAYS), end, now=now,
                                      coverage_file=paths.tips_prices_coverage))
    return sorted(days)


def backfill_daily_treasury_px(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                               fetch_missing: bool = True, fetch=None, now=None) -> dict:
    paths = paths or CyclePaths.default()
    days = plan_daily_treasury_px(start, end, paths=paths, force_refetch=force_refetch, now=now) if fetch_missing else []
    if not days:
        return {"planned": [], "stored": [], "holiday": [], "pending": [], "errors": {}, "tips": {}}
    from infra.pipeline import tips
    base = fetch if fetch is not None else ptp.FETCH
    ref, cpi, tips_status = None, None, {"stored": 0, "holiday": 0, "pending": 0, "errors": {}}

    def fetch_and_keep_tips(day):
        """The nominal fetch, with the page's TIPS rows stored on the way (one request)."""
        nonlocal ref, cpi
        page = base(day)
        try:
            if ref is None:
                ref, cpi = tips.tips_reference(auctions_root=paths.tsy_auctions_dir), tips.cpi_nsa()
            tips_status[tips.store_tips_day(day, page, ref, cpi, now=now, root=paths.tips_prices_dir,
                                            coverage_file=paths.tips_prices_coverage)] += 1
        except Exception as exc:  # never costs the nominal prices
            tips_status["errors"][str(pd.Timestamp(day).date())] = f"{type(exc).__name__}: {exc}"
        return page

    out = ptp.fetch_and_store_prices(days, root=paths.treasury_prices_dir, coverage_file=paths.treasury_prices_coverage,
                                     securities_root=paths.treasury_securities_dir, fetch=fetch_and_keep_tips, now=now)
    return {"planned": days, **out, "tips": tips_status}


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext) -> dict:
    return ctx.output.get("treasuries", {})


def _prices(ctx: StepContext, lookback: pd.Timedelta = pd.Timedelta(0)) -> pd.DataFrame:
    return ptp.read_prices(ctx.start - lookback, ctx.end + _ONE_DAY, root=ctx.paths.treasury_prices_dir)


def _check_fetch_ok(ctx: StepContext):
    errors = _out(ctx).get("errors", {})
    o = _out(ctx)
    if not errors:
        return True, (f"stored {len(o.get('stored', []))}, holidays {len(o.get('holiday', []))}, "
                      f"END OF DAY not posted yet {len(o.get('pending', []))}"), None
    return False, f"{len(errors)} FedInvest day(s) failed (retried next run)", \
        pd.DataFrame({"day": list(errors), "error": list(errors.values())})


def _check_fresh(ctx: StepContext):
    """Warning: the latest stored day lags the window's last weekday by more than
    STALE_BUSINESS_DAYS (FedInvest down, or END OF DAY posting later than usual)."""
    df = _prices(ctx, lookback=pd.Timedelta(days=14))
    latest = df["timestamp"].max() if len(df) else None
    due = last_weekday(ctx.end) - pd.offsets.BDay(STALE_BUSINESS_DAYS)
    if latest is not None and latest >= due:
        return True, f"latest Treasury prices {pd.Timestamp(latest).date()}", None
    return False, f"latest Treasury prices {None if latest is None else pd.Timestamp(latest).date()}, due {due.date()}", None


def _check_sane(ctx: StepContext):
    """Prices and yields within bounds, and every stored day prices its on-the-run issues."""
    df = _prices(ctx)
    if df.empty:
        return True, "no Treasury prices in the window", None
    bad = df[~df["price_eod"].between(*PRICE_BOUNDS) | (df["yield_eod"].notna()
                                                      & ~df["yield_eod"].between(*YIELD_BOUNDS_PCT))]
    otr = potr.read_otr(ctx.start, ctx.end + _ONE_DAY, rank=0, root=ctx.paths.treasury_otr_dir)
    otr = otr[otr["timestamp"].isin(set(df["timestamp"]))]
    priced = set(zip(df["timestamp"], df["cusip"].astype(str)))
    unpriced = otr[[(t, c) not in priced for t, c in zip(otr["timestamp"], otr["cusip"].astype(str))]]
    if bad.empty and unpriced.empty:
        return True, f"{len(df)} prices sane; every on-the-run issue priced", None
    details = pd.concat([bad[["timestamp", "cusip", "price_eod", "yield_eod"]].assign(problem="out of bounds"),
                         unpriced[["timestamp", "cusip"]].assign(problem="on-the-run issue has no price")],
                        ignore_index=True)
    return False, f"{len(bad)} price(s) out of bounds, {len(unpriced)} on-the-run issue(s) unpriced", details


TIPS_YIELD_BOUNDS_PCT = (-5.0, 10.0)
TIPS_RATIO_BOUNDS = (0.9, 3.0)


def _check_tips_fetch_ok(ctx: StepContext):
    t = _out(ctx).get("tips", {})
    if not t.get("errors"):
        return True, f"TIPS: stored {t.get('stored', 0)}, END OF DAY not posted yet {t.get('pending', 0)}", None
    return False, f"{len(t['errors'])} TIPS day(s) failed", pd.DataFrame({"day": list(t["errors"]), "error": list(t["errors"].values())})


def _check_tips_fresh(ctx: StepContext):
    from infra.pipeline import tips
    df = tips.read_tips(ctx.start - pd.Timedelta(days=14), ctx.end + _ONE_DAY, root=ctx.paths.tips_prices_dir)
    latest = df["timestamp"].max() if len(df) else None
    due = last_weekday(ctx.end) - pd.offsets.BDay(STALE_BUSINESS_DAYS)
    if latest is not None and latest >= due:
        return True, f"latest TIPS prices {pd.Timestamp(latest).date()}", None
    return False, f"latest TIPS prices {None if latest is None else pd.Timestamp(latest).date()}, due {due.date()}", None


def _check_tips_sane(ctx: StepContext):
    """Real yields and index ratios within bounds; every priced TIPS in the reference."""
    from infra.pipeline import tips
    df = tips.read_tips(ctx.start, ctx.end + _ONE_DAY, root=ctx.paths.tips_prices_dir)
    if df.empty:
        return True, "no TIPS prices in the window", None
    bad = df[(df["real_yield"].notna() & ~df["real_yield"].between(*TIPS_YIELD_BOUNDS_PCT))
             | (df["index_ratio"].notna() & ~df["index_ratio"].between(*TIPS_RATIO_BOUNDS))
             | (df["price_eod"].notna() & df["index_ratio"].isna())]
    if bad.empty:
        return True, f"{len(df)} TIPS prices sane (real yield, index ratio, reference)", None
    return False, f"{len(bad)} TIPS row(s) out of bounds or without reference", bad[["timestamp", "cusip", "price_eod", "real_yield", "index_ratio"]]


TREASURY_CHECKS: tuple[Check, ...] = (
    Check("treasuries_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("treasuries_fresh", _check_fresh, severity=Severity.WARN),
    # warn, not fail, while nothing downstream consumes Treasury prices yet (the OTR yield
    # benchmark joins `derived` later): a bad FedInvest day must not block bmk
    Check("treasuries_sane", _check_sane, severity=Severity.WARN),
    revision_check(lambda p: p.treasury_prices_dir, ["timestamp", "cusip"], name="treasuries_no_revisions"),
    Check("tips_fetch_ok", _check_tips_fetch_ok, severity=Severity.WARN),
    Check("tips_fresh", _check_tips_fresh, severity=Severity.WARN),
    Check("tips_sane", _check_tips_sane, severity=Severity.WARN),
    revision_check(lambda p: p.tips_prices_dir, ["timestamp", "cusip"], name="tips_no_revisions"),
)
