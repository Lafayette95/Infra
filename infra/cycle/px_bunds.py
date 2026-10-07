"""German Federal securities in the ``px`` step (``infra.pipeline.bunds``): refresh the
Finanzagentur's issuance history (one small workbook), then the Bundesbank's per-ISIN prices
over the window plus never-covered days in ``GAP_LOOKBACK_DAYS`` before it - one request.
All checks warn: nothing downstream consumes these yet, and a missed day heals next run.
"""
from __future__ import annotations

import pandas as pd

from infra.cycle.bad_prints import last_weekday
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import bunds

GAP_LOOKBACK_DAYS = 30
STALE_BUSINESS_DAYS = 2
YIELD_BOUNDS_PCT = (-3.0, 15.0)
PRICE_BOUNDS = (20.0, 250.0)
NOMINAL_TYPES = ("Schatz", "Bobl", "Bund", "Green")
_ONE_DAY = pd.Timedelta(days=1)


def backfill_daily_bund_px(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                           fetch_missing: bool = True) -> dict:
    paths = paths or CyclePaths.default()
    out = {"auctions": 0, "planned": [], "rows": 0, "error": None}
    if not fetch_missing:
        return out
    try:
        out["auctions"] = bunds.update_auctions(root=paths.de_auctions_dir)
        lo = pd.Timestamp(start) - pd.Timedelta(days=GAP_LOOKBACK_DAYS)
        ranges = bunds.plan_prices(start, end, coverage_file=paths.bund_prices_coverage, force_refetch=force_refetch)
        ranges += [r for r in bunds.plan_prices(lo, pd.Timestamp(start) - _ONE_DAY, coverage_file=paths.bund_prices_coverage)]
        out["planned"] = [(str(s.date()), str((e - _ONE_DAY).date())) for s, e in ranges]
        out["rows"] = bunds.fetch_and_store_prices(ranges, root=paths.bund_prices_dir, coverage_file=paths.bund_prices_coverage)
    except Exception as exc:  # noqa: BLE001 - reported by bunds_fetch_ok
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _out(ctx: StepContext) -> dict:
    return ctx.output.get("bunds", {})


def _prices(ctx: StepContext, lookback=pd.Timedelta(0)) -> pd.DataFrame:
    return bunds.read_bund_prices(ctx.start - lookback, ctx.end + _ONE_DAY, root=ctx.paths.bund_prices_dir)


def _check_fetch_ok(ctx: StepContext):
    o = _out(ctx)
    if o.get("error"):
        return False, f"Bund prices: {o['error']}", None
    return True, f"Bund prices: {o.get('rows', 0)} rows over {o.get('planned', [])}; {o.get('auctions', 0)} auction rows", None


def _check_fresh(ctx: StepContext):
    df = _prices(ctx, pd.Timedelta(days=14))
    latest = pd.Timestamp(df["timestamp"].max()) if len(df) else None
    due = last_weekday(ctx.end) - pd.offsets.BDay(STALE_BUSINESS_DAYS)
    if latest is not None and latest >= due:
        return True, f"latest Bund prices {latest.date()}", None
    return False, f"latest Bund prices {None if latest is None else latest.date()}, due {due.date()}", None


def _check_sane(ctx: StepContext):
    df = _prices(ctx)
    bad = df[~df["yield"].between(*YIELD_BOUNDS_PCT) & df["yield"].notna()
             | ~df["price_clean"].between(*PRICE_BOUNDS) & df["price_clean"].notna()]
    return (bad.empty, "Bund prices and yields in bounds" if bad.empty else f"{len(bad)} row(s) out of bounds",
            None if bad.empty else bad)


def _check_complete(ctx: StepContext):
    """Every outstanding nominal Federal coupon security (issued, not matured) has a price on
    each stored day of the window."""
    df = _prices(ctx)
    if df.empty:
        return True, "no Bund prices in the window", None
    sec = bunds.read_securities(root=ctx.paths.de_auctions_dir)
    sec = sec[sec["type"].isin(NOMINAL_TYPES)]
    missing = []
    for d, g in df.groupby("timestamp"):
        live = sec[(sec["issue_date"] <= d) & (sec["maturity_date"] > d + pd.Timedelta(days=3))]
        for isin in sorted(set(live["isin"]) - set(g.loc[g["yield"].notna(), "isin"])):
            missing.append({"timestamp": d, "isin": isin})
    if not missing:
        return True, f"every outstanding nominal Federal security priced on {df['timestamp'].nunique()} day(s)", None
    return False, f"{len(missing)} outstanding security-day(s) without a price", pd.DataFrame(missing)


BUND_CHECKS = (
    Check("bunds_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("bunds_fresh", _check_fresh, severity=Severity.WARN),
    Check("bunds_sane", _check_sane, severity=Severity.WARN),
    Check("bunds_complete", _check_complete, severity=Severity.WARN),
    revision_check(lambda p: p.bund_prices_dir, ["timestamp", "isin"], name="bunds_no_revisions"),
)
