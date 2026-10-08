"""FTSE-Tradeweb gilt and EuroGov closing prices in the ``px`` step
(``infra.pipeline.tradeweb_prices``): one export a run - every security over the last 5 working
days, so a missed run heals itself. Needs the user's InSite login in ``.env``
(``TRADEWEB_USER`` / ``TRADEWEB_PASSWORD``); without it the source is skipped and
``tradeweb_fetch_ok`` warns. Warn-level checks: nothing downstream consumes these yet."""
from __future__ import annotations

import pandas as pd

from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import tradeweb_prices as tw

YIELD_BOUNDS = (-2.0, 25.0)
PRICE_BOUNDS = (0.0, 300.0)


def backfill_daily_tradeweb(start, end, *, paths: CyclePaths | None = None, fetch_missing: bool = True) -> dict:
    from infra.api.tradeweb_client import has_credentials
    paths = paths or CyclePaths.default()
    out = {"rows": 0, "days": [], "error": None}
    if not fetch_missing:
        return out
    if tw.SESSION is None and not has_credentials():
        out["error"] = "TRADEWEB_USER / TRADEWEB_PASSWORD not set in .env - skipped"
        return out
    try:
        out.update(tw.update_daily(raw_root=paths.tradeweb_raw_dir, root=paths.tradeweb_prices_dir))
    except Exception as exc:  # noqa: BLE001 - reported by tradeweb_fetch_ok
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _check_fetch_ok(ctx: StepContext):
    o = ctx.output.get("tradeweb", {})
    if o.get("error"):
        return False, f"Tradeweb closing prices: {o['error']}", None
    return True, f"Tradeweb closing prices: {o.get('rows', 0)} row(s) over {len(o.get('days', []))} day(s)", None


def _check_fresh(ctx: StepContext):
    """The latest stored day is at least the business day before the window's end (a day is
    published at 12:00 London the next day, before the 10:45 New York run)."""
    p = tw.read_prices(ctx.end - pd.Timedelta(days=14), ctx.end, root=ctx.paths.tradeweb_prices_dir)
    want = (pd.Timestamp(ctx.end).normalize() - pd.offsets.BDay(1)).normalize()
    last = p["timestamp"].max() if len(p) else None
    if last is not None and last >= want:
        return True, f"Tradeweb prices through {last.date()}", None
    return False, f"Tradeweb prices through {None if last is None else last.date()}, expected {want.date()}", None


def _check_sane(ctx: StepContext):
    p = tw.read_prices(ctx.start, ctx.end, root=ctx.paths.tradeweb_prices_dir)
    lo, hi = YIELD_BOUNDS
    plo, phi = PRICE_BOUNDS
    bad = p[(p["yield"].notna() & ~p["yield"].between(lo, hi))
            | (p["price_clean"].notna() & ~p["price_clean"].between(plo, phi))]
    if bad.empty:
        return True, f"{len(p)} Tradeweb price(s) in bounds", None
    return False, f"{len(bad)} Tradeweb price(s) out of bounds", bad[["timestamp", "isin", "name", "price_clean", "yield"]]


TRADEWEB_CHECKS = (
    Check("tradeweb_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("tradeweb_fresh", _check_fresh, severity=Severity.WARN),
    Check("tradeweb_sane", _check_sane, severity=Severity.WARN),
    revision_check(lambda p: p.tradeweb_prices_dir, tw.KEYS, name="tradeweb_no_revisions"),
)
