"""Montréal Exchange futures (Canada's CGB) in the ``px`` step (``infra.pipeline.mx_futures``):
this year's window through yesterday, one download per root (at the site's 15-second crawl
delay). Warn-level checks: nothing downstream consumes these yet."""
from __future__ import annotations

import pandas as pd

from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import mx_futures as mx

STALE_BUSINESS_DAYS = 3


def backfill_daily_mx(start, end, *, paths: CyclePaths | None = None, fetch_missing: bool = True) -> dict:
    paths = paths or CyclePaths.default()
    out = {"rows": {}, "latest": {}, "error": None}
    if not fetch_missing:
        return out
    try:
        out.update(mx.update(raw_root=paths.mx_raw_dir, root=paths.mx_futures_dir))
    except Exception as exc:  # noqa: BLE001 - reported by mx_fetch_ok
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _check_fetch_ok(ctx: StepContext):
    o = ctx.output.get("mx", {})
    if o.get("error"):
        return False, f"Montreal Exchange futures: {o['error']}", None
    return True, f"Montreal Exchange futures: {o.get('rows', {})} row(s)", None


def _check_fresh(ctx: StepContext):
    d = mx.read_mx_futures(ctx.end - pd.Timedelta(days=21), ctx.end, root=ctx.paths.mx_futures_dir)
    last = d["timestamp"].max() if len(d) else None
    want = (pd.Timestamp(ctx.end).normalize() - pd.offsets.BDay(STALE_BUSINESS_DAYS)).normalize()
    if last is not None and last >= want:
        return True, f"MX futures through {last.date()}", None
    return False, f"MX futures through {None if last is None else last.date()} (> {STALE_BUSINESS_DAYS} business days old)", None


def _check_sane(ctx: StepContext):
    """Every day in the window has a settlement for its held contract, within (50, 250)."""
    d = mx.read_mx_futures(ctx.start, ctx.end, root=ctx.paths.mx_futures_dir)
    bad = d[d["settlement"].notna() & ~d["settlement"].between(50, 250)]
    if bad.empty:
        return True, f"{len(d)} MX settlement row(s) in bounds", None
    return False, f"{len(bad)} MX settlement(s) out of bounds", bad[["timestamp", "ticker", "settlement"]]


MX_CHECKS = (
    Check("mx_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("mx_fresh", _check_fresh, severity=Severity.WARN),
    Check("mx_sane", _check_sane, severity=Severity.WARN),
    revision_check(lambda p: p.mx_futures_dir, mx.KEYS, name="mx_no_revisions"),
)
