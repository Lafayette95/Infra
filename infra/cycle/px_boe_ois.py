"""The BoE SONIA OIS curve in the ``px`` step (``infra.pipeline.boe_ois``): the window plus
never-covered days in the ``GAP_LOOKBACK_DAYS`` before it (the BoE's archive is refreshed
once a month closes, as for the gilt curves). Warn-level checks: a missed day heals itself."""
from __future__ import annotations

import pandas as pd

from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import boe_ois

GAP_LOOKBACK_DAYS = 30
STALE_DAYS = 6


def backfill_daily_boe_ois(start, end, *, paths: CyclePaths | None = None, fetch_missing: bool = True) -> dict:
    paths = paths or CyclePaths.default()
    ranges = boe_ois.plan(pd.Timestamp(start) - pd.Timedelta(days=GAP_LOOKBACK_DAYS), end,
                          coverage_file=paths.boe_ois_coverage) if fetch_missing else []
    try:
        n = boe_ois.fetch_and_store(ranges, root=paths.boe_ois_dir, coverage_file=paths.boe_ois_coverage)
        return {"planned": len(ranges), "rows": n, "error": None}
    except Exception as exc:
        return {"planned": len(ranges), "rows": 0, "error": f"{type(exc).__name__}: {exc}"}


def _check_fetch_ok(ctx: StepContext):
    o = ctx.output.get("boe_ois", {})
    if not o.get("error"):
        return True, f"BoE OIS: {o.get('rows', 0)} rows stored ({o.get('planned', 0)} range(s) planned)", None
    return False, f"BoE OIS fetch failed: {o['error']}", None


def _check_fresh(ctx: StepContext):
    df = boe_ois.read_boe_ois(ctx.end - pd.Timedelta(days=30), ctx.end + pd.Timedelta(days=1), root=ctx.paths.boe_ois_dir)
    latest = df["timestamp"].max() if len(df) else None
    if latest is not None and (ctx.end - pd.Timestamp(latest)).days <= STALE_DAYS:
        return True, f"latest BoE OIS curve {pd.Timestamp(latest).date()}", None
    return False, f"latest BoE OIS curve {None if latest is None else pd.Timestamp(latest).date()}", None


BOE_OIS_CHECKS = (
    Check("boe_ois_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("boe_ois_fresh", _check_fresh, severity=Severity.WARN),
)
