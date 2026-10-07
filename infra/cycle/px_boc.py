"""Canada's benchmark-switch inputs in the ``px`` step (``infra.pipeline.boc_benchmarks``):
today's snapshot of the BoC benchmark page (it names each benchmark bond and the date it
became the benchmark) and the BoC zero curve since the last stored day. Warn-level checks."""
from __future__ import annotations

import pandas as pd

from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import boc_benchmarks as bb

ZERO_STALE_DAYS = 28    # weekly, two weeks late: a curve older than four weeks is overdue


def backfill_daily_boc(start, end, *, paths: CyclePaths | None = None, fetch_missing: bool = True) -> dict:
    paths = paths or CyclePaths.default()
    out = {"snapshot": 0, "zero_rows": 0, "error": None}
    if not fetch_missing:
        return out
    try:
        out["snapshot"] = bb.snapshot_today(root=paths.ca_benchmarks_dir)
        out["zero_rows"] = bb.update_zero_curve(root=paths.boc_zero_dir)
    except Exception as exc:  # noqa: BLE001 - reported by boc_fetch_ok
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _check_fetch_ok(ctx: StepContext):
    o = ctx.output.get("boc", {})
    if o.get("error"):
        return False, f"BoC benchmarks / zero curve: {o['error']}", None
    return True, f"BoC page: {o.get('snapshot', 0)} benchmark(s); zero curve: {o.get('zero_rows', 0)} new row(s)", None


def _check_zero_fresh(ctx: StepContext):
    c = bb.zero_curve_before(ctx.end, lag_days=0, root=ctx.paths.boc_zero_dir)
    if c is not None and (ctx.end - c[0]).days <= ZERO_STALE_DAYS:
        return True, f"latest BoC zero curve {c[0].date()}", None
    return False, f"latest BoC zero curve {None if c is None else c[0].date()} (published weekly, two weeks late)", None


BOC_CHECKS = (
    Check("boc_fetch_ok", _check_fetch_ok, severity=Severity.WARN),
    Check("boc_zero_fresh", _check_zero_fresh, severity=Severity.WARN),
)
