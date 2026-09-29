"""Step 2 - ``backfill_daily_derived``: metrics computed purely from the cycle's own
stored data (no API). Unlike the dashboard's on-demand views, derived values ARE
persisted here, because the revision check (b) needs yesterday's values on disk.

Each metric is a ``DerivedMetric`` in ``DERIVED_METRICS``; adding one means adding an
entry (compute function, store, keys, custom checks) - the step, its presence check and
its revision check are generated from the registry.

Every recomputed day REPLACES that day's rows wholesale (prune on ``timestamp``) rather
than upserting: a day whose output shrinks (e.g. a meeting going from two outcome levels
to one) would otherwise keep a stale row from the previous run.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline.wirp import available_days, build_schedule
from infra.storage import parquet_store

# (rows with a "timestamp" column = the as-of day, {day: why no rows}) over [start, end]
ComputeFn = Callable[[pd.Timestamp, pd.Timestamp, CyclePaths], "tuple[pd.DataFrame, dict]"]


@dataclass(frozen=True)
class DerivedMetric:
    name: str
    store: Callable[[CyclePaths], Path]
    key_columns: tuple[str, ...]
    compute: ComputeFn
    checks: tuple[Check, ...] = ()  # custom checks (c); presence/revision are generated


# ----------------------------------------------------------------------------- WIRP
WIRP_KEYS = ("timestamp", "meeting_date", "outcome_step")
WIRP_MAX_ABS_MOVE_BPS = 100.0  # a single scheduled meeting beyond this is a bug, not a market


def compute_wirp(start: pd.Timestamp, end: pd.Timestamp, paths: CyclePaths) -> tuple[pd.DataFrame, dict]:
    """WIRP (CLOSE, i.e. settlement-based) for every day in ``[start, end]`` that has ZQ
    settlements on disk - the exact function the dashboard uses, as of each day."""
    kw = dict(contracts_file=paths.contracts_file, close_root=paths.daily_futures_dir)
    days = [d for d in available_days("close", **kw) if start <= d <= end]
    kw["adjustments_dir"] = paths.adjustments_dir  # WIRP reads cleaned settlements
    frames, empty = [], {}
    for day in days:
        schedule, meta = build_schedule("close", today=day, **kw)
        if schedule.empty:
            empty[day] = meta["status"]
            continue
        frames.append(schedule.assign(timestamp=day, anchor_month=meta["anchor_month"],
                                      anchor_rate=meta["anchor_rate"]))
    if not frames:
        return pd.DataFrame(columns=[*WIRP_KEYS]), {"days": days, "empty_days": empty}
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    df["meeting_date"] = pd.to_datetime(df["meeting_date"]).astype("datetime64[ms]")
    df["outcome_step"] = df["outcome_step"].astype("int32")
    return df, {"days": days, "empty_days": empty}


def _wirp_rows(ctx: StepContext) -> pd.DataFrame:
    df = parquet_store.read_partitioned(ctx.paths.wirp_dir, start=ctx.start,
                                        end=ctx.end + pd.Timedelta(days=1))
    return pd.DataFrame() if df is None else df


def _check_wirp_probabilities(ctx: StepContext):
    df = _wirp_rows(ctx)
    if df.empty:
        return True, "no rows in window", None
    in_range = df["probability"].between(0.0, 1.0)
    sums = df.groupby(["timestamp", "meeting_date"])["probability"].sum()
    bad_sum = sums[~np.isclose(sums, 1.0, atol=1e-9)]
    if in_range.all() and bad_sum.empty:
        return True, f"{sums.size} meeting distributions, all in [0,1] and summing to 1", None
    details = pd.concat([df.loc[~in_range, ["timestamp", "meeting_date", "probability"]],
                         bad_sum.rename("probability").reset_index()], ignore_index=True)
    return False, f"{(~in_range).sum()} out-of-range probability(ies), {len(bad_sum)} bad sum(s)", details


def _check_wirp_moves(ctx: StepContext):
    df = _wirp_rows(ctx)
    if df.empty:
        return True, "no rows in window", None
    bad = df.loc[df["change_bps"].abs() > WIRP_MAX_ABS_MOVE_BPS,
                 ["timestamp", "meeting_date", "method", "change_bps"]].drop_duplicates()
    if bad.empty:
        return True, f"every implied move within +/-{WIRP_MAX_ABS_MOVE_BPS:g}bp", None
    return False, f"{len(bad)} implausible implied move(s)", bad


WIRP = DerivedMetric(
    "wirp", lambda p: p.wirp_dir, WIRP_KEYS, compute_wirp,
    checks=(Check("wirp_probabilities_valid", _check_wirp_probabilities),
            Check("wirp_moves_plausible", _check_wirp_moves)),
)

DERIVED_METRICS: dict[str, DerivedMetric] = {"wirp": WIRP}


# ------------------------------------------------------------------------------ step
def backfill_daily_derived(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    metrics: dict[str, DerivedMetric] | None = None,
) -> dict:
    """Recompute every registered metric over ``[start, end]`` (inclusive) and replace
    those days in its store. Returns per-metric diagnostics for the checks."""
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    metrics = DERIVED_METRICS if metrics is None else metrics
    out = {}
    for name, metric in metrics.items():
        df, diag = metric.compute(start, end, paths)
        store = metric.store(paths)
        parquet_store.prune_rows(store, "timestamp", pd.date_range(start, end, freq="D"))
        if not df.empty:
            parquet_store.write_partitioned(df, store, list(metric.key_columns))
        out[name] = {**diag, "rows": len(df)}
    return out


def _presence_check(metric: DerivedMetric) -> Check:
    """Test (a): every day in the window that HAS input data produced output."""

    def fn(ctx: StepContext):
        diag = ctx.output.get(metric.name, {})
        empty = diag.get("empty_days", {})
        if not empty:
            return True, f"{len(diag.get('days', []))} day(s) computed, {diag.get('rows', 0)} rows", None
        details = pd.DataFrame({"timestamp": list(empty), "reason": list(empty.values())})
        return False, f"{len(empty)} day(s) with input data produced no {metric.name}", details

    return Check(f"{metric.name}_present", fn)


def derived_checks(metrics: dict[str, DerivedMetric]) -> tuple[Check, ...]:
    checks: list[Check] = []
    for m in metrics.values():
        checks += [_presence_check(m),
                   revision_check(m.store, list(m.key_columns), name=f"{m.name}_no_revisions",
                                  rtol=0.0, atol=1e-10),
                   *m.checks]
    return tuple(checks)


def _run(ctx: StepContext) -> dict:
    return backfill_daily_derived(ctx.start, ctx.end, paths=ctx.paths,
                                  metrics=ctx.options.get("derived_metrics"))


DERIVED_STEP = Step("derived", _run, depends_on=("px", "raw"), checks=derived_checks(DERIVED_METRICS))
