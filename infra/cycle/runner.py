"""Runs the daily cycle: each step, then its checks; a blocking (``fail``) check or a step
that raises marks it not-ok and every step depending on it is skipped.

Two entry points over the SAME steps:
* ``run_daily_cycle(start, end)`` - manual / history backfill: every step over one
  ``[start, end]`` window, ``force_refetch`` off by default (Rule 2.1 applies as usual).
* ``run_scheduled_daily(today)`` - what the scheduler calls: each step over
  ``[T - its revision_window_days business days, T]`` with ``force_refetch=True``, so
  the last few days are re-queried to catch upstream revisions.
"""
from __future__ import annotations

import dataclasses
import logging

import pandas as pd

from infra.cycle import vintage
from infra.cycle.backup import BACKUP_STEP
from infra.cycle.core import CheckResult, CycleReport, Severity, Step, StepContext, StepOutcome
from infra.cycle.derived import DERIVED_STEP
from infra.cycle.paths import CyclePaths
from infra.cycle.px import PX_STEP
from infra.cycle.raw import RAW_STEP

log = logging.getLogger(__name__)


def _with_backup_last(steps: tuple[Step, ...]) -> tuple[Step, ...]:
    """Backup runs last and depends on every other step, whatever the registry holds."""
    others = tuple(s for s in steps if s.name != BACKUP_STEP.name)
    return (*others, dataclasses.replace(BACKUP_STEP, depends_on=tuple(s.name for s in others)))


DEFAULT_STEPS: tuple[Step, ...] = _with_backup_last((PX_STEP, RAW_STEP, DERIVED_STEP))


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def _log_check(step: str, r: CheckResult) -> None:
    if r.passed:
        log.info("[%s] %s: pass - %s", step, r.name, r.message)
    elif r.severity is Severity.INFO:
        log.info("[%s] %s: %s", step, r.name, r.message)
    elif r.severity is Severity.WARN:
        log.warning("[%s] %s: %s", step, r.name, r.message)
    else:
        log.error("[%s] %s: FAILED - %s", step, r.name, r.message)


def run_steps(
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    *,
    run_day: pd.Timestamp,
    paths: CyclePaths,
    force_refetch: bool,
    options: dict | None = None,
    registry: tuple[Step, ...] = DEFAULT_STEPS,
) -> CycleReport:
    """Run the registry's steps named in ``windows`` (in registry order - which is also
    dependency order), each over its own inclusive ``(start, end)``."""
    known = {s.name for s in registry}
    unknown = set(windows) - known
    if unknown:
        raise KeyError(f"unknown step(s) {sorted(unknown)}; registry has {sorted(known)}")
    reference = vintage.latest_before(run_day, paths)
    outcomes: dict[str, StepOutcome] = {}
    for step in registry:
        if step.name not in windows:
            continue
        start, end = (pd.Timestamp(d).normalize() for d in windows[step.name])
        blocked = [d for d in step.depends_on if d in outcomes and outcomes[d].status != "ok"]
        if blocked:
            outcomes[step.name] = StepOutcome(step.name, "skipped", start, end,
                                              error=f"upstream not ok: {', '.join(blocked)}")
            log.warning("[%s] skipped - upstream not ok: %s", step.name, blocked)
            continue
        ctx = StepContext(start, end, run_day, paths, force_refetch, reference, dict(options or {}))
        try:
            ctx.output = step.fn(ctx) or {}
        except Exception as exc:
            log.exception("[%s] step raised", step.name)
            outcomes[step.name] = StepOutcome(step.name, "error", start, end,
                                              error=f"{type(exc).__name__}: {exc}")
            continue
        results = [c.run(ctx) for c in step.checks]
        for r in results:
            _log_check(step.name, r)
        status = "failed" if any(r.blocking for r in results) else "ok"
        outcomes[step.name] = StepOutcome(step.name, status, start, end, results, ctx.output)
    report = CycleReport(run_day, list(outcomes.values()))
    log.info("%s", report.summary())
    return report


def run_daily_cycle(
    start,
    end,
    *,
    steps: list[str] | None = None,
    force_refetch: bool = False,
    run_day=None,
    paths: CyclePaths | None = None,
    options: dict | None = None,
    registry: tuple[Step, ...] = DEFAULT_STEPS,
) -> CycleReport:
    """Every requested step (default: all, backup included) over the same inclusive
    ``[start, end]`` - the history-backfill entry point."""
    names = [s.name for s in registry] if steps is None else steps
    return run_steps(
        {n: (start, end) for n in names},
        run_day=pd.Timestamp(run_day).normalize() if run_day is not None else _today(),
        paths=paths or CyclePaths.default(), force_refetch=force_refetch,
        options=options, registry=registry,
    )


def scheduled_windows(
    today: pd.Timestamp,
    registry: tuple[Step, ...] = DEFAULT_STEPS,
    overrides: dict[str, int] | None = None,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """``{step: (T - N business days, T)}`` with N = the step's own revision window (or an
    override for this run)."""
    overrides = overrides or {}
    today = pd.Timestamp(today).normalize()
    return {
        s.name: (today - pd.offsets.BDay(overrides.get(s.name, s.revision_window_days)), today)
        for s in registry
    }


def run_scheduled_daily(
    today=None,
    *,
    paths: CyclePaths | None = None,
    revision_window_days: dict[str, int] | None = None,
    options: dict | None = None,
    registry: tuple[Step, ...] = DEFAULT_STEPS,
) -> CycleReport:
    """The scheduler's entry point: T-N..T per step, ``force_refetch=True``."""
    today = pd.Timestamp(today).normalize() if today is not None else _today()
    return run_steps(
        scheduled_windows(today, registry, revision_window_days),
        run_day=today, paths=paths or CyclePaths.default(), force_refetch=True,
        options=options, registry=registry,
    )
