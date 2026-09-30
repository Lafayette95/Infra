"""Prefect wrapper around infra.cycle.runner - the ONLY scheduler-aware module.

The runner stays the single source of truth for step order, checks and halting
dependents; Prefect contributes the schedule, one task per step (shown failed in the UI
when that step's blocking checks fail), run history and logs. Nothing here decides
anything the runner doesn't - swap Prefect out and only this file changes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE

from infra.cycle.core import StepContext, StepOutcome
from infra.cycle.network import wait_for_network
from infra.cycle.runner import execute_step, run_daily_cycle, run_scheduled_daily

# Tue-Sat 06:00 New York time (10:00 UTC in EDT, 11:00 UTC in EST), processing through
# the previous trading day. Databento's queryable end runs BEHIND now (verified
# 2026-09-28: GLBX exactly 8h, Eurex about a day - see api.available_end), so a
# same-evening run could never see that day's settlement; by 10:00 UTC, GLBX has fully
# published T-1. Eurex's longer lag means it's judged a day older - the presence check
# handles that per dataset, and the T-3 re-fetch window picks each day up once it's
# published. LOCAL time on purpose (changed from 10:00 UTC 2026-09-30): the Mac's
# scheduled wake (`pmset repeat`, 05:55) is local time, so a UTC schedule would drift an
# hour against it at every DST change - from November the run would come BEFORE the wake.
SCHEDULE_CRON = "0 6 * * 2-6"
SCHEDULE_TZ = "America/New_York"


class StepNotOk(Exception):
    """Raised inside a step's task so Prefect marks the task failed; carries the
    outcome back to the runner, which decides what to skip."""

    def __init__(self, outcome: StepOutcome):
        super().__init__(f"{outcome.name}: {outcome.status}" + (f" - {outcome.error}" if outcome.error else ""))
        self.outcome = outcome


@task(cache_policy=NO_CACHE)  # never cache: every run must actually re-run its step
def _step_task(step, ctx: StepContext) -> StepOutcome:
    outcome = execute_step(step, ctx)
    if outcome.status != "ok":
        raise StepNotOk(outcome)
    return outcome


def prefect_execute(step, ctx: StepContext) -> StepOutcome:
    """Executor for the runner: each step as its own named Prefect task."""
    state = _step_task.with_options(name=step.name, task_run_name=step.name)(step, ctx, return_state=True)
    if state.is_completed():
        return state.result()
    exc = state.result(raise_on_failure=False)
    if isinstance(exc, StepNotOk):
        return exc.outcome
    return StepOutcome(step.name, "error", ctx.start, ctx.end, error=f"{type(exc).__name__}: {exc}")


@contextmanager
def _stay_awake():
    """Hold off idle sleep for the length of a run (macOS ``caffeinate -i``, tied to this
    process). On a laptop the scheduled wake only guarantees the run STARTS - a Mac going
    back to sleep mid-fetch would stall it. No-op where caffeinate doesn't exist."""
    exe = shutil.which("caffeinate")
    proc = subprocess.Popen([exe, "-i", "-w", str(os.getpid())]) if exe else None
    try:
        yield
    finally:
        if proc is not None:
            proc.terminate()


@flow(name="daily-cycle")
def daily_cycle_flow(today: str | None = None) -> str:
    """What the schedule runs: every step over T-N..T (per-step N), force_refetch on."""
    with _stay_awake():
        wait_for_network()  # a dark-woken Mac has no network - wait / fail clearly (infra.cycle.network)
        report = run_scheduled_daily(today, execute=prefect_execute)
    get_run_logger().info(report.summary())
    report.raise_for_status()  # a failed cycle is a failed flow run
    return report.summary()


@flow(name="daily-cycle-backfill")
def backfill_flow(start: str, end: str, steps: list[str] | None = None, force_refetch: bool = False) -> str:
    """History backfill through Prefect (for run tracking in the UI)."""
    with _stay_awake():
        wait_for_network()
        report = run_daily_cycle(start, end, steps=steps, force_refetch=force_refetch, execute=prefect_execute)
    get_run_logger().info(report.summary())
    report.raise_for_status()
    return report.summary()
