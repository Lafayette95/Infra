"""The Prefect layer (infra/cycle/flows.py): task states map back to step outcomes, and
the runner's ordering/halting survives being executed as Prefect tasks. Runs against
Prefect's own throwaway test API - no server, nothing touches the real ~/Database."""
from __future__ import annotations

import pandas as pd
import pytest
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from infra.cycle.core import Check, Step
from infra.cycle.flows import SCHEDULE_CRON, SCHEDULE_TZ, prefect_execute
from infra.cycle.paths import CyclePaths
from infra.cycle.runner import _with_backup_last, run_daily_cycle


@pytest.fixture(scope="module", autouse=True)
def harness():
    with prefect_test_harness():
        yield


def _registry():
    fails = Check("always_fails", lambda ctx: (False, "nope", None))
    return _with_backup_last((
        Step("good", lambda ctx: {"n": 1}),
        Step("bad", lambda ctx: {}, depends_on=("good",), checks=(fails,)),
        Step("after_bad", lambda ctx: {}, depends_on=("bad",)),
        Step("independent", lambda ctx: {}, depends_on=("good",)),
    ))


def test_prefect_executor_preserves_outcomes_and_dependent_skipping(tmp_path):
    paths = CyclePaths.under(tmp_path / "db")

    @flow
    def run():
        return run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths,
                               registry=_registry(), execute=prefect_execute)

    report = run()
    status = {o.name: o.status for o in report.outcomes}
    assert status == {"good": "ok", "bad": "failed", "after_bad": "skipped",
                      "independent": "ok", "backup": "skipped"}
    assert report.outcome("good").output == {"n": 1}  # task results flow back intact


def test_a_step_that_raises_inside_a_task_becomes_an_error_outcome(tmp_path):
    paths = CyclePaths.under(tmp_path / "db")

    def boom(ctx):
        raise RuntimeError("down")

    @flow
    def run():
        return run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths,
                               registry=(Step("x", boom),), execute=prefect_execute)

    report = run()
    assert report.outcome("x").status == "error" and "down" in report.outcome("x").error


def test_schedule_runs_tuesday_to_saturday_mornings_for_the_prior_trading_day():
    # 06:00 New York, local like the Mac's scheduled wake - so they never drift apart at DST
    assert (SCHEDULE_CRON, SCHEDULE_TZ) == ("0 6 * * 2-6", "America/New_York")
