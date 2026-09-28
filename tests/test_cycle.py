"""Offline tests for the daily cycle (infra/cycle): universe, px step + checks, vintages,
revision detection, dependency halting. Fake API throughout - no cost, and every path is
under tmp_path (CyclePaths.under)."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.api import databento_client as api
from infra.config import DailyBackfillSpec
from infra.cycle import vintage
from infra.cycle.checks import compare_to_vintage
from infra.cycle.core import Check, Step
from infra.cycle.paths import CyclePaths
from infra.cycle.px import backfill_daily_px_data, last_weekday
from infra.cycle.runner import (
    DEFAULT_STEPS, _with_backup_last, run_daily_cycle, run_scheduled_daily, scheduled_windows,
)
from infra.cycle.universe import daily_universe, snapshot_grid_floor
from infra.pipeline import daily as dl
from infra.storage import contract_store, parquet_store

D = pd.Timestamp
ZN = [("ZNH5", "2025-03-20"), ("ZNM5", "2025-06-18"), ("ZNU5", "2025-09-19")]
FGBL = [("FGBL SI 20250306 PS", "2025-03-06"), ("FGBL SI 20250606 PS", "2025-06-06")]
SPECS = {"ZN": DailyBackfillSpec(2), "FGBL": DailyBackfillSpec(1)}


def _contracts(paths: CyclePaths, root: str, rows, activation="2024-01-01") -> None:
    contract_store.write_contracts(paths.contracts_file, pd.DataFrame({
        "root": root, "ticker": [t for t, _ in rows], "instrument_id": range(len(rows)),
        "expiry": pd.to_datetime([e for _, e in rows]).astype("datetime64[ms]"),
        "activation": pd.to_datetime([activation] * len(rows)).astype("datetime64[ms]"),
    }))


class FakeStats:
    """Stand-in for api.fetch_statistics: one settlement per weekday. ``prices`` overrides
    a (ticker, day) price; ``silent`` tickers return nothing (missing data)."""

    def __init__(self):
        self.calls, self.prices, self.silent = [], {}, set()

    def __call__(self, dataset, symbol, start, end, **_):
        self.calls.append((symbol, D(start), D(end)))
        days = [d for d in pd.date_range(start, end, freq="D", inclusive="left") if d.weekday() < 5]
        if symbol in self.silent or not days:
            return pd.DataFrame(columns=["ts_recv", "ts_ref", "stat_type", "price", "quantity"])
        base = 100.0 + (hash(symbol) % 7)
        return pd.DataFrame({
            "ts_recv": pd.to_datetime([d + pd.Timedelta(hours=20) for d in days], utc=True),
            "ts_ref": pd.to_datetime(days, utc=True),
            "stat_type": 3,
            # a function of the DATE, never of the requested window - re-fetching a
            # different window must return identical values for the same day
            "price": [self.prices.get((symbol, d), base + 0.01 * (d - D("2024-01-01")).days) for d in days],
            "quantity": None,
        })


@pytest.fixture
def env(tmp_path, monkeypatch):
    paths = CyclePaths.under(tmp_path / "db")
    _contracts(paths, "ZN", ZN)
    _contracts(paths, "FGBL", FGBL)
    fake = FakeStats()
    monkeypatch.setattr(api, "fetch_statistics", fake)
    opts = {"specs": SPECS, "refresh_contracts": False}
    return paths, fake, opts


# ------------------------------------------------------------------------ paths / grid
def test_cycle_paths_rebase_keeps_the_config_layout(tmp_path):
    p, d = CyclePaths.under(tmp_path), CyclePaths.default()
    assert p.daily_futures_dir == tmp_path / d.daily_futures_dir.relative_to(d.database_root)
    assert p.vintage_root.parent == tmp_path


def test_snapshot_grid_is_stable_across_consecutive_days():
    days = pd.date_range("2025-01-06", "2025-01-10")
    assert len({snapshot_grid_floor(d) for d in days}) <= 2  # at most one grid boundary crossed
    assert all(snapshot_grid_floor(d) <= d for d in days)


# ----------------------------------------------------------------------------- universe
def test_universe_is_point_in_time_and_rolls_as_contracts_expire(env):
    paths, _, _ = env
    members, errors = daily_universe("2025-03-17", "2025-03-25", paths=paths,
                                     specs={"ZN": DailyBackfillSpec(2)}, refresh_contracts=False)
    assert not errors
    assert members["ZNH5"].last == D("2025-03-20")      # leaves the universe on expiry
    assert members["ZNU5"].first == D("2025-03-21")     # rolls in the day after
    assert members["ZNM5"].first == D("2025-03-17") and members["ZNM5"].last == D("2025-03-25")


def test_universe_skips_disabled_roots_and_reports_unknown_contracts(env):
    paths, _, _ = env
    members, errors = daily_universe(
        "2025-01-06", "2025-01-10", paths=paths, refresh_contracts=False,
        specs={"ZN": DailyBackfillSpec(1, enabled=False), "ZB": DailyBackfillSpec(2)})
    assert members == {} and "ZB" in errors  # nothing cached for ZB


def test_member_not_expected_before_its_listing_date(env):
    paths, _, _ = env
    _contracts(paths, "ZF", [("ZFH5", "2025-03-31")], activation="2025-01-08")
    members, _ = daily_universe("2025-01-06", "2025-01-10", paths=paths,
                                specs={"ZF": DailyBackfillSpec(1)}, refresh_contracts=False)
    assert not members["ZFH5"].expected_on(D("2025-01-07"))
    assert members["ZFH5"].expected_on(D("2025-01-08"))


# ------------------------------------------------------------------------------ px step
def test_px_step_fetches_only_universe_windows_and_all_checks_pass(env):
    paths, fake, opts = env
    report = run_daily_cycle("2025-01-06", "2025-01-10", steps=["px"], run_day="2025-01-10",
                             paths=paths, options=opts)
    assert report.ok, report.summary()
    assert {c[0] for c in fake.calls} == {"ZNH5", "ZNM5", "FGBL SI 20250306 PS"}
    df = dl.read_daily_from_disk(["ZNH5"], D("2025-01-01"), D("2025-02-01"), root=paths.daily_futures_dir)
    assert len(df) == 5


def test_px_present_fails_when_one_contract_is_missing(env):
    paths, fake, opts = env
    fake.silent = {"ZNM5"}
    report = run_daily_cycle("2025-01-06", "2025-01-10", steps=["px"], run_day="2025-01-10",
                             paths=paths, options=opts)
    px = report.outcome("px")
    check = next(c for c in px.checks if c.name == "px_present")
    assert px.status == "failed" and not check.passed
    assert list(check.details["ticker"]) == ["ZNM5"]


def test_whole_dataset_missing_is_a_warning_not_a_failure(env):
    paths, fake, opts = env
    fake.silent = {"FGBL SI 20250306 PS"}  # the only FGBL member: looks like a holiday
    report = run_daily_cycle("2025-01-06", "2025-01-10", steps=["px"], run_day="2025-01-10",
                             paths=paths, options=opts)
    px = report.outcome("px")
    warn = next(c for c in px.checks if c.name == "px_dataset_present")
    assert px.status == "ok" and not warn.passed and "XEUR.EOBI" in warn.message


def test_px_outlier_check_flags_a_bad_print(env):
    paths, fake, opts = env
    run_daily_cycle("2024-10-01", "2025-01-03", steps=["px"], run_day="2025-01-03", paths=paths, options=opts)
    fake.prices[("ZNH5", D("2025-01-08"))] = 150.0  # absurd jump vs a steady 0.01/day drift
    report = run_daily_cycle("2025-01-06", "2025-01-10", steps=["px"], run_day="2025-01-10",
                             paths=paths, options=opts)
    out = next(c for c in report.outcome("px").checks if c.name == "px_outliers")
    assert not out.passed and "ZNH5" in set(out.details["ticker"])


def test_last_weekday():
    assert last_weekday(D("2025-01-11")) == D("2025-01-10")  # Sat -> Fri
    assert last_weekday(D("2025-01-08")) == D("2025-01-08")


# --------------------------------------------------------------- vintages / revisions
def test_full_cycle_writes_a_vintage_and_second_day_compares_against_it(env):
    paths, fake, opts = env
    day1 = run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths, options=opts)
    assert day1.ok, day1.summary()
    assert vintage.list_vintages(paths) == [D("2025-01-10")]

    # next run day, nothing revised upstream: revision check passes against day-1 vintage
    day2 = run_scheduled_daily("2025-01-13", paths=paths, options=opts)
    assert day2.ok, day2.summary()
    rev = next(c for c in day2.outcome("px").checks if c.name == "px_no_revisions")
    assert rev.passed and "2025-01-10" in rev.message
    assert vintage.list_vintages(paths) == [D("2025-01-10"), D("2025-01-13")]


def test_upstream_revision_is_caught_fails_px_and_skips_backup(env):
    paths, fake, opts = env
    assert run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths, options=opts).ok
    fake.prices[("ZNH5", D("2025-01-09"))] = 99.0  # exchange revises an already-published day

    report = run_scheduled_daily("2025-01-13", paths=paths, options=opts)
    px = report.outcome("px")
    rev = next(c for c in px.checks if c.name == "px_no_revisions")
    assert not rev.passed and px.status == "failed"
    assert (rev.details["ticker"] == "ZNH5").all() and (rev.details["column"] == "settlement_price").all()
    assert report.outcome("backup").status == "skipped"  # a failed day never becomes a baseline
    assert vintage.list_vintages(paths) == [D("2025-01-10")]
    assert not report.ok


def test_without_force_refetch_a_revision_is_never_even_seen(env):
    """Documents WHY the scheduled run force-refetches: a normal (Rule 2.1) run never
    re-queries covered days, so the revision check has nothing new to compare."""
    paths, fake, opts = env
    run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths, options=opts)
    fake.prices[("ZNH5", D("2025-01-09"))] = 99.0
    report = run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-13", paths=paths, options=opts)
    assert report.ok  # covered days not re-queried -> revision invisible


def test_compare_to_vintage_flags_changes_and_deletions_but_not_new_rows(tmp_path):
    keys = ["timestamp", "ticker"]
    ts = lambda *d: pd.to_datetime(list(d)).astype("datetime64[ms]")  # noqa: E731
    old = pd.DataFrame({"timestamp": ts("2025-01-06", "2025-01-07", "2025-01-08"),
                        "ticker": ["A", "A", "A"], "v": [1, 2, 3]})
    new = pd.DataFrame({"timestamp": ts("2025-01-06", "2025-01-07", "2025-01-09"),
                        "ticker": ["A", "A", "A"], "v": [1, 5, 4]})
    parquet_store.write_partitioned(old, tmp_path / "old", keys)
    parquet_store.write_partitioned(new, tmp_path / "new", keys)
    diffs = compare_to_vintage(tmp_path / "new", tmp_path / "old", keys, D("2025-01-01"), D("2025-01-31"))
    assert sorted(diffs["column"]) == ["<row deleted>", "v"]
    assert set(diffs["timestamp"]) == {D("2025-01-07"), D("2025-01-08")}  # Jan 9 is new history


def test_latest_vintage_before_excludes_same_day(env):
    paths, _, _ = env
    vintage.snapshot(D("2025-01-10"), paths)
    vintage.snapshot(D("2025-01-13"), paths)
    assert vintage.latest_before(D("2025-01-13"), paths).name == "2025-01-10"
    assert vintage.latest_before(D("2025-01-10"), paths) is None


def test_snapshot_does_not_copy_vintages_into_themselves(env):
    paths, _, _ = env
    vintage.snapshot(D("2025-01-10"), paths)
    second = vintage.snapshot(D("2025-01-13"), paths)
    assert not (second / "_vintages").exists()


# ------------------------------------------------------------------------ orchestration
def test_step_that_raises_is_error_and_its_dependents_are_skipped(env):
    paths, _, opts = env

    def boom(ctx):
        raise RuntimeError("upstream down")

    registry = _with_backup_last((Step("px", boom), Step("derived", lambda ctx: {}, depends_on=("px",))))
    report = run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths,
                             options=opts, registry=registry)
    assert [o.status for o in report.outcomes] == ["error", "skipped", "skipped"]


def test_warn_and_info_checks_do_not_block_dependents(env):
    paths, _, opts = env
    from infra.cycle.core import Severity
    failing = lambda ctx: (False, "meh", None)  # noqa: E731
    registry = _with_backup_last((
        Step("a", lambda ctx: {}, checks=(Check("w", failing, Severity.WARN), Check("i", failing, Severity.INFO))),
        Step("b", lambda ctx: {}, depends_on=("a",)),
    ))
    report = run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths,
                             options=opts, registry=registry)
    assert report.ok and [o.status for o in report.outcomes] == ["ok", "ok", "ok"]


def test_a_crashing_check_counts_as_failed_not_passed(env):
    paths, _, opts = env

    def crash(ctx):
        raise ValueError("bug in check")

    registry = (Step("a", lambda ctx: {}, checks=(Check("c", crash),)),)
    report = run_daily_cycle("2025-01-06", "2025-01-10", run_day="2025-01-10", paths=paths,
                             options=opts, registry=registry)
    assert report.outcome("a").status == "failed"


def test_scheduled_windows_are_business_days_with_per_step_overrides():
    w = scheduled_windows(D("2025-01-13"))  # a Monday
    assert w["px"] == (D("2025-01-08"), D("2025-01-13"))  # T-3 business days = Wed
    w = scheduled_windows(D("2025-01-13"), overrides={"px": 1})
    assert w["px"] == (D("2025-01-10"), D("2025-01-13"))


def test_default_registry_order_and_backup_depends_on_everything():
    names = [s.name for s in DEFAULT_STEPS]
    assert names[-1] == "backup" and names[0] == "px"
    assert set(DEFAULT_STEPS[-1].depends_on) == set(names[:-1])


def test_unknown_step_name_raises(env):
    paths, _, opts = env
    with pytest.raises(KeyError):
        run_daily_cycle("2025-01-06", "2025-01-10", steps=["nope"], paths=paths, options=opts)


def test_px_standalone_function_usable_without_runner(env):
    paths, fake, _ = env
    out = backfill_daily_px_data("2025-01-06", "2025-01-10", paths=paths, specs=SPECS, refresh_contracts=False)
    assert out["rows"] > 0 and not out["fetch_errors"]
