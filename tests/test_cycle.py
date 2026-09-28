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
from infra.dashboard.wirp_selectors import build_schedule
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


# ------------------------------------------------------------------------- derived: WIRP
from infra.cycle.derived import DERIVED_METRICS, backfill_daily_derived  # noqa: E402
from infra.processing import statistics as stats  # noqa: E402

ZQ = [("ZQQ6", "2026-08-31"), ("ZQU6", "2026-09-30"), ("ZQV6", "2026-10-30"),
      ("ZQX6", "2026-11-30"), ("ZQZ6", "2026-12-31")]
WEEK = pd.date_range("2026-09-21", "2026-09-25")  # after Sep 16's meeting, before Oct 28's


def _settle(paths, rows) -> None:
    """rows: (ticker, day, price) written straight to the daily store (no API)."""
    df = pd.DataFrame(rows, columns=["ticker", "timestamp", "settlement_price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    df["open_interest"] = pd.array([None] * len(df), dtype="Int64")
    parquet_store.write_partitioned(stats.encode_daily(df[stats.DAILY_COLUMNS]),
                                    paths.daily_futures_dir, stats.DAILY_KEYS)


@pytest.fixture
def zq_env(tmp_path):
    """A flat 4.00% curve: every meeting prices an exact hold. Uses the REAL
    FOMC_MEETINGS schedule (Sep 16 past, Oct 28 and Dec 9 upcoming)."""
    paths = CyclePaths.under(tmp_path / "db")
    _contracts(paths, "ZQ", ZQ)
    _settle(paths, [("ZQQ6", "2026-08-28", 96.0)])  # August: the flat anchor month
    _settle(paths, [(t, d, 96.0) for t, _ in ZQ[1:] for d in WEEK])
    return paths


def _wirp(paths) -> pd.DataFrame:
    return parquet_store.read_partitioned(paths.wirp_dir)


def test_derived_persists_wirp_for_every_day_with_settlements(zq_env):
    report = run_daily_cycle(WEEK[0], WEEK[-1], steps=["derived"], run_day=WEEK[-1], paths=zq_env)
    assert report.ok, report.summary()
    df = _wirp(zq_env)
    assert sorted(df["timestamp"].unique()) == list(WEEK)
    held = df[df["outcome_step"] == 0]
    assert (held["probability"] == 1.0).all()  # flat curve -> certain hold everywhere
    assert set(df["meeting_date"].dt.date.astype(str)) == {"2026-10-28", "2026-12-09"}


def test_derived_output_matches_the_dashboard_function_exactly(zq_env):
    """No second WIRP implementation: the stored rows ARE build_schedule's output."""
    backfill_daily_derived(WEEK[0], WEEK[-1], paths=zq_env)
    stored = _wirp(zq_env)
    stored = stored[stored["timestamp"] == WEEK[2]].sort_values(["meeting_date", "outcome_step"])
    direct, _ = build_schedule("close", today=WEEK[2], contracts_file=zq_env.contracts_file,
                               close_root=zq_env.daily_futures_dir)
    direct = direct.sort_values(["meeting_date", "outcome_step"])
    assert list(stored["probability"]) == pytest.approx(list(direct["probability"]))
    assert list(stored["implied_rate"]) == pytest.approx(list(direct["implied_rate"]))


def test_recomputed_day_replaces_its_rows_no_stale_outcome_levels(zq_env):
    day = WEEK[2]
    # November (backing October's meeting) priced 12.5bp higher -> a 50/50 hold/+25 split
    _settle(zq_env, [("ZQX6", day, 95.875)])
    backfill_daily_derived(day, day, paths=zq_env)
    before = _wirp(zq_env)
    assert len(before[(before["timestamp"] == day) & (before["meeting_date"] == D("2026-10-28"))]) == 2

    _settle(zq_env, [("ZQX6", day, 96.0)])  # revised back to flat -> a single hold level
    backfill_daily_derived(day, day, paths=zq_env)
    after = _wirp(zq_env)
    octo = after[(after["timestamp"] == day) & (after["meeting_date"] == D("2026-10-28"))]
    assert list(octo["outcome_step"]) == [0] and octo["probability"].iloc[0] == 1.0


def test_derived_revision_detected_against_previous_vintage(zq_env):
    assert run_daily_cycle(WEEK[0], WEEK[-1], steps=["derived", "backup"], run_day=WEEK[-1], paths=zq_env).ok
    _settle(zq_env, [("ZQX6", WEEK[1], 95.875)])  # an input changes for an already-computed day
    report = run_daily_cycle(WEEK[0], WEEK[-1], steps=["derived", "backup"],
                             run_day=WEEK[-1] + pd.Timedelta(days=3), paths=zq_env)
    rev = next(c for c in report.outcome("derived").checks if c.name == "wirp_no_revisions")
    assert not rev.passed and set(rev.details["timestamp"]) == {WEEK[1]}
    assert report.outcome("backup").status == "skipped"


def test_derived_presence_fails_with_the_reason_when_anchor_is_missing(tmp_path):
    paths = CyclePaths.under(tmp_path / "db")
    _contracts(paths, "ZQ", ZQ)
    _settle(paths, [(t, d, 96.0) for t, _ in ZQ[1:] for d in WEEK])  # no August anchor
    report = run_daily_cycle(WEEK[0], WEEK[-1], steps=["derived"], run_day=WEEK[-1], paths=paths)
    pres = next(c for c in report.outcome("derived").checks if c.name == "wirp_present")
    assert not pres.passed and len(pres.details) == len(WEEK)
    assert pres.details["reason"].str.contains("anchor").all()


def test_wirp_implausible_move_check(zq_env):
    _settle(zq_env, [("ZQX6", d, 93.0) for d in WEEK])  # November at 7% vs a 4% anchor
    report = run_daily_cycle(WEEK[0], WEEK[-1], steps=["derived"], run_day=WEEK[-1], paths=zq_env)
    moves = next(c for c in report.outcome("derived").checks if c.name == "wirp_moves_plausible")
    assert not moves.passed


def test_derived_depends_on_px_and_raw():
    from infra.cycle.derived import DERIVED_STEP
    assert set(DERIVED_STEP.depends_on) == {"px", "raw"}
    assert "wirp" in DERIVED_METRICS


# ----------------------------------------------------------------------------- bmk
from infra.cycle.bmk import backfill_daily_pnl, backfill_daily_risk  # noqa: E402

ZQ25 = [("ZQF5", "2025-01-31"), ("ZQG5", "2025-02-28"), ("ZQH5", "2025-03-31")]
ZQ_SPECS = {"ZQ": DailyBackfillSpec(2)}


@pytest.fixture
def stir_env(tmp_path, monkeypatch):
    """ZQ, 2 nearest months. FakeStats prices rise 0.01 per CALENDAR day, so a
    weekday-to-weekday move is exactly 1bp and Friday-to-Monday is 3bp."""
    paths = CyclePaths.under(tmp_path / "db")
    _contracts(paths, "ZQ", ZQ25)
    monkeypatch.setattr(api, "fetch_statistics", FakeStats())
    return paths, {"specs": ZQ_SPECS, "refresh_contracts": False}


def _store(paths, sub) -> pd.DataFrame:
    df = parquet_store.read_partitioned(paths.bmk_root / sub)
    df["ticker"] = df["ticker"].astype(str)
    return df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def test_stir_dv01_is_point_value_x_one_bp(stir_env):
    paths, opts = stir_env
    run_daily_cycle("2025-01-06", "2025-01-10", steps=["px", "bmk_risk"], run_day="2025-01-10",
                    paths=paths, options=opts)
    risk = _store(paths, "Risk")
    assert set(risk["ticker"]) == {"ZQF5", "ZQG5"}
    assert risk["value"].tolist() == pytest.approx([41.67] * len(risk))
    assert (risk["currency"] == "USD").all()


def test_pnl_is_price_change_x_point_value_and_per_dv01_is_the_bp_move(stir_env):
    paths, opts = stir_env
    report = run_daily_cycle("2025-01-06", "2025-01-13", steps=["px", "bmk_risk", "bmk_pnl"],
                             run_day="2025-01-13", paths=paths, options=opts)
    assert report.ok, report.summary()
    pnl = _store(paths, "Pnl").set_index(["ticker", "timestamp"])
    tue = pnl.loc[("ZQF5", D("2025-01-07"))]
    assert tue["price_change"] == pytest.approx(0.01) and tue["pnl"] == pytest.approx(41.67)
    assert tue["pnl_per_dv01"] == pytest.approx(1.0)          # 1bp, weekday to weekday
    mon = pnl.loc[("ZQF5", D("2025-01-13"))]
    assert mon["prev_timestamp"] == D("2025-01-10") and mon["pnl_per_dv01"] == pytest.approx(3.0)


def test_first_day_of_a_history_has_no_pnl_rather_than_a_guess(stir_env):
    paths, opts = stir_env
    report = run_daily_cycle("2025-01-06", "2025-01-10", steps=["px", "bmk_risk", "bmk_pnl"],
                             run_day="2025-01-10", paths=paths, options=opts)
    assert report.ok, report.summary()
    assert D("2025-01-06") not in set(_store(paths, "Pnl")["timestamp"])


def test_contract_rolling_in_mid_window_still_gets_first_day_pnl(stir_env):
    paths, opts = stir_env
    # ZQF5 expires Fri Jan 31 -> ZQH5 enters the 2-contract universe Sat Feb 1
    report = run_daily_cycle("2025-01-27", "2025-02-07", steps=["px", "bmk_risk", "bmk_pnl"],
                             run_day="2025-02-07", paths=paths, options=opts)
    assert report.ok, report.summary()
    pnl = _store(paths, "Pnl").set_index(["ticker", "timestamp"])
    first = pnl.loc[("ZQH5", D("2025-02-03"))]
    assert first["prev_timestamp"] == D("2025-01-31")  # fetched thanks to PRIOR_SETTLEMENT_DAYS


def test_bond_futures_dv01_unavailable_with_reason_and_only_warns(env):
    paths, fake, opts = env
    report = run_daily_cycle("2025-01-06", "2025-01-10", steps=["px", "bmk_risk", "bmk_pnl"],
                             run_day="2025-01-10", paths=paths, options=opts)
    assert report.ok, report.summary()
    cov = next(c for c in report.outcome("bmk_risk").checks if c.name == "dv01_coverage")
    assert not cov.passed and cov.severity.value == "warn"
    assert cov.details["method"].str.contains("cheapest-to-deliver").all()
    pnl = _store(paths, "Pnl")
    assert pnl["pnl"].notna().all() and pnl["pnl_per_dv01"].isna().all()  # pnl yes, per-DV01 no
    assert set(pnl.loc[pnl["root"] == "FGBL", "currency"]) == {"EUR"}


def test_pnl_consistency_check_catches_a_stored_row_that_disagrees_with_the_spec(stir_env):
    """The check guards the COMPUTATION: run it directly against a store holding a row
    whose pnl != price_change x point value (e.g. a future code change breaking it)."""
    from infra.cycle.bmk import _check_pnl_consistent
    from infra.cycle.core import StepContext
    paths, opts = stir_env
    run_daily_cycle("2025-01-06", "2025-01-10", steps=["px", "bmk_risk", "bmk_pnl"],
                    run_day="2025-01-10", paths=paths, options=opts)
    ctx = StepContext(D("2025-01-06"), D("2025-01-10"), D("2025-01-10"), paths)
    assert _check_pnl_consistent(ctx)[0]
    tampered = _store(paths, "Pnl").head(1).assign(pnl=1.0)
    parquet_store.write_partitioned(tampered, paths.bmk_root / "Pnl", ["timestamp", "ticker", "bmk"])
    passed, _, details = _check_pnl_consistent(ctx)
    assert not passed and len(details) == 1


def test_carry_is_recognised_but_not_implemented_and_unknown_risk_rejected(stir_env):
    paths, _ = stir_env
    with pytest.raises(NotImplementedError):
        backfill_daily_risk("2025-01-06", "2025-01-10", risk="carry", paths=paths, specs=ZQ_SPECS)
    with pytest.raises(KeyError):
        backfill_daily_risk("2025-01-06", "2025-01-10", risk="vega", paths=paths, specs=ZQ_SPECS)


def test_bmk_steps_registered_in_dependency_order():
    names = [s.name for s in DEFAULT_STEPS]
    assert names == ["px", "raw", "derived", "bmk_risk", "bmk_pnl", "backup"]
