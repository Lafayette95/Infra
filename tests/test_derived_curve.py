"""The derived step's multi-store metrics (the Treasury curve: TreasuryCurves + TreasuryRV
from one fit) and the curve metric's registration - stub computations, temp stores."""
from __future__ import annotations

import pandas as pd

from infra.cycle import derived
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.storage import parquet_store

D = pd.Timestamp


def _stub_compute(start, end, paths):
    days = pd.bdate_range(start, end)
    main = pd.DataFrame({"timestamp": days.astype("datetime64[ms]"), "method": "spline", "rmse_bp": 1.0, "par_10y": 4.0})
    extra = pd.DataFrame({"timestamp": days.astype("datetime64[ms]"), "cusip": "X", "method": "spline", "zspread_bp": 0.5})
    return main, {"days": list(days), "empty_days": {D("2026-09-30"): "too few priced notes / bonds to fit (3 with a price)"},
                  "extra": {"treasury_rv": extra}}


def test_one_compute_fills_both_stores_and_presence_lists_empty_days(tmp_path):
    paths = CyclePaths.under(tmp_path)
    metric = derived.DerivedMetric("treasury_curve", lambda p: p.treasury_curves_dir, derived.CURVE_KEYS, _stub_compute,
                                   extra_stores=(derived.ExtraStore("treasury_rv", lambda p: p.treasury_rv_dir, derived.RV_KEYS),))
    out = derived.backfill_daily_derived("2026-09-28", "2026-10-02", paths=paths, metrics={"treasury_curve": metric})
    assert len(parquet_store.read_partitioned(paths.treasury_curves_dir)) == 5
    assert len(parquet_store.read_partitioned(paths.treasury_rv_dir)) == 5
    assert "extra" not in out["treasury_curve"]  # frames never ride into the checks' output
    ctx = StepContext(D("2026-09-28"), D("2026-10-02"), D("2026-10-05"), paths, output=out)
    ok, msg, details = derived._presence_check(metric).fn(ctx)
    assert not ok and list(details["timestamp"]) == [D("2026-09-30")]


def test_curve_metric_registered_with_both_revision_checks():
    names = [c.name for c in derived.DERIVED_STEP.checks]
    assert {"treasury_curve_present", "treasury_curve_no_revisions", "treasury_rv_no_revisions",
            "treasury_curve_fit_sane"} <= set(names)


def test_swap_closes_replace_only_pure_rows_by_instant(tmp_path):
    """The cycle owns the PURE closes only: a run replaces them (keyed by the snap instant,
    not midnight) and leaves the hand-built adjusted rows alone."""
    from infra.storage import parquet_store as ps
    store = tmp_path / "SwapCloses"
    def rows(method, rate):
        return pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-30 19:30"]).astype("datetime64[ms]"), "close": "NY1530",
                             "currency": "USD", "tenor": pd.Series([10], dtype="int32"), "method": method, "rate": rate,
                             "n_trades": pd.Series([5], dtype="int32"), "half_window_min": pd.Series([30], dtype="int32"),
                             "dispersion_bp": 0.3, "se_bp": 0.2})
    ps.write_partitioned(pd.concat([rows("pure", 3.50), rows("adjusted", 3.51)]), store, list(derived.SWAP_CLOSE_KEYS))
    derived._replace_pure_closes(store, rows("pure", 3.55), {"range": (D("2026-09-30"), D("2026-09-30"))}, None, None)
    back = ps.read_partitioned(store).set_index("method")["rate"].to_dict()
    assert back == {"pure": 3.55, "adjusted": 3.51}


def test_swap_closes_metric_registered():
    names = [c.name for c in derived.DERIVED_STEP.checks]
    assert {"swap_closes_present", "swap_closes_no_revisions", "swap_closes_sane"} <= set(names)


# ----------------------------------------------------------------------- OIS (SOFR) curve
def test_ois_bootstrap_reprices_every_pillar_and_short_nodes_hold():
    from infra.analytics import swap_curve as sc
    day = D("2026-10-05")
    quotes = {1: 4.49, 2: 4.69, 3: 4.74, 5: 4.78, 7: 4.84, 10: 4.92, 30: 5.00}
    short = [(1 / 12, 0.9966), (0.5, 0.9793)]
    curve, nodes = sc.bootstrap_ois(day, quotes, short_nodes=short)
    assert nodes.loc[nodes.source == "swap", "reprice_bp"].abs().max() < 1e-8
    assert abs(curve.discount([0.5])[0] - 0.9793) < 1e-12  # short-end nodes are kept exactly
    # beyond the last pillar the last forward continues
    f1, f2 = curve.forward(25.0, 30.0)[0], curve.forward(30.5, 35.0)[0]
    assert abs(f1 - f2) < 1e-9


def test_fill_missing_tenor_carries_its_fly_residual_point_in_time():
    from infra.analytics import swap_curve as sc
    days = pd.bdate_range("2026-09-01", periods=3)
    r = pd.DataFrame({10: [4.0, 4.1, 4.2], 15: [4.3, None, None], 20: [4.4, 4.5, 4.6]}, index=days)
    out, filled = sc.fill_missing_tenors(r, max_age_days=1)
    # day 1: line 10y-20y at 15y = 4.20, residual +0.10 carried (1 day old) -> 4.30 + 0.10
    assert abs(out.loc[days[1], 15] - 4.40) < 1e-12 and filled.loc[days[1], 15]
    # day 2: the residual is 2 days old (> max_age_days): not filled; nothing learnt from a fill
    assert pd.isna(out.loc[days[2], 15]) and not filled.loc[days[2], 15]
    assert not filled[10].any() and not filled[20].any()  # end tenors never filled


def test_ois_curve_metric_registered_after_swap_closes_with_warn_presence():
    names = list(derived.DERIVED_METRICS)
    assert names.index("ois_curve") > names.index("swap_closes")
    checks = {c.name: c for c in derived.DERIVED_STEP.checks}
    assert {"ois_curve_present", "ois_curve_no_revisions", "ois_curve_sane"} <= set(checks)
    assert checks["ois_curve_present"].severity.value == "warn"
    assert checks["swap_closes_present"].severity.value == "warn"
    assert checks["treasury_curve_present"].severity.value == "fail"
