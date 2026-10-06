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
