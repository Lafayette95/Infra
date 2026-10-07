"""Zero-coupon inflation swaps: the package switch, the curve's forwards and real rates.
No network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.storage import parquet_store

D = pd.Timestamp


def test_par_trades_keeps_package_legs_only_when_allowed():
    from infra.config import SwapCurveSpec
    from infra.processing import dtcc_trades as dt
    spec = SwapCurveSpec("NA/Swap Infl Idx USD", "USA-CPI-U", 2, (1, 10), fixed_frequencies=("YEAR", "EXPI"))
    t = pd.DataFrame({"executed": [D("2026-09-15 15:00")] * 2, "effective": [D("2026-09-17")] * 2,
                      "maturity": [D("2036-09-17")] * 2, "rate": [2.51, 2.52], "notional": [1e8] * 2,
                      "notional_capped": [False] * 2, "cleared": ["I"] * 2, "platform": ["TWSF"] * 2, "block": [False] * 2,
                      "trade_id": ["S1", "S2"], "package": [False, True], "non_standard": [False] * 2,
                      "upfront": [False] * 2, "fixed_freq": ["EXPI", "YEAR"], "currency": ["USD"] * 2})
    assert dt.par_trades(t, spec, "USD")["trade_id"].tolist() == ["S1"]
    assert dt.par_trades(t, spec, "USD", allow_packages=True)["trade_id"].tolist() == ["S1", "S2"]


def test_curve_forward_breakevens_and_real_rates(tmp_path, monkeypatch):
    from infra.analytics import swap_curve as sc
    from infra.pipeline import inflation_swaps as isw
    closes = tmp_path / "closes"
    ts = D("2026-09-15 19:30")
    rows = pd.DataFrame({"timestamp": [ts] * 2, "close": "NY1530", "curve": "USD_CPI", "tenor": [5, 10], "method": "pure",
                         "rate": [2.50, 2.40], "n_trades": [3, 3], "half_window_min": [30, 30], "dispersion_bp": [0.1, 0.1],
                         "se_bp": [0.2, 0.2]})
    parquet_store.write_partitioned(rows.astype({"timestamp": "datetime64[ms]"}), closes, isw.CLOSE_KEYS)
    t = np.array([1.0, 40.0])
    flat = sc.OisCurve(t, np.exp(-0.045 * t))  # 4.5% continuously compounded nominal
    monkeypatch.setattr(isw, "read_ois_curves", lambda *a, **k: pd.DataFrame(
        {"timestamp": [ts, ts], "t_years": t, "df": flat.df}))
    cv, _ = isw.compute_curves("2026-09-15", "2026-09-15", closes_root=closes)
    r = cv.set_index("tenor")
    # 5y5y forward breakeven from the 5y and 10y zero-coupon rates
    assert r.loc[10, "fwd_pct"] == pytest.approx(((1.024 ** 10 / 1.025 ** 5) ** 0.2 - 1) * 100)
    assert r.loc[5, "fwd_from"] == 0 and r.loc[10, "fwd_from"] == 5
    # real = nominal (annual) / inflation (annual), compounded
    assert r.loc[10, "real_pct"] == pytest.approx((np.exp(0.045) / 1.024 - 1) * 100)


def test_inflation_metrics_registered_after_the_ois_curve():
    from infra.cycle import derived
    names = list(derived.DERIVED_METRICS)
    assert names.index("ois_curve") < names.index("inflation_swap_closes") < names.index("inflation_curve")
