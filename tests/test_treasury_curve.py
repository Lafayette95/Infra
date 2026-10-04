"""Our Treasury curve's math (infra/analytics/treasury_curve.py) on synthetic bonds priced
off a KNOWN curve - no data, no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.analytics import treasury_curve as tc

SETTLE = pd.Timestamp("2024-06-04")
TRUE = tc.SvenssonCurve(np.array([0.045, -0.005, -0.01, 0.01, 1.5, 12.0]))


def _universe(curve=TRUE, rich=None, rich_bp=0.0):
    bonds, dirty = [], []
    for i, m in enumerate(np.linspace(0.6, 29.5, 70)):
        mat = SETTLE + pd.Timedelta(days=int(m * 365.25))
        cpn = 1.0 + (i % 5)  # coupons 1..5%: a coupon effect a YTM regression would see
        t, a = tc.cash_flows(cpn, mat, SETTLE)
        d = float((a * curve.discount(t)).sum())
        if rich is not None and i == rich:
            d = float((a * curve.discount(t) * np.exp(-(rich_bp / 1e4) * t)).sum())
        bonds.append((t, a)); dirty.append(d)
    return bonds, np.array(dirty)


@pytest.mark.parametrize("method", ["spline", "svensson"])
def test_fits_recover_the_true_par_curve(method):
    bonds, dirty = _universe()
    w = np.ones(len(dirty))
    curve = tc.fit_spline(bonds, dirty, w)[0] if method == "spline" else tc.fit_svensson(bonds, dirty, w)[0]
    for m in (2, 5, 10, 30):
        assert tc.par_yield(curve, m) == pytest.approx(tc.par_yield(TRUE, m), abs=0.02)  # within 2bp


def test_zspread_and_leave_one_out_find_a_rich_bond():
    bonds, dirty = _universe(rich=35, rich_bp=-8.0)  # one bond 8bp rich
    w = np.ones(len(dirty))
    curve, res, h = tc.fit_spline(bonds, dirty, w)
    t, a = bonds[35]
    z = tc.zspread(curve, dirty[35], t, a)
    assert -8.5 < z / (1 - h[35]) < -6.0 and z < 0  # LOO undoes the bond pulling the curve to itself
    t, a = bonds[10]
    assert abs(tc.zspread(TRUE, dirty[10], t, a)) < 1e-6


def test_rolldown_is_zero_on_a_flat_curve_and_carry_positive_with_coupon_above_it():
    flat = tc.SvenssonCurve(np.array([0.04, 0.0, 0.0, 0.0, 1.0, 10.0]))
    t, a = tc.cash_flows(5.0, SETTLE + pd.Timedelta(days=3650), SETTLE)
    d = float((a * flat.discount(t)).sum())
    m = tc.bond_metrics(flat, d, 0.0, t, a, horizon_days=91)
    assert abs(m["rolldown_bp"]) < 0.5 and abs(m["zspread_bp"]) < 1e-6
    steep = tc.SvenssonCurve(np.array([0.05, -0.03, 0.0, 0.0, 2.0, 10.0]))  # upward sloping
    d = float((a * steep.discount(t)).sum())
    assert tc.bond_metrics(steep, d, 0.0, t, a, horizon_days=91)["rolldown_bp"] > 0  # rolls to lower yields: a gain
