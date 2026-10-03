"""Treasury event study (add-on MS's first event-profile provider): curve residuals and
event paths - pure, synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.analytics.event_study import EVENT_PROFILE_COLUMNS, curve_residuals, event_paths, event_profiles

D = pd.Timestamp


def _market(days, rich_bond="RICH", rich_bp=-2.0):
    mats = np.linspace(1.5, 29.5, 40)
    cusips = [f"B{i}" for i in range(40)]
    rows, maturity = [], {}
    for c, m in zip(cusips, mats):
        maturity[c] = D("2024-01-01") + pd.Timedelta(days=int(m * 365.25))
    maturity[rich_bond] = D("2024-01-01") + pd.Timedelta(days=int(10.2 * 365.25))
    for d in days:
        for c in [*cusips, rich_bond]:
            m = (maturity[c] - d).days / 365.25
            y = 3.0 + 1.5 * (1 - np.exp(-m / 5))  # a smooth curve, %
            rows.append((d, c, y + (rich_bp / 100 if c == rich_bond else 0.0)))
    return pd.DataFrame(rows, columns=["timestamp", "cusip", "yield_eod"]), pd.Series(maturity)


def test_residual_finds_a_rich_bond_on_a_smooth_curve():
    days = pd.bdate_range("2024-01-02", periods=3)
    prices, mat = _market(days)
    r = curve_residuals(prices, mat)
    rich = r[r["cusip"] == "RICH"]["residual_bp"]
    assert np.allclose(rich, -2.0, atol=0.3)
    assert r[r["cusip"] != "RICH"]["residual_bp"].abs().max() < 0.3


def test_event_path_is_measured_from_the_anchor_day():
    days = pd.bdate_range("2024-01-02", periods=10)
    res = pd.DataFrame({"timestamp": days, "cusip": "X", "residual_bp": np.arange(10.0)})
    ev = pd.DataFrame({"event_type": ["auction"], "tenor": ["10y"], "cusip": ["X"], "event_day": [days[5]]})
    p = event_paths(res, ev, window=(-2, 2), anchor_offset=-1)
    assert list(p["rel_day"]) == [-2, -1, 0, 1, 2] and list(p["change_bp"]) == [-1.0, 0.0, 1.0, 2.0, 3.0]
    prof = event_profiles(p)
    assert list(prof.columns) == EVENT_PROFILE_COLUMNS
