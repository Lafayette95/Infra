"""Treasury event study (add-on MS's first event-profile provider): curve residuals and
event paths - pure, synthetic data."""
from __future__ import annotations

import numpy as np
import pytest
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


def _profiles():
    rows = [("new_issue", "10y", k, 0.02 * k) for k in range(0, 61)]  # cheapens 0.02bp a day for 60 days
    rows += [("otr_roll", "10y", k, (0.01 * k if k >= -1 else -0.03 * (-1 - k))) for k in range(-20, 41)]
    return pd.DataFrame(rows, columns=["event_type", "tenor", "rel_day", "mean_bp"])


def test_drift_rules_new_issue_then_roll_then_nothing():
    from infra.analytics.event_drift import bond_drift_bp
    bd = pd.bdate_range("2024-01-01", "2024-12-31")
    p = _profiles()
    new = {"tenor": "10y", "issue_date": bd[0], "is_otr": True, "successor_issue": bd[63]}
    d, rule = bond_drift_bp(new, bd[10], bd[30], bd, p)
    assert rule == "new_issue" and d == pytest.approx(0.02 * 20)
    old_otr = {"tenor": "10y", "issue_date": bd[0], "is_otr": True, "successor_issue": bd[80]}
    d, rule = bond_drift_bp(old_otr, bd[70], bd[90], bd, p)  # age 70 > 60: the roll rule; successor at 80
    assert rule == "otr_roll" and d == pytest.approx(0.01 * 10 - (-0.03 * 9))
    off = {"tenor": "10y", "issue_date": bd[0], "is_otr": False, "successor_issue": pd.NaT}
    assert bond_drift_bp(off, bd[70], bd[90], bd, p) == (0.0, "none")
    assert bond_drift_bp({**off, "tenor": "4y"}, bd[5], bd[9], bd, p)[1] == "none"  # no profile for the tenor


def test_aging_profile_and_drift_apply_to_every_bond_relative_only():
    from infra.analytics.event_drift import aging_drift_bp, aging_profile
    daily = pd.DataFrame({"tenor": "10y", "age_bd": np.arange(1, 601), "d1_bp": np.where(np.arange(1, 601) <= 60, 0.02, 0.0)})
    prof = aging_profile(daily)
    bd = pd.bdate_range("2023-01-02", "2026-12-31")
    young = {"tenor": "10y", "issue_date": bd[100]}
    old = {"tenor": "10y", "issue_date": bd[0]}
    d_young, _ = aging_drift_bp(young, bd[110], bd[130], bd, prof)  # ages 10 -> 30: inside the cheapening phase
    d_old, _ = aging_drift_bp(old, bd[110], bd[130], bd, prof)      # ages 110 -> 130: none
    assert d_young == pytest.approx(0.02 * 20) and d_old == pytest.approx(0.0)
    assert aging_drift_bp({"tenor": "4y", "issue_date": bd[0]}, bd[1], bd[5], bd, prof) == (0.0, "none")
