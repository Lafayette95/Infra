"""FX-hedged yields (infra.analytics.hedged_yields): conversions to semi-annual, the hedge
identity and its basis sign, and the basis fill rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.analytics import hedged_yields as hy

D = pd.Timestamp


def test_conversions_to_semiannual():
    assert hy.annual_to_semiannual(4.0) == pytest.approx(200 * (np.sqrt(1.04) - 1))
    # SOFR 1y par (annual ACT/360) vs CORRA 5y (semi-annual ACT/365): the latter is already semi
    assert hy.par_ois_to_semiannual(4.0, day_basis=360, freq_months=12, tenor_years=1) == pytest.approx(
        hy.annual_to_semiannual(4.0 * 365 / 360))
    assert hy.par_ois_to_semiannual(3.0, day_basis=365, freq_months=6, tenor_years=5) == pytest.approx(3.0)
    assert hy.df_to_semiannual((1 + 0.02) ** (-2 * 0.5), 0.5) == pytest.approx(4.0)


def test_a_negative_basis_is_a_pickup_for_the_usd_investor_and_a_cost_for_the_foreign_one():
    # USD investor in a JGB: y_JP - (TONA + b_JPY) + SOFR; b_JPY = -40bp adds 40bp
    usd_in_jgb = hy.hedged_yield(3.0, 1.0, -40.0, 4.0, 0.0)
    assert usd_in_jgb == pytest.approx(3.0 - 1.0 + 0.40 + 4.0)
    # Japanese investor in a Treasury: y_US - SOFR + (TONA + b_JPY); the same -40bp costs 40bp
    jp_in_ust = hy.hedged_yield(4.5, 4.0, 0.0, 1.0, -40.0)
    assert jp_in_ust == pytest.approx(4.5 - 4.0 + 1.0 - 0.40)
    # the pickups over each base's own yield mirror each other (same tenor, same day)
    assert (usd_in_jgb - 4.5) == pytest.approx(-(jp_in_ust - 3.0))


def test_basis_fill_same_day_then_interpolated_then_carried():
    days = pd.bdate_range("2026-09-01", periods=4)
    obs = pd.DataFrame({60: [-10.0, np.nan, np.nan, np.nan], 24: [np.nan, -6.0, np.nan, np.nan],
                        120: [np.nan, -12.0, np.nan, np.nan]}, index=days)
    f = hy.fill_basis(obs, 60, carry_days=10)
    assert f.loc[days[0], "basis_source"] == "same_day"
    assert f.loc[days[1], "basis_bp"] == pytest.approx(-6 + (-12 + 6) * (60 - 24) / (120 - 24))
    assert f.loc[days[2], "basis_bp"] == -10.0 and f.loc[days[2], "basis_source"].startswith("carried")


def test_short_basis_prefers_a_recent_3m_over_a_same_day_1y():
    days = pd.bdate_range("2026-09-01", periods=3)
    obs = pd.DataFrame({3: [-20.0, np.nan, np.nan], 12: [-30.0, -31.0, -32.0]}, index=days)
    s = hy.short_basis(obs, carry_days=1)
    assert list(s["basis_bp"]) == [-20.0, -20.0, -32.0]          # carried 1 day, then the 1y
    assert s["basis_source"].iloc[2] == "1y same day"
