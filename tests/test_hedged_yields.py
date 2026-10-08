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


def test_fx_hedge_cost_same_day_then_carries_the_spread_over_ois_not_the_level():
    days = pd.bdate_range("2026-01-05", periods=15)
    ois = pd.Series(2.0, index=days)
    ois.iloc[3:] = 2.5                                     # the OIS 3m moves 50bp after the last FX day
    fx = pd.Series({days[0]: 1.70, days[2]: 1.80})         # FX swaps: 70 then 20bp below OIS (on day 2: 2.0)
    out = hy.fx_hedge_cost(fx, ois, carry_days=10)
    assert out.at[days[2], "cost"] == pytest.approx(1.80) and out.at[days[2], "source"] == "same_day"
    assert out.at[days[1], "cost"] == pytest.approx(2.0 - 0.30)     # spread from day 0
    assert out.at[days[5], "cost"] == pytest.approx(2.5 - 0.20)     # today's OIS + day 2's spread
    assert out.at[days[5], "source"].startswith("spread carried from")
    assert np.isnan(out.at[days[13], "cost"])                       # beyond 10 business days


def test_fx_swap_pairing_orientation_and_implied_rate():
    from infra.processing import dtcc_fx as fx
    day = D("2026-09-14")
    t = "2026-09-14T14:00:11Z"

    def leg(fisn, exp, rate, n1, c1, n2, c2, when=t):
        return {"UPI FISN": fisn, "Action type": "NEWT", "Package indicator": "True", "Execution Timestamp": when,
                "Expiration Date": exp, "Exchange rate": rate, "Notional amount-Leg 1": n1, "Notional currency-Leg 1": c1,
                "Notional amount-Leg 2": n2, "Notional currency-Leg 2": c2, "Platform identifier": "XOFF"}
    raw = pd.DataFrame([
        leg("NA/Swaps JPY USD", "2026-09-16", "150.00", "1,500,000,000", "JPY", "10,000,000", "USD"),
        leg("NA/Swaps JPY USD", "2026-12-16", "148.89", "1,488,900,000", "JPY", "10,000,000", "USD"),
        # a lone leg and a 1-year swap: not a spot-start 3-month swap
        leg("NA/Swaps EUR USD", "2026-09-16", "1.16", "1,000,000", "EUR", "1,160,000", "USD", "2026-09-14T15:00:00Z"),
        leg("NA/Swaps EUR USD", "2026-09-16", "1.16", "1,000,000", "EUR", "1,160,000", "USD", "2026-09-14T16:00:00Z"),
        leg("NA/Swaps EUR USD", "2027-09-16", "1.18", "1,000,000", "EUR", "1,180,000", "USD", "2026-09-14T16:00:00Z"),
    ])
    sw = fx.spot_start_swaps(fx.legs(raw, day), day)
    assert list(sw["ccy"]) == ["JPY"] and bool(sw["orientation_ok"].iloc[0])
    assert sw["usd_notional"].iloc[0] == 10_000_000
    # USDJPY falls 1.11 big figures over 3 months with USD growing 1%: JPY grows 1.01 x 148.89 / 150
    g = fx.implied_growth(150.0, 148.89, 1.01, "JPY")
    assert g == pytest.approx(1.01 * 148.89 / 150.0)
    # EURUSD-style: forward above spot = EUR grows less than USD
    assert fx.implied_growth(1.16, 1.17, 1.01, "EUR") == pytest.approx(1.01 * 1.16 / 1.17)


def test_rolling_fx_hedge_mirrors_both_ways():
    # with all-in costs A (r + b) per currency, F->B and B->F pickups mirror exactly
    a_jp, a_us, y_jp, y_us = 0.9, 4.0, 1.6, 4.2
    jp_into_usd = hy.hedged_yield(y_jp, a_jp, 0.0, a_us, 0.0) - y_us
    us_into_jpy = hy.hedged_yield(y_us, a_us, 0.0, a_jp, 0.0) - y_jp
    assert jp_into_usd == pytest.approx(-us_into_jpy)
