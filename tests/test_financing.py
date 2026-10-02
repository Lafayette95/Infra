"""Financing model (CLAUDE.md 20): the SR1-implied SOFR path, the layered composition and
the per-CUSIP specialness model - on synthetic inputs with known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.analytics import financing as fa
from infra.analytics import sofr_curve as sc
from infra.analytics import specialness as sa
from infra.config import FOMCMeeting, FinancingSpec
from infra.pipeline.financing import change_dates


# ------------------------------------------------------------------ layer 1
def test_business_days_follow_the_sifma_approximation():
    days = sc.business_days("2026-04-01", "2026-04-07")
    assert pd.Timestamp("2026-04-03") not in days  # Good Friday
    assert pd.Timestamp("2026-10-12") not in sc.business_days("2026-10-09", "2026-10-13")  # Columbus Day


def _synthetic(levels: dict, turn_bp: float, months, fixed_through):
    """Prices of SR1 contracts for a known step path (levels by effective date), plus the
    fixings published through ``fixed_through``."""
    first = pd.Timestamp(months[0])
    last = pd.Timestamp(months[-1]) + pd.offsets.MonthEnd(0)
    bd = sc.business_days(first - pd.Timedelta(days=10), last)
    levels = {pd.Timestamp(k): v for k, v in levels.items()}
    cuts = sorted(pd.Timestamp(k) for k in levels)
    fix = pd.Series([levels[max([c for c in cuts if c <= d] or [cuts[0]])] for d in bd], index=bd)
    fix[np.isin(bd, sc.year_end_days(bd))] += turn_bp / 100
    cal = pd.date_range(first, last)
    daily = pd.Series(fix.reindex(sc.governing_days(cal, bd)).to_numpy(), index=cal)
    fut = pd.DataFrame({"month": pd.to_datetime(months),
                        "price": [100 - daily[pd.Timestamp(m):pd.Timestamp(m) + pd.offsets.MonthEnd(0)].mean() for m in months]})
    return fut, fix[fix.index <= pd.Timestamp(fixed_through)], daily, cuts


def test_the_fit_recovers_a_known_step_path_with_its_year_end_turn():
    levels = {"2026-01-01": 3.90, "2026-10-29": 4.15, "2026-12-10": 4.40, "2027-01-28": 4.40}
    months = pd.date_range("2026-10-01", "2027-03-01", freq="MS")
    fut, fix, daily, cuts = _synthetic(levels, 12.0, months, "2026-10-06")
    path = sc.fit_sofr_path("2026-10-07", fut, fix, cuts[1:], year_end_turn=12.0)
    assert path.residuals_bp.abs().max() < 1e-6
    assert np.allclose(path.daily.loc["2026-10-07":"2027-03-31"], daily.loc["2026-10-07":"2027-03-31"], atol=1e-9)
    assert path.levels["level"].round(6).tolist()[:3] == [3.9, 4.15, 4.4]
    flat = sc.fit_sofr_path("2026-10-07", fut, fix, cuts[1:], year_end_turn=0.0)
    assert flat.residuals_bp.abs().max() > 0.01  # a missing turn shows up as misfit


def test_rolling_overnight_compounds_daily():
    daily = pd.Series(4.0, index=pd.date_range("2026-01-01", "2026-12-31"))
    path = sc.SofrPath(pd.Timestamp("2026-01-01"), daily, pd.Timestamp("2025-12-31"), pd.DataFrame(), pd.Series(), 0.0)
    r = path.compounded("2026-01-01", "2026-04-01")
    assert r == pytest.approx(((1 + 0.04 / 360) ** 90 - 1) * 360 / 90 * 100)
    assert r > 4.0


def test_year_end_turn_is_point_in_time_and_recent():
    bd = sc.business_days("2022-12-01", "2026-01-15")
    fix = pd.Series(4.0, index=bd)
    for ye, jump in (("2022-12-30", 0.0), ("2023-12-29", 0.03), ("2024-12-31", 0.09), ("2025-12-31", 0.16)):
        fix[pd.Timestamp(ye)] += jump
    assert sc.year_end_turn_bp(fix, "2025-12-31", last_n=3) == pytest.approx(3.0)  # 2025's not known yet
    assert sc.year_end_turn_bp(fix, "2026-01-05", last_n=3) == pytest.approx(9.0)


def test_unscheduled_meetings_count_only_once_announced():
    meetings = (FOMCMeeting("2020-01-28", "2020-01-29", False),
                FOMCMeeting("2020-03-03", "2020-03-03", False, scheduled=False),
                FOMCMeeting("2020-04-28", "2020-04-29", False))
    assert change_dates("2020-03-02", meetings) == [pd.Timestamp("2020-01-30"), pd.Timestamp("2020-04-30")]
    assert pd.Timestamp("2020-03-04") in change_dates("2020-03-03", meetings)


# ------------------------------------------------------------ composition
def test_layers_compose_and_two_models_share_inputs(monkeypatch):
    daily = pd.Series(4.0, index=pd.date_range("2026-01-01", "2026-12-31"))
    path = sc.SofrPath(pd.Timestamp("2026-01-02"), daily, pd.Timestamp("2026-01-01"), pd.DataFrame(), pd.Series(), 0.0)
    fixings = pd.DataFrame({"timestamp": pd.bdate_range("2025-12-01", "2026-01-01"), "rate": 4.0, "p75": 4.06})
    inputs = fa.FinancingInputs(as_of=pd.Timestamp("2026-01-02"), sofr_path=path, sofr_fixings=fixings)
    v1 = fa.financing_rate(inputs, "v1", FinancingSpec("sofr_futures", "sofr_p75", "none"), "2026-01-05", "2026-02-05")
    gc = fa.financing_rate(inputs, "gc", FinancingSpec("sofr_futures", "none", "none"), "2026-01-05", "2026-02-05")
    assert v1.base == gc.base and v1.basis == pytest.approx(0.06) and gc.basis == 0.0
    assert v1.rate == pytest.approx(gc.rate + 0.06)


def test_client_basis_is_a_rolling_median():
    fx = pd.DataFrame({"timestamp": pd.bdate_range("2026-01-01", periods=30), "rate": 4.0,
                       "p75": [4.20] * 10 + [4.05] * 20})
    assert fa.client_basis_bp(fx, 20) == pytest.approx(5.0)
    assert fa.client_basis_bp(fx, 30) == pytest.approx(5.0)


# ------------------------------------------------------------------ layer 3
def _profile():
    idx = pd.MultiIndex.from_tuples([("5y", 0, b) for b in range(6)] + [("5y", 1, b) for b in range(7)],
                                    names=["tenor", "rank", "bucket"])
    return pd.Series([0, 0, 1, 3, 10, 6] + [8, 3, 1, 0, 0, 0, 0], index=idx, dtype="float64")


def test_an_on_the_run_bond_becomes_1_old_at_its_successors_issue():
    bd = sc.business_days("2026-08-20", "2026-12-31")
    state = sa.BondState("X", "5y", 0, pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-30"), observed_bp=10.0)
    days = bd[(bd >= pd.Timestamp("2026-09-25")) & (bd <= pd.Timestamp("2026-10-30"))]
    path = sa.expected_path(state, "2026-09-25", days, _profile(), 1e-9, bd)  # deviation gone at once
    assert path[pd.Timestamp("2026-09-29")] == 6.0  # last 10% of its on-the-run life
    assert path[pd.Timestamp("2026-09-30")] == 8.0  # day 0 as 1-old
    assert path[pd.Timestamp("2026-10-30")] == 0.0


def test_a_deviation_halves_every_half_life():
    bd = sc.business_days("2026-01-01", "2026-03-31")
    state = sa.BondState("Y", None, None, None, None, observed_bp=40.0)  # untracked: profile 0
    days = bd[(bd >= pd.Timestamp("2026-02-02")) & (bd <= pd.Timestamp("2026-02-12"))]
    path = sa.expected_path(state, "2026-02-02", days, pd.Series(dtype="float64"), 3.0, bd)
    assert path.iloc[0] == 40.0 and path.iloc[3] == pytest.approx(20.0) and path.iloc[6] == pytest.approx(10.0)


def test_half_life_estimator_recovers_a_known_persistence():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(40):
        x = 0.0
        for t in range(200):
            x = 0.8 * x + rng.normal()
            rows.append({"timestamp": t, "cusip": str(c), "tenor": "10y", "rank": 0, "bucket": 1, "sp": x})
    panel = pd.DataFrame(rows)
    hl = sa.half_life(panel, pd.Series({("10y", 0, 1): 0.0}))
    assert hl == pytest.approx(np.log(0.5) / np.log(0.8), rel=0.1)


def test_term_specialness_weights_calendar_days():
    bd = sc.business_days("2026-09-01", "2026-10-31")
    path = pd.Series(0.0, index=bd)
    path[pd.Timestamp("2026-10-02")] = 30.0  # a Friday: governs Sat and Sun too
    assert sa.term_specialness_bp(path, "2026-10-01", "2026-10-08", bd) == pytest.approx(90.0 / 7)
