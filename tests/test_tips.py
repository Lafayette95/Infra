"""TIPS: reference CPI, index ratios and real yields; stored from the nominal fetch's page;
the real-curve metric's place. No network."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from infra.processing import tips as tp

D = pd.Timestamp
CPI = pd.Series([300.0, 301.0, 302.0, 303.0, 304.0], index=pd.date_range("2026-01-01", periods=5, freq="MS"))


def test_reference_cpi_interpolates_months_minus_3_and_2():
    # 2026-05-16: month m-3 = Feb (301), m-2 = Mar (302); 15 of 31 days through May
    assert tp.reference_cpi([D("2026-05-16")], CPI).iloc[0] == pytest.approx(301.0 + 15 / 31 * 1.0)
    assert tp.reference_cpi([D("2026-05-01")], CPI).iloc[0] == pytest.approx(301.0)
    assert np.isnan(tp.reference_cpi([D("2026-09-01")], CPI).iloc[0])  # months not in the series


def _auction(cusip, reopening="No", rate="1.250000", dated="2026-01-15", cpi="300.000000"):
    return json.dumps({"cusip": cusip, "inflation_index_security": "Yes", "reopening": reopening, "int_rate": rate,
                       "dated_date": dated, "issue_date": "2026-01-30", "maturity_date": "2036-01-15",
                       "ref_cpi_on_dated_date": cpi, "original_security_term": "10-Year"})


def test_reference_takes_the_original_issue_and_prices_carry_ratio_and_real_yield():
    a = pd.DataFrame({"raw_json": [_auction("912828TIP"), _auction("912828TIP", reopening="Yes", rate="1.300000"),
                                   json.dumps({"cusip": "912828NOM", "inflation_index_security": "No"})]})
    ref = tp.reference(a)
    assert ref["cusip"].tolist() == ["912828TIP"] and ref["coupon"].iloc[0] == 1.25
    page = pd.DataFrame({"cusip": ["912828TIP", "912828NOM"], "security_type": ["TIPS", "MARKET BASED NOTE"],
                         "buy": ["100.0", "99.0"], "sell": ["99.9", "98.9"], "end_of_day": ["100.0", "99.0"]})
    out = tp.prices(page, D("2026-05-15"), ref, CPI)  # settles 2026-05-18
    assert out["cusip"].tolist() == ["912828TIP"]
    rc = 301.0 + 17 / 31
    assert out["index_ratio"].iloc[0] == pytest.approx(rc / 300.0)
    assert out["real_yield"].iloc[0] == pytest.approx(1.25, abs=0.02)  # priced near par
    back = tp.decode(tp.encode(out))
    assert back["index_ratio"].iloc[0] == pytest.approx(rc / 300.0, abs=1e-6)


def test_px_stores_tips_from_the_same_fedinvest_page(tmp_path, monkeypatch):
    from infra.cycle.paths import CyclePaths
    from infra.cycle import px_treasuries
    from infra.pipeline import tips
    paths = CyclePaths.under(tmp_path)
    monkeypatch.setattr(tips, "tips_reference", lambda **k: tp.reference(pd.DataFrame({"raw_json": [_auction("912828TIP")]})))
    monkeypatch.setattr(tips, "cpi_nsa", lambda **k: CPI)
    calls = []

    def fetch(day):
        calls.append(day)
        return pd.DataFrame({"cusip": ["912828TIP"], "security_type": ["TIPS"], "rate": ["1.250%"], "maturity": ["01/15/2036"],
                             "call_date": [""], "buy": ["100.0"], "sell": ["99.9"], "end_of_day": ["100.0"]})
    out = px_treasuries.backfill_daily_treasury_px("2026-05-15", "2026-05-15", paths=paths, fetch=fetch, now=D("2026-05-20"))
    assert len(calls) == len(set(calls)) == len(out["planned"])  # ONE request per day, for both stores
    assert out["tips"]["stored"] == len(calls) and not out["tips"]["errors"]
    t = tips.read_tips(root=paths.tips_prices_dir)
    assert set(t["cusip"]) == {"912828TIP"} and t["timestamp"].nunique() == len(calls)


def test_tips_curve_metric_after_the_nominal_curve():
    from infra.cycle import derived
    names = list(derived.DERIVED_METRICS)
    assert names.index("treasury_curve") < names.index("tips_curve")
    checks = {c.name for c in derived.DERIVED_STEP.checks}
    assert {"tips_curve_present", "tips_rv_no_revisions", "tips_curve_sane"} <= checks


def test_a_never_published_cpi_month_takes_the_treasury_fallback():
    """October 2025 (shutdown): CPI(m) = CPI(m-1) x (CPI(m-1)/CPI(m-13))^(1/12), 3 decimals;
    a month not published YET (after the last) is never filled."""
    cpi = pd.Series(np.linspace(300.0, 314.0, 16), index=pd.date_range("2024-09-01", periods=16, freq="MS")).round(3)
    gap = cpi.drop(D("2025-10-01"))
    filled = tp.fill_missing_months(gap)
    expect = round(gap[D("2025-09-01")] * (gap[D("2025-09-01")] / gap[D("2024-09-01")]) ** (1 / 12), 3)
    assert filled[D("2025-10-01")] == pytest.approx(expect)
    assert filled.index.max() == gap.index.max()  # no extrapolation past the last published month
    # the reference CPI on a December day now exists (it needs months m-3 and m-2: Sep, Oct)
    assert not np.isnan(tp.reference_cpi([D("2025-12-15")], gap).iloc[0])
