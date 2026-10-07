"""Asymmetric reaction measures (infra/analytics/positioning, infra/pipeline/positioning):
each family recovers an asymmetry built into synthetic data, reads ~0 without one, and
never uses data after the day it reports."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from infra.analytics.positioning import asymmetry as asym
from infra.analytics.positioning.config import ASYMMETRY_SPECS, get_spec

IDX = pd.bdate_range("2015-01-01", periods=3000)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ family 2

def test_semivariance_reads_zero_when_symmetric_and_positive_when_selloffs_are_amplified():
    x = pd.Series(_rng().standard_normal(len(IDX)), index=IDX)
    sym, n = asym.semivariance_asymmetry(x, 250, 100)
    assert abs(sym.dropna().mean()) < 0.08 and n.max() == 250   # 250-day windows: SE ~0.09
    skew = x.where(x < 0, x * 1.5)
    up, _ = asym.semivariance_asymmetry(skew, 250, 100)
    assert up.dropna().mean() > 0.3


def test_tail_asymmetry_counts_only_big_moves():
    x = pd.Series(_rng(1).standard_normal(len(IDX)), index=IDX)
    x = x.where(x < 2, x * 2)                       # only big up moves amplified
    vol = asym.trailing_vol(x, 63, 40)
    a, nu, nd = asym.tail_asymmetry(x, vol, 1.5, 250, 100, 5)
    assert a.dropna().mean() > 0.2
    assert (nu.dropna() > 0).all()


# ------------------------------------------------------------------ family 3

def _panel(seed=2, extra=0.0):
    r = _rng(seed)
    f = pd.Series(r.standard_normal(len(IDX)), index=IDX)
    big_up = f > 1.0
    y = 1.2 * f + extra * f.where(big_up, 0.0) + 0.3 * r.standard_normal(len(IDX))
    z = 0.8 * f + 0.3 * r.standard_normal(len(IDX))
    return pd.DataFrame({"y": y, "z": z, "f": f})


def test_pc1_is_a_level_factor_and_point_in_time():
    p = _panel()
    fac, load = asym.trailing_pc1(p[["y", "z"]], 250, 21, 100)
    assert np.allclose(load.sum(axis=1), 1.0)
    shocked = p.copy()
    shocked.iloc[2000:] *= 5
    fac2, _ = asym.trailing_pc1(shocked[["y", "z"]], 250, 21, 100)
    pd.testing.assert_series_equal(fac.iloc[:2000], fac2.iloc[:2000])


def test_trailing_beta_never_sees_its_own_window():
    p = _panel()
    b = asym.trailing_beta(p["y"], p["f"], 250, 63, 100)
    shocked = p.copy()
    shocked.loc[IDX[2000]:, "y"] += 50.0
    b2 = asym.trailing_beta(shocked["y"], shocked["f"], 250, 63, 100)
    pd.testing.assert_series_equal(b.iloc[:2000 + 63], b2.iloc[:2000 + 63])
    assert b.dropna().iloc[-1] == pytest.approx(1.2, abs=0.1)


def test_relative_asymmetry_finds_the_amplified_instrument_only():
    p = _panel(extra=0.8)
    out = asym.relative_asymmetry(p[["y", "z"]], p["f"], beta_window=500, window=250, k=1.0, vol_span=63,
                                  min_obs=100, min_big=10)
    ra = out["rel_asym"].dropna()
    assert ra["y"].mean() > 0.3          # sells off more than usual in big sell-offs
    assert abs(ra["z"].mean()) < 0.1     # a plain 0.8 beta: no asymmetry


# ------------------------------------------------------------------ family 1

def test_standardised_surprises_use_earlier_prints_only():
    s = pd.DataFrame({"A": np.r_[np.ones(20), 100.0]}, index=IDX[:21])
    z = asym.standardise_surprises(s, 8)
    assert z["A"].iloc[-1] == pytest.approx(100.0)   # judged against the earlier RMS of 1


def test_impact_betas_fit_coincident_releases_jointly_and_point_in_time():
    r = _rng(3)
    n = 1500
    t = pd.date_range("2010-01-01", periods=n, freq="2D")
    za = r.standard_normal(n)
    zb = np.where(r.random(n) < 0.5, 0.7 * za + 0.3 * r.standard_normal(n), 0.0)  # sometimes coincident
    z = pd.DataFrame({"A": za, "B": zb}, index=t)
    y = pd.DataFrame({"I": 2.0 * za + 1.0 * zb + 0.2 * r.standard_normal(n)}, index=t)
    refit = pd.date_range(t[0], t[-1], freq="MS")
    e = asym.impact_betas(z, y, refit=refit, lookback=pd.Timedelta(days=5 * 365), ridge=0.1, min_events=20)
    late = e.index > t[800]
    assert np.corrcoef(e.loc[late, "I"], (2.0 * za + 1.0 * zb)[late])[0, 1] > 0.99
    y2 = y.copy()
    y2.iloc[1000:] += 100
    e2 = asym.impact_betas(z, y2, refit=refit, lookback=pd.Timedelta(days=5 * 365), ridge=0.1, min_events=20)
    cut = refit[refit > t[1000]][0]
    pd.testing.assert_frame_equal(e[e.index < cut], e2[e2.index < cut])


def test_surprise_asymmetry_recovers_side_slopes():
    r = _rng(4)
    x = pd.DataFrame({"I": r.standard_normal(400)}, index=pd.date_range("2015-01-01", periods=400))
    y = x.where(x < 0, 2 * x) + 0.01 * r.standard_normal((400, 1))
    out = asym.surprise_asymmetry(y, x, 100, 20, 10)
    last = {k: v["I"].iloc[-1] for k, v in out.items()}
    assert last["b_up"] == pytest.approx(2.0, abs=0.05) and last["b_down"] == pytest.approx(1.0, abs=0.05)
    assert last["surprise_asym"] == pytest.approx(1 / 3, abs=0.02)


def test_continuation_asymmetry():
    r = _rng(5)
    first = pd.DataFrame({"I": r.standard_normal(300)})
    after = first.where(first < 0, 0.5 * first).where(first > 0, -0.3 * first)
    out = asym.continuation_asymmetry(first, after, 100, 20, 10)
    assert out["cont_up"]["I"].iloc[-1] == pytest.approx(0.5)
    assert out["cont_down"]["I"].iloc[-1] == pytest.approx(-0.3)
    assert out["cont_asym"]["I"].iloc[-1] == pytest.approx(0.8)


# ------------------------------------------------------------------ config / pipeline

def test_specs_valid():
    for s in ASYMMETRY_SPECS.values():
        assert s.instruments
    with pytest.raises(ValueError):
        replace(get_spec("ust_daily"), factor="series:nope")
    with pytest.raises(KeyError):
        get_spec("nope")


def test_compute_end_to_end_and_store_roundtrip(monkeypatch, tmp_path):
    from infra.pipeline import positioning as pos
    p = _panel(extra=0.5)
    days = IDX
    moves = pd.DataFrame({"a": p["y"], "b": p["z"], "c": p["f"]}, index=days)
    spec = replace(get_spec("ust_daily"), name="t", instruments=("a", "b", "c"), releases=("R",),
                   impact_min_events=10, event_window=30, event_min_side=5)
    monkeypatch.setattr(pos, "daily_moves", lambda s, a, b: moves)
    r = _rng(6)
    inst = pd.DatetimeIndex(days[::5]) + pd.Timedelta(hours=13, minutes=30)
    sur = pd.DataFrame({"R": r.standard_normal(len(inst))}, index=inst)
    monkeypatch.setattr(pos, "surprises", lambda s, a, b: sur)
    df = pos.compute(spec, days[0], days[-1])
    assert set(df["family"]) == {"single", "relative", "surprise"}
    assert {"rel_asym", "semivar_asym", "tail_asym", "surprise_asym", "b_all"} <= set(df["measure"])
    assert not df.duplicated(pos.KEYS).any()
    pos.store(df, root=tmp_path)
    back = pos.read_asymmetry("t", "rel_asym", "a", root=tmp_path)
    assert len(back) == len(df[(df["measure"] == "rel_asym") & (df["instrument"] == "a")])
    assert back["value"].mean() > 0.1        # "a" amplified in sell-offs (vs a PC that partly contains it)


def test_skew_ignores_a_trend_that_semivariance_reads_as_asymmetry():
    """Found on real data 2026-10-07: around 0, a trending window reads as 'asymmetric'."""
    x = pd.Series(_rng(7).standard_normal(len(IDX)) + 0.8, index=IDX)   # steady sell-off, symmetric noise
    sv, _ = asym.semivariance_asymmetry(x, 250, 100)
    sk = asym.rolling_skew(x, 250, 100)
    assert sv.dropna().mean() > 0.5
    assert abs(sk.dropna().mean()) < 0.1
