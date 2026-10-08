"""CTA model (infra/models/cta): back-adjustment, prepare/fit/predict split, point in time,
signal and position bounds, forecasts, reaction functions, walk-forward, config, inputs."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from infra.models.base import NotFittedError
from infra.models.cta import paper
from infra.models.cta.config import CTA_MODELS, CTAAsset, CTAUniverse, get_spec
from infra.models.cta.model import CTAModel, walk_forward
from infra.models.cta.prep import prepare
from infra.processing import continuous

# Short windows so a synthetic history of a few years exercises every stage.
SPEC = replace(get_spec("ubs2022"), name="test", ewma_pairs=((4, 12), (8, 24)), norm_vol_window=126,
               sizing_vol_window=42, forecast_vol_window=21, vol_min_obs=20, response_min_obs=100,
               pvs_window=252, pvs_min_obs=100, position_scale_window=750, mc_paths=400,
               flow_horizons=(1, 5, 10), change_horizons=(1, 3, 5))


def _prices(n=1200, seed=0, assets=("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    drift = np.repeat(rng.choice([-0.08, 0.0, 0.08], size=n // 100 + 1), 100)[:n]
    out = {}
    for i, a in enumerate(assets):
        steps = drift * (1 + 0.3 * i) + rng.standard_normal(n) * (0.5 + 0.2 * i)
        out[a] = 100 + np.cumsum(steps)
    return pd.DataFrame(out, index=idx)


UNI = CTAUniverse("t", (CTAAsset("A", liquidity=1), CTAAsset("B", liquidity=4), CTAAsset("C", liquidity=8)),
                  {"default": 0.3})


def _fit(prices, as_of=None, spec=SPEC):
    m = CTAModel(spec)
    return m.fit(m.prepare(prices, universe=UNI), as_of=as_of)


# ---------------------------------------------------------------- back-adjustment

def test_back_adjusted_removes_roll_gap_and_keeps_last_price_real():
    days = pd.bdate_range("2024-01-01", periods=6)
    rel = pd.DataFrame({"timestamp": days, "ticker": "X.v.0",
                        "contract": ["XH4"] * 3 + ["XM4"] * 3,
                        "settlement_price": [100.0, 101.0, 102.0, 110.5, 111.0, 110.0]})
    absolute = pd.DataFrame({"timestamp": list(days[:3]) + list(days[2:]), "ticker": ["XH4"] * 3 + ["XM4"] * 4,
                             "settlement_price": [100.0, 101.0, 102.0, 110.0, 110.5, 111.0, 110.0]})
    chg = continuous.same_contract_changes(rel, absolute)
    assert chg["change"].tolist()[1:] == [1.0, 1.0, 0.5, 0.5, -1.0]  # roll day measured on XM4
    wide = continuous.back_adjusted(chg)
    assert wide["X.v.0"].iloc[-1] == 110.0
    assert wide["X.v.0"].diff().dropna().tolist() == [1.0, 1.0, 0.5, 0.5, -1.0]
    assert continuous.unknown_changes(chg).empty


def test_unknown_roll_prior_is_reported_not_guessed():
    days = pd.bdate_range("2024-01-01", periods=3)
    rel = pd.DataFrame({"timestamp": days, "ticker": "X.v.0", "contract": ["XH4", "XM4", "XM4"],
                        "settlement_price": [100.0, 110.0, 111.0]})
    absolute = pd.DataFrame({"timestamp": [days[0], days[1], days[2]], "ticker": ["XH4", "XM4", "XM4"],
                             "settlement_price": [100.0, 110.0, 111.0]})
    chg = continuous.same_contract_changes(rel, absolute)
    assert len(continuous.unknown_changes(chg)) == 1
    assert continuous.back_adjusted(chg)["X.v.0"].tolist() == [110.0, 110.0, 111.0]


# ---------------------------------------------------------------- prepare

def test_prepare_levels_returns_and_anchor():
    px = _prices(5)
    d = prepare(px, assets={"A": CTAAsset("A", returns="log")})
    assert np.allclose(d.levels["A"], np.log(px["A"]))
    assert np.isnan(d.returns["B"].iloc[0])
    anchored = prepare(px, anchor={"B": 99.0})
    assert anchored.returns["B"].iloc[0] == pytest.approx(px["B"].iloc[0] - 99.0)
    with pytest.raises(ValueError):
        prepare(px.tz_localize("UTC"))


def test_each_series_keeps_its_own_calendar():
    px = _prices(10)
    px.iloc[4, 0] = np.nan  # A has a holiday
    lvl, ret = prepare(px).series("A")
    assert len(lvl) == 9
    assert ret.iloc[4] == pytest.approx(px["A"].iloc[5] - px["A"].iloc[3])


# ---------------------------------------------------------------- fit / predict

def test_predict_continues_the_fit_exactly():
    """EWMAs, vols and crossovers from predict-after-fit equal a single pass over all data."""
    px = _prices()
    cut = px.index[800]
    m = _fit(px, as_of=cut)
    stepped = m._step(m._as_data(px[px.index > cut]))
    full = _fit(px).fitted_.frames
    cols = [f"ema_{n}" for n in SPEC.spans] + ["norm_vol", "vol", "forecast_vol", "z0", "z1", "level", "ret"]
    for a in px.columns:
        pd.testing.assert_frame_equal(stepped[a][cols], full[a].loc[stepped[a].index, cols], check_exact=False,
                                      rtol=1e-10, atol=1e-12)


def test_fit_is_point_in_time():
    px = _prices()
    cut = px.index[900]
    changed = px.copy()
    changed.loc[changed.index > cut] *= 1.5
    a, b = _fit(px, as_of=cut), _fit(changed, as_of=cut)
    for asset in px.columns:
        pd.testing.assert_frame_equal(a.fitted_.frames[asset], b.fitted_.frames[asset])
        assert a.fitted_.params[asset].pos_scale == b.fitted_.params[asset].pos_scale
    assert a.fitted_.pvs == b.fitted_.pvs


def test_signal_and_position_bounds_and_uniform_signal():
    m = _fit(_prices())
    fr = m.predict(forecast_at="none").frame
    assert fr["signal"].dropna().between(-1, 1).all()
    assert fr["position"].dropna().between(-1, 1).all()
    for a, g in fr.groupby("asset"):
        assert g["position"].abs().max() == pytest.approx(1.0)  # the 10y max IS the unit
    s = fr.loc[fr["asset"] == "A", "signal"].dropna()
    assert np.quantile(s, [0.25, 0.5, 0.75]) == pytest.approx([-0.5, 0.0, 0.5], abs=0.03)


def test_trend_direction():
    idx = pd.bdate_range("2018-01-01", periods=900)
    rng = np.random.default_rng(1)
    noise = np.cumsum(rng.standard_normal(900))
    up = pd.DataFrame({"A": noise + np.r_[np.zeros(800), np.arange(100) * 0.8]}, index=idx)
    m = CTAModel(SPEC)
    m.fit(m.prepare(up))
    last = m.predict().latest().loc["A"]
    assert last["signal"] > 0.9 and last["position"] > 0


def test_changes_are_position_differences():
    fr = _fit(_prices()).predict(forecast_at="none").frame
    g = fr[fr["asset"] == "B"].set_index("timestamp")
    for h in SPEC.change_horizons:
        pd.testing.assert_series_equal(g[f"chg_{h}"], g["position"] - g["position"].shift(h), check_names=False)


def test_predict_out_of_sample_changes_reach_back_into_the_fit():
    px = _prices()
    cut = px.index[1000]
    m = _fit(px, as_of=cut)
    res = m.predict(px[px.index > cut], forecast_at="none")
    first = res.frame[res.frame["asset"] == "A"].iloc[0]
    fitted_last = m.fitted_.frames["A"]["position"].iloc[-1]
    assert first["chg_1"] == pytest.approx(first["position"] - fitted_last)


def test_predict_requires_fit():
    with pytest.raises(NotFittedError):
        CTAModel(SPEC).predict()


# ---------------------------------------------------------------- forecast

def test_flow_decomposition_sums_and_is_reproducible():
    m = _fit(_prices())
    a = m.predict().forecast
    b = m.predict().forecast
    pd.testing.assert_frame_equal(a, b)
    unclipped = a[a["exp_position"].abs() < 1]
    assert np.allclose(unclipped["flow"], unclipped["flow_from_signal"] + unclipped["flow_from_vol"])
    assert set(a["horizon"]) == set(SPEC.flow_horizons)


def test_saturated_signal_is_expected_to_mean_revert():
    """Paths are zero-drift: from a pinned signal the only way is back (UBS's
    'initial conditions matter')."""
    idx = pd.bdate_range("2018-01-01", periods=900)
    rng = np.random.default_rng(2)
    px = pd.DataFrame({"A": np.cumsum(rng.standard_normal(900)) + np.r_[np.zeros(820), np.arange(80) * 1.5]},
                      index=idx)
    m = CTAModel(SPEC)
    m.fit(m.prepare(px))
    res = m.predict()
    s = res.states["A"].signal
    fc = res.forecast.set_index("horizon")
    assert s > 0.95 and fc.loc[10, "exp_signal"] < s


# ---------------------------------------------------------------- reaction

def test_price_reaction_is_monotone_and_zero_at_zero():
    m = _fit(_prices())
    r = m.reaction(axis="price", assets=["A"])
    assert r.loc[r["price_shock"] == 0, "d_position"].abs().max() < 1e-12
    assert (r["position"].diff().dropna() >= -1e-12).all()


def test_vol_reaction_scales_position_inversely():
    m = _fit(_prices())
    res = m.predict(forecast_at="none")
    st = res.states["B"]
    r = m.reaction(res, "vol", assets=["B"]).set_index("vol_shock")
    if abs(st.position) < 0.5:  # far from the clip
        assert r.loc[1.0, "position"] == pytest.approx(st.position / 2.0)
        assert (r["signal"] == st.signal).all()


def test_path_reaction_and_unknown_axis():
    m = _fit(_prices())
    r = m.reaction(axis="path", assets=["C"])
    assert {"d_position_vs_flat"} <= set(r.columns)
    with pytest.raises(KeyError):
        m.reaction(axis="nope")


# ---------------------------------------------------------------- walk-forward

def test_walk_forward_matches_a_fit_on_its_fit_dates():
    px = _prices()
    data = prepare(px, universe=UNI)
    start, end = px.index[1000], px.index[1100]
    wf = walk_forward(SPEC, data, start, end, refit="W-FRI")
    assert wf["timestamp"].min() >= start and wf["timestamp"].max() <= end
    fri = [d for d in pd.date_range(start, end, freq="W-FRI") if d in px.index][1]
    direct = _fit(px, as_of=fri).predict(forecast_at="none").frame
    direct = direct[direct["timestamp"] == fri].set_index("asset")["position"]
    got = wf[wf["timestamp"] == fri].set_index("asset")["position"]
    pd.testing.assert_series_equal(got.sort_index(), direct.sort_index(), check_names=False)


# ---------------------------------------------------------------- config / paper / inputs

def test_every_configured_model_is_valid():
    for spec in CTA_MODELS.values():
        assert spec.weights().sum() == pytest.approx(1.0)
        assert all(s < l for s, l in spec.ewma_pairs)
    with pytest.raises(KeyError):
        get_spec("nope")


def test_paper_positions_follow_the_stated_formula():
    """UBS's position = 30% x liquidity x signal x 809.9 / vol, up to a common ~1.075."""
    ratios = [pos / (paper.ASSET_CLASS_WEIGHT * liq * s * paper.PORTFOLIO_VOL_SCALING / vol)
              for s, vol, liq, pos, _ in paper.FIG63.values() if abs(s) >= 0.5]
    assert 1.04 < min(ratios) and max(ratios) < 1.11


def test_futures_prices_reads_disk_only(monkeypatch):
    from infra.models.cta import inputs
    from infra.pipeline import series_panel
    days = pd.bdate_range("2024-01-01", periods=4)
    rel = pd.DataFrame({"timestamp": days, "ticker": "ZN.v.0", "contract": ["ZNH4", "ZNH4", "ZNM4", "ZNM4"],
                        "settlement_price": [110.0, 110.5, 111.5, 111.0]})
    absolute = pd.DataFrame({"timestamp": [days[0], days[1], days[1], days[2], days[3]],
                             "ticker": ["ZNH4", "ZNH4", "ZNM4", "ZNM4", "ZNM4"],
                             "settlement_price": [110.0, 110.5, 111.25, 111.5, 111.0]})
    seen = {}

    def fake_rel(specs, start, end, fetch_missing):
        seen["fetch_missing"] = fetch_missing
        return rel

    monkeypatch.setattr(series_panel, "load_relative_daily", fake_rel)
    monkeypatch.setattr(series_panel.dl, "read_daily_from_disk", lambda tickers, start, end: absolute)
    out = inputs.futures_prices(["ZN.v.0"], days[0], days[-1])
    assert seen["fetch_missing"] is False
    assert out["ZN.v.0"].diff().dropna().tolist() == [0.5, 0.25, -0.5]
    assert out["ZN.v.0"].iloc[-1] == 111.0


def test_log_returns_use_the_held_contracts_own_prices():
    """Back-adjusted levels shift by the roll gaps: their log differences are wrong."""
    days = pd.bdate_range("2024-01-01", periods=4)
    rel = pd.DataFrame({"timestamp": days, "ticker": "CL.v.0", "contract": ["CLF4", "CLF4", "CLG4", "CLG4"],
                        "settlement_price": [10.0, 11.0, 20.0, 22.0]})
    absolute = pd.DataFrame({"timestamp": [days[0], days[1], days[1], days[2], days[3]],
                             "ticker": ["CLF4", "CLF4", "CLG4", "CLG4", "CLG4"],
                             "settlement_price": [10.0, 11.0, 19.0, 20.0, 22.0]})
    chg = continuous.same_contract_changes(rel, absolute)
    r = continuous.log_returns(chg)["CL.v.0"]
    assert np.allclose(r.dropna().to_numpy(), np.log([1.1, 20 / 19, 1.1]))
    lvl = continuous.back_adjusted(chg)["CL.v.0"]
    assert not np.allclose(np.log(lvl).diff().dropna().to_numpy(), r.dropna().to_numpy())
