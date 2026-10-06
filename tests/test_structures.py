"""Curve structures and layered sizing (infra/reference/structures.py, infra/analytics/structures.py,
infra/pipeline/structures.py, infra/strategies/layered.py, the STRUCT_BPS_BBO source). Synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.analytics import structures as an
from infra.pipeline.structures import structure_state
from infra.reference.structures import BBG_FUTURES, STRUCTURE_SETS, STRUCTURES

D = pd.Timestamp
LEGS = STRUCTURE_SETS["ust_layers"].legs()


def _leg_moves(n=700, seed=0):
    """Six legs driven by level / slope / curvature / front factors + noise (bp per day)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-02", periods=n)
    lvl, slope, curv, front = (rng.normal(0, s, n) for s in (6.0, 2.0, 0.6, 1.5))
    load = {"TU": (1, 1.0, -0.6, 1), "FV": (1, 0.5, 0.3, 0), "TY": (1, 0.1, 0.5, 0), "UXY": (1, -0.1, 0.3, 0),
            "US": (1, -0.4, 0.0, 0), "WN": (1, -0.6, -0.5, 0)}
    out = {BBG_FUTURES[k]: a * lvl + b * slope + c * curv + f * front + rng.normal(0, 0.3, n)
           for k, (a, b, c, f) in load.items()}
    return pd.DataFrame(out, index=days)[LEGS]


def test_registry_spans_the_six_futures_and_hedges_on_the_macro_layer():
    s = STRUCTURE_SETS["ust_layers"]
    assert sorted(s.legs()) == sorted(BBG_FUTURES.values())
    assert s.layers() == {"macro": ["DUR__TY", "CURVE__FV__WN", "FLY__FV__UXY__WN"], "front": ["FRONT__TU__H"],
                          "micro": ["MICRO__TY__FV__H", "MICRO__US__WN__H"]}
    assert s.hedge_targets("FRONT__TU__H") == ["DUR__TY", "CURVE__FV__WN", "FLY__FV__UXY__WN"]
    assert STRUCTURE_SETS["ust_layers_uxy"].hedge_targets("MICRO__US__WN__H")[0] == "DUR__UXY"
    assert STRUCTURES["FLY__FV__UXY__WN"].base_weights() == {"TN.v.0": 1.0, "ZF.v.0": -0.5, "UB.v.0": -0.5}


def test_rolling_betas_recover_planted_slopes_and_use_rows_up_to_each_day():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(400, 2)), columns=["a", "b"])
    y = 0.7 * X["a"] - 1.5 * X["b"] + rng.normal(0, 0.1, 400)
    b = an.rolling_betas(y, X, 250, 100)
    assert b.iloc[:99].isna().all().all() and b.iloc[-1].round(1).tolist() == [0.7, -1.5]
    y2 = y.copy(); y2.iloc[300:] += 50.0                          # shock after row 299
    pd.testing.assert_frame_equal(an.rolling_betas(y2, X, 250, 100).iloc[:300], b.iloc[:300])


def test_state_hedged_structures_are_residual_and_point_in_time():
    lm = _leg_moves()
    st = structure_state("ust_layers", "2021-01-04", lm.index[-1], leg_moves=lm, with_dv01=False)
    m = st.moves.loc[st.days]
    for s in ("FRONT__TU__H", "MICRO__TY__FV__H", "MICRO__US__WN__H"):
        for t in ("DUR__TY", "CURVE__FV__WN", "FLY__FV__UXY__WN"):
            assert abs(m[s].corr(m[t])) < 0.15, (s, t)
    # weights for day D depend only on moves before D
    d = st.days[100]
    lm2 = lm.copy(); lm2.loc[lm2.index >= d] += 30.0
    st2 = structure_state("ust_layers", "2021-01-04", lm.index[-1], leg_moves=lm2, with_dv01=False)
    pd.testing.assert_frame_equal(st.W(d), st2.W(d))
    assert st.sigma.loc[d].equals(st2.sigma.loc[d]) and st.cov[d].equals(st2.cov[d])


def test_structures_and_futures_map_into_each_other_exactly():
    lm = _leg_moves()
    st = structure_state("ust_layers", "2021-01-04", lm.index[-1], leg_moves=lm, with_dv01=False)
    W = st.W(st.days[-1])
    e = pd.Series([3.0, -1.0, 2.0, 0.5, -4.0, 1.0], index=W.columns)
    x = an.to_structures(e, W)
    pd.testing.assert_series_equal(an.to_legs(x, W), e, check_names=False)
    with pytest.raises(ValueError):
        an.to_structures(e, W.iloc[:3])
    assert an.basis_condition(W) < 50


def _state_for_sizing():
    lm = _leg_moves()
    st = structure_state("ust_layers", "2021-01-04", lm.index[-1], leg_moves=lm, with_dv01=False)
    st.dv01 = pd.DataFrame(100.0, index=st.days, columns=LEGS)
    st.contracts = pd.DataFrame("X", index=st.days, columns=LEGS)
    return st


def test_size_layers_budget_per_layer_netting_and_cap():
    st = _state_for_sizing()
    d = st.days[200]
    labels = pd.DatetimeIndex([d + pd.Timedelta(hours=14)] * 1)
    layers = STRUCTURE_SETS["ust_layers"].layers()
    budgets = {"macro": 1e6, "front": 3e5, "micro": 5e5}
    v = pd.DataFrame({"DUR__TY": [1.0]}, index=labels)
    r = an.size_layers(v, pd.Series([d], index=labels), layers, budgets, st.sigma, st.weights, st.cov, st.dv01, None)
    x = 1e6 / np.sqrt(3) / st.sigma.at[d, "DUR__TY"]
    assert r.positions.iloc[0]["DUR__TY"] == pytest.approx(x)
    assert r.exposures.iloc[0]["ZN.v.0"] == pytest.approx(x) and r.contracts.iloc[0]["ZN.v.0"] == pytest.approx(x / 100)
    # a full-strength DUR view's standalone vol = budget / sqrt(n) (single structure: its own sigma)
    assert r.diagnostics.iloc[0]["vol:macro"] == pytest.approx(1e6 / np.sqrt(3), rel=0.15)
    # netting: long duration + a TY-FV micro view put TY exposure from two structures into one number
    v2 = pd.DataFrame({"DUR__TY": [1.0], "MICRO__TY__FV__H": [1.0]}, index=labels)
    r2 = an.size_layers(v2, pd.Series([d], index=labels), layers, budgets, st.sigma, st.weights, st.cov, st.dv01, None)
    W = st.W(d)
    exp = r2.positions.iloc[0].fillna(0) @ W.reindex(r2.positions.columns).fillna(0)
    pd.testing.assert_series_equal(r2.exposures.iloc[0], exp, check_names=False)
    # cap: everything scales down to the cap
    full = pd.DataFrame({s: [1.0] for s in STRUCTURE_SETS["ust_layers"].structures}, index=labels)
    r3 = an.size_layers(full, pd.Series([d], index=labels), layers, budgets, st.sigma, st.weights, st.cov, st.dv01, 2e5)
    assert r3.diagnostics.iloc[0]["vol_total"] == pytest.approx(2e5) and r3.diagnostics.iloc[0]["scale"] < 1


def test_layered_positions_map_labels_to_their_trading_day_and_carry_into_the_plan():
    from infra.strategies.layered import layered_positions
    st = _state_for_sizing()
    d = st.days[300]
    # 13:00 UTC on d is trading day d; 23:00 UTC on d (19:00 ET) is CME trading day d + 1
    labels = pd.DatetimeIndex([d + pd.Timedelta(hours=13), d + pd.Timedelta(hours=23), st.days[-1] + pd.Timedelta(days=7)])
    v = pd.DataFrame({"CURVE__FV__WN": [1.0, 1.0, 1.0]}, index=labels)
    c, diag = layered_positions(v, "ust_layers", state=st)
    nxt = st.days[301]
    unit = 1e6 / np.sqrt(3)
    assert c.iloc[0]["ZF.v.0"] == pytest.approx(unit / st.sigma.at[d, "CURVE__FV__WN"] / 100)
    assert c.iloc[1]["ZF.v.0"] == pytest.approx(unit / st.sigma.at[nxt, "CURVE__FV__WN"] / 100)
    assert c.iloc[2]["ZF.v.0"] == pytest.approx(unit / st.sigma.at[st.days[-1], "CURVE__FV__WN"] / 100)  # plan: last state
    assert c.iloc[0]["UB.v.0"] == pytest.approx(-c.iloc[0]["ZF.v.0"]) and c.iloc[0]["ZN.v.0"] == 0
    with pytest.raises(KeyError):
        layered_positions(pd.DataFrame({"ZN.v.0": [1.0]}, index=labels[:1]), "ust_layers", state=st)


def test_struct_pnl_source_combines_legs_with_the_days_weights(monkeypatch):
    import infra.pipeline.event_pnl as ep
    st = _state_for_sizing()
    d = st.days[250]
    pts = pd.date_range(d + pd.Timedelta(hours=13), periods=4, freq="15min")
    steps = pd.DataFrame(np.arange(24, dtype=float).reshape(4, 6), index=pts, columns=LEGS)
    monkeypatch.setattr(ep, "_futures_bbo", lambda legs, grid, bps, **kw: steps[legs])
    out = ep._structures_bbo(["CURVE__FV__WN", "FRONT__TU__H"], None, state=st)
    W = st.W(d)
    np.testing.assert_allclose(out["CURVE__FV__WN"], steps["ZF.v.0"] - steps["UB.v.0"])
    np.testing.assert_allclose(out["FRONT__TU__H"], steps[LEGS].to_numpy() @ W.loc["FRONT__TU__H", LEGS].to_numpy())


def test_a_strategy_with_layers_sizes_structure_views_into_futures(monkeypatch):
    from infra.strategies.cevt import CEVT
    from infra.strategies.config.cevt import CEVTSpec
    st = _state_for_sizing()
    s = CEVT(CEVTSpec("t", instruments=("CURVE__FV__WN", "FLY__FV__UXY__WN"), layers="ust_layers"))
    assert sorted(s.futures()) == sorted(LEGS)
    labels = pd.DatetimeIndex([st.days[300] + pd.Timedelta(hours=14)])
    sig = pd.DataFrame({"CURVE__FV__WN": [0.5], "FLY__FV__UXY__WN": [-1.0]}, index=labels)
    pos = s.size(sig, labels, state=st)
    assert list(pos.columns) == LEGS and pos.iloc[0]["TN.v.0"] < 0 and pos.iloc[0]["ZT.v.0"] == 0
    assert "vol:macro" in s.sizing_diagnostics_.columns
