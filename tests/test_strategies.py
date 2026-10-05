"""The strategy layer (infra/strategies, infra/jobs, infra/storage/strategy_store) and family runs:
the t-signal, the aggregators, the single P&L shift, positions, the point-in-time family FDR,
family runs incremental == rebuild, CEVT views end to end. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.models.event_study.config import EVENT_FAMILIES, EventStudySpec, FamilySpec, LegRule
from infra.models.event_study.family import fdr_table
from infra.storage import strategy_store as sstore
from infra.strategies.base import Strategy, StrategySpec, asof, pnl
from infra.strategies.cevt import CEVT, aggregate, t_signal
from infra.strategies.config.cevt import CEVT_STRATEGIES, CEVTSpec
from tests.test_event_study_model import NFP, _synthetic

D = pd.Timestamp


# --------------------------------------------------------------------------- pure pieces
def test_t_signal_is_concave_and_full_strength_at_the_cap():
    s = t_signal([0, 1.5, 3, 10, -3, np.nan], cap=3, a=0.5)
    assert s[0] == 0 and s[2] == pytest.approx(1) and s[3] == pytest.approx(1) and s[4] == pytest.approx(-1)
    assert 0.5 < s[1] < 1 and np.isnan(s[5])


def test_aggregators():
    v = pd.Series([0.9, 0.5, 0.6, 0.8, -0.8, 0.3])
    k = [pd.Series(list("aaabbc"))]
    sm = aggregate(v, k, "smooth")
    assert sm["a"] == pytest.approx(0.9) and sm["b"] == pytest.approx(0.0) and sm["c"] == pytest.approx(0.3)
    # partial agreement: a = 0.5/1.5 = 1/3 -> 1/3 * 1.0 + 2/3 * mean(1.0, -0.5) = 0.5
    assert aggregate(pd.Series([1.0, -0.5]), [pd.Series(["x", "x"])], "smooth")["x"] == pytest.approx(0.5)
    assert aggregate(v, k, "mean")["a"] == pytest.approx(2 / 3)
    assert aggregate(v, k, "absmax")["b"] == pytest.approx(0.8)
    assert aggregate(pd.Series([1.0, -0.5]), [pd.Series(["x", "x"])], "threshold:0.8")["x"] == pytest.approx(0.25)
    with pytest.raises(KeyError):
        aggregate(v, k, "nope")


def test_pnl_is_the_one_label_shift():
    idx = pd.date_range("2024-01-02 11:00", periods=4, freq="15min")
    pos = pd.DataFrame({"X": [1.0, 2.0, 0.0, 0.0]}, index=idx)          # decided AT the label
    steps = pd.DataFrame({"X": [5.0, 7.0, 11.0, 13.0]}, index=idx)      # step ENDING at the label
    np.testing.assert_allclose(pnl(pos, steps)["X"].to_numpy(), [np.nan, 7.0, 22.0, 0.0])
    np.testing.assert_allclose(pnl(pos, steps, lag=1)["X"].to_numpy()[2:], [11.0, 26.0])


def test_asof_uses_only_values_known_at_the_label():
    s = pd.Series([1.0, 2.0], index=[D("2024-01-02 22:00"), D("2024-01-03 22:00")])
    out = asof(s, pd.DatetimeIndex([D("2024-01-02 21:59"), D("2024-01-02 22:00"), D("2024-01-03 12:00")]))
    assert np.isnan(out[0]) and out[1] == 1.0 and out[2] == 1.0


class _Plain(Strategy):
    def views(self, x):
        return x


def test_full_strength_scaling_and_realised_targeting():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2020-01-01", periods=800)
    vols = pd.DataFrame({"A": 10_000.0, "B": 40_000.0}, index=days)
    sig = pd.DataFrame({"A": 1.0, "B": -0.5}, index=days)
    s = _Plain(StrategySpec(instruments=("A", "B"), target_vol_usd=1e6, scaling="full_strength"))
    pos = s.positions(sig, vols)
    assert pos["A"].iloc[0] == pytest.approx(1e6 / np.sqrt(2) / 10_000) and pos["B"].iloc[0] == pytest.approx(
        -0.5 * 1e6 / np.sqrt(2) / 40_000)
    # realised: steps with the stated $ vol per contract -> the scaled strategy hits the target
    steps = pd.DataFrame({"A": rng.normal(0, 10_000 / np.sqrt(252), len(days)),
                          "B": rng.normal(0, 40_000 / np.sqrt(252), len(days))}, index=days)
    s2 = _Plain(StrategySpec(instruments=("A", "B"), target_vol_usd=1e6, scaling="realised"))
    p2 = s2.positions(sig * 0.3, vols, steps)          # weak views get levered back up to the target
    real = pnl(p2, steps).sum(axis=1, min_count=1).iloc[300:].std() * np.sqrt(252)
    assert real == pytest.approx(1e6, rel=0.15)
    assert p2.iloc[:60].isna().all().all()             # no scale before min_obs days of P&L


def test_fdr_table_is_per_fit_day_and_needs_the_own_test():
    def block(fit, p_t, passed, row="X"):
        return pd.DataFrame({"section": "event_stat", "row": row, "col": ["p_t", "passed"], "value": [p_t, passed],
                             "fit_as_of": D(fit)})
    params = {0: pd.concat([block("2024-01-05 12:00", 0.001, 1.0), block("2024-02-02 12:00", 0.001, 1.0)]),
              1: pd.concat([block("2024-01-05 12:15", 0.04, 1.0), block("2024-02-02 12:15", 0.001, 0.0)]),
              2: block("2024-01-05 13:00", 0.9, 0.0)}
    t = fdr_table(params, 0.10)
    jan = t[pd.to_datetime(t["fit_as_of"]).dt.month == 1].set_index("code")
    # one BH set per fit DAY across codes (instants differ): q = [0.003, 0.06, 0.9]
    assert jan.loc[0, "q"] == pytest.approx(0.003) and jan.loc[1, "q"] == pytest.approx(0.06)
    assert jan.loc[0, "passed_fdr"] == 1 and jan.loc[1, "passed_fdr"] == 1 and jan.loc[2, "passed_fdr"] == 0
    feb = t[pd.to_datetime(t["fit_as_of"]).dt.month == 2].set_index("code")
    assert feb.loc[1, "passed_fdr"] == 0                # q passes, but the code's own tests don't


def test_strategy_store_upsert_counts_changes(tmp_path):
    idx = pd.DatetimeIndex([D("2024-01-02 11:00"), D("2024-01-02 11:15")])
    a = pd.DataFrame({"X": [1.0, 2.0]}, index=idx)
    assert sstore.upsert_series("s", "positions", a, root=tmp_path)["new"] == 2
    b = pd.DataFrame({"X": [1.0, 3.0]}, index=idx)
    assert sstore.upsert_series("s", "positions", b, root=tmp_path) == {"new": 0, "unchanged": 1, "changed": 1}
    pd.testing.assert_frame_equal(sstore.read_series("s", "positions", root=tmp_path), b.rename_axis("label"),
                                  check_freq=False)


# --------------------------------------------------------------------------- families + CEVT end to end
FAM = FamilySpec("t_fam", (NFP,), start=LegRule(step_lags=(-1, 0)), end=LegRule(step_lags=(1, 4)),
                 study=EventStudySpec(instruments=("X",), ev_abs_min=0.5, min_obs=5))


@pytest.fixture
def family_env(monkeypatch, tmp_path):
    import infra.models.event_study.model as mm
    panel, occ = _synthetic(effect=1.0, start="2022-01-01", end="2024-12-31")
    monkeypatch.setattr(mm.EventStudy, "occurrences", lambda self, as_of=None: occ)
    monkeypatch.setitem(EVENT_FAMILIES, FAM.name, FAM)
    return panel, occ, tmp_path


def test_family_run_incremental_equals_rebuild(family_env):
    from infra.jobs import family_runs as fr
    panel, _, root = family_env
    fr.create("f", FAM.name, start="2024-01-05", history_start="2022-01-01", refit="ME", root=root)
    days = pd.DatetimeIndex(sorted(set(panel.index.normalize())))
    for day in list(days[days >= D("2024-01-05")][::4]) + [days[-1]]:
        fr.run_daily("f", day, root=root, panel=panel[panel.index < day + pd.Timedelta(days=1)])
    rep = fr.rebuild("f", days[-1], root=root, panel=panel)
    assert rep["identical"] is True, rep
    fdr = fr.read_fdr("f", root=root)
    assert set(fdr["code"]) == {0, 1, 2, 3} and fdr["passed_fdr"].sum() > 0
    preds = fr.predictions("f", root=root)
    assert {"t:X", "passed:X", "anchor", "code", "fit_as_of"} <= set(preds.columns)


def test_cevt_views_positions_and_rebuild(family_env, monkeypatch):
    from infra.jobs import family_runs as fr, strategy_runs as sr
    panel, occ, root = family_env
    fr.create("f", FAM.name, start="2024-01-05", history_start="2022-01-01", refit="ME", root=root)
    fr.rebuild("f", "2024-12-31", root=root, panel=panel, promote=True)
    spec = CEVTSpec("t_cevt", families=("f",), instruments=("X",), target_vol_usd=1e6, plan_vintages=False)
    monkeypatch.setitem(CEVT_STRATEGIES, spec.name, spec)
    vols = {"X": pd.Series([20_000.0], index=[D("2020-01-01")])}
    labels_all = pd.date_range("2022-01-01", "2025-01-01", freq="15min")
    contracts = {"X": pd.Series("XH4", index=labels_all)}
    kw = dict(root=root / "S", models_root=root, vols=vols, contracts=contracts)
    sr.run_daily("t_cevt", "2024-12-31", **kw)
    sig = sstore.read_series("t_cevt", "signals", root=root / "S")
    pos = sstore.read_series("t_cevt", "positions", root=root / "S")
    preds = fr.predictions("f", root=root)
    fdr = fr.read_fdr("f", root=root)
    # a view is held over start <= L < end, with the windows' sign; flat elsewhere
    on = sig["n:X"] > 0
    assert on.any() and (sig.loc[on, "tsig:X"] > 0).all() and (sig.loc[~on, "tsig:X"] == 0).all()
    w = preds[(preds["legal"] == 1) & (preds["code"] == 1)].iloc[-1]       # &-1 .. &+4 (codes: start x end)
    inside = sig.loc[(sig.index >= w.name) & (sig.index < pd.Timestamp(w["end"]))]
    assert len(inside) == 5 and (inside["n:X"] >= 1).all()
    assert sig.loc[pd.Timestamp(w["end"]), "n:X"] == 0         # the end point is the next decision, flat
    # gate: every contributing code passed the family's FDR at the fit its row used
    assert fdr["passed_fdr"].sum() > 0
    # positions: full strength = target / vol; abs positions carry the contract
    assert pos["X"].max() <= 1e6 / 20_000 + 1e-9 and pos.loc[on, "X"].min() > 0
    pabs = sstore.read_series("t_cevt", "positions_abs", root=root / "S")
    assert set(pabs["contract"].dropna()) == {"XH4"}
    assert sr.rebuild("t_cevt", "2024-12-31", **kw)["identical"] is True
    # include_failing turns excluded codes into zeros, which dilute the mean
    s_mean = CEVT(spec, gate="none", within="mean")
    v = s_mean.views({"f": preds}, {"f": fdr}, sig.index)
    assert (v["n:X"] >= sig["n:X"]).all()
