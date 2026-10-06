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
    contracts = {"X": pd.Series("ZNH4", index=labels_all)}
    marks = _quotes(labels_all, "ZNH4", np.full(len(labels_all), 110.0), half=1 / 128)
    cal = pd.DataFrame({"point_value": [1000.0], "last_trade": [pd.NaT], "first_notice": [pd.NaT]}, index=["ZNH4"])
    kw = dict(root=root / "S", models_root=root, vols=vols, contracts=contracts, marks=marks, calendar=cal)
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
    assert set(pabs["contract"].dropna()) == {"ZNH4"}
    pnl_t = sstore.read_series("t_cevt", "pnl", root=root / "S")
    traded = pos["X"].diff().abs().fillna(pos["X"].abs()).sum()
    assert pnl_t["gross"].sum() == 0 and pnl_t["cost"].sum() == pytest.approx(traded / 128 * 0.5 * 1000)
    assert sr.rebuild("t_cevt", "2024-12-31", **kw)["identical"] is True
    # include_failing turns excluded codes into zeros, which dilute the mean
    s_mean = CEVT(spec, gate="none", within="mean")
    v = s_mean.views({"f": preds}, {"f": fdr}, sig.index)
    assert (v["n:X"] >= sig["n:X"]).all()


# --------------------------------------------------------------------------- accounting (P&L and costs)
from infra.strategies import accounting as acc  # noqa: E402
from infra.strategies.config.accounting import get_accounting_spec  # noqa: E402


def _labels(n=6):
    return pd.date_range("2024-03-05 14:00", periods=n, freq="15min")


def _quotes(labels, contract, mids, half=0.01, sizes=500.0, age_min=0.0, halt=None, bid_nan=None):
    n = len(labels)
    mids = np.asarray(mids, dtype=float)
    bid, ask = mids - half, mids + half
    if bid_nan is not None:
        bid = np.where(bid_nan, np.nan, bid)
    return pd.DataFrame({"timestamp": labels, "contract": contract, "bid": bid, "ask": ask, "bid_size": sizes,
                         "ask_size": sizes, "quote_time": labels - pd.to_timedelta(np.broadcast_to(age_min, n), "min"),
                         "mark": mids, "halt": False if halt is None else halt})


CAL = pd.DataFrame({"point_value": [1000.0, 1000.0], "last_trade": [D("2024-06-18"), D("2024-09-19")],
                    "first_notice": [D("2024-05-31"), D("2024-08-30")]}, index=["ZNM4", "ZNU4"])


def _pabs(labels, contract, pos, inst="ZN.v.0"):
    return pd.DataFrame({"label": labels, "instrument": inst, "contract": contract, "position": pos})


def test_accounting_align_and_multiply_with_half_the_half_spread():
    lab = _labels(4)
    marks = _quotes(lab, "ZNM4", [110.0, 110.5, 110.25, 110.25], half=1 / 64)
    pabs = _pabs(lab, "ZNM4", [10.0, 10.0, -5.0, 0.0])
    r = acc.account(pabs, lab, marks, CAL, get_accounting_spec("bbo_mid"))
    t = r.totals
    # step 2: 10 x 0.5 x 1000 = 5000; step 3: 10 x -0.25 x 1000 = -2500; step 4: -5 x 0 = 0
    np.testing.assert_allclose(t["gross"].to_numpy(), [0, 5000, -2500, 0])
    # costs: |trade| x 1/64 x 0.5 x 1000 at labels 1, 3, 4: 10, 15, 5 contracts
    np.testing.assert_allclose(t["cost"].to_numpy(), np.array([10, 0, 15, 5]) / 64 * 0.5 * 1000)
    np.testing.assert_allclose(t["net"], t["gross"] - t["cost"])
    full = acc.account(pabs, lab, marks, CAL, get_accounting_spec("bbo_mid", spread_paid=1.0))
    np.testing.assert_allclose(full.totals["cost"], 2 * t["cost"])
    assert r.checks.empty or not (r.checks["severity"] == "fail").any()


def test_a_roll_is_two_trades_and_pnl_stays_on_each_contract():
    lab = _labels(3)
    marks = pd.concat([_quotes(lab, "ZNM4", [110.0, 110.5, 111.0]), _quotes(lab, "ZNU4", [109.0, 109.0, 120.0])])
    pabs = pd.concat([_pabs(lab[:2], "ZNM4", [4.0, 4.0]), _pabs(lab[2:], "ZNU4", [4.0])])
    r = acc.account(pabs, lab, marks, CAL, get_accounting_spec("bbo_mid", spread_paid=0.0))
    # step 3 is earned by the M4 contract held over it (+0.5); U4's jump to 120 is not ours yet
    np.testing.assert_allclose(r.totals["gross"].to_numpy(), [0, 2000, 2000])
    c = r.contracts.set_index(["label", "contract"])
    assert c.loc[(lab[2], "ZNM4"), "trade"] == -4 and c.loc[(lab[2], "ZNU4"), "trade"] == 4


def test_non_executable_trades_wait_and_are_reported():
    lab = _labels(6)
    stale = np.array([0, 0, 10, 10, 0, 0], dtype=float)               # minutes old at labels 3-4
    marks = _quotes(lab, "ZNM4", [110.0, 110.0, 110.5, 111.0, 111.0, 111.0], age_min=stale)
    pabs = _pabs(lab, "ZNM4", [0.0, 0.0, 5.0, 5.0, 5.0, 5.0])
    r = acc.account(pabs, lab, marks, CAL, get_accounting_spec("bbo_mid"))
    ex = r.contracts.set_index("label")["executed"].reindex(lab).fillna(0)
    np.testing.assert_allclose(ex.to_numpy(), [0, 0, 0, 0, 5, 5])       # bought once the quote is fresh
    assert r.totals["gross"].sum() == 0                                 # missed the move: no P&L for it
    d = r.checks[r.checks["check"] == "deferred"].iloc[0]
    assert d["n"] == 2 and "no_fresh_quote" in d["detail"]
    ig = acc.account(pabs, lab, marks, CAL, get_accounting_spec("bbo_mid_ignore"))
    assert ig.totals["gross"].sum() == pytest.approx(5 * 0.5 * 1000)


@pytest.mark.parametrize("kw,reason", [({"halt": np.array([0, 0, 1, 0], bool)}, "halt"),
                                       ({"bid_nan": np.array([0, 0, 1, 0], bool)}, "one_sided")])
def test_halt_and_one_sided_books_block_trades(kw, reason):
    lab = _labels(4)
    marks = _quotes(lab, "ZNM4", [110.0] * 4, **kw)
    r = acc.account(_pabs(lab, "ZNM4", [0.0, 0.0, 3.0, 3.0]), lab, marks, CAL, get_accounting_spec("bbo_mid"))
    assert reason in r.checks.loc[r.checks["check"] == "deferred", "detail"].iloc[0]


def test_wide_spread_uses_only_earlier_spreads():
    lab = pd.date_range("2024-03-05 14:00", periods=40, freq="15min")
    half = np.full(40, 1 / 128)
    half[-1] = 10 / 128                                                 # 10x the usual spread at the last label
    marks = _quotes(lab, "ZNM4", np.full(40, 110.0), half=half)
    pos = np.zeros(40)
    pos[-1] = 1.0
    r = acc.account(_pabs(lab, "ZNM4", pos), lab, marks, CAL, get_accounting_spec("bbo_mid"))
    assert r.contracts["executed"].abs().sum() == 0
    assert r.checks.loc[r.checks["check"] == "target_not_reached"].shape[0] == 1


def test_checks_delivery_top_of_book_and_missing_marks():
    lab = pd.DatetimeIndex([D("2024-05-30 15:00"), D("2024-05-31 15:00"), D("2024-06-03 15:00")])
    marks = _quotes(lab, "ZNM4", [110.0, np.nan, 110.0], sizes=50.0)
    r = acc.account(_pabs(lab, "ZNM4", [100.0, 100.0, 100.0]), lab, marks, CAL,
                    get_accounting_spec("bbo_mid_ignore"))
    checks = set(r.checks["check"])
    assert {"held_in_delivery", "exceeds_top_of_book", "no_mark"} <= checks
    assert np.isnan(r.totals["gross"].iloc[1])                          # never a silent zero


def test_daily_settlement_marks_execute_on_the_right_day():
    from infra.pipeline.futures_marks import settlement_marks
    days = pd.bdate_range("2024-03-04", periods=5)
    sett = pd.DataFrame({"timestamp": days, "ticker": "ZNM4", "settlement_price": [110.0, 110.5, 111.0, 110.0, 110.0]})
    snaps = pd.DataFrame({"timestamp": days + pd.Timedelta(hours=19), "ticker": "ZNM4", "bid": 109.99, "ask": 110.01})
    # decisions at 12:00 UTC (before 14:00 CT = 19:00/20:00 UTC) trade that day; 21:00 UTC trades the next
    lab = pd.DatetimeIndex([days[0] + pd.Timedelta(hours=12), days[1] + pd.Timedelta(hours=21)])
    m = settlement_marks(["ZNM4"], lab, settlements=sett, snaps=snaps)
    assert list(m["exec_day"]) == [days[0], days[2]]
    labels = days + pd.Timedelta(hours=12)
    m = settlement_marks(["ZNM4"], labels, settlements=sett, snaps=snaps.iloc[[0, 1, 2, 3]])
    r = acc.account(_pabs(labels, "ZNM4", [2.0, 2.0, 0.0, 0.0, 1.0]), labels, m, CAL,
                    get_accounting_spec("settlement"))
    np.testing.assert_allclose(r.totals["gross"].to_numpy(), [0, 1000, 1000, 0, 0])
    assert r.totals["cost"].iloc[-1] == pytest.approx(0.01 * 0.5 * 1000)    # fallback: trailing median spread
    assert "cost_fallback" in set(r.checks["check"])


def test_mark_jump_flags_a_reversed_spike_not_a_lasting_move():
    lab = pd.date_range("2024-03-05 14:00", periods=80, freq="15min")
    mids = 110.0 + np.tile([0.0, 1 / 64], 40)
    mids[60] += 1.0                                                     # spike that reverses
    mids[70:] += 1.0                                                    # news that stays
    r = acc.account(_pabs(lab, "ZNM4", np.ones(80)), lab, _quotes(lab, "ZNM4", mids), CAL,
                    get_accounting_spec("bbo_mid"))
    jumps = r.checks[r.checks["check"] == "mark_jump"]
    assert list(jumps["label"]) == [lab[60]]


def test_cevt_views_have_no_nan_where_one_instrument_is_flat():
    lab = pd.date_range("2024-01-05 13:00", periods=4, freq="15min")
    preds = pd.DataFrame({"end": lab[2], "legal": 1.0, "code": 0, "fit_as_of": D("2024-01-01"),
                          "t:A": 2.0, "expected:A": 1.0, "passed:A": 1.0,
                          "t:B": np.nan, "expected:B": np.nan, "passed:B": 0.0}, index=lab[:1])
    v = CEVT(CEVTSpec("t", instruments=("A", "B"), gate="passed")).views({"f": preds}, {}, lab)
    assert not v.isna().any().any() and v.loc[lab[0], "tsig:B"] == 0 and v.loc[lab[0], "tsig:A"] > 0
