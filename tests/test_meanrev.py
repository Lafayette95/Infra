"""Cross-sectional mean reversion of factor residuals (infra/models/meanrev): a planted OU residual is
found with its half-life and faded; random-walk residuals fail the gates; the netted book is
factor-neutral; fit modes; point in time; params round trip. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.models.meanrev.config import get_meanrev_spec
from infra.models.meanrev.model import MeanRevModel

COLS = tuple(f"i{j}" for j in range(5))


def _panel(b=0.9, n=1500, seed=0, random_walk=False):
    """Daily moves of 5 instruments: 2 large factors + residual LEVELS that are AR(1) with
    coefficient b (half-life ln2 / -ln b) - or random walks."""
    rng = np.random.default_rng(seed)
    L = np.array([[1, 1, 1, 1, 1], [-2, -1, 0, 1, 2]], dtype=float).T
    f = rng.normal(0, [5.0, 2.0], (n, 2))
    r = np.zeros((n, 5))
    for t in range(1, n):
        r[t] = (r[t - 1] if random_walk else b * r[t - 1]) + rng.normal(0, 1.0, 5)
    moves = f @ L.T + np.diff(np.vstack([np.zeros(5), r]), axis=0)
    return pd.DataFrame(moves, index=pd.bdate_range("2015-01-01", periods=n), columns=COLS), L


def _spec(name="default", **kw):
    return get_meanrev_spec(name, columns=COLS, window=250, factor_overrides=(("prep", ()), ("n_components", 2)), **kw)


def _fit(panel, as_of="2019-12-31", **kw):
    m = MeanRevModel(_spec(**kw))
    data = m.prepare(panel)
    return m.fit(data, as_of=as_of), data


def test_a_planted_ou_residual_is_found_faded_and_netted_factor_neutral():
    panel, L = _panel(b=0.9)
    m, data = _fit(panel)
    t = m.fitted_.ou
    assert (t["tradable"] == 1).mean() >= 0.6
    hl_true = np.log(2) / -np.log(0.9)
    assert t.loc[t["tradable"] == 1, "half_life"].median() == pytest.approx(hl_true, rel=0.5)
    assert (t["r2_proj"] > 0.999).all()                                     # a PCA residual is an exact portfolio
    out = m.predict(data, start="2019-12-31")
    s, sig = out[[f"s:{c}" for c in COLS]].to_numpy(), out[[f"signal:{c}" for c in COLS]].to_numpy()
    live = np.isfinite(s) & (sig != 0)
    assert (np.sign(sig[live]) == -np.sign(s[live])).all()                  # fades the s-score
    pos = out[[f"pos:{c}" for c in COLS]].to_numpy()
    # factor-neutral: exposure to each true factor tiny next to the gross position (loadings are estimated)
    expo = np.abs(pos @ (L / np.linalg.norm(L, axis=0))).max(axis=0)
    assert (expo < 0.05 * np.abs(pos).sum(axis=1).max()).all()
    # the trade metrics: shorter half-life -> faster expected passage; p in [0, 1]
    c = t.index[t["tradable"] == 1][0]
    assert ((out[f"p_target:{c}"] >= 0) & (out[f"p_target:{c}"] <= 1)).all()
    assert (out[f"fpt_median:{c}"] <= out[f"fpt_mean:{c}"] + 1e-9).all()


def test_the_book_makes_money_out_of_sample_on_a_reverting_panel_and_nothing_on_random_walks():
    from infra.models.walk_forward import walk_forward
    for rw, want in ((False, 1.0), (True, None)):
        panel, _ = _panel(b=0.85, random_walk=rw, seed=1)
        spec = _spec()
        res = walk_forward(lambda: MeanRevModel(spec), panel, "2017-01-02", panel.index[-1], refit="ME")
        pos = res.predictions[[f"pos:{c}" for c in COLS]].shift(2)          # decided on t, earns t+2's move
        pnl = (pos.to_numpy() * panel.reindex(pos.index).to_numpy()).sum(axis=1)
        sharpe = np.nanmean(pnl) / np.nanstd(pnl) * np.sqrt(252)
        tradable_share = res.predictions["n_tradable"].mean() / len(COLS)
        if want:
            assert sharpe > want
        else:
            assert tradable_share < 0.25                                    # a unit root rarely passes the gates


def test_fit_modes_and_the_threshold_rule():
    panel, _ = _panel(random_walk=True, seed=2)
    ex, data = _fit(panel, fit_mode="exante", gates=())
    assert (ex.fitted_.ou["tradable"] == 1).all()                          # exante: traded regardless
    out = ex.predict(data, start="2019-12-31")
    z = out[[f"z:{c}" for c in COLS]].to_numpy()
    sig = out[[f"signal:{c}" for c in COLS]].to_numpy()
    assert (np.sign(sig[np.abs(z) > 0.1]) == -np.sign(z[np.abs(z) > 0.1])).all()
    pr, _ = _fit(panel, fit_mode="prior", gates=("min_obs", "reverting"))
    assert ((pr.fitted_.ou["tradable"] == 1) <= (pr.fitted_.ou["b"] < 1)).all()
    th, data2 = _fit(_panel(b=0.9)[0], signal_rule="threshold", entry=1.25, exit=0.5)
    o = th.predict(data2, start="2019-12-31")
    for c in COLS:
        assert set(np.unique(o[f"signal:{c}"])) <= {-1.0, 0.0, 1.0}
        if th.fitted_.ou.loc[c, "tradable"] == 0:
            assert (o[f"signal:{c}"] == 0).all()
            continue
        sc = o[f"s:{c}"]
        assert (o.loc[sc.abs() <= 0.5, f"signal:{c}"] == 0).all()           # closed inside the exit band
        assert (o.loc[sc.abs() >= 1.25, f"signal:{c}"] == -np.sign(sc[sc.abs() >= 1.25])).all()


def test_pooled_point_in_time_and_round_trip():
    panel, _ = _panel(b=0.9)
    m, data = _fit(panel, pooling="pooled")
    assert m.fitted_.ou["b"].nunique() == 1
    cut = pd.Timestamp("2019-12-31")
    shocked = panel.copy()
    shocked.loc[shocked.index > cut] *= 5
    a, _ = _fit(panel)
    b, _ = _fit(shocked)
    pd.testing.assert_frame_equal(a.fitted_.ou, b.fitted_.ou)
    r = MeanRevModel.from_params(a.params(), _spec())
    pd.testing.assert_frame_equal(r.predict(data, start=cut), a.predict(data, start=cut))
