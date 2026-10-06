"""The central feature maker (infra/processing/features.py, infra/pipeline/features.py): the
grammar, every step point in time, the normalisers' scale, parity with the code it replaced
(B1's features, the event-study partitions), alignment with a forward-fill limit, derived inputs.
Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.processing import features as fx

D = pd.Timestamp
SPECS = ["lvl", "gap:ewm:10", "chg:5", "chg:ewm:5", "x:5:20", "acc:5:5", "accr:5:20", "range:20", "dd:20",
         "chg:5|norm:vol:60", "chg:5|norm:z:60", "chg:5|norm:z0:60", "chg:5|norm:rank:60", "chg:5|norm:robust:60",
         "chg:5|norm:vol:60|abs", "chg:5|sign", "chg:5|clip:2", "chg:5|pow:0.5", "chg:5|lag:2",
         "chg:20|norm:vol:60|part:tercile:100", "lvl|part:z:50:1", "chg:5|part:sign", "lvl|part:fixed:-1:1"]


def _x(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2020-01-01", periods=n))


def test_grammar_parse_and_errors():
    assert fx.parse("chg:20 | norm:vol:60 | abs") == [("chg", ["20"]), ("norm", ["vol", "60"]), ("abs", [])]
    with pytest.raises(ValueError):
        fx.apply(_x(), "norm:vol:60")                      # must start with a kind
    with pytest.raises(KeyError):
        fx.apply(_x(), "chg:5|nope")
    with pytest.raises(ValueError):
        fx.apply(_x(), "lvl", kind="returns")


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("kind", ["level", "moves"])
def test_every_step_is_point_in_time(spec, kind):
    x = _x()
    cut = x.index[400]
    shocked = x.copy()
    shocked[shocked.index > cut] += 50.0
    a, b = fx.apply(x, spec, kind), fx.apply(shocked, spec, kind)
    pd.testing.assert_series_equal(a[a.index <= cut], b[b.index <= cut], check_names=False)


def test_kinds_change_means_difference_or_sum():
    x = _x()
    pd.testing.assert_series_equal(fx.apply(x, "chg:5", "moves"), x.rolling(5).sum(), check_names=False)
    pd.testing.assert_series_equal(fx.apply(x, "chg:5", "level"), x - x.shift(5), check_names=False)
    pd.testing.assert_series_equal(fx.apply(x, "lvl", "moves"), x.cumsum(), check_names=False)


def test_normalisers_have_their_stated_scale():
    x = _x(5000, seed=1)                                   # iid N(0,1) daily moves
    for spec in ("chg:20|norm:vol:60", "chg:ewm:10|norm:vol:60", "accr:5:20|norm:vol:60"):
        assert fx.apply(x, spec, "moves").std() == pytest.approx(1.0, abs=0.12), spec
    z = fx.apply(x, "chg:5|norm:z:250", "moves").dropna()
    assert abs(z.mean()) < 0.1 and z.std() == pytest.approx(1.0, abs=0.1)
    r = fx.apply(x, "chg:5|norm:rank:100", "moves").dropna()
    assert r.between(0, 1).all()
    p = fx.apply(x, "chg:20|norm:vol:60|part:tercile:250", "moves").dropna()
    assert set(p.unique()) == {-1.0, 0.0, 1.0}


def test_b1_features_are_unchanged_by_the_migration():
    from infra.models.autocorr.config import get_autocorr_spec
    from infra.models.autocorr.model import ConditionalAutocorr
    xs = _x(800, seed=2)
    span = 60
    vol = xs.ewm(span=span, min_periods=max(span // 2, 10)).std()
    legacy = {
        "move:20": xs.rolling(20, min_periods=20).sum() / (vol * np.sqrt(20)),
        "absmove:5": (xs.rolling(5, min_periods=5).sum() / (vol * np.sqrt(5))).abs(),
        "level": xs,
        "z:50": (xs - xs.rolling(50, min_periods=25).mean()) / xs.rolling(50, min_periods=25).std(),
    }
    for feat, old in legacy.items():
        m = ConditionalAutocorr(get_autocorr_spec("default", x_feature=feat, vol_span=span))
        pd.testing.assert_series_equal(m._x_feature(xs), old.astype("float64"), check_names=False, rtol=1e-12)


def test_event_study_partitions_are_the_central_ones():
    from infra.models.event_study.conditions import PARTITIONERS, partition
    x = _x(300)
    np.testing.assert_array_equal(partition(x, "rolling_tercile:100"), fx.rolling_tercile(x, "100"))
    assert PARTITIONERS["zscore"] is fx.zscore_bands


def test_align_respects_the_forward_fill_limit():
    from infra.pipeline.features import align
    s = pd.Series([1.0, 2.0], index=[D("2024-01-01 15:00"), D("2024-02-01 15:00")])
    idx = pd.DatetimeIndex([D("2024-01-03"), D("2024-01-20"), D("2024-02-02")])
    a = align(s, idx, limit="5D")
    assert a.iloc[0] == 1.0 and np.isnan(a.iloc[1]) and a.iloc[2] == 2.0
    assert align(s, idx, source="release").iloc[1] == 1.0          # monthly macro: 100 days


def test_pipeline_feature_and_derived_inputs(monkeypatch):
    import infra.pipeline.features as pf
    import infra.pipeline.series_panel as sp
    rng = np.random.default_rng(4)
    days = pd.bdate_range("2021-01-01", periods=400)
    a = pd.Series(rng.normal(0, 1, 400), index=days)
    b = 0.5 * a + pd.Series(rng.normal(0, 1, 400), index=days)
    tl = {"bmk:curve:A": a, "bmk:curve:B": b}
    monkeypatch.setattr(pf, "_timeline", lambda sid, start, end: tl[sid].set_axis(tl[sid].index + pd.Timedelta(hours=34)))
    f = pf.feature("bmk:curve:A | chg:5 | norm:vol:60", "2021-01-01", "2022-12-31")
    assert f.attrs["input_kind"] == "moves" and f.index[0] == days[0] + pd.Timedelta(hours=34)  # availability index
    beta = pf.feature("beta(bmk:curve:B,bmk:curve:A,200) | lvl", "2021-01-01", "2022-12-31").dropna()
    assert beta.iloc[-1] == pytest.approx(0.5, abs=0.15)
    corr = pf.feature("corr(bmk:curve:A,bmk:curve:B,200)", "2021-01-01", "2022-12-31").dropna()
    assert 0.2 < corr.iloc[-1] < 0.7
    assert sp.kind_of("bmk:curve:A") == "moves" and sp.kind_of("bond:US_BOND_10y") == "level"


def test_event_time_features_use_what_was_known(monkeypatch):
    import infra.pipeline.features as pf
    import infra.pipeline.release_calendar as rc
    from infra.processing import event_windows as ew
    cal = pd.DataFrame({"event": "E", "timestamp": [D("2024-03-06 13:30"), D("2024-03-15 13:30")],
                        "stage": "", "source": "x", "time_source": "registry",
                        "known_from": [D("2024-01-01"), D("2024-03-12")], "last_seen": D("2024-12-31")})
    monkeypatch.setattr(rc, "read_release_calendar", lambda as_of=None, events=None: cal)
    monkeypatch.setattr(ew, "consolidate", lambda c: c.assign(day=pd.to_datetime(c["timestamp"]).dt.normalize()))
    to = pf._events("E", "2024-03-01", "2024-03-29", "to")
    assert to[D("2024-03-01")] == 3 and to[D("2024-03-07")] != to[D("2024-03-07")]   # 03-15 not known yet on 03-07
    assert to[D("2024-03-12")] == 3                                                 # known from 03-12
    since = pf._events("E", "2024-03-01", "2024-03-29", "since")
    assert since[D("2024-03-08")] == 2 and np.isnan(since[D("2024-03-01")])


def test_event_time_reaches_past_the_window(monkeypatch):
    import infra.pipeline.features as pf
    import infra.pipeline.release_calendar as rc
    from infra.processing import event_windows as ew
    cal = pd.DataFrame({"event": "E", "timestamp": [D("2024-05-01 18:00")], "stage": "", "source": "x",
                        "time_source": "registry", "known_from": [D("2024-01-01")], "last_seen": D("2024-12-31")})
    monkeypatch.setattr(rc, "read_release_calendar", lambda as_of=None, events=None: cal)
    monkeypatch.setattr(ew, "consolidate", lambda c: c.assign(day=pd.to_datetime(c["timestamp"]).dt.normalize()))
    to = pf._events("E", "2024-03-01", "2024-03-29", "to")
    assert to.notna().all() and to.iloc[-1] > 20                    # the occurrence lies after the window
