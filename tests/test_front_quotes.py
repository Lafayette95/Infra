"""Front-contract quote planning: unbroken runs (decade-safe), roll switches, and the
other contract only around each roll. Pure - no API."""
from __future__ import annotations

import pandas as pd

from infra.pipeline.front_quotes import quote_windows
from infra.relative.rolls import around_switch_windows, contract_runs, roll_switches

D = pd.Timestamp


def _mapping(spans):
    """spans: (ticker, first, last inclusive) -> rank-0 mapping on a daily index."""
    idx = pd.date_range(spans[0][1], spans[-1][2], freq="D")
    col = pd.Series(None, index=idx, dtype=object)
    for t, a, b in spans:
        col[D(a):D(b)] = t
    return pd.DataFrame({0: col})


def test_a_reused_symbol_is_two_runs_never_one_decade_long_window():
    m = _mapping([("ZNH5", "2015-01-01", "2015-02-26"), ("ZNM5", "2015-02-27", "2015-03-10")])
    m2 = _mapping([("ZNH5", "2025-01-01", "2025-01-10")])
    runs = contract_runs(pd.concat([m, m2]), 0)
    assert [(t, a.year, (b - a).days) for t, a, b in runs] == [("ZNH5", 2015, 57), ("ZNM5", 2015, 12), ("ZNH5", 2025, 10)]


def test_the_other_contract_is_held_only_around_each_roll():
    m = _mapping([("ZNH5", "2015-01-01", "2015-02-26"), ("ZNM5", "2015-02-27", "2015-05-27"),
                  ("ZNU5", "2015-05-28", "2015-06-30")])
    assert [(d.strftime("%m-%d"), a, b) for d, a, b in roll_switches(m)] == [("02-27", "ZNH5", "ZNM5"),
                                                                             ("05-28", "ZNM5", "ZNU5")]
    pad = pd.offsets.BDay(5)
    w = quote_windows(m, pad=pad)
    assert w["ZNH5"] == [(D("2015-01-01"), D("2015-02-27") + pad)]  # its run, then 5 bdays as the outgoing leg
    assert w["ZNM5"] == [(D("2015-02-27") - pad, D("2015-05-28") + pad)]  # incoming early, outgoing late
    assert w["ZNU5"] == [(D("2015-05-28") - pad, D("2015-07-01"))]  # clipped to the mapping's end
    assert len(around_switch_windows(m, pad=pad)) == 4
