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


def test_fetches_run_in_parallel_but_every_store_happens_on_the_calling_thread():
    import threading
    import time

    from infra.pipeline.front_quotes import _parallel
    main, stored, fetch_threads = threading.get_ident(), [], set()

    def fetch(key, job):
        fetch_threads.add(threading.get_ident())
        time.sleep(0.2)
        if key == "bad":
            raise ConnectionError("504 gateway timeout")
        return job * 10

    def store(key, job, result):
        assert threading.get_ident() == main  # shared files: never written concurrently
        stored.append((key, result))

    t0 = time.monotonic()
    errors = _parallel({"a": 1, "b": 2, "c": 3, "bad": 4}, fetch, store, workers=4)
    assert time.monotonic() - t0 < 0.6  # 4 x 0.2s fetches overlapped, not serial
    assert sorted(stored) == [("a", 10), ("b", 20), ("c", 30)] and len(fetch_threads) > 1
    assert errors == {"bad": "ConnectionError: 504 gateway timeout"}  # collected, the rest still stored


def test_long_quote_windows_are_cut_into_weekly_pieces():
    from infra.coverage.intervals import split_intervals
    pieces = split_intervals([(D("2015-01-01"), D("2015-01-20")), (D("2015-03-01"), D("2015-03-03"))],
                             pd.Timedelta(days=7))
    assert pieces == [(D("2015-01-01"), D("2015-01-08")), (D("2015-01-08"), D("2015-01-15")),
                      (D("2015-01-15"), D("2015-01-20")), (D("2015-03-01"), D("2015-03-03"))]


def test_a_contract_is_never_front_before_it_lists():
    from infra.relative.rolls import drop_unlisted
    m = _mapping([("TNH6", "2015-12-01", "2016-03-01")])
    contracts = pd.DataFrame({"ticker": ["TNH6"], "activation": [D("2015-12-18")]})
    out = drop_unlisted(m, contracts)[0]
    assert out[:D("2015-12-17")].isna().all() and (out[D("2015-12-18"):] == "TNH6").all()
    assert quote_windows(drop_unlisted(m, contracts))["TNH6"] == [(D("2015-12-18"), D("2016-03-02"))]


def test_a_window_databento_cannot_resolve_is_skipped_not_fatal(monkeypatch):
    from databento.common.error import BentoClientError

    from infra.api import databento_client as api
    from infra.pipeline.front_quotes import price_front_quotes

    def cost(dataset, schema, symbols, start, end, stype, client):
        if start < D("2016-01-11"):
            raise BentoClientError(http_status=422, message="422 symbology_invalid_request")
        return 0.1
    monkeypatch.setattr(api, "estimate_cost", cost)
    gaps = {"TNH6": [(D("2015-12-18"), D("2016-01-22"))]}  # one window, partly before it trades
    skipped = []
    assert round(price_front_quotes("TN", gaps, unresolvable=skipped), 6) == 0.1  # only the week from 01-15 resolves
    assert [w[1] for w in skipped] == [D("2015-12-18"), D("2015-12-25"), D("2016-01-01"), D("2016-01-08")]
    assert gaps["TNH6"] == [(D("2016-01-15"), D("2016-01-22"))]
