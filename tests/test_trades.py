"""Tick trades with aggressor side (infra/processing/trades.py, infra/pipeline/trades.py): no API."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.api import databento_client as api
from infra.pipeline import trades as tp
from infra.processing import trades as tr

D = pd.Timestamp


def _raw(rows):
    """Databento-shaped trades: (ts_event, side, price, size, sequence)."""
    df = pd.DataFrame(rows, columns=["ts_event", "side", "price", "size", "sequence"])
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    df["ts_recv"] = df["ts_event"] + pd.Timedelta(microseconds=300)
    df["action"], df["flags"] = "T", 0
    return df.set_index("ts_recv")


SWEEP = _raw([
    ("2026-08-21 14:00:00.001872769", "B", 108.421875, 6, 100),
    ("2026-08-21 14:00:54.183302459", "A", 108.40625, 26, 200),     # one sweep: same sequence,
    ("2026-08-21 14:00:54.183303419", "A", 108.40625, 204, 200),    # a microsecond apart
    ("2026-08-21 14:00:53.923300769", "N", 108.40625, 10, 150),     # a spread leg: no aggressor
    ("2026-08-21 14:01:10.000000000", "B", 108.4375, 5, 300),
])


def test_clean_keeps_nanoseconds_and_makes_unique_keys():
    t = tr.clean_trades(SWEEP, "ZNU6")
    assert t["timestamp"].dtype == "datetime64[ns]"
    assert t["timestamp"].iloc[-1] == D("2026-08-21 14:01:10")
    assert not t.duplicated(tr.TRADE_KEYS).any()
    assert set(t["side"]) == {"A", "B", "N"}


def test_encode_rounds_to_fixed_point_and_keeps_ticks_distinct():
    t = tr.clean_trades(SWEEP, "ZNU6")
    enc = tr.encode_trades(t)
    assert enc["price"].dtype == "int32"
    back = tr.decode_trades(enc)
    assert (back["price"] - t["price"]).abs().max() <= 0.5 / tr.PRICE_SCALE + 1e-12   # 1/64 ticks sit on a half unit
    assert back["price"].nunique() == t["price"].nunique()


def test_signed_bars():
    bars = tr.signed_bars(tr.clean_trades(SWEEP, "ZNU6")).set_index("timestamp")
    b0 = bars.loc[D("2026-08-21 14:00")]
    assert (b0["buy_volume"], b0["sell_volume"], b0["none_volume"]) == (6, 230, 10)
    assert b0["signed_volume"] == -224 and b0["imbalance"] == pytest.approx(-224 / 236)
    assert b0["vwap_sell"] == pytest.approx(108.40625)
    assert bars.loc[D("2026-08-21 14:01"), "imbalance"] == 1.0


def test_month_pieces():
    p = tp.month_pieces([(D("2026-04-07"), D("2026-06-10"))])
    assert p == [(D("2026-04-07"), D("2026-05-01")), (D("2026-05-01"), D("2026-06-01")),
                 (D("2026-06-01"), D("2026-06-10"))]


def test_load_stores_reads_and_never_asks_twice(tmp_path, monkeypatch):
    root, cov = tmp_path / "trades", tmp_path / "cov.parquet"
    calls = []

    def fake(dataset, symbol, start, end, **_):
        calls.append((symbol, start, end))
        return SWEEP

    monkeypatch.setattr(api, "fetch_futures_trades", fake)
    kw = dict(dataset="GLBX.MDP3", root=root, coverage_file=cov)
    df = tp.load_trades(["ZNU6"], "2026-08-21", "2026-08-22", **kw)
    assert len(calls) == 1 and len(df) == 5
    tp.load_trades(["ZNU6"], "2026-08-21", "2026-08-22", **kw)
    assert len(calls) == 1                                   # covered: no second request
    n = tp.build_signed_bars(["ZNU6"], "2026-08-21", "2026-08-22", trades_root=root, root=tmp_path / "bars")
    assert n == 2
    bars = tp.read_signed_bars(["ZNU6"], "2026-08-21", "2026-08-22", root=tmp_path / "bars")
    assert bars["signed_volume"].tolist() == [-224, 5]
