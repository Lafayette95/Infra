"""Futures snaps (CLAUDE.md 23): selection rule, snap instants across DST, store roundtrip."""
from __future__ import annotations

import pandas as pd

from infra.pipeline import futures_snaps as ps
from infra.processing import futures_snaps as fs

D = pd.Timestamp
TOL = pd.Timedelta(minutes=5)


def _quotes(rows):
    return pd.DataFrame(rows, columns=["timestamp", "ticker", "bid", "ask"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"]), mid=lambda d: (d["bid"] + d["ask"]) / 2)


def test_the_snap_is_the_last_two_sided_quote_at_or_before_the_instant():
    q = _quotes([("2026-09-30 19:28", "ZNZ6", 104.0, 104.02), ("2026-09-30 19:30", "ZNZ6", 104.1, 104.12),
                 ("2026-09-30 19:31", "ZNZ6", 105.0, 105.02),  # after the instant: never used
                 ("2026-09-30 19:30", "ZBZ6", None, 102.0),  # one-sided
                 ("2026-09-30 19:20", "ZTZ6", 101.0, 101.01)])  # 10 minutes old: too stale
    out = fs.select_snaps(q, pd.DataFrame({"snap": ["NY1530"], "timestamp": [D("2026-09-30 19:30")]}), TOL)
    assert out["ticker"].tolist() == ["ZNZ6"] and out["mid"].iloc[0] == 104.11
    assert out["quote_time"].iloc[0] == D("2026-09-30 19:30")


def test_snap_instants_follow_each_zone_through_dst():
    t = ps.snap_times(pd.to_datetime(["2026-07-01", "2026-12-01"]), snaps=("NY1530", "LDN1615"))
    by = {(r.snap, r.timestamp.month): r.timestamp for r in t.itertuples()}
    assert by[("NY1530", 7)] == D("2026-07-01 19:30") and by[("NY1530", 12)] == D("2026-12-01 20:30")
    assert by[("LDN1615", 7)] == D("2026-07-01 15:15") and by[("LDN1615", 12)] == D("2026-12-01 16:15")


def test_weekends_get_no_snap():
    assert ps.snap_times(pd.to_datetime(["2026-10-03"])).empty  # a Saturday


def test_store_roundtrip(tmp_path):
    rows = pd.DataFrame({"timestamp": [D("2026-09-30 19:30")], "snap": ["NY1530"], "ticker": ["ZNZ6"],
                         "bid": [104.21875], "ask": [104.234375], "mid": [104.2265625],
                         "quote_time": [D("2026-09-30 19:30")], "settlement": [104.203125]})
    from infra.storage import parquet_store
    parquet_store.write_partitioned(fs.encode(rows), tmp_path, fs.SNAP_KEYS)
    back = ps.read_futures_snaps("2026-09-30", "2026-10-01", root=tmp_path)
    assert back["mid"].iloc[0] == round(104.2265625, 4) and back["snap"].iloc[0] == "NY1530"
