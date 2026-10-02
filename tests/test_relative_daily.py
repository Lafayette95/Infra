"""Offline tests for relative DAILY settlement/open-interest (pipeline/relative_daily.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.api import databento_client as api
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline import relative_daily as reld
from infra.relative.symbology import RelativeSpec

D = pd.Timestamp


def _contracts() -> pd.DataFrame:
    """SRZ4 (front through its Mar 18 expiry, inclusive), SRH5 takes over Mar 19."""
    return pd.DataFrame({
        "root": "SR3", "ticker": ["SRZ4", "SRH5"], "instrument_id": [1, 2],
        "expiry": pd.to_datetime(["2025-03-18", "2025-06-17"]).astype("datetime64[ms]"),
        "activation": pd.NaT,
    })


def _fake_defs(dataset, parents, day, **_):
    c = _contracts()
    return pd.DataFrame({
        "raw_symbol": c["ticker"], "instrument_class": "F", "instrument_id": c["instrument_id"],
        "expiration": pd.to_datetime(c["expiry"], utc=True), "activation": pd.NaT,
    })


def _fake_statistics(dataset, symbol, start, end, **_):
    """Per UTC day: one settlement, one OI and one cleared-volume row, comfortably inside
    the trading day (ts_ref = the day). SRH5 out-trades SRZ4 every day."""
    days = list(pd.date_range(start, end, freq="D", inclusive="left"))
    n = len(days)
    volume = 100 if symbol == "SRH5" else 10
    return pd.DataFrame({
        "ts_recv": pd.to_datetime([d + pd.Timedelta(hours=h) for h in (19, 2, 20) for d in days], utc=True),
        "ts_ref": pd.to_datetime(days * 3, utc=True),
        "stat_type": [3] * n + [9] * n + [6] * n,
        "price": [95.0 + i * 0.01 for i in range(n)] + [None] * 2 * n,
        "quantity": [None] * n + [1000 + i for i in range(n)] + [volume] * n,
    })


def _no_1m(*a, **k):
    raise AssertionError("daily relative series must never fetch 1-minute data")


def _paths(tmp_path):
    return dict(
        contracts_file=tmp_path / "contracts.parquet",
        defs_coverage_file=tmp_path / "defs_cov.parquet",
        daily_root=tmp_path / "Daily",
        daily_coverage_file=tmp_path / "daily_cov.parquet",
    )


def _plan_paths(paths):
    return {k: paths[k] for k in ("contracts_file", "defs_coverage_file", "daily_root", "daily_coverage_file")}


def _stub(monkeypatch, calls=None):
    monkeypatch.setattr(api, "fetch_definitions", _fake_defs)
    monkeypatch.setattr(api, "fetch_statistics",
                        lambda *a, **k: ((calls.append(a) if calls is not None else None), _fake_statistics(*a, **k))[1])
    monkeypatch.setattr(api, "fetch_futures_ohlcv", _no_1m)
    monkeypatch.setattr(contracts_pipe, "snapshot_days", lambda s, e, step=30: [D("2025-03-01")])


def test_calendar_ranked_daily_series(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    df = reld.load_relative_daily([RelativeSpec("SR3", "c", 0)], "2025-03-15", "2025-03-21", **_paths(tmp_path))
    assert set(df["ticker"].astype(str)) == {"SR3.c.0"}
    by_day = df.assign(day=df["timestamp"].dt.strftime("%m-%d")).set_index("day")["contract"].to_dict()
    assert by_day["03-17"] == "SRZ4" and by_day["03-19"] == "SRH5"  # matches the c.0 roll on Mar 19
    assert {"settlement_price", "open_interest", "volume"} <= set(df.columns)
    assert len(calls) == 2  # one per absolute contract needed (SRZ4, SRH5)


def test_volume_ranked_daily_series_ranks_on_cleared_volume(tmp_path, monkeypatch):
    _stub(monkeypatch)
    df = reld.load_relative_daily([RelativeSpec("SR3", "v", 0)], "2025-03-15", "2025-03-21", **_paths(tmp_path))
    assert set(df["ticker"].astype(str)) == {"SR3.v.0"}
    # SRH5 out-trades SRZ4 every day, and volume from before the start is read too, so
    # even the first day ranks on real prior volume rather than falling back to calendar
    assert set(df["contract"]) == {"SRH5"}
    assert list(df.columns) == ["timestamp", "ticker", "settlement_price", "open_interest", "volume", "contract"]


def test_repeat_request_costs_nothing(tmp_path, monkeypatch):
    calls = []
    _stub(monkeypatch, calls)
    paths = _paths(tmp_path)
    for spec in (RelativeSpec("SR3", "c", 0), RelativeSpec("SR3", "v", 0)):
        reld.load_relative_daily([spec], "2025-03-15", "2025-03-21", **paths)
        n_calls = len(calls)
        reld.load_relative_daily([spec], "2025-03-15", "2025-03-21", **paths)
        assert len(calls) == n_calls
        assert reld.plan_relative_daily_update([spec], "2025-03-15", "2025-03-21", **_plan_paths(paths)) == {}


def test_plan_for_a_volume_spec_includes_the_volume_history(tmp_path, monkeypatch):
    _stub(monkeypatch)
    paths = _paths(tmp_path)
    contracts_pipe.ensure_contracts(
        __import__("infra.config", fromlist=["FUTURES_ROOTS"]).FUTURES_ROOTS["SR3"],
        "2025-03-01", "2025-03-21", fetch_missing=True,
        contracts_file=paths["contracts_file"], coverage_file=paths["defs_coverage_file"],
    )
    plan_c = reld.plan_relative_daily_update([RelativeSpec("SR3", "c", 0)], "2025-03-15", "2025-03-21",
                                             **_plan_paths(paths))
    plan_v = reld.plan_relative_daily_update([RelativeSpec("SR3", "v", 0)], "2025-03-15", "2025-03-21",
                                             **_plan_paths(paths))
    assert min(g0 for gaps in plan_c.values() for g0, _ in gaps if not isinstance(g0, str)) == D("2025-03-15")
    assert min(g0 for k, gaps in plan_v.items() if not k.startswith("definitions") for g0, _ in gaps) < D("2025-03-15")
    assert not any(k.startswith("1m:") for k in plan_v)
