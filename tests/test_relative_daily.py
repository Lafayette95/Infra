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
    """One settlement + one OI row per UTC day, comfortably inside the trading day
    (10:00 UTC -> 05:00 CT in March, well before the 16:00 CT close)."""
    days = pd.date_range(start, end, freq="D", inclusive="left")
    ts_recv = [d + pd.Timedelta(hours=19) for d in days] + [d + pd.Timedelta(hours=2) for d in days]
    ts_ref = list(days) + list(days)
    return pd.DataFrame({
        "ts_recv": pd.to_datetime(ts_recv, utc=True),
        "ts_ref": pd.to_datetime(ts_ref, utc=True),
        "stat_type": [3] * len(days) + [9] * len(days),
        "price": [95.0 + i * 0.01 for i in range(len(days))] + [None] * len(days),
        "quantity": [None] * len(days) + [1000 + i for i in range(len(days))],
    })


def _fake_1m_bars(dataset, symbol, start, end, **_):
    """SRH5 out-trades SRZ4 every day, so v.0 should pick SRH5 once both are on disk."""
    idx = pd.DatetimeIndex(
        [start.tz_localize("UTC") + pd.Timedelta(days=d, hours=10) for d in range((end - start).days)],
        name="ts_event",
    )
    base = np.full(len(idx), 95.0)
    volume = 100 if symbol == "SRH5" else 10
    return pd.DataFrame({"open": base, "high": base, "low": base, "close": base,
                         "volume": volume, "symbol": symbol, "instrument_id": 1}, index=idx)


def _paths(tmp_path):
    return dict(
        contracts_file=tmp_path / "contracts.parquet",
        defs_coverage_file=tmp_path / "defs_cov.parquet",
        daily_root=tmp_path / "Daily",
        daily_coverage_file=tmp_path / "daily_cov.parquet",
        futures_root_dir=tmp_path / "Futures",
        futures_coverage_file=tmp_path / "futures_cov.parquet",
    )


def test_calendar_ranked_daily_series_never_touches_1m_data(tmp_path, monkeypatch):
    daily_calls, min_calls = [], []
    monkeypatch.setattr(api, "fetch_definitions", _fake_defs)
    monkeypatch.setattr(api, "fetch_statistics", lambda *a, **k: (daily_calls.append(a), _fake_statistics(*a, **k))[1])
    monkeypatch.setattr(api, "fetch_futures_ohlcv", lambda *a, **k: (min_calls.append(a), _fake_1m_bars(*a, **k))[1])
    monkeypatch.setattr(contracts_pipe, "snapshot_days", lambda s, e, step=30: [D("2025-03-10")])

    df = reld.load_relative_daily([RelativeSpec("SR3", "c", 0)], "2025-03-15", "2025-03-21", **_paths(tmp_path))
    assert set(df["ticker"].astype(str)) == {"SR3.c.0"}
    by_day = df.assign(day=df["timestamp"].dt.strftime("%m-%d")).set_index("day")["contract"].to_dict()
    assert by_day["03-17"] == "SRZ4" and by_day["03-19"] == "SRH5"  # matches the c.0 roll on Mar 19
    assert "settlement_price" in df.columns and "open_interest" in df.columns
    assert not min_calls, "c.N ranking needs no 1-minute data - it must never be fetched"
    assert len(daily_calls) == 2  # one per absolute contract needed (SRZ4, SRH5)


def test_volume_ranked_daily_series_fetches_1m_only_for_ranking(tmp_path, monkeypatch):
    daily_calls, min_calls = [], []
    monkeypatch.setattr(api, "fetch_definitions", _fake_defs)
    monkeypatch.setattr(api, "fetch_statistics", lambda *a, **k: (daily_calls.append(a), _fake_statistics(*a, **k))[1])
    monkeypatch.setattr(api, "fetch_futures_ohlcv", lambda *a, **k: (min_calls.append(a), _fake_1m_bars(*a, **k))[1])
    monkeypatch.setattr(contracts_pipe, "snapshot_days", lambda s, e, step=30: [D("2025-03-10")])

    df = reld.load_relative_daily([RelativeSpec("SR3", "v", 0)], "2025-03-15", "2025-03-21", **_paths(tmp_path))
    assert set(df["ticker"].astype(str)) == {"SR3.v.0"}
    # SRH5 always out-trades SRZ4 in the fixture; the very first day has no prior-day
    # volume yet and falls back to calendar order (the same day-1 behaviour already
    # covered by test_volume_mapping_uses_prior_day_volume_only), every day after picks SRH5.
    by_day = df.set_index(df["timestamp"].dt.strftime("%m-%d"))["contract"]
    assert by_day["03-15"] == "SRZ4" and set(by_day.iloc[1:]) == {"SRH5"}
    assert min_calls, "v.N ranking needs 1-minute data for the volume signal"
    assert list(df.columns) == ["timestamp", "ticker", "settlement_price", "open_interest", "contract"], \
        "1-minute OHLCV columns must never leak into the daily-series output"


def test_repeat_request_costs_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "fetch_definitions", _fake_defs)
    monkeypatch.setattr(api, "fetch_statistics", lambda *a, **k: (calls.append(a), _fake_statistics(*a, **k))[1])
    monkeypatch.setattr(api, "fetch_futures_ohlcv", _fake_1m_bars)
    monkeypatch.setattr(contracts_pipe, "snapshot_days", lambda s, e, step=30: [D("2025-03-10")])
    specs = [RelativeSpec("SR3", "c", 0)]
    paths = _paths(tmp_path)

    reld.load_relative_daily(specs, "2025-03-15", "2025-03-21", **paths)
    n_calls = len(calls)
    reld.load_relative_daily(specs, "2025-03-15", "2025-03-21", **paths)
    assert len(calls) == n_calls
    assert reld.plan_relative_daily_update(
        specs, "2025-03-15", "2025-03-21",
        contracts_file=paths["contracts_file"], defs_coverage_file=paths["defs_coverage_file"],
        daily_coverage_file=paths["daily_coverage_file"], futures_coverage_file=paths["futures_coverage_file"],
    ) == {}


def test_plan_surfaces_1m_gaps_only_for_volume_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "fetch_definitions", _fake_defs)
    monkeypatch.setattr(contracts_pipe, "snapshot_days", lambda s, e, step=30: [D("2025-03-10")])
    paths = _paths(tmp_path)
    contracts_pipe.ensure_contracts(
        __import__("infra.config", fromlist=["FUTURES_ROOTS"]).FUTURES_ROOTS["SR3"],
        "2025-03-15", "2025-03-21", fetch_missing=True,
        contracts_file=paths["contracts_file"], coverage_file=paths["defs_coverage_file"],
    )
    plan_c = reld.plan_relative_daily_update(
        [RelativeSpec("SR3", "c", 0)], "2025-03-15", "2025-03-21",
        contracts_file=paths["contracts_file"], defs_coverage_file=paths["defs_coverage_file"],
        daily_coverage_file=paths["daily_coverage_file"], futures_coverage_file=paths["futures_coverage_file"],
    )
    assert not any(k.startswith("1m:") for k in plan_c)

    plan_v = reld.plan_relative_daily_update(
        [RelativeSpec("SR3", "v", 0)], "2025-03-15", "2025-03-21",
        contracts_file=paths["contracts_file"], defs_coverage_file=paths["defs_coverage_file"],
        daily_coverage_file=paths["daily_coverage_file"], futures_coverage_file=paths["futures_coverage_file"],
    )
    assert any(k.startswith("1m:") for k in plan_v)
