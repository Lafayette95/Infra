"""Offline tests for the relative-ticker layer (roll rules, definitions, end-to-end)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from infra.api import databento_client as api
from infra.pipeline import contracts as contracts_pipe
from infra.pipeline import relative as rel
from infra.processing.definitions import normalize_futures_definitions
from infra.relative.rolls import calendar_mapping, day_index, volume_mapping
from infra.relative.series import apply_mapping, daily_volume
from infra.relative.symbology import RelativeSpec, parse_relative

D = pd.Timestamp


def _contracts() -> pd.DataFrame:
    """Two quarterlies and one serial (April expiry, must be ignored by the quarterly cycle)."""
    return pd.DataFrame({
        "root": "SR3",
        "ticker": ["SRZ4", "SRF5", "SRH5", "SRM5"],
        "instrument_id": [1, 2, 3, 4],
        "expiry": pd.to_datetime(["2025-03-18", "2025-04-15", "2025-06-17", "2025-09-16"]).astype("datetime64[ms]"),
        "activation": pd.NaT,
    })


# ------------------------------------------------------------------ symbology
def test_parse_relative():
    assert parse_relative("SR3.c.0") == RelativeSpec("SR3", "c", 0)
    assert parse_relative("FGBL.v.2").label == "FGBL.v.2"
    assert parse_relative("SRZ4") is None and parse_relative("SR3*") is None


# ------------------------------------------------------------------ definitions
def test_normalize_drops_spreads_and_keeps_outrights():
    raw = pd.DataFrame({
        "raw_symbol": ["SRZ4", "SRZ4-SRZ5", "SR3:BF Z4-M5-Z5"],
        "instrument_class": ["F", "S", "S"],
        "instrument_id": [1, 2, 3],
        "expiration": pd.to_datetime(["2025-03-18 21:00", "2025-03-18 21:00", "2025-03-18 21:00"], utc=True),
        "activation": pd.to_datetime(["2019-09-27 21:30", "2019-09-27 21:30", "2019-09-27 21:30"], utc=True),
    })
    out = normalize_futures_definitions(raw, "SR3")
    assert out["ticker"].tolist() == ["SRZ4"]
    assert out.loc[0, "expiry"] == D("2025-03-18")


# ------------------------------------------------------------------ roll rules
def test_calendar_mapping_quarterly_cycle_and_roll_day():
    dates = day_index("2025-03-17", "2025-03-21")
    m = calendar_mapping(_contracts(), dates, max_rank=1)
    # serial SRF5 (April) is not on the quarterly cycle; SRZ4 stays front through its expiry day
    assert m[0].tolist() == ["SRZ4", "SRZ4", "SRH5", "SRH5"]
    assert m[1].tolist() == ["SRH5", "SRH5", "SRM5", "SRM5"]


def test_calendar_mapping_offset_rolls_early():
    dates = day_index("2025-03-10", "2025-03-13")
    m = calendar_mapping(_contracts(), dates, max_rank=0, roll_offset_days=7)
    assert m[0].tolist() == ["SRZ4", "SRZ4", "SRH5"]  # Mar 12 + 7d = Mar 19 > Mar 18 expiry


def test_calendar_mapping_runs_out_of_contracts():
    m = calendar_mapping(_contracts(), day_index("2025-09-17", "2025-09-18"), max_rank=0)
    assert m[0].tolist() == [None]


def test_volume_mapping_uses_prior_day_volume_only():
    dates = day_index("2025-03-10", "2025-03-14")
    vol = pd.DataFrame(
        {"SRZ4": [100, 100, 10, 10], "SRH5": [10, 10, 500, 500]},
        index=pd.to_datetime(["2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13"]).astype("datetime64[ms]"),
    )
    m = volume_mapping(vol, _contracts(), dates, max_rank=0)
    # H5 first out-trades Z4 on Mar 12; it becomes v.0 only on Mar 13 (prior-day volume)
    assert m[0].tolist() == ["SRZ4", "SRZ4", "SRZ4", "SRH5"]


# ------------------------------------------------------------------ apply mapping
def test_apply_mapping_selects_contract_per_day():
    ts = pd.to_datetime(["2025-03-17 10:00", "2025-03-17 10:00", "2025-03-19 10:00", "2025-03-19 10:00"]).astype("datetime64[ms]")
    bars = pd.DataFrame({"timestamp": ts, "ticker": ["SRZ4", "SRH5", "SRZ4", "SRH5"],
                         "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": [1, 2, 3, 4]})
    m = calendar_mapping(_contracts(), day_index("2025-03-17", "2025-03-21"), max_rank=0)
    out = apply_mapping(bars, m, 0, "SR3.c.0")
    assert out["contract"].tolist() == ["SRZ4", "SRH5"] and set(out["ticker"]) == {"SR3.c.0"}
    assert daily_volume(bars).loc[D("2025-03-17"), "SRZ4"] == 1


# ------------------------------------------------------------------ end to end
def test_relative_pipeline_fetches_only_needed_contracts_once(tmp_path, monkeypatch):
    paths = dict(
        contracts_file=tmp_path / "contracts.parquet",
        defs_coverage_file=tmp_path / "defs_cov.parquet",
        coverage_file=tmp_path / "cov.parquet",
        root_dir=tmp_path / "Futures",
    )
    def_calls, bar_calls = [], []

    def fake_defs(dataset, parents, day, **_):
        def_calls.append(day)
        c = _contracts()
        return pd.DataFrame({
            "raw_symbol": c["ticker"], "instrument_class": "F", "instrument_id": c["instrument_id"],
            "expiration": pd.to_datetime(c["expiry"], utc=True), "activation": pd.NaT,
        })

    def fake_bars(dataset, symbol, start, end, **_):
        bar_calls.append((symbol, start, end))
        idx = pd.DatetimeIndex([start.tz_localize("UTC") + pd.Timedelta(days=d, hours=10) for d in range((end - start).days)], name="ts_event")
        base = np.full(len(idx), 95.0)
        return pd.DataFrame({"open": base, "high": base, "low": base, "close": base,
                             "volume": 1, "symbol": symbol, "instrument_id": 1}, index=idx)

    monkeypatch.setattr(api, "fetch_definitions", fake_defs)
    monkeypatch.setattr(api, "fetch_futures_ohlcv", fake_bars)
    monkeypatch.setattr(contracts_pipe, "snapshot_days", lambda s, e, step=30: [D("2025-03-10")])
    specs = [RelativeSpec("SR3", "c", 0)]

    df = rel.load_relative_futures(specs, "2025-03-15", "2025-03-21", **paths)
    assert set(df["ticker"].astype(str)) == {"SR3.c.0"}
    by_day = df.assign(day=df["timestamp"].dt.strftime("%m-%d")).set_index("day")["contract"].to_dict()
    assert by_day["03-17"] == "SRZ4" and by_day["03-19"] == "SRH5"
    assert {c[0] for c in bar_calls} == {"SRZ4", "SRH5"}  # serial and SRM5 never requested
    n_defs, n_bars = len(def_calls), len(bar_calls)

    rel.load_relative_futures(specs, "2025-03-15", "2025-03-21", **paths)  # identical -> no spend
    assert (len(def_calls), len(bar_calls)) == (n_defs, n_bars)
    assert rel.plan_relative_update(specs, "2025-03-15", "2025-03-21", **{k: v for k, v in paths.items() if k != "root_dir"}) == {}


def test_unknown_root_fails_clearly():
    with pytest.raises(KeyError):
        rel.group_by_root([RelativeSpec("XX", "c", 0)])
