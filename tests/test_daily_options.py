"""Offline tests: daily option settlement/open-interest transform + pipeline."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.api import databento_client as api
from infra.pipeline import daily_options as opt
from infra.processing.statistics import (
    DAILY_OPTIONS_COLUMNS,
    clean_daily_option_statistics,
    decode_daily_options,
    empty_daily_options,
    encode_daily_options,
)

D = pd.Timestamp


def _definitions() -> pd.DataFrame:
    return pd.DataFrame({
        "instrument_id": [101, 102],
        "underlying": ["SR3H5", "SR3H5"],
        "option_type": ["C", "P"],
        "strike": [96.0, 96.0],
        "expiry": pd.to_datetime(["2025-06-17", "2025-06-17"]).astype("datetime64[ms]"),
    })


def _raw_rows(rows: list[tuple]) -> pd.DataFrame:
    """rows: (ts_recv, ts_ref, instrument_id, stat_type, price, quantity)"""
    ts_recv, ts_ref, iid, stype, price, qty = zip(*rows)
    return pd.DataFrame({
        "ts_recv": pd.to_datetime(ts_recv, utc=True),
        "ts_ref": pd.to_datetime(ts_ref, utc=True),
        "instrument_id": iid,
        "stat_type": stype,
        "price": price,
        "quantity": qty,
    })


# ---------------------------------------------------------- clean_daily_option_statistics
def test_pivots_two_instruments_and_joins_definitions():
    raw = _raw_rows([
        ("2025-03-10 19:02", "2025-03-10", 101, 3, 3.25, None),
        ("2025-03-10 13:28", "2025-03-07", 101, 9, None, 5115),
        ("2025-03-10 19:03", "2025-03-10", 102, 3, 1.75, None),
    ])
    out = clean_daily_option_statistics(raw, _definitions(), "GLBX.MDP3")
    call = out[(out.underlying == "SR3H5") & (out.option_type == "C") & (out.timestamp == D("2025-03-10"))]
    assert call["settlement_price"].iloc[0] == 3.25
    call_oi = out[(out.option_type == "C") & (out.timestamp == D("2025-03-07"))]
    assert call_oi["open_interest"].iloc[0] == 5115  # OI correctly on the PRIOR trading day
    put = out[(out.option_type == "P") & (out.timestamp == D("2025-03-10"))]
    assert put["strike"].iloc[0] == 96.0 and put["expiry"].iloc[0] == D("2025-06-17")


def test_settlement_last_wins_per_instrument():
    raw = _raw_rows([
        ("2025-03-10 19:00", "2025-03-10", 101, 3, 3.00, None),
        ("2025-03-10 21:00", "2025-03-10", 101, 3, 3.25, None),  # final settlement
    ])
    out = clean_daily_option_statistics(raw, _definitions(), "GLBX.MDP3")
    assert out["settlement_price"].iloc[0] == 3.25


def test_unknown_instrument_id_is_dropped_not_kept_unjoined():
    raw = _raw_rows([("2025-03-10 19:00", "2025-03-10", 999, 3, 3.25, None)])
    out = clean_daily_option_statistics(raw, _definitions(), "GLBX.MDP3")
    assert out.empty


def test_empty_input():
    out = clean_daily_option_statistics(pd.DataFrame(), _definitions(), "GLBX.MDP3")
    assert out.empty and list(out.columns) == DAILY_OPTIONS_COLUMNS


# --------------------------------------------------------------------- encode / decode
def test_encode_decode_roundtrip():
    clean = clean_daily_option_statistics(
        _raw_rows([("2025-03-10 19:00", "2025-03-10", 101, 3, 3.25, None),
                  ("2025-03-10 13:28", "2025-03-07", 101, 9, None, 5115)]),
        _definitions(), "GLBX.MDP3",
    )
    enc = encode_daily_options(clean)
    assert str(enc["strike"].dtype) == "int32" and enc.loc[0, "strike"] == 960000
    assert str(enc["settlement_price"].dtype) == "Int32"
    dec = decode_daily_options(enc)
    assert abs(dec.loc[dec.timestamp == D("2025-03-10"), "settlement_price"].iloc[0] - 3.25) < 1e-9
    assert dec["expiry"].iloc[0] == D("2025-06-17")
    assert dec["underlying"].dtype.name == "category"


def test_encode_rejects_sub_precision_strike():
    clean = pd.DataFrame({
        "timestamp": [D("2025-03-10")], "underlying": ["SR3H5"], "option_type": ["C"],
        "strike": [0.00001], "expiry": [D("2025-06-17")],
        "settlement_price": [1.0], "open_interest": pd.array([None], dtype="Int64"),
    })
    with pytest.raises(ValueError):
        encode_daily_options(clean)


def test_empty_daily_options_shape():
    assert empty_daily_options().empty and list(empty_daily_options().columns) == DAILY_OPTIONS_COLUMNS


# ------------------------------------------------------------------------- API batching
def test_fetch_statistics_by_instrument_ids_batches_over_the_symbol_cap(monkeypatch):
    monkeypatch.setattr(api, "MAX_SYMBOLS_PER_REQUEST", 2)
    calls = []

    def fake_get_range(dataset, schema, symbols, start, end, stype_in, max_cost_usd, client):
        calls.append(list(symbols))
        return pd.DataFrame({"stat_type": [3] * len(symbols), "instrument_id": symbols})

    monkeypatch.setattr(api, "_get_range", fake_get_range)
    out = api.fetch_statistics_by_instrument_ids(
        "GLBX.MDP3", [1, 2, 3, 4, 5], D("2025-03-10"), D("2025-03-11"), client=object()
    )
    assert [len(c) for c in calls] == [2, 2, 1]  # 5 ids batched at size 2
    assert len(out) == 5


def test_fetch_statistics_by_instrument_ids_rejects_empty():
    with pytest.raises(ValueError):
        api.fetch_statistics_by_instrument_ids("GLBX.MDP3", [], D("2025-03-10"), D("2025-03-11"))


# --------------------------------------------------------------------------- pipeline
def test_load_daily_options_end_to_end_and_cache_first(tmp_path, monkeypatch):
    calls = []

    def fake_defs(dataset, parents, day, **_):
        d = _definitions()
        return pd.DataFrame({
            "raw_symbol": ["SR3H5 C9600", "SR3H5 P9600"], "instrument_class": d["option_type"],
            "instrument_id": d["instrument_id"], "strike_price": d["strike"],
            "expiration": pd.to_datetime(d["expiry"], utc=True), "underlying": d["underlying"],
        })

    def fake_stats(dataset, symbols, start, end, **_):
        calls.append(list(symbols))
        rows = [(str(start + pd.Timedelta(hours=19)), str(start), iid, 3, 3.25, None) for iid in symbols]
        return _raw_rows(rows)

    monkeypatch.setattr(api, "fetch_definitions", fake_defs)
    monkeypatch.setattr(api, "fetch_statistics_by_instrument_ids", fake_stats)

    paths = dict(
        definitions_directory=tmp_path / "defs", root=tmp_path / "Daily" / "Options",
        coverage_file=tmp_path / "cov.parquet",
    )
    df = opt.load_daily_options(
        "SR3.OPT", "2025-03-12", "2025-03-10", "2025-03-11", max_cost_usd=5.0, **paths,
    )
    assert len(df) == 2 and set(df["option_type"]) == {"C", "P"}
    assert len(calls) == 1

    df2 = opt.load_daily_options("SR3.OPT", "2025-03-12", "2025-03-10", "2025-03-11", **paths)
    assert len(calls) == 1  # identical request -> no repeat API call
    assert len(df2) == 2


def test_load_daily_options_filters_locally(tmp_path, monkeypatch):
    def fake_defs(dataset, parents, day, **_):
        d = _definitions()
        return pd.DataFrame({
            "raw_symbol": ["SR3H5 C9600", "SR3H5 P9600"], "instrument_class": d["option_type"],
            "instrument_id": d["instrument_id"], "strike_price": d["strike"],
            "expiration": pd.to_datetime(d["expiry"], utc=True), "underlying": d["underlying"],
        })

    monkeypatch.setattr(api, "fetch_definitions", fake_defs)
    df = opt.load_daily_options(
        "SR3.OPT", "2025-03-12", "2025-03-10", "2025-03-11",
        option_types=["C"], fetch_missing=False,
        definitions_directory=tmp_path / "defs", root=tmp_path / "Daily" / "Options",
        coverage_file=tmp_path / "cov.parquet",
    )
    assert df.empty  # nothing on disk yet, but the filter itself didn't error
