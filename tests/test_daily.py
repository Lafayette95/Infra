"""Offline tests: daily settlement/open-interest transform + pipeline (no API, no cost)."""
from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import pytest

from infra.api import databento_client as api
from infra.pipeline import daily as dl
from infra.processing.statistics import (
    DAILY_COLUMNS,
    clean_daily_statistics,
    decode_daily,
    empty_daily,
    encode_daily,
    resolve_trading_day,
)

D = pd.Timestamp


def _raw_stat_rows(ts_recv, ts_ref, stat_type, price=None, quantity=None) -> pd.DataFrame:
    n = len(ts_recv)
    return pd.DataFrame({
        "ts_recv": pd.to_datetime(ts_recv, utc=True),
        "ts_ref": pd.to_datetime(ts_ref, utc=True),
        "stat_type": stat_type,
        "price": price if price is not None else [None] * n,
        "quantity": quantity if quantity is not None else [None] * n,
    })


# ---------------------------------------------------------------- resolve_trading_day
def test_prefers_ts_ref_over_ts_recv_day():
    # A CME open-interest update published early Mar 12 but referencing Mar 11's close -
    # the exact quirk that would silently mis-bucket the data if ts_recv were used instead.
    raw = _raw_stat_rows(["2025-03-12 01:50:06"], ["2025-03-11"], [9], quantity=[1056906])
    out = resolve_trading_day(raw, "GLBX.MDP3")
    assert out.iloc[0] == D("2025-03-11")


def test_falls_back_to_trading_day_when_ts_ref_is_null():
    # Eurex/ICE settlement commonly has no ts_ref; trading_day(ts_recv) is used instead,
    # which is exact for these venues since neither session crosses midnight.
    raw = _raw_stat_rows(["2025-06-05 15:21:59"], [pd.NaT], [3], price=[124.84])
    out = resolve_trading_day(raw, "XEUR.EOBI")
    assert out.iloc[0] == D("2025-06-05")


# ---------------------------------------------------------------- clean_daily_statistics
def test_settlement_takes_the_last_update_not_the_first():
    raw = _raw_stat_rows(
        ["2025-03-12 19:02:34", "2025-03-12 21:38:00", "2025-03-12 22:57:42"],
        ["2025-03-12"] * 3, [3, 3, 3], price=[95.63, 95.6325, 95.6325],
    )
    out = clean_daily_statistics(raw, "SR3Z4", "GLBX.MDP3")
    assert out.loc[0, "settlement_price"] == 95.6325


def test_oi_prior_day_and_settlement_same_day_produce_separate_rows():
    raw = pd.concat([
        _raw_stat_rows(["2025-03-12 19:02:34"], ["2025-03-12"], [3], price=[95.6325]),
        _raw_stat_rows(["2025-03-12 01:50:06"], ["2025-03-11"], [9], quantity=[1056906]),
    ], ignore_index=True)
    out = clean_daily_statistics(raw, "SR3Z4", "GLBX.MDP3").set_index("timestamp")
    assert out.loc[D("2025-03-11"), "open_interest"] == 1056906
    assert pd.isna(out.loc[D("2025-03-11"), "settlement_price"])
    assert out.loc[D("2025-03-12"), "settlement_price"] == 95.6325
    assert pd.isna(out.loc[D("2025-03-12"), "open_interest"])


def test_ignores_other_stat_types():
    raw = _raw_stat_rows(["2025-03-12 08:29:58"], ["2025-03-12"], [4], price=[95.63])  # session low
    out = clean_daily_statistics(raw, "SR3Z4", "GLBX.MDP3")
    assert out.empty


def test_empty_input():
    out = clean_daily_statistics(pd.DataFrame(), "SR3Z4", "GLBX.MDP3")
    assert out.empty and list(out.columns) == DAILY_COLUMNS


# ---------------------------------------------------------------------- encode / decode
def test_encode_decode_roundtrip_and_nullable_dtypes():
    clean = clean_daily_statistics(
        pd.concat([
            _raw_stat_rows(["2025-03-12 19:02:34"], ["2025-03-12"], [3], price=[95.6325]),
            _raw_stat_rows(["2025-03-12 01:50:06"], ["2025-03-11"], [9], quantity=[1056906]),
        ], ignore_index=True),
        "SR3Z4", "GLBX.MDP3",
    )
    enc = encode_daily(clean)
    assert str(enc["settlement_price"].dtype) == "Int32" and str(enc["open_interest"].dtype) == "Int32"
    assert enc.loc[enc["timestamp"] == D("2025-03-12"), "settlement_price"].iloc[0] == 956325
    dec = decode_daily(enc)
    assert abs(dec.loc[dec["timestamp"] == D("2025-03-12"), "settlement_price"].iloc[0] - 95.6325) < 1e-9
    assert pd.isna(dec.loc[dec["timestamp"] == D("2025-03-11"), "settlement_price"].iloc[0])
    assert dec["ticker"].dtype.name == "category"


def test_empty_daily_shape():
    assert empty_daily().empty and list(empty_daily().columns) == DAILY_COLUMNS


# --------------------------------------------------------------------------- pipeline
def test_layout_flat_partitioned_and_no_repeat_queries(tmp_path, monkeypatch):
    root, cov = tmp_path / "Daily", tmp_path / "cov.parquet"
    calls: list[tuple] = []

    def fake_fetch(dataset, symbol, start, end, **_):
        calls.append((symbol, start, end))
        days = pd.date_range(start, end, freq="D", inclusive="left")
        return pd.concat([
            _raw_stat_rows([d + pd.Timedelta(hours=19) for d in days], list(days), [3] * len(days),
                          price=[95.0 + i * 0.01 for i in range(len(days))]),
            _raw_stat_rows([d + pd.Timedelta(hours=1, minutes=50) for d in days], list(days), [9] * len(days),
                          quantity=[1000 + i for i in range(len(days))]),
        ], ignore_index=True)

    monkeypatch.setattr(api, "fetch_statistics", fake_fetch)
    kw = dict(root=root, coverage_file=cov)

    df = dl.load_daily(["SR3Z4"], "2024-12-30", "2025-01-03", dataset="GLBX.MDP3", **kw)
    assert len(calls) == 1 and len(df) == 4  # Dec30,31,Jan1,2 (end exclusive)
    assert (root / "year=2024" / "quarter=4").exists() and (root / "year=2025" / "quarter=1").exists()

    for f in root.rglob("*.parquet"):
        schema = pq.read_schema(f)
        assert "__index_level_0__" not in schema.names
        assert pq.ParquetFile(f).metadata.row_group(0).column(0).compression == "ZSTD"

    dl.load_daily(["SR3Z4"], "2024-12-30", "2025-01-03", dataset="GLBX.MDP3", **kw)
    assert len(calls) == 1  # identical request -> no API call

    df2 = dl.load_daily(["SR3Z4"], "2024-12-30", "2025-01-05", dataset="GLBX.MDP3", **kw)
    assert len(calls) == 2 and calls[1][1:] == (D("2025-01-03"), D("2025-01-05"))
    assert len(df2) == 6 and not df2.duplicated(["timestamp", "ticker"]).any()


def test_fetch_missing_false_never_calls_api(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("API must not be called")

    monkeypatch.setattr(api, "fetch_statistics", boom)
    df = dl.load_daily(["SR3Z4"], "2025-01-01", "2025-01-05", fetch_missing=False,
                       root=tmp_path / "D", coverage_file=tmp_path / "c.parquet")
    assert df.empty


def test_dataset_required_when_fetching(tmp_path):
    with pytest.raises(ValueError):
        dl.load_daily(["SR3Z4"], "2025-01-01", "2025-01-05",
                      root=tmp_path / "D", coverage_file=tmp_path / "c.parquet")


# ------------------------------------------------------------ force_refetch / prune
def _fake_stats_fetch(calls, price_fn):
    def fake_fetch(dataset, symbol, start, end, **_):
        calls.append((symbol, start, end))
        days = pd.date_range(start, end, freq="D", inclusive="left")
        return _raw_stat_rows([d + pd.Timedelta(hours=19) for d in days], list(days), [3] * len(days),
                              price=[price_fn(d) for d in days])
    return fake_fetch


def test_force_refetch_requeries_covered_range_and_overwrites_revisions(tmp_path, monkeypatch):
    root, cov = tmp_path / "Daily", tmp_path / "cov.parquet"
    calls: list[tuple] = []
    monkeypatch.setattr(api, "fetch_statistics", _fake_stats_fetch(calls, lambda d: 95.0))
    kw = dict(dataset="GLBX.MDP3", root=root, coverage_file=cov)
    dl.load_daily(["SR3Z4"], "2025-01-06", "2025-01-09", **kw)
    assert len(calls) == 1

    # Upstream revises Jan 7. A normal call never sees it (range covered, Rule 2.1)...
    monkeypatch.setattr(api, "fetch_statistics", _fake_stats_fetch(
        calls, lambda d: 95.5 if d == D("2025-01-07") else 95.0))
    df = dl.load_daily(["SR3Z4"], "2025-01-06", "2025-01-09", **kw)
    assert len(calls) == 1 and (df["settlement_price"] == 95.0).all()

    # ...force_refetch re-queries the whole window and the revised value wins.
    df = dl.load_daily(["SR3Z4"], "2025-01-06", "2025-01-09", force_refetch=True, **kw).set_index("timestamp")
    assert len(calls) == 2 and calls[1][1:] == (D("2025-01-06"), D("2025-01-09"))
    assert df.loc[D("2025-01-07"), "settlement_price"] == 95.5
    assert df.loc[D("2025-01-06"), "settlement_price"] == 95.0


def test_force_refetch_defaults_off():
    import inspect
    assert inspect.signature(dl.load_daily).parameters["force_refetch"].default is False
    assert inspect.signature(dl.plan_daily_update).parameters["force_refetch"].default is False


def test_prune_replaces_one_tickers_whole_history_and_its_coverage(tmp_path, monkeypatch):
    from infra.storage import coverage_store
    root, cov = tmp_path / "Daily", tmp_path / "cov.parquet"
    calls: list[tuple] = []
    monkeypatch.setattr(api, "fetch_statistics", _fake_stats_fetch(calls, lambda d: 95.0))
    kw = dict(dataset="GLBX.MDP3", root=root, coverage_file=cov)
    # two quarters of history for SR3Z4, plus another ticker that must be untouched
    dl.load_daily(["SR3Z4", "SR3H5"], "2024-12-30", "2025-01-03", **kw)

    dl.load_daily(["SR3Z4"], "2025-01-02", "2025-01-03", force_refetch=True, prune=True, **kw)
    out = dl.read_daily_from_disk(["SR3Z4", "SR3H5"], D("2024-01-01"), D("2026-01-01"), root=root)
    z4 = out[out["ticker"] == "SR3Z4"]
    assert list(z4["timestamp"]) == [D("2025-01-02")]  # 2024 partition rows gone too
    assert len(out[out["ticker"] == "SR3H5"]) == 4       # other ticker untouched
    assert coverage_store.read_covered(cov, "SR3Z4") == [(D("2025-01-02"), D("2025-01-03"))]
    assert coverage_store.read_covered(cov, "SR3H5") == [(D("2024-12-30"), D("2025-01-03"))]


# ------------------------------------------------------------ availability bounding
def test_old_ranges_never_look_up_availability(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no metadata lookup for historical ranges")
    monkeypatch.setattr(api, "available_end", boom)
    now = D("2026-09-28 23:00")
    assert dl.bounded_by_availability("GLBX.MDP3", D("2026-09-01"), now=now) == (D("2026-09-01"), D("2026-09-01"))


def test_range_past_advertised_availability_is_clamped_and_covers_only_complete_days(monkeypatch):
    monkeypatch.setattr(api, "available_end", lambda *a, **k: D("2026-09-28 11:29:43"))
    now = D("2026-09-28 23:00")
    # "today inclusive" -> clamped to the advertised end; coverage stops at the start of that day
    assert dl.bounded_by_availability("GLBX.MDP3", D("2026-09-29"), now=now) == (
        D("2026-09-28 11:29:43"), D("2026-09-28"))
    # a recent range already inside availability is untouched
    assert dl.bounded_by_availability("GLBX.MDP3", D("2026-09-28"), now=now) == (D("2026-09-28"), D("2026-09-28"))


def test_fetch_past_availability_never_marks_the_partial_day_covered(tmp_path, monkeypatch):
    from infra.storage import coverage_store
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    monkeypatch.setattr(api, "available_end", lambda *a, **k: today + pd.Timedelta(hours=11))
    seen = []

    def fake_fetch(dataset, symbol, start, end, **_):
        seen.append(end)
        return _raw_stat_rows([start + pd.Timedelta(hours=19)], [start], [3], price=[95.0])

    monkeypatch.setattr(api, "fetch_statistics", fake_fetch)
    cov = tmp_path / "cov.parquet"
    dl.fetch_and_store_daily("SR3Z6", [(today - pd.Timedelta(days=2), today + pd.Timedelta(days=1))],
                             dataset="GLBX.MDP3", root=tmp_path / "D", coverage_file=cov)
    assert seen == [today + pd.Timedelta(hours=11)]  # clamped, never an end Databento rejects
    assert coverage_store.read_covered(cov, "SR3Z6") == [(today - pd.Timedelta(days=2), today)]


def test_range_entirely_beyond_availability_is_skipped_and_not_covered(tmp_path, monkeypatch):
    from infra.storage import coverage_store
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    monkeypatch.setattr(api, "available_end", lambda *a, **k: today - pd.Timedelta(hours=2))

    def boom(*a, **k):
        raise AssertionError("must not query a range with nothing available yet")

    monkeypatch.setattr(api, "fetch_statistics", boom)
    cov = tmp_path / "cov.parquet"
    rows = dl.fetch_and_store_daily("SR3Z6", [(today, today + pd.Timedelta(days=1))], dataset="GLBX.MDP3",
                                    root=tmp_path / "D", coverage_file=cov)
    assert rows == 0 and coverage_store.read_covered(cov, "SR3Z6") == []
