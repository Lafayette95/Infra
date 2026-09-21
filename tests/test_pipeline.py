"""Offline tests: intervals, transforms, parquet layout and the no-duplicate-query rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from infra.api import databento_client as api
from infra.coverage.intervals import find_missing_ranges, merge_intervals, subtract_intervals
from infra.pipeline import futures as fut
from infra.processing import transforms as tf

D = pd.Timestamp


def _raw_bars(start: str, days: int, symbol: str = "SRZ4") -> pd.DataFrame:
    """Fake Databento ohlcv-1m frame: 3 bars/day, tz-aware ts_event index."""
    idx = pd.DatetimeIndex(
        [D(start, tz="UTC") + pd.Timedelta(days=d, minutes=m) for d in range(days) for m in range(3)],
        name="ts_event",
    )
    n = len(idx)
    base = 95.0 + np.arange(n) * 0.005
    return pd.DataFrame({
        "open": base, "high": base + 0.01, "low": base - 0.01, "close": base + 0.005,
        "volume": np.arange(n) + 1, "symbol": symbol, "instrument_id": 1,
    }, index=idx)


# ---------------------------------------------------------------- intervals
def test_subtract_and_merge():
    target = (D("2025-01-01"), D("2025-01-11"))
    covered = [(D("2025-01-03"), D("2025-01-05")), (D("2025-01-05"), D("2025-01-07"))]
    assert merge_intervals(covered) == [(D("2025-01-03"), D("2025-01-07"))]
    assert subtract_intervals(target, covered) == [
        (D("2025-01-01"), D("2025-01-03")), (D("2025-01-07"), D("2025-01-11")),
    ]


def test_missing_ranges_never_reach_today():
    now = D("2025-01-10 15:00")
    gaps = find_missing_ranges((D("2025-01-01"), D("2025-02-01")), [], now=now)
    assert gaps == [(D("2025-01-01"), D("2025-01-10"))]


# -------------------------------------------------------------- transforms
def test_encode_decode_roundtrip_and_dtypes():
    clean = tf.clean_futures_ohlcv(_raw_bars("2025-01-06", 2))
    enc = tf.encode_futures(clean)
    assert all(enc[c].dtype == "int32" for c in ("open", "high", "low", "close", "volume"))
    assert enc["open_interest"].dtype == "Int32"
    dec = tf.decode_futures(enc)
    np.testing.assert_allclose(dec["close"], clean["close"], atol=1e-4)
    assert isinstance(tf.to_multiindex(dec).index, pd.MultiIndex)


# ------------------------------------------------------- parquet + pipeline
def test_layout_flat_partitioned_and_no_repeat_queries(tmp_path, monkeypatch):
    root, cov = tmp_path / "Futures", tmp_path / "cov.parquet"
    calls: list[tuple] = []

    def fake_fetch(dataset, symbol, start, end, **_):
        calls.append((symbol, start, end))
        span = (end - start).days
        return _raw_bars(start.strftime("%Y-%m-%d"), span, symbol)

    monkeypatch.setattr(api, "fetch_futures_ohlcv", fake_fetch)
    kw = dict(root=root, coverage_file=cov)

    # spans a year and a quarter boundary (Dec 30 -> Jan 3)
    df = fut.load_futures(["SRZ4"], "2024-12-30", "2025-01-03", dataset="GLBX.MDP3", **kw)
    assert len(calls) == 1 and len(df) == 12
    assert (root / "year=2024" / "quarter=4").exists() and (root / "year=2025" / "quarter=1").exists()

    # partition dirs are year/quarter only; files are flat with no pandas index column
    for f in root.rglob("*.parquet"):
        schema = pq.read_schema(f)
        assert "__index_level_0__" not in schema.names
        assert pq.ParquetFile(f).metadata.row_group(0).column(0).compression == "ZSTD"

    # identical request again -> zero API calls; superset -> only the new tail
    fut.load_futures(["SRZ4"], "2024-12-30", "2025-01-03", dataset="GLBX.MDP3", **kw)
    assert len(calls) == 1
    df2 = fut.load_futures(["SRZ4"], "2024-12-30", "2025-01-05", dataset="GLBX.MDP3", **kw)
    assert len(calls) == 2 and calls[1][1:] == (D("2025-01-03"), D("2025-01-05"))
    assert len(df2) == 18 and not df2.duplicated(["timestamp", "ticker"]).any()


def test_fetch_missing_false_never_calls_api(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("API must not be called")

    monkeypatch.setattr(api, "fetch_futures_ohlcv", boom)
    df = fut.load_futures(["SRZ4"], "2025-01-01", "2025-01-05", fetch_missing=False,
                          root=tmp_path / "F", coverage_file=tmp_path / "c.parquet")
    assert df.empty


# ---------------------------------------------------------------- guardrails
@pytest.mark.parametrize("bad", ["SR3*", "", "  ", "*", "SR3.c.0", "ZN.v.1"])
def test_wildcards_and_relative_rejected_at_api_boundary(bad):
    with pytest.raises(ValueError):
        api.validate_absolute_symbol(bad)


@pytest.mark.parametrize("ok", ["SRZ4", "ZNH5", "FGBL SI 20250606 PS"])
def test_absolute_symbols_accepted(ok):
    assert api.validate_absolute_symbol(ok) == ok


def test_dataset_required_when_fetching(tmp_path):
    with pytest.raises(ValueError):
        fut.load_futures(["SRZ4"], "2025-01-01", "2025-01-05",
                         root=tmp_path / "F", coverage_file=tmp_path / "c.parquet")
