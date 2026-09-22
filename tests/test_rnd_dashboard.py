"""Offline tests: RND dashboard page (selectors, chart builder, app structure)."""
from __future__ import annotations

import pandas as pd
import pytest

from infra.dashboard import rnd_charts
from infra.dashboard.app import create_app
from infra.dashboard.rnd_selectors import day_options, expiry_options, underlying_options
from infra.processing import statistics as stats
from infra.storage import parquet_store

D = pd.Timestamp


def _write_chain(root, underlying: str, expiry: str, days: list[str]) -> None:
    """A tiny but valid option chain (needs both C and P, >=4 strikes for the smile)."""
    rows = []
    strikes = [95.0, 95.5, 96.0, 96.5, 97.0]
    for day in days:
        for k in strikes:
            call = max(96.2 - k, 0) + 0.3
            put = max(k - 96.2, 0) + 0.3
            rows.append((day, underlying, "C", k, expiry, call, None))
            rows.append((day, underlying, "P", k, expiry, put, None))
    df = pd.DataFrame(rows, columns=["timestamp", "underlying", "option_type", "strike", "expiry",
                                     "settlement_price", "open_interest"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    df["expiry"] = pd.to_datetime(df["expiry"]).astype("datetime64[ms]")
    df["open_interest"] = pd.array([None] * len(df), dtype="Int64")
    parquet_store.write_partitioned(stats.encode_daily_options(df), root, stats.DAILY_OPTIONS_KEYS)


# ------------------------------------------------------------------------- selectors
def test_underlying_expiry_day_options_roundtrip(tmp_path):
    root = tmp_path / "Options"
    _write_chain(root, "SR3Z9", "2029-12-14", ["2029-11-01", "2029-11-02", "2029-12-14"])

    assert underlying_options(root) == ["SR3Z9"]
    assert expiry_options("SR3Z9", root=root) == ["2029-12-14"]
    # the expiry day itself is excluded (T=0, extract_rnd would reject it)
    assert day_options("SR3Z9", "2029-12-14", root=root) == ["2029-11-01", "2029-11-02"]


def test_expiry_decodes_correctly_not_epoch_1970(tmp_path):
    """Regression test: the stored `expiry` column is an int32 day-epoch, not a
    datetime - reading it without infra.processing.statistics.decode_expiry_column
    gives nonsense near 1970-01-01 (pd.to_datetime reads a bare int as nanoseconds)."""
    root = tmp_path / "Options"
    _write_chain(root, "SR3H8", "2028-06-15", ["2028-05-01"])
    out = expiry_options("SR3H8", root=root)
    assert out == ["2028-06-15"]
    assert "1970" not in out[0]


def test_empty_when_nothing_cached(tmp_path):
    root = tmp_path / "Options"
    assert underlying_options(root) == []
    assert expiry_options("SR3Z9", root=root) == []
    assert day_options("SR3Z9", "2029-12-14", root=root) == []


# ---------------------------------------------------------------------------- charts
def _fake_density() -> pd.DataFrame:
    x = [95.5, 96.0, 96.2, 96.4, 97.0]
    return pd.DataFrame({"strike": x, "density": [0.1, 1.0, 2.5, 1.0, 0.1]})


def _meta() -> dict:
    return {"underlying": "SR3U6", "expiry": "2026-09-11", "valuation_day": "2026-08-27",
            "forward": 96.2, "p5": 95.8, "p25": 96.05, "p75": 96.35, "p95": 96.6}


def test_density_figure_builds_expected_traces():
    fig = rnd_charts.density_figure(_fake_density(), _meta(), "light")
    assert len(fig.data) == 3  # 90% band, 50% band, density line
    assert fig.data[2].mode == "lines"
    assert "SR3U6" in fig.layout.title.text
    assert "American" not in fig.layout.title.text  # caveat lives in the page note, not the chart


def test_density_figure_empty_input():
    fig = rnd_charts.density_figure(pd.DataFrame(), {}, "light")
    assert fig.layout.annotations  # empty_figure's placeholder message


def test_density_figure_no_weekend_rangebreak():
    """The x-axis is strike, not a date - the weekend rangebreak from charts._style
    must be disabled here (date_axis=False)."""
    fig = rnd_charts.density_figure(_fake_density(), _meta(), "light")
    assert fig.layout.xaxis.rangebreaks in (None, ())


# ------------------------------------------------------------------------- app shell
def test_app_builds_with_both_pages_registered():
    import dash
    app = create_app()
    layout = app.layout()
    assert layout is not None
    paths = {p["relative_path"] for p in dash.page_registry.values()}
    assert paths == {"/", "/rnd"}


def test_create_app_is_idempotent_across_calls():
    """create_app() may be called more than once in a process (e.g. across tests) -
    must not raise on re-registration."""
    app1 = create_app()
    app2 = create_app()
    assert app1.layout() is not None
    assert app2.layout() is not None
