"""Smoke tests: the app builds and figure builders handle real and empty data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.dashboard import charts
from infra.dashboard.app import create_app
from infra.processing.resample import coarsen_to_fit, resample_ohlcv


def _df(days: int = 5) -> pd.DataFrame:
    ts = pd.date_range("2025-01-06", periods=days * 1440, freq="min")
    px = 95 + np.cumsum(np.random.default_rng(0).normal(0, 0.002, len(ts)))
    return pd.DataFrame({"timestamp": ts, "open": px, "high": px + .01, "low": px - .01,
                         "close": px, "volume": 10})


def test_app_builds_and_layout_renders():
    app = create_app()
    assert app.layout() is not None


def test_figures_light_dark_and_empty():
    df = _df()
    bars, tf_used = coarsen_to_fit(df, "5m", 10_000)
    daily = resample_ohlcv(df, "1D")
    for theme in ("light", "dark"):
        assert len(charts.price_figure(bars, "SR3.c.0", tf_used, theme).data) == 2
        assert len(charts.daily_change_figure(daily, "SR3.c.0", theme).data) == 1
    assert charts.summary_metrics(bars, daily)
    assert charts.price_figure(bars.iloc[0:0], "X", "5m").layout.annotations


def test_coarsen_when_too_many_bars():
    _, used = coarsen_to_fit(_df(5), "1m", 100)
    assert used != "1m"


def test_resample_carries_contract_through_with_one_per_bucket():
    """A roll only ever happens at UTC midnight, so every bucket (up to 1D) stays
    within a single contract - 'first' aggregation never hides a mid-bucket switch."""
    df = _df(days=2).assign(contract=["SRZ4"] * 1440 + ["SRH5"] * 1440)
    for tf in ("1h", "4h", "1D"):
        bars = resample_ohlcv(df, tf)
        assert list(bars["contract"].unique()) == ["SRZ4", "SRH5"]
    # a bucket never straddles the roll: each row has exactly the day's own contract
    hourly = resample_ohlcv(df, "1h")
    assert (hourly.loc[hourly.index.normalize() == df["timestamp"].iloc[0].normalize(), "contract"] == "SRZ4").all()
    assert (hourly.loc[hourly.index.normalize() == df["timestamp"].iloc[-1].normalize(), "contract"] == "SRH5").all()


def test_resample_without_contract_column_is_unaffected():
    bars = resample_ohlcv(_df(1), "1h")
    assert "contract" not in bars.columns


def test_price_figure_hover_shows_contract_when_present():
    df = _df(days=1).assign(contract="SRZ4")
    bars = resample_ohlcv(df, "1h")
    fig = charts.price_figure(bars, "SR3.v.0", "1h")
    candle = fig.data[0]
    assert candle.customdata is not None and (candle.customdata[:, 0] == "SRZ4").all()
    assert "Contract" in candle.hovertemplate


def test_price_figure_hover_omits_contract_when_absent():
    bars = resample_ohlcv(_df(1), "1h")
    fig = charts.price_figure(bars, "SRZ4", "1h")
    assert fig.data[0].customdata is None
    assert "Contract" not in fig.data[0].hovertemplate


def test_price_figure_display_timezone_shifts_x_without_mutating_bars():
    bars = resample_ohlcv(_df(days=1), "1h")
    before_index = bars.index.copy()
    fig = charts.price_figure(bars, "SR3.v.0", "1h", tz="America/New_York")
    assert list(fig.data[0].x)[0] == bars.index[0] - pd.Timedelta(hours=5)  # _df() is in January: EST (UTC-5)
    assert bars.index.equals(before_index)  # caller's frame is untouched
    assert "America/New_York" in fig.layout.title.text


def test_price_figure_defaults_to_utc():
    bars = resample_ohlcv(_df(days=1), "1h")
    fig = charts.price_figure(bars, "SR3.v.0", "1h")
    assert list(fig.data[0].x)[0] == bars.index[0]
    assert "UTC" in fig.layout.title.text


def test_daily_change_figure_always_labelled_utc():
    daily = resample_ohlcv(_df(days=3), "1D")
    fig = charts.daily_change_figure(daily, "SR3.v.0")
    assert "UTC" in fig.layout.title.text
