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
