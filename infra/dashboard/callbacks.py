"""Dash callbacks. Data access goes ONLY through infra.pipeline (never the API directly).

API spend is opt-in: it happens only when the user ticks "Fetch missing" AND presses
Load. Theme/timeframe changes re-read from disk with ``fetch_missing=False``.
"""
from __future__ import annotations

import logging

import pandas as pd
from dash import Dash, Input, Output, State, ctx, html

from infra.api.databento_client import CostLimitExceeded
from infra.dashboard import charts
from infra.dashboard.theme import tokens
from infra.pipeline import futures as fut
from infra.processing.resample import coarsen_to_fit, resample_ohlcv

log = logging.getLogger(__name__)
MAX_BARS = 8_000  # beyond this the timeframe is coarsened automatically


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("root", "className"),
        Output("price-chart", "figure"),
        Output("change-chart", "figure"),
        Output("kpis", "children"),
        Output("status", "children"),
        Input("theme", "value"),
        Input("ticker", "value"),
        Input("timeframe", "value"),
        Input("load-btn", "n_clicks"),
        State("dates", "start_date"),
        State("dates", "end_date"),
        State("fetch", "value"),
    )
    def refresh(theme, ticker, timeframe, _clicks, start, end, fetch):
        theme_class = f"theme-{theme}"
        if not (ticker and start and end):
            return theme_class, charts.empty_figure("Pick a ticker and dates", theme), \
                charts.empty_figure("", theme), [], ""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)  # date picker end is inclusive
        want_fetch = ctx.triggered_id == "load-btn" and "yes" in (fetch or [])

        try:
            df = fut.load_futures([ticker], start_ts, end_ts, fetch_missing=want_fetch)
        except CostLimitExceeded as exc:
            return theme_class, charts.empty_figure("Blocked by cost guardrail", theme), \
                charts.empty_figure("", theme), [], f"⚠ {exc}"
        except Exception as exc:  # surface API/key problems in the UI, keep the app alive
            log.exception("load failed")
            return theme_class, charts.empty_figure("Load failed", theme), \
                charts.empty_figure("", theme), [], f"⚠ {type(exc).__name__}: {exc}"

        bars, used_tf = coarsen_to_fit(df, timeframe, MAX_BARS)
        daily = resample_ohlcv(df, "1D")
        gaps = fut.plan_futures_update(ticker, start_ts, end_ts)

        notes = [f"{len(df):,} 1-min bars on disk"]
        if used_tf != timeframe:
            notes.append(f"showing {used_tf} (too many {timeframe} bars)")
        if gaps:
            notes.append(f"{len(gaps)} date range(s) never queried - tick 'Fetch missing' and press Load")

        kpis = [
            html.Div(className="kpi", children=[html.Span(label), html.Strong(value)])
            for label, value in charts.summary_metrics(bars, daily)
        ]
        return (
            theme_class,
            charts.price_figure(bars, ticker, used_tf, theme),
            charts.daily_change_figure(daily, ticker, theme) if not df.empty
            else charts.empty_figure("", theme),
            kpis,
            " · ".join(notes),
        )
