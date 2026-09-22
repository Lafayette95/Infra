"""Dash callbacks. Data access goes ONLY through infra.pipeline (never the API directly).

API spend is opt-in: it happens only when the user ticks "Fetch missing" AND presses
Load. Theme/timeframe changes re-read from disk with ``fetch_missing=False``.
"""
from __future__ import annotations

import logging

import pandas as pd
from dash import Dash, Input, Output, State, ctx, html

from infra.api.databento_client import CostLimitExceeded
from infra.config import FUTURES_ROOTS, TRADING_HOURS
from infra.dashboard import charts
from infra.dashboard.selectors import default_expiry, expiry_options
from infra.dashboard.theme import tokens
from infra.pipeline.series import load_series, plan_series
from infra.processing.resample import coarsen_to_fit, resample_ohlcv

log = logging.getLogger(__name__)
MAX_BARS = 8_000  # beyond this the timeframe is coarsened automatically


def register_callbacks(app: Dash) -> None:
    @app.callback(
        Output("expiry", "options"),
        Output("expiry", "value"),
        Input("ticker-root", "value"),
        State("expiry", "value"),
    )
    def update_expiry(root, current):
        """Rebuild the Expiry list for the chosen root, keeping the relative choice if any."""
        return expiry_options(root), default_expiry(root, current)

    @app.callback(
        Output("price-chart", "figure"),
        Output("change-chart", "figure"),
        Output("kpis", "children"),
        Output("status", "children"),
        Input("theme", "value"),
        Input("expiry", "value"),
        Input("timeframe", "value"),
        Input("display-tz", "value"),
        Input("load-btn", "n_clicks"),
        State("ticker-root", "value"),
        State("dates", "start_date"),
        State("dates", "end_date"),
        State("fetch", "value"),
    )
    def refresh(theme, ticker, timeframe, tz, _clicks, ticker_root, start, end, fetch):
        if not (ticker and start and end):
            return charts.empty_figure("Pick a root, expiry and dates", theme), \
                charts.empty_figure("", theme), [], ""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)  # date picker end is inclusive
        want_fetch = ctx.triggered_id == "load-btn" and "yes" in (fetch or [])

        try:
            df = load_series([ticker], start_ts, end_ts, fetch_missing=want_fetch)
        except CostLimitExceeded as exc:
            return charts.empty_figure("Blocked by cost guardrail", theme), \
                charts.empty_figure("", theme), [], f"⚠ {exc}"
        except Exception as exc:  # surface API/key problems in the UI, keep the app alive
            log.exception("load failed")
            return charts.empty_figure("Load failed", theme), \
                charts.empty_figure("", theme), [], f"⚠ {type(exc).__name__}: {exc}"

        dataset = FUTURES_ROOTS[ticker_root].dataset
        bars, used_tf = coarsen_to_fit(df, timeframe, MAX_BARS, dataset=dataset)
        daily = resample_ohlcv(df, "1D", dataset=dataset)
        gaps = plan_series([ticker], start_ts, end_ts)

        notes = [f"{len(df):,} 1-min bars on disk"]
        if not df.empty and ticker != df["contract"].iloc[0]:
            notes.append(f"{df['contract'].nunique()} contract(s): {', '.join(sorted(df['contract'].unique()))}")
        if used_tf != timeframe:
            notes.append(f"showing {used_tf} (too many {timeframe} bars)")
        if gaps:
            notes.append(f"{len(gaps)} contract/definition set(s) missing on disk - tick 'Fetch missing' and press Load")

        kpis = [
            html.Div(className="kpi", children=[html.Span(label), html.Strong(value)])
            for label, value in charts.summary_metrics(bars, daily)
        ]
        return (
            charts.price_figure(bars, ticker, used_tf, theme, tz=tz),
            charts.daily_change_figure(daily, ticker, theme, exchange=TRADING_HOURS[dataset].exchange)
            if not df.empty else charts.empty_figure("", theme),
            kpis,
            " · ".join(notes),
        )
