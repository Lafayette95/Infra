"""Static Dash layout (component tree only; behaviour lives in callbacks.py)."""
from __future__ import annotations

import pandas as pd
from dash import dcc, html

from infra.config import FUTURES_DIR, FUTURES_UNIVERSE
from infra.processing.resample import TIMEFRAMES
from infra.storage import parquet_store


def available_tickers() -> list[str]:
    """Configured universe plus any ticker already present on disk."""
    return sorted(set(FUTURES_UNIVERSE) | set(parquet_store.list_values(FUTURES_DIR, "ticker")))


def build_layout() -> html.Div:
    tickers = available_tickers()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return html.Div(id="root", className="theme-light", children=[
        html.Header(className="bar", children=[
            html.H1("STIR & Rates · 1-minute bars"),
            dcc.RadioItems(
                id="theme", value="light", inline=True, className="theme-toggle",
                options=[{"label": "Light", "value": "light"}, {"label": "Dark", "value": "dark"}],
            ),
        ]),
        html.Section(className="controls", children=[
            html.Label(["Ticker", dcc.Dropdown(
                id="ticker", options=tickers, value=tickers[0], clearable=False)]),
            html.Label(["Date range (UTC)", dcc.DatePickerRange(
                id="dates", start_date=(today - pd.Timedelta(days=30)).date(),
                end_date=today.date(), max_date_allowed=today.date(), display_format="YYYY-MM-DD")]),
            html.Label(["Timeframe", dcc.Dropdown(
                id="timeframe", options=list(TIMEFRAMES), value="1h", clearable=False)]),
            html.Div(className="fetch", children=[
                dcc.Checklist(
                    id="fetch", value=[],
                    options=[{"label": " Fetch missing from Databento (costs money)", "value": "yes"}]),
                html.Button("Load", id="load-btn", n_clicks=0),
            ]),
        ]),
        html.Div(id="status", className="status", role="status"),
        html.Section(id="kpis", className="kpis"),
        dcc.Loading(children=[
            dcc.Graph(id="price-chart", config={"displaylogo": False}),
            dcc.Graph(id="change-chart", config={"displaylogo": False}),
        ]),
    ])
