"""Static Dash layout for the Futures page (component tree only; behaviour lives in
callbacks.py). Registered as the app's index page - see infra/dashboard/shell.py for
the persistent nav + theme toggle shared across pages.
"""
from __future__ import annotations

import dash
import pandas as pd
from dash import dcc, html

from infra.config import FUTURES_ROOTS
from infra.dashboard.selectors import default_expiry, expiry_options, root_options
from infra.dashboard.timezones import DEFAULT_TIMEZONE, DISPLAY_TIMEZONES
from infra.processing.resample import TIMEFRAMES

def build_layout() -> html.Div:
    first_root = next(iter(FUTURES_ROOTS))
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return html.Div(className="page", children=[
        html.H2("Futures · 1-minute bars"),
        html.Section(className="controls", children=[
            html.Label(["Ticker Root", dcc.Dropdown(
                id="ticker-root", options=root_options(), value=first_root, clearable=False)]),
            html.Label(["Expiry", dcc.Dropdown(
                id="expiry", options=expiry_options(first_root),
                value=default_expiry(first_root), clearable=False)]),
            html.Label(["Date range (UTC)", dcc.DatePickerRange(
                id="dates", start_date=(today - pd.Timedelta(days=30)).date(),
                end_date=today.date(), max_date_allowed=today.date(), display_format="YYYY-MM-DD")]),
            html.Label(["Timeframe", dcc.Dropdown(
                id="timeframe",
                options=[{"label": ("1D (trading day)" if k == "1D" else k), "value": k} for k in TIMEFRAMES],
                value="1h", clearable=False)]),
            html.Label(["Display timezone", dcc.Dropdown(
                id="display-tz",
                options=[{"label": k, "value": v} for k, v in DISPLAY_TIMEZONES.items()],
                value=DEFAULT_TIMEZONE, clearable=False)]),
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


dash.register_page(__name__, path="/", name="Futures", layout=build_layout)
