"""Static Dash layout for the Risk-Neutral Density page (component tree only;
behaviour lives in rnd_callbacks.py). Display-only: reads whatever option settlement
data is already cached under Database/Daily/Options (infra.dashboard.rnd_selectors),
never fetches - populate data first via scripts/update_daily_options.py.
"""
from __future__ import annotations

import dash
from dash import dcc, html

from infra.dashboard.rnd_selectors import day_options, expiry_options, underlying_options


def build_layout() -> html.Div:
    underlyings = underlying_options()
    first_u = underlyings[0] if underlyings else None
    expiries = expiry_options(first_u) if first_u else []
    first_e = expiries[0] if expiries else None
    days = day_options(first_u, first_e) if first_u and first_e else []
    first_d = days[-1] if days else None  # most recent by default

    return html.Div(className="page", children=[
        html.H2("Risk-Neutral Density"),
        html.P(
            "The market-implied probability distribution for a SOFR future at its "
            "option's expiry, extracted from real settlement prices via "
            "Breeden-Litzenberger. Display-only - use scripts/update_daily_options.py "
            "to add more underlyings, expiries or days.",
            className="page-intro",
        ),
        html.Section(className="controls", children=[
            html.Label(["Underlying", dcc.Dropdown(
                id="rnd-underlying", options=underlyings, value=first_u, clearable=False)]),
            html.Label(["Option expiry", dcc.Dropdown(
                id="rnd-expiry", options=expiries, value=first_e, clearable=False)]),
            html.Label(["Valuation day", dcc.Dropdown(
                id="rnd-day", options=days, value=first_d, clearable=False)]),
        ]),
        html.Div(id="rnd-status", className="status", role="status"),
        html.Section(id="rnd-stats", className="kpis"),
        dcc.Loading(children=[
            dcc.Graph(id="rnd-chart", config={"displaylogo": False}),
        ]),
        html.Div(className="note", children=[
            html.Strong("Methodology & known limitations"),
            html.P([
                "Forward price and discount factor come from the chain's own put-call "
                "parity (no separate rate curve needed). Out-of-the-money implied "
                "vols are fit with a cubic spline; the discrete second difference of "
                "the fitted call-price curve gives the density (Breeden-Litzenberger), "
                "normalized to integrate to 1 over the observed strike range.",
            ]),
            html.P([
                html.Strong("SR3 options are American-exercise, not European"),
                " - this method is derived for European options, so the result is a "
                "documented approximation. The put-call parity fit is typically very "
                "clean on real data; treat deep in-the-money or longer-dated tails "
                "with more skepticism than the body of the distribution.",
            ]),
        ]),
    ])


dash.register_page(__name__, path="/rnd", name="Risk-Neutral Density", layout=build_layout)
