"""Static Dash layout for the WIRP (Fed rate-probability) page (component tree only;
behaviour lives in wirp_callbacks.py). Read-only - never fetches; populate ZQ data
first via scripts/update_futures.py (LIVE) and scripts/update_daily.py (CLOSE).
CLAUDE.md section 11.
"""
from __future__ import annotations

import dash
from dash import dcc, html


def build_layout() -> html.Div:
    return html.Div(className="page", children=[
        html.H2("Fed Funds Rate Probabilities (WIRP)"),
        html.P(
            "The market-implied probability of a Fed rate move at each upcoming FOMC "
            "meeting, backed out of 30-Day Fed Funds futures (ZQ) prices - the same "
            "idea as Bloomberg's WIRP screen or CME's own FedWatch Tool. Display-only - "
            "use scripts/update_futures.py and scripts/update_daily.py to add more ZQ "
            "contract months.",
            className="page-intro",
        ),
        html.Section(className="controls", children=[
            html.Label(["Source", dcc.RadioItems(
                id="wirp-mode", value="close", inline=True,
                options=[
                    {"label": "Close (official settlement)", "value": "close"},
                    {"label": "Live (latest cached 1m bar)", "value": "live"},
                ],
            )]),
        ]),
        html.Div(id="wirp-status", className="status", role="status"),
        html.Section(id="wirp-stats", className="kpis"),
        dcc.Loading(children=[
            dcc.Graph(id="wirp-chart", config={"displaylogo": False}),
        ]),
        html.Div(className="note", children=[
            html.Strong("Methodology & known limitations"),
            html.P([
                "Each contract month's price implies its average daily Fed Funds rate "
                "(100 - price). For each meeting, the post-meeting rate is read from "
                "the FLAT month right after it when available (its whole-month average "
                "IS the post-decision rate, no day-weighting) - falling back to "
                "day-weighting the meeting's own month (Geraty 2000, via Keasler & "
                "Goff 2007) only when that's not cached. Hover a bar to see which was "
                "used. The pre-meeting rate is anchored to the nearest flat month's own "
                "contract, then chained meeting to meeting - no external Fed Funds rate "
                "feed needed.",
            ]),
            html.P([
                html.Strong("Day-weighting a late-month meeting is noisy."),
                " Its amplification factor is days-in-month / days-after-the-meeting - "
                "large when few days remain in the month. This is why the next-month "
                "flat read is preferred whenever it's available, not just for "
                "late-month meetings.",
            ]),
            html.P([
                html.Strong("Per-meeting, not a full joint tree."),
                " CME's own FedWatch builds a cross-meeting probability tree enforcing "
                "a consistent joint path; this page decomposes each meeting's own "
                "implied change independently instead. The nearest meeting is "
                "effectively bootstrapped from the front of the curve either way, so "
                "the two methods should land close together there (checked against "
                "real FedWatch numbers: ~0.2bp apart) - the difference matters more for "
                "later meetings, and for genuinely joint questions like \"probability "
                "of at least 50bp of total cuts by year-end\", which this page doesn't "
                "attempt.",
            ]),
            html.P([
                html.Strong("Live mode is not a real-time feed"),
                " - this project only ever calls Databento's Historical API, so "
                '"Live" means the freshest 1-minute bar already cached on disk, shown '
                "with its own \"as of\" timestamp, not a live market print.",
            ]),
            html.P([
                html.Strong("The toggle only affects upcoming meetings."),
                " The current/anchor rate and any already-past meeting used to chain "
                "it forward always read the official settlement, in both modes - "
                "they're settled fact, not a live prediction. Only the genuinely "
                "upcoming meetings (and their own reference months) follow Live/Close.",
            ]),
        ]),
    ])


dash.register_page(__name__, path="/wirp", name="WIRP", layout=build_layout)
