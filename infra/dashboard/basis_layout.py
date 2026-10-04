"""Static Dash layout for the Treasury futures BASIS page (component tree only; behaviour
in basis_callbacks.py). Read-only - never runs a model: it shows what a stored model run
holds (``Derived/BasisRuns/<model>``, written by ``scripts/run_basis.py --persist``).
Methodology: infra/models/basis/CLAUDE.md; root CLAUDE.md 22.
"""
from __future__ import annotations

import dash
from dash import dcc, html

ROOTS = ["ZT", "Z3N", "ZF", "ZN", "TN", "ZB", "UB"]


def build_layout() -> html.Div:
    return html.Div(className="page", children=[
        html.H2("Treasury Futures Basis"),
        html.P(
            "Per contract and basis model: every deliverable bond in order of its probability of "
            "being cheapest to deliver, with its basis metrics, and over time the delivery "
            "probabilities and the model's delivery-option value split into its components. "
            "Display-only - run scripts/run_basis.py --persist to add days or models.",
            className="page-intro",
        ),
        html.Section(className="controls", children=[
            html.Label(["Model", dcc.Dropdown(id="basis-model", clearable=False)]),
            html.Label(["Root", dcc.Dropdown(id="basis-root", options=ROOTS, value="ZN", clearable=False)]),
            html.Label(["Contract", dcc.Dropdown(id="basis-contract", clearable=False)]),
            html.Label(["Day", dcc.Dropdown(id="basis-day", clearable=False)]),
        ]),
        html.Div(id="basis-status", className="status", role="status"),
        html.Section(id="basis-kpis", className="kpis"),
        html.Div(id="basis-table", className="table-wrap"),
        dcc.Loading(children=[
            dcc.Graph(id="basis-prob-chart", config={"displaylogo": False}),
            dcc.Graph(id="basis-option-chart", config={"displaylogo": False}),
        ]),
        html.Div(className="note", children=[
            html.Strong("How to read it"),
            html.P("Implied futures = the bond's forward price to its delivery day / its conversion "
                   "factor; the lowest is the cheapest to deliver. Net basis and carry use the "
                   "funding model (futures-implied SOFR + client basis - the bond's specialness); "
                   "z-spread, curve carry and rolldown come from our own Treasury curve (spline, "
                   "leave-one-out; + = gain). Basis figures in 32nds of a point."),
            html.P("Optionality = what the delivery options take off the carry-only (M0) fair price: "
                   "quality macro (switches from parallel level moves - bonds' DV01 / CF differ), "
                   "quality spread (switches from relative moves), and the timing options (wild card "
                   "and end of month). 'Observed' is the M0 fair price minus the market - it moves "
                   "with much more than the options (futures richness, cash marks), see the basis doc."),
        ]),
    ])


dash.register_page(__name__, path="/basis", name="Basis", layout=build_layout)
