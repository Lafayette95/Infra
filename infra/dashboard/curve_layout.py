"""Static Dash layout for the BOND CURVE page (component tree only; behaviour in
curve_callbacks.py). Read-only: our fitted Treasury curve and its per-bond metrics from
``Derived/TreasuryCurves`` / ``TreasuryRV`` (scripts/build_treasury_curves.py), the OTR map,
the futures delivery baskets and any stored basis run. Methodology: infra/models/curves/CLAUDE.md.
"""
from __future__ import annotations

import dash
from dash import dcc, html


def build_layout() -> html.Div:
    return html.Div(className="page", children=[
        html.H2("Treasury Bond Curve"),
        html.P("Our fitted Treasury curve with every note and bond on it: market yield against the par "
               "curve, and rich / cheap as the leave-one-out z-spread. On-the-runs, older issues and "
               "each futures contract's CTD marked; futures-basket sectors shaded. Hover a bond for its "
               "details. Display-only - rebuild with scripts/build_treasury_curves.py.",
               className="page-intro"),
        html.Section(className="controls", children=[
            html.Label(["Day", dcc.Dropdown(id="curve-day", clearable=False)]),
            html.Label(["Method", dcc.RadioItems(id="curve-method", value="spline", inline=True, options=[
                {"label": "Spline (default)", "value": "spline"}, {"label": "Svensson", "value": "svensson"}])]),
            html.Label(["Table: sector", dcc.Dropdown(id="curve-sector", value="all", clearable=False)]),
        ]),
        html.Div(id="curve-status", className="status", role="status"),
        dcc.Loading(children=[dcc.Graph(id="curve-chart", config={"displaylogo": False})]),
        html.Div(id="curve-table", className="table-wrap"),
        html.Div(className="note", children=[
            html.Strong("How to read it"),
            html.P("The line is the par curve fitted to off-the-run notes and bonds (on-the-runs and first "
                   "off-the-runs are left out of the fit, as in the Fed's GSW curve). A bond's z-spread is the "
                   "constant that, added to the curve's zero rates, reprices it - negative = rich. The "
                   "leave-one-out version refits without the bond, so it can't pull the curve to itself. "
                   "Curve carry and rolldown are over 3 months, + = gain; repo carry needs the funding "
                   "model and is on the basis page."),
        ]),
    ])


dash.register_page(__name__, path="/curve", name="Bond Curve", layout=build_layout)
