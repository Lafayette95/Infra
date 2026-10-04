"""Dash callbacks for the basis page: read a stored basis-model run
(``infra.storage.basis_runs``) and our curve's per-bond metrics
(``infra.pipeline.treasury_curves.read_rv``) - never runs a model or fetches.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, html

from infra.dashboard import basis_charts
from infra.pipeline.treasury_curves import read_rv
from infra.storage import basis_runs

log = logging.getLogger(__name__)
TICKS = 32.0


def _fmt(v, f="{:.2f}"):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def bond_table(bonds: pd.DataFrame, rv: pd.DataFrame) -> html.Table:
    """One day's deliverables (one delivery kind), most likely CTD first."""
    b = bonds.merge(rv[["cusip", "zspread_loo_bp", "carry_curve_bp", "rolldown_bp"]], on="cusip", how="left") \
        if not rv.empty else bonds.assign(zspread_loo_bp=np.nan, carry_curve_bp=np.nan, rolldown_bp=np.nan)
    b = b.sort_values(["prob", "implied_futures"], ascending=[False, True])
    cols = [("Bond", lambda r: basis_charts.bond_label(r.cusip, r.coupon, r.maturity)),
            ("P(CTD)", lambda r: _fmt(r.prob, "{:.0%}")),
            ("CF", lambda r: _fmt(r.cf, "{:.4f}")),
            ("Price", lambda r: _fmt(r.price, "{:.4f}")),
            ("Forward", lambda r: _fmt(r.fwd, "{:.4f}")),
            ("Implied fut.", lambda r: _fmt(r.implied_futures, "{:.4f}")),
            ("Gross basis /32", lambda r: _fmt(r.gross_basis * TICKS)),
            ("Carry /32", lambda r: _fmt(r.carry * TICKS)),
            ("Net basis /32", lambda r: _fmt(r.net_basis * TICKS)),
            ("Impl. repo %", lambda r: _fmt(r.irr, "{:.3f}")),
            ("Funding %", lambda r: _fmt(r.repo, "{:.3f}")),
            ("z-spread bp", lambda r: _fmt(r.zspread_loo_bp, "{:.1f}")),
            ("Curve carry bp", lambda r: _fmt(r.carry_curve_bp, "{:.1f}")),
            ("Rolldown bp", lambda r: _fmt(r.rolldown_bp, "{:.1f}"))]
    head = html.Thead(html.Tr([html.Th(c) for c, _ in cols]))
    body = html.Tbody([html.Tr([html.Td(f(r)) for _, f in cols]) for r in b.itertuples()])
    return html.Table(className="data-table", children=[head, body])


def register_basis_callbacks(app: Dash) -> None:
    @app.callback(Output("basis-model", "options"), Output("basis-model", "value"), Input("basis-root", "value"))
    def models(_root):
        ms = basis_runs.models()
        return ms, ("M2T" if "M2T" in ms else (ms[0] if ms else None))

    @app.callback(Output("basis-contract", "options"), Output("basis-contract", "value"),
                  Input("basis-model", "value"), Input("basis-root", "value"))
    def contracts(model, root):
        if not model:
            return [], None
        c = basis_runs.read(model, "contracts")
        if c.empty:
            return [], None
        c = c[c["root"] == root]
        if c.empty:
            return [], None
        last = c[c["day"] == c["day"].max()].sort_values("delivery")
        opts = sorted(c["contract"].unique(), key=lambda k: c.loc[c["contract"] == k, "delivery"].min())
        return opts, last["contract"].iloc[0]

    @app.callback(Output("basis-day", "options"), Output("basis-day", "value"),
                  Input("basis-model", "value"), Input("basis-contract", "value"))
    def days(model, contract):
        if not model or not contract:
            return [], None
        c = basis_runs.read(model, "contracts", contracts=[contract])
        ds = [d.strftime("%Y-%m-%d") for d in sorted(c["day"].unique(), reverse=True)]
        return ds, (ds[0] if ds else None)

    @app.callback(Output("basis-kpis", "children"), Output("basis-table", "children"),
                  Output("basis-prob-chart", "figure"), Output("basis-option-chart", "figure"),
                  Output("basis-timing-chart", "figure"), Output("basis-status", "children"),
                  Input("basis-model", "value"), Input("basis-contract", "value"), Input("basis-day", "value"),
                  Input("theme", "value"))
    def refresh(model, contract, day, theme):
        empty = basis_charts.empty_figure("Pick a model, a contract and a day", theme)
        if not model or not contract or not day:
            return [], None, empty, empty, empty, "No stored basis run - see scripts/run_basis.py --persist"
        try:
            c = basis_runs.read(model, "contracts", contracts=[contract])
            b = basis_runs.read(model, "bonds", contracts=[contract])
            d = pd.Timestamp(day)
            row = c[c["day"] == d].iloc[0]
            # each day's bonds at the delivery day the model picked (first / last)
            kinds = c.set_index("day")["delivery_kind"]
            b = b[b["delivery_kind"].to_numpy() == b["day"].map(kinds).to_numpy()]
            today = b[b["day"] == d]
            rv = read_rv(d, d, method="spline", cusips=list(today["cusip"].unique()))
            g = lambda k: row.get(k, np.nan)
            tiles = [("CTD", basis_charts.bond_label(row["ctd"], g("ctd_coupon"), g("ctd_maturity"))),
                     ("P(CTD)", _fmt(g("ctd_prob"), "{:.0%}")),
                     ("Delivery", f"{pd.Timestamp(g('delivery')):%Y-%m-%d} ({g('delivery_kind')})"),
                     ("Futures", _fmt(g("futures"), "{:.4f}")),
                     ("Model option value", _fmt(g("option_value_model_32"), "{:.2f}/32")),
                     ("Observed", _fmt(g("option_value_obs_32"), "{:.2f}/32")),
                     ("CTD implied repo", _fmt(g("ctd_irr"), "{:.3f}%")),
                     ("Futures DV01", _fmt(g("futures_dv01"), "{:.4f}"))]
            kpis = [html.Div(className="kpi", children=[html.Span(k), html.Strong(v)]) for k, v in tiles]
            return (kpis, bond_table(today, rv), basis_charts.probability_figure(b, theme),
                    basis_charts.optionality_figure(c, theme, part="quality"),
                    basis_charts.optionality_figure(c, theme, part="timing"),
                    f"{model} - {contract} on {day}: {today['cusip'].nunique()} deliverables")
        except Exception as exc:  # a data surprise must not crash the page
            log.exception("basis page refresh failed")
            return [], None, empty, empty, empty, f"⚠ {type(exc).__name__}: {exc}"
