"""Dash callbacks for the bond-curve page - reads stores through
``infra.dashboard.curve_selectors``; never fits or runs anything."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, html

from infra.dashboard import curve_charts, curve_selectors

log = logging.getLogger(__name__)


def _f(v, fmt):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) else fmt.format(v)


def bond_table(b: pd.DataFrame, sector: str, sectors: pd.DataFrame) -> html.Table:
    if sector != "all" and not sectors.empty:
        s = sectors[sectors["contract"] == sector]
        b = b[b["baskets"].str.contains(sector, regex=False)] if not s.empty else b
    cols = [("Bond", lambda r: f"{_f(r.coupon, '{:.3f}')}% {pd.Timestamp(r.maturity_date):%b-%d-%Y} ({r.cusip})"),
            ("Years", lambda r: _f(r.maturity_years, "{:.2f}")),
            ("Status", lambda r: (f"{r.tenor} OTR" if r.status == "on-the-run" else f"{r.tenor} {int(r.rank)}-old")
             if pd.notna(r.tenor) else ""),
            ("Yield %", lambda r: _f(r.ytm, "{:.3f}")),
            ("z-spread bp", lambda r: _f(r.zspread_bp, "{:+.1f}")),
            ("LOO z bp", lambda r: _f(r.zspread_loo_bp, "{:+.1f}")),
            ("Curve carry bp", lambda r: _f(r.carry_curve_bp, "{:+.1f}")),
            ("Rolldown bp", lambda r: _f(r.rolldown_bp, "{:+.1f}")),
            ("In fit", lambda r: "yes" if r.in_fit else "no"),
            ("Baskets", lambda r: r.baskets),
            ("CTD", lambda r: r.ctd_of)]
    head = html.Thead(html.Tr([html.Th(c) for c, _ in cols]))
    body = html.Tbody([html.Tr([html.Td(f(r)) for _, f in cols]) for r in b.itertuples()])
    return html.Table(className="data-table", children=[head, body])


def register_curve_callbacks(app: Dash) -> None:
    @app.callback(Output("curve-day", "options"), Output("curve-day", "value"), Input("curve-method", "value"))
    def days(method):
        ds = [d.strftime("%Y-%m-%d") for d in curve_selectors.curve_days(method)]
        return ds, (ds[0] if ds else None)

    @app.callback(Output("curve-chart", "figure"), Output("curve-table", "children"), Output("curve-sector", "options"),
                  Output("curve-status", "children"),
                  Input("curve-day", "value"), Input("curve-method", "value"), Input("curve-sector", "value"),
                  Input("theme", "value"))
    def refresh(day, method, sector, theme):
        if not day:
            return curve_charts.empty_figure("No curve stored", theme), None, [{"label": "All", "value": "all"}], \
                "No curve stored - run scripts/build_treasury_curves.py"
        try:
            v = curve_selectors.curve_view(day, method)
            opts = [{"label": "All", "value": "all"}] + [{"label": f"{s.root} {s.contract}", "value": s.contract}
                                                          for s in v["sectors"].itertuples()]
            fit = v.get("fit", {})
            ctd_note = f", CTDs from the stored {v['ctds']['model'].iloc[0]} run" if not v["ctds"].empty else ", no stored basis run that day"
            status = f"{day} ({method}): {len(v['bonds'])} notes and bonds, {fit.get('n_fit', 0)} in the fit, fit error {fit.get('rmse_bp', np.nan):.2f}bp{ctd_note}"
            return curve_charts.curve_figure(v, theme), bond_table(v["bonds"], sector or "all", v["sectors"]), opts, status
        except Exception as exc:  # a data surprise must not crash the page
            log.exception("curve page refresh failed")
            return curve_charts.empty_figure(str(exc), theme), None, [{"label": "All", "value": "all"}], f"⚠ {type(exc).__name__}: {exc}"
