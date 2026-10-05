"""Plotly figures for the bond-curve page: pure functions, DataFrame in, Figure out, on the
shared chrome of ``infra.dashboard.charts``. Two stacked panels sharing the maturity axis:
the par curve with every bond's market yield, and every bond's leave-one-out z-spread
(rich below zero). On-the-runs, older issues and CTDs marked; futures-basket sectors shaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from infra.dashboard.charts import _style, empty_figure
from infra.dashboard.theme import series_colors, tokens

SYMBOLS = {"on-the-run": "star", "old": "circle", "other": "circle-open"}


def _hover(b: pd.DataFrame) -> list[str]:
    f = lambda v, fmt: "" if pd.isna(v) else fmt.format(v)
    out = []
    for r in b.itertuples():
        lines = [f"<b>{f(r.coupon, '{:.3f}')}% {pd.Timestamp(r.maturity_date):%b-%d-%Y}</b> ({r.cusip})",
                 f"{r.security_type or ''} {r.original_term or ''}, issued {f(pd.Timestamp(r.issue_date) if pd.notna(r.issue_date) else np.nan, '{:%Y-%m-%d}')}",
                 (f"{r.tenor} on-the-run" if r.status == "on-the-run" else f"{r.tenor} {int(r.rank)}-old") if pd.notna(r.tenor) else "",
                 f"yield {f(r.ytm, '{:.3f}')}% | z-spread {f(r.zspread_bp, '{:+.1f}')}bp (LOO {f(r.zspread_loo_bp, '{:+.1f}')})",
                 f"curve carry {f(r.carry_curve_bp, '{:+.1f}')}bp | rolldown {f(r.rolldown_bp, '{:+.1f}')}bp (3m)",
                 f"in fit: {'yes' if r.in_fit else 'no'}"]
        if r.baskets:
            lines.append(f"baskets: {r.baskets}")
        if r.ctd_of:
            lines.append(f"<b>CTD: {r.ctd_of}</b>")
        out.append("<br>".join(l for l in lines if l))
    return out


def curve_figure(view: dict, theme: str = "light") -> go.Figure:
    t = tokens(theme)
    b, line, sectors = view.get("bonds", pd.DataFrame()), view.get("line", pd.DataFrame()), view.get("sectors", pd.DataFrame())
    if b is None or b.empty:
        return empty_figure("No curve stored for this day - see scripts/build_treasury_curves.py", theme)
    colors = series_colors(theme)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                        subplot_titles=("Par curve and every bond's yield (%)", "Rich / cheap: leave-one-out z-spread (bp; below 0 = rich)"))
    # basket sectors: shaded maturity spans, labelled at the top
    for k, s in enumerate(sectors.itertuples() if sectors is not None else []):
        for row in (1, 2):
            fig.add_vrect(x0=s.min, x1=s.max, fillcolor=colors[k % len(colors)], opacity=0.08, line_width=0, row=row, col=1)
        fig.add_annotation(x=(s.min + s.max) / 2, y=1.0, yref="paper", text=s.contract, showarrow=False,
                           font=dict(size=10, color=colors[k % len(colors)]))
    fig.add_trace(go.Scatter(x=line["maturity_years"], y=line["par_yield"], mode="lines", name="Fitted par curve",
                             line=dict(color=t["ink"], width=1.5), hoverinfo="skip"), row=1, col=1)
    hover = _hover(b)
    for status in ("other", "old", "on-the-run"):
        m = (b["status"] == status).to_numpy()
        if not m.any():
            continue
        x = b.loc[m]
        color = t["muted"] if status == "other" else (colors[0] if status == "old" else colors[1])
        size = 6 if status != "on-the-run" else 12
        h = [hover[i] for i in np.where(m)[0]]
        for row, ycol in ((1, "ytm"), (2, "zspread_loo_bp")):
            fig.add_trace(go.Scatter(x=x["maturity_years"], y=x[ycol], mode="markers", name=status, showlegend=row == 1,
                                     marker=dict(symbol=SYMBOLS[status], size=size, color=color,
                                                 line=dict(width=1, color=color)),
                                     hovertext=h, hoverinfo="text"), row=row, col=1)
    ctd = b[b["ctd_of"] != ""]
    if not ctd.empty:
        h = [hover[i] for i in ctd.index]
        for row, ycol in ((1, "ytm"), (2, "zspread_loo_bp")):
            fig.add_trace(go.Scatter(x=ctd["maturity_years"], y=ctd[ycol], mode="markers+text" if row == 2 else "markers",
                                     name="CTD", showlegend=row == 1, text=ctd["ctd_of"].str.split(" ").str[0],
                                     textposition="top center", textfont=dict(size=10, color=t["ink"]),
                                     marker=dict(symbol="diamond-open", size=14, color=t["ink"], line=dict(width=2)),
                                     hovertext=h, hoverinfo="text"), row=row, col=1)
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1), row=2, col=1)
    fig = _style(fig, t, 720, date_axis=False)
    fig.update_layout(showlegend=True, hovermode="closest", legend=dict(orientation="h", y=-0.08))
    fig.update_xaxes(title_text="Years to maturity", row=2, col=1)
    zs = b["zspread_loo_bp"].dropna()
    if len(zs):
        lim = max(3.0, float(np.nanpercentile(zs.abs(), 98)) * 1.2)
        fig.update_yaxes(range=[-lim, lim], row=2, col=1)  # a few far outliers don't flatten the rest
    return fig
