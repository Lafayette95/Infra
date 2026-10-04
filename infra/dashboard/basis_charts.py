"""Plotly figure builders for the Treasury futures basis page (``/basis``): pure functions,
DataFrame in, Figure out, on the shared chrome of ``infra.dashboard.charts``.
Methodology: infra/models/basis/CLAUDE.md.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from infra.dashboard.charts import _style, empty_figure
from infra.dashboard.theme import series_colors, tokens

# the modelled optionality's components, in stacking order, with their labels
COMPONENTS = [("quality_macro_32", "Quality: macro (level)"), ("quality_spread_32", "Quality: spread (relative)"),
              ("wildcard_32", "Timing: wild card"), ("eom_32", "Timing: end of month")]
MIN_PROB = 0.02  # a bond enters the probability chart once it reaches this


def bond_label(cusip: str, coupon: float, maturity) -> str:
    return f"{coupon:.3f}% {pd.Timestamp(maturity):%b-%y} ({cusip})"


def probability_figure(bonds: pd.DataFrame, theme: str = "light") -> go.Figure:
    """``bonds``: one contract's rows over time (``day, cusip, coupon, maturity, prob``),
    one delivery kind per day. Stacked area of the CTD probability per bond."""
    t = tokens(theme)
    if bonds.empty or bonds["prob"].sum() == 0:
        return empty_figure("No delivery probabilities stored for this contract", theme)
    keep = bonds.groupby("cusip")["prob"].max()
    keep = keep[keep >= MIN_PROB].index
    b = bonds[bonds["cusip"].isin(keep)]
    labels = b.drop_duplicates("cusip").set_index("cusip").apply(lambda r: bond_label(r.name, r["coupon"], r["maturity"]), axis=1)
    order = b.groupby("cusip")["prob"].mean().sort_values(ascending=False).index
    colors = series_colors(theme)
    fig = go.Figure()
    for k, c in enumerate(order):
        s = b[b["cusip"] == c].groupby("day")["prob"].sum()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=labels[c], stackgroup="p", mode="lines",
                                 line=dict(width=0.5, color=colors[k % len(colors)]),
                                 hovertemplate="%{y:.0%}<extra>" + labels[c] + "</extra>"))
    fig = _style(fig, t, 360)
    fig.update_layout(title="Cheapest-to-deliver probability by bond", showlegend=True,
                      legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def optionality_figure(contracts: pd.DataFrame, theme: str = "light") -> go.Figure:
    """``contracts``: one contract over time (``day`` + the components, ``option_value_model_32``,
    ``option_value_obs_32``). Stacked components (32nds), model total and observed as lines."""
    t = tokens(theme)
    if contracts.empty:
        return empty_figure("No model run stored for this contract", theme)
    c = contracts.sort_values("day")
    colors = series_colors(theme)
    fig = go.Figure()
    for k, (col, label) in enumerate(COMPONENTS):
        if col in c.columns and c[col].notna().any():
            fig.add_trace(go.Scatter(x=c["day"], y=c[col].fillna(0.0), name=label, stackgroup="o", mode="lines",
                                     line=dict(width=0.5, color=colors[k % len(colors)]),
                                     hovertemplate="%{y:.2f}/32<extra>" + label + "</extra>"))
    fig.add_trace(go.Scatter(x=c["day"], y=c["option_value_model_32"], name="Model total", mode="lines",
                             line=dict(color=t["ink"], width=1.5, dash="dot"),
                             hovertemplate="%{y:.2f}/32<extra>Model total</extra>"))
    if "option_value_obs_32" in c.columns:
        fig.add_trace(go.Scatter(x=c["day"], y=c["option_value_obs_32"], name="Observed (M0 fair - market)", mode="lines",
                                 line=dict(color=t["muted"], width=1.5),
                                 hovertemplate="%{y:.2f}/32<extra>Observed</extra>"))
    fig = _style(fig, t, 360)
    fig.update_layout(title="Modelled optionality by component (32nds)", showlegend=True,
                      legend=dict(orientation="h", y=-0.15))
    return fig
