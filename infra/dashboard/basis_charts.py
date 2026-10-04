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


QUALITY = COMPONENTS[:2]
TIMING = COMPONENTS[2:]


def _switch_days(contracts: pd.DataFrame) -> list:
    """Days on which the model's delivery day switched (first <-> last): the horizon jumps
    by ~a month there, and the option values with it."""
    c = contracts.sort_values("day")
    k = c["delivery_kind"].astype(str)
    return list(c.loc[k.ne(k.shift()) & k.shift().notna(), "day"])


def optionality_figure(contracts: pd.DataFrame, theme: str = "light", *, part: str = "quality") -> go.Figure:
    """One part of the modelled optionality over time (32nds): ``part`` = "quality" (macro +
    spread - decays with the horizon and moves with the CTD / runner-up gap) or "timing"
    (wild card + end of month - concentrated in the delivery month, so roughly flat until
    then). Stacked components; the timing panel adds the model total and the observed
    value (M0 fair - market) as lines; dotted verticals where the model's delivery
    day switched first <-> last."""
    t = tokens(theme)
    if contracts.empty:
        return empty_figure("No model run stored for this contract", theme)
    c = contracts.sort_values("day")
    comps = QUALITY if part == "quality" else TIMING
    colors = series_colors(theme)
    offset = 0 if part == "quality" else len(QUALITY)
    fig = go.Figure()
    for k, (col, label) in enumerate(comps):
        if col in c.columns and c[col].notna().any():
            y = c[col].fillna(0.0)
            fig.add_trace(go.Scatter(x=c["day"], y=y, name=label, stackgroup="o", mode="lines",
                                     line=dict(width=0.5, color=colors[(offset + k) % len(colors)]),
                                     hovertemplate="%{y:.2f}/32<extra>" + label + "</extra>"))
    if part == "timing":
        fig.add_trace(go.Scatter(x=c["day"], y=c["option_value_model_32"], name="Model total (quality + timing)",
                                 mode="lines", line=dict(color=t["ink"], width=1.5, dash="dot"),
                                 hovertemplate="%{y:.2f}/32<extra>Model total</extra>"))
        if "option_value_obs_32" in c.columns:
            fig.add_trace(go.Scatter(x=c["day"], y=c["option_value_obs_32"], name="Observed (M0 fair - market)",
                                     mode="lines", line=dict(color=t["muted"], width=1.5),
                                     hovertemplate="%{y:.2f}/32<extra>Observed</extra>"))
    for d in _switch_days(c):
        fig.add_vline(x=d, line=dict(color=t["muted"], width=1, dash="dot"))
    fig = _style(fig, t, 320)
    title = ("Quality option: macro (level) + spread (relative), 32nds" if part == "quality"
             else "Timing options: wild card + end of month, 32nds (with the model total and the observed value)")
    fig.update_layout(title=title, showlegend=True, legend=dict(orientation="h", y=-0.18))
    return fig
