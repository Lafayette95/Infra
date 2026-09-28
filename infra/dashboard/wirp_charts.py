"""Plotly figure builder for the WIRP page: a stacked probability bar per FOMC
meeting, one segment per discrete rate-outcome level (CLAUDE.md section 11). Pure
function - mirrors infra.dashboard.rnd_charts's shape (DataFrame in, Figure out),
reusing infra.dashboard.charts's shared chrome.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from infra.dashboard.charts import _style, empty_figure
from infra.dashboard.theme import tokens


def _level_style(level: int, t: dict[str, str]) -> tuple[str, float]:
    """Direction (cut/hold/hike) is encoded blue/gray/orange - the project's existing
    up/down pair (infra.dashboard.theme), never red/green. Magnitude within the same
    direction is a sequential intensity ramp (one hue, deeper for a bigger discrete
    move) rather than a second hue, per the dataviz skill's "sequential = one hue" rule."""
    if level == 0:
        return t["muted"], 0.9
    color = t["down"] if level < 0 else t["up"]
    return color, min(1.0, 0.45 + 0.25 * abs(level))


def probability_figure(schedule: pd.DataFrame, meta: dict, theme: str = "light") -> go.Figure:
    """``schedule``: infra.analytics.wirp.meeting_schedule's long-format output.
    ``meta``: dict with ``mode``, ``as_of``, ``anchor_month``, ``anchor_rate`` (see
    infra.pipeline.wirp.build_schedule) and, when empty, ``status`` as the
    placeholder message.
    """
    t = tokens(theme)
    if schedule.empty:
        return empty_figure(meta.get("status", "No data"), theme)

    labels = schedule[["meeting_date", "month"]].drop_duplicates().sort_values("meeting_date").copy()
    labels["label"] = labels["meeting_date"].dt.strftime("%b %d '%y")
    order = labels["label"].tolist()

    fig = go.Figure()
    for level in sorted(schedule["outcome_step"].unique()):
        seg = schedule[schedule["outcome_step"] == level].merge(labels, on=["meeting_date", "month"])
        bps = int(round(seg["outcome_bps"].iloc[0]))
        name = "unchanged" if level == 0 else f"{bps:+d}bp"
        color, opacity = _level_style(level, t)
        method_label = seg["method"].map({"next_month_flat": "next-month read", "day_weighted": "day-weighted"})
        fig.add_trace(go.Bar(
            x=seg["label"], y=seg["probability"], name=name,
            marker_color=color, marker_line_width=0, opacity=opacity,
            customdata=pd.DataFrame({"rate": seg["outcome_rate"], "method": method_label}),
            hovertemplate=(
                f"{name}<br>implied rate %{{customdata[0]:.3f}}%<br>%{{y:.0%}}"
                "<br><span style='opacity:0.7'>%{customdata[1]}</span><extra></extra>"
            ),
        ))

    as_of = meta.get("as_of")
    as_of_text = as_of.strftime("%Y-%m-%d %H:%M") + " UTC" if as_of is not None else "n/a"
    fig.update_layout(
        barmode="stack",
        xaxis=dict(categoryorder="array", categoryarray=order),
        title=dict(
            text="Fed Funds futures (ZQ) · implied FOMC move probabilities"
                 f"<br><sup>{meta.get('mode', '').upper()} · as of {as_of_text} · "
                 f"anchor {meta.get('anchor_month', '?')} @ {meta.get('anchor_rate') or float('nan'):.3f}%</sup>",
            x=0, font=dict(color=t["ink"], size=16),
        ),
    )
    fig.update_yaxes(title_text="probability", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title_text="FOMC meeting")
    fig = _style(fig, t, 460, date_axis=False)
    # _style suppresses the legend by default (single-series charts elsewhere in this
    # app don't need one); a stacked multi-outcome bar does, so restore it here.
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.18))
    return fig
