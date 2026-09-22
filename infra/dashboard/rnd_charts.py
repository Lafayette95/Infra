"""Plotly figure builder for the risk-neutral density page. Pure function: DataFrame
in, Figure out - mirrors infra.dashboard.charts's convention, reusing its shared chart
chrome (``_style``) with the weekend rangebreak disabled (the x-axis here is strike,
not a date).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from infra.dashboard.charts import _style, empty_figure
from infra.dashboard.theme import tokens


def density_figure(
    density: pd.DataFrame, meta: dict, theme: str = "light",
) -> go.Figure:
    """``density``: columns ``strike``, ``density`` (infra.analytics.rnd.extract_rnd's
    output). ``meta``: dict with ``forward``, ``p5``, ``p25``, ``p75``, ``p95``,
    ``underlying``, ``expiry``, ``valuation_day`` (see infra.analytics.forward /
    rnd.extract_rnd and CLAUDE.md section 9 for what these mean and their limits -
    SR3 options are American-exercise; this is a European approximation).
    """
    t = tokens(theme)
    if density.empty:
        return empty_figure("No density to show", theme)

    fig = go.Figure()

    def band(lo: float, hi: float, opacity: float) -> None:
        seg = density[(density["strike"] >= lo) & (density["strike"] <= hi)]
        if seg.empty:
            return
        fig.add_trace(go.Scatter(
            x=seg["strike"], y=seg["density"], fill="tozeroy", mode="none",
            fillcolor=t["up"], opacity=opacity, hoverinfo="skip", showlegend=False,
        ))

    band(meta["p5"], meta["p95"], 0.10)
    band(meta["p25"], meta["p75"], 0.18)

    fig.add_trace(go.Scatter(
        x=density["strike"], y=density["density"], mode="lines",
        line=dict(color=t["up"], width=2), name="density",
        hovertemplate="strike %{x:.4f}<br>density %{y:.4f}<extra></extra>",
    ))

    fig.add_vline(
        x=meta["forward"], line=dict(color=t["down"], width=1.5, dash="dash"),
        annotation_text=f"forward {meta['forward']:.4f}", annotation_position="top",
        annotation_font=dict(color=t["down"], size=11),
    )

    fig.update_layout(
        title=dict(
            text=f"{meta['underlying']} · risk-neutral density"
                 f"<br><sup>valued {meta['valuation_day']} · expiry {meta['expiry']} · "
                 f"blue = density · orange = forward · bands = 50% / 90% probability</sup>",
            x=0, font=dict(color=t["ink"], size=16),
        ),
    )
    fig.update_yaxes(title_text="density")
    fig.update_xaxes(title_text="strike (rate/price scale)")
    return _style(fig, t, 460, date_axis=False)
