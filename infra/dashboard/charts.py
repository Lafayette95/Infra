"""Plotly figure builders. Pure functions: DataFrame in, Figure out."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from infra.dashboard.theme import tokens


def _style(fig: go.Figure, t: dict[str, str], height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(color=t["ink2"], size=12),
        margin=dict(l=56, r=24, t=48, b=32),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor=t["grid"], linecolor=t["axis"], zeroline=False,
        rangebreaks=[dict(bounds=["sat", "sun"])],  # Globex has no Saturday session
    )
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["axis"], zeroline=False)
    return fig


def empty_figure(message: str, theme: str = "light") -> go.Figure:
    """Placeholder figure with a centred message (nothing loaded yet)."""
    t = tokens(theme)
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=t["muted"], size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, t, 420)


def price_figure(bars: pd.DataFrame, ticker: str, timeframe: str, theme: str = "light") -> go.Figure:
    """Candlesticks (top) with volume (bottom). Two stacked panels, one y-scale each."""
    t = tokens(theme)
    if bars.empty:
        return empty_figure("No data on disk for this selection", theme)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=bars.index, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
        name=ticker,
        increasing=dict(line=dict(color=t["up"], width=1), fillcolor=t["up"]),
        decreasing=dict(line=dict(color=t["down"], width=1), fillcolor=t["down"]),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=bars.index, y=bars["volume"], name="Volume", marker_color=t["volume"], marker_line_width=0,
    ), row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"{ticker} · {timeframe} bars"
                 f"<br><sup>blue = close ≥ open · orange = close &lt; open</sup>",
            x=0, font=dict(color=t["ink"], size=16),
        ),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return _style(fig, t, 560)


def daily_change_figure(bars_1d: pd.DataFrame, ticker: str, theme: str = "light") -> go.Figure:
    """Close-to-close daily change (price points). Sign = colour, plus the bar direction."""
    t = tokens(theme)
    change = bars_1d["close"].diff().dropna()
    if change.empty:
        return empty_figure("Need at least two trading days", theme)
    fig = go.Figure(go.Bar(
        x=change.index, y=change,
        marker_color=[t["up"] if v >= 0 else t["down"] for v in change],
        marker_line_width=0, name="Daily change",
        hovertemplate="%{y:+.4f}<extra></extra>",
    ))
    fig.update_layout(title=dict(
        text=f"{ticker} · daily close-to-close change", x=0, font=dict(color=t["ink"], size=16),
    ))
    fig.add_hline(y=0, line_color=t["axis"], line_width=1)
    return _style(fig, t, 300)


def summary_metrics(bars: pd.DataFrame, bars_1d: pd.DataFrame) -> list[tuple[str, str]]:
    """Headline (label, value) pairs for the KPI tiles."""
    if bars.empty:
        return []
    first_close, last_close = bars_1d["close"].iloc[0], bars_1d["close"].iloc[-1]
    change = last_close - first_close
    return [
        ("Last close", f"{last_close:.4f}"),
        ("Period change", f"{change:+.4f}"),
        ("High", f"{bars['high'].max():.4f}"),
        ("Low", f"{bars['low'].min():.4f}"),
        ("Volume", f"{int(bars['volume'].sum()):,}"),
        ("Trading days", f"{len(bars_1d):,}"),
    ]
