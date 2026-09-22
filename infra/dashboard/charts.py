"""Plotly figure builders. Pure functions: DataFrame in, Figure out."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from infra.dashboard.theme import tokens
from infra.dashboard.timezones import DEFAULT_TIMEZONE, to_display_index


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


def price_figure(
    bars: pd.DataFrame, ticker: str, timeframe: str, theme: str = "light", tz: str = DEFAULT_TIMEZONE
) -> go.Figure:
    """Candlesticks (top) with volume (bottom). Two stacked panels, one y-scale each.

    ``tz`` (an infra.dashboard.timezones.DISPLAY_TIMEZONES value) only changes how bar
    times are RENDERED, on a display-only copy of the index - see
    infra/dashboard/timezones.py and CLAUDE.md section 7. ``bars`` itself is untouched.
    """
    t = tokens(theme)
    if bars.empty:
        return empty_figure("No data on disk for this selection", theme)
    x = to_display_index(bars.index, tz)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    # "contract" is the absolute ticker each bar actually came from (see
    # infra.pipeline.series.load_series); shown on hover so a relative ticker like
    # SR3.v.0 reveals which real contract is behind each bar, especially around a roll.
    has_contract = "contract" in bars.columns and bars["contract"].notna().any()
    hovertemplate = (
        f"%{{x|%Y-%m-%d %H:%M}} {tz}<br>O %{{open:.4f}}  H %{{high:.4f}}<br>L %{{low:.4f}}  C %{{close:.4f}}"
        + ("<br>Contract: %{customdata[0]}" if has_contract else "")
        + "<extra></extra>"
    )
    fig.add_trace(go.Candlestick(
        x=x, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
        name=ticker,
        customdata=bars[["contract"]].to_numpy() if has_contract else None,
        hovertemplate=hovertemplate,
        increasing=dict(line=dict(color=t["up"], width=1), fillcolor=t["up"]),
        decreasing=dict(line=dict(color=t["down"], width=1), fillcolor=t["down"]),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x, y=bars["volume"], name="Volume", marker_color=t["volume"], marker_line_width=0,
    ), row=2, col=1)
    fig.update_layout(
        title=dict(
            text=f"{ticker} · {timeframe} bars ({tz})"
                 f"<br><sup>blue = close ≥ open · orange = close &lt; open</sup>",
            x=0, font=dict(color=t["ink"], size=16),
        ),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return _style(fig, t, 560)


def daily_change_figure(bars_1d: pd.DataFrame, ticker: str, theme: str = "light", exchange: str = "UTC") -> go.Figure:
    """Close-to-close daily change (price points). Sign = colour, plus the bar direction.

    Always bucketed by the exchange TRADING day (infra.trading_calendar, CLAUDE.md
    section 6e), regardless of the chart's display timezone: a "day" here is a DATA
    bucket, so shifting its label would misstate which bucket a bar belongs to. Only
    price_figure's intraday x-axis is timezone-convertible. ``exchange`` is a label only
    (e.g. "CME Globex / CBOT" from infra.config.TRADING_HOURS) - it does not affect the
    bucketing itself, which the caller already applied via resample_ohlcv's `dataset`.
    """
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
        text=f"{ticker} · daily close-to-close change ({exchange} trading day)",
        x=0, font=dict(color=t["ink"], size=16),
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
