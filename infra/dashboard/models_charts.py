"""Plotly figures for the Models page (regressions and PCA, infra/models/stats). Pure
functions: frames in, figure out, sharing infra.dashboard.charts's chrome. Identity
colours come from the fixed categorical order (theme.series_colors), direction from the
up/down pair; one y-axis per chart, always."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from infra.dashboard.charts import _style, empty_figure
from infra.dashboard.theme import series_colors, tokens


def _title(fig, t, text, sub=""):
    fig.update_layout(title=dict(text=text + (f"<br><sup>{sub}</sup>" if sub else ""), x=0,
                                 font=dict(color=t["ink"], size=15)))


def _as_of_marker(fig, t, as_of, x_max):
    """Shade the out-of-sample region (rows after the fit date)."""
    if as_of is None or pd.isna(as_of) or x_max is None or pd.Timestamp(x_max) <= pd.Timestamp(as_of):
        return
    fig.add_vrect(x0=as_of, x1=x_max, fillcolor=t["grid"], opacity=0.35, line_width=0, layer="below")
    fig.add_vline(x=as_of, line=dict(color=t["axis"], width=1, dash="dot"))
    fig.add_annotation(x=as_of, y=1, yref="paper", text="  out of sample →", showarrow=False, xanchor="left",
                       font=dict(color=t["ink2"], size=11))


def _legend(fig):
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    return fig


def fit_figure(out: pd.DataFrame, as_of, theme: str, label: str, binary: bool = False) -> go.Figure:
    t = tokens(theme)
    if out.empty:
        return empty_figure("No rows", theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=out.index, y=out["y"], name="actual", mode="markers" if binary else "lines",
                             line=dict(color=t["muted"], width=1.5), marker=dict(size=4, color=t["muted"])))
    fig.add_trace(go.Scatter(x=out.index, y=out["fitted"], name="probability" if binary else "fitted",
                             line=dict(color=t["up"], width=2)))
    _as_of_marker(fig, t, as_of, out.index.max())
    _title(fig, t, f"{label} · actual vs fitted", "fitted with parameters as of the fit date; shaded = after it")
    return _legend(_style(fig, t, 380))


def residual_figure(out: pd.DataFrame, as_of, theme: str, column: str = "resid_z", label: str = "") -> go.Figure:
    t = tokens(theme)
    if out.empty or column not in out:
        return empty_figure("No residuals", theme)
    s = out[column]
    fig = go.Figure()
    for lvl in (-2, 2):
        fig.add_hline(y=lvl, line=dict(color=t["axis"], width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    fig.add_trace(go.Scatter(x=s.index, y=s, name="residual z", line=dict(color=t["down"], width=1.5),
                             hovertemplate="%{x|%Y-%m-%d}<br>z %{y:.2f}<extra></extra>"))
    _as_of_marker(fig, t, as_of, s.index.max())
    _title(fig, t, f"{label} residual, in fit-sample standard deviations", "dashed: ±2")
    return _style(fig, t, 300)


def lines_figure(frame: pd.DataFrame, theme: str, title: str, sub: str = "", as_of=None,
                 y_title: str = "", height: int = 340) -> go.Figure:
    """One line per column, categorical colours in fixed order (max 8; more are cut)."""
    t = tokens(theme)
    if frame is None or frame.empty:
        return empty_figure("No data", theme)
    cols = list(frame.columns)[:8]
    fig = go.Figure()
    for c, color in zip(cols, series_colors(theme)):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[c], name=str(c), line=dict(color=color, width=2)))
    if as_of is not None:
        _as_of_marker(fig, t, as_of, frame.index.max())
    _title(fig, t, title, sub)
    fig.update_yaxes(title_text=y_title)
    return _legend(_style(fig, t, height))


def loadings_figure(loadings: pd.DataFrame, theme: str, units: str) -> go.Figure:
    """Each PC's effect across the input columns (the curve shape of the factor)."""
    t = tokens(theme)
    fig = go.Figure()
    labels = [c.split(":", 1)[-1] for c in loadings.index]
    for pc, color in zip(loadings.columns, series_colors(theme)):
        fig.add_trace(go.Scatter(x=labels, y=loadings[pc], name=pc, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=8, color=color,
                                                                                line=dict(color=t["surface"], width=2))))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    _title(fig, t, "Loadings: a one-standard-deviation move of each PC", f"in model units ({units})")
    fig = _legend(_style(fig, t, 360, date_axis=False))
    fig.update_layout(hovermode="closest")
    return fig


def explained_figure(explained: pd.DataFrame, mp_share: float, theme: str) -> go.Figure:
    t = tokens(theme)
    ex = explained.head(10)
    colors = [t["up"] if a else t["muted"] for a in ex["above_mp"]]
    fig = go.Figure(go.Bar(x=ex.index, y=ex["explained"], marker_color=colors, marker_line_width=0,
                           hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=mp_share, line=dict(color=t["axis"], width=1, dash="dash"),
                  annotation_text="noise edge (Marchenko-Pastur, noise from the eigenvalues below it)", annotation_font=dict(color=t["ink2"], size=11))
    _title(fig, t, "Variance explained per PC", "blue = above the noise edge; gray = indistinguishable from noise")
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, t, 300, date_axis=False)


def scatter_fit_figure(x: pd.Series, y: pd.Series, fitted: pd.Series, theme: str, x_label: str, y_label: str,
                       knots=()) -> go.Figure:
    """y against one regressor with the fitted relation (univariate / hockey stick)."""
    t = tokens(theme)
    ok = x.notna() & y.notna() & fitted.notna()
    x, y, fitted = x[ok], y[ok], fitted[ok]
    order = np.argsort(x.to_numpy())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="observations",
                             marker=dict(size=5, color=t["muted"], opacity=0.6),
                             hovertemplate="%{x:.3f}, %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x.to_numpy()[order], y=fitted.to_numpy()[order], mode="lines", name="fitted",
                             line=dict(color=t["up"], width=2)))
    for k in knots:
        fig.add_vline(x=k, line=dict(color=t["down"], width=1, dash="dot"))
    _title(fig, t, f"{y_label} against {x_label}", "fit sample" + (" · dotted: estimated knot" if len(knots) else ""))
    fig.update_xaxes(title_text=x_label)
    fig = _style(fig, t, 340, date_axis=False)
    fig.update_layout(hovermode="closest")
    return fig


def factor_corr_figure(fc: pd.DataFrame, theme: str) -> go.Figure:
    """Diverging heatmap (blue = +1, gray = 0, orange = -1) of PC-pair correlations by period."""
    t = tokens(theme)
    if fc.empty:
        return empty_figure("Not enough rows per period", theme)
    fc = fc.assign(pair=fc["pc_i"] + "–" + fc["pc_j"], label=pd.to_datetime(fc["period"]).dt.year.astype(str))
    wide = fc.pivot_table(index="pair", columns="label", values="corr")
    fig = go.Figure(go.Heatmap(
        z=wide.to_numpy(), x=list(wide.columns), y=list(wide.index), zmin=-1, zmax=1,
        colorscale=[[0, t["down"]], [0.5, t["grid"]], [1, t["up"]]], xgap=2, ygap=2,
        text=np.round(wide.to_numpy(), 2), texttemplate="%{text}", textfont=dict(color=t["ink"], size=11),
        hovertemplate="%{y} in %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=10)))
    _title(fig, t, "Factor correlation inside each year (fitted loadings)",
           "0 over the fit sample by construction; far from 0 in a year = the factors weren't separate risks then")
    fig = _style(fig, t, 280, date_axis=False)
    fig.update_layout(hovermode="closest")
    return fig
