"""Dash callbacks for the Risk-Neutral Density page. Reads directly from cached
Database/Daily/Options parquet (never through infra.pipeline.daily_options.
load_daily_options, which needs a matching `definitions_day` snapshot - the valuation
day picked here usually isn't that exact day). Never fetches (CLAUDE.md section 9);
this page is a read-only viewer - populate data via scripts/update_daily_options.py.

Strike/density are shown on the /100 "natural" scale (matching the underlying future's
own price scale) rather than the raw stored scale - see CLAUDE.md section 9's scale note.
"""
from __future__ import annotations

import logging

import pandas as pd
from dash import Dash, Input, Output, State, html

from infra.analytics.forward import implied_forward_and_discount
from infra.analytics.rnd import extract_rnd, percentiles
from infra.config import DAILY_OPTIONS_DIR
from infra.dashboard import rnd_charts
from infra.dashboard.rnd_selectors import day_options, expiry_options, underlying_options
from infra.processing import statistics as stats
from infra.storage import parquet_store

log = logging.getLogger(__name__)


def _load_chain(underlying: str, expiry: str, day: str) -> pd.DataFrame:
    """Decoded option rows for one (underlying, expiry, day) - direct from disk."""
    raw = parquet_store.read_partitioned(DAILY_OPTIONS_DIR, equals_in={"underlying": [underlying]})
    if raw is None or raw.empty:
        return stats.empty_daily_options()
    dec = stats.decode_daily_options(raw[stats.DAILY_OPTIONS_COLUMNS])
    chain = dec[(dec["expiry"] == pd.Timestamp(expiry)) & (dec["timestamp"] == pd.Timestamp(day))]
    return chain.reset_index(drop=True)


def register_rnd_callbacks(app: Dash) -> None:
    @app.callback(
        Output("rnd-expiry", "options"),
        Output("rnd-expiry", "value"),
        Input("rnd-underlying", "value"),
    )
    def update_expiry(underlying):
        if not underlying:
            return [], None
        expiries = expiry_options(underlying)
        return expiries, (expiries[0] if expiries else None)

    @app.callback(
        Output("rnd-day", "options"),
        Output("rnd-day", "value"),
        Input("rnd-expiry", "value"),
        State("rnd-underlying", "value"),
    )
    def update_day(expiry, underlying):
        if not (underlying and expiry):
            return [], None
        days = day_options(underlying, expiry)
        return days, (days[-1] if days else None)  # most recent by default

    @app.callback(
        Output("rnd-chart", "figure"),
        Output("rnd-stats", "children"),
        Output("rnd-status", "children"),
        Input("rnd-day", "value"),
        Input("theme", "value"),
        State("rnd-underlying", "value"),
        State("rnd-expiry", "value"),
    )
    def refresh(day, theme, underlying, expiry):
        empty_msg = "No cached option data yet - run scripts/update_daily_options.py first."
        if not (underlying and expiry and day):
            return rnd_charts.density_figure(pd.DataFrame(), {}, theme), [], empty_msg

        try:
            chain = _load_chain(underlying, expiry, day)
            if chain.empty:
                raise ValueError(f"no cached chain for {underlying} expiring {expiry} on {day}")
            forward, discount = implied_forward_and_discount(chain)
            density = extract_rnd(chain)
            p5, p25, p75, p95 = percentiles(density, [0.05, 0.25, 0.75, 0.95])
        except Exception as exc:  # a thin/degenerate chain must not crash the page
            log.exception("RND extraction failed")
            return rnd_charts.density_figure(pd.DataFrame(), {}, theme), [], f"⚠ {type(exc).__name__}: {exc}"

        # Display scale: /100 for strike (matches the future's own price scale, CLAUDE.md
        # section 9), x100 for density so the curve still integrates to 1 on that axis.
        scaled = density.assign(strike=density["strike"] / 100, density=density["density"] * 100)
        meta = {
            "underlying": underlying, "expiry": expiry, "valuation_day": day,
            "forward": forward / 100, "p5": p5 / 100, "p25": p25 / 100, "p75": p75 / 100, "p95": p95 / 100,
        }
        T_days = (pd.Timestamp(expiry) - pd.Timestamp(day)).days
        mode = float(scaled.loc[scaled["density"].idxmax(), "strike"])

        stats_tiles = [
            html.Div(className="kpi", children=[html.Span(label), html.Strong(value)])
            for label, value in [
                ("Forward", f"{meta['forward']:.4f}"),
                ("Mode", f"{mode:.4f}"),
                ("90% range", f"{meta['p5']:.3f} – {meta['p95']:.3f}"),
                ("Discount factor", f"{discount:.5f}"),
                ("Days to expiry", f"{T_days:,}"),
                ("Strikes used", f"{chain['strike'].nunique():,}"),
            ]
        ]
        status = f"{len(chain):,} settlement rows · {chain['strike'].nunique()} strikes"
        return rnd_charts.density_figure(scaled, meta, theme), stats_tiles, status
