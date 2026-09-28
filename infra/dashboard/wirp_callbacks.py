"""Dash callbacks for the WIRP page. Reads directly from cached Database data via
infra.dashboard.wirp_selectors (LIVE or CLOSE, switched by the mode toggle) - never
fetches (CLAUDE.md section 11); this page is a read-only viewer.
"""
from __future__ import annotations

import logging

import pandas as pd
from dash import Dash, Input, Output, html

from infra.analytics.wirp import modal_outcome
from infra.dashboard import wirp_charts
from infra.dashboard.wirp_selectors import build_schedule

log = logging.getLogger(__name__)


def register_wirp_callbacks(app: Dash) -> None:
    @app.callback(
        Output("wirp-chart", "figure"),
        Output("wirp-stats", "children"),
        Output("wirp-status", "children"),
        Input("wirp-mode", "value"),
        Input("theme", "value"),
    )
    def refresh(mode, theme):
        try:
            schedule, meta = build_schedule(mode)
        except Exception as exc:  # a data/config surprise must not crash the page
            log.exception("WIRP schedule build failed")
            return wirp_charts.probability_figure(pd.DataFrame(), {}, theme), [], f"⚠ {type(exc).__name__}: {exc}"

        if schedule.empty:
            return wirp_charts.probability_figure(schedule, meta, theme), [], meta["status"]

        modal = modal_outcome(schedule)
        next_meeting = modal.iloc[0]
        as_of = meta.get("as_of")
        stats_tiles = [
            html.Div(className="kpi", children=[html.Span(label), html.Strong(value)])
            for label, value in [
                ("Next meeting", pd.Timestamp(next_meeting["meeting_date"]).strftime("%Y-%m-%d")),
                ("Most likely outcome", f"{next_meeting['outcome_bps']:+.0f}bp ({next_meeting['probability']:.0%})"),
                ("Implied rate after", f"{next_meeting['implied_rate']:.3f}%"),
                ("Anchor rate", f"{meta['anchor_rate']:.3f}% ({meta['anchor_month']})"),
                ("As of", (as_of.strftime("%Y-%m-%d %H:%M") + " UTC") if as_of is not None else "n/a"),
                ("Meetings priced", f"{schedule['meeting_date'].nunique():,}"),
            ]
        ]
        return wirp_charts.probability_figure(schedule, meta, theme), stats_tiles, meta["status"]
