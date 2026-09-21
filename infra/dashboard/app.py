"""Dash application factory."""
from __future__ import annotations

from pathlib import Path

from dash import Dash

from infra.dashboard.callbacks import register_callbacks
from infra.dashboard.layout import build_layout


def create_app() -> Dash:
    app = Dash(
        __name__,
        title="STIR & Rates",
        assets_folder=str(Path(__file__).parent / "assets"),
    )
    app.layout = build_layout
    register_callbacks(app)
    return app
