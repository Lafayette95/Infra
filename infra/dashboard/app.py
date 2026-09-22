"""Dash application factory - multi-page (infra.dashboard.shell + Dash Pages).

Page modules register themselves via ``dash.register_page`` at import time (no
``pages/`` folder - kept flat with the rest of infra/dashboard, one file per concern,
matching the project's existing convention). ``dash.register_page`` requires the
``Dash(use_pages=True)`` instance to already exist, so page modules are imported
AFTER construction, before ``app.layout``/callbacks are wired up.
"""
from __future__ import annotations

from pathlib import Path

from dash import Dash


def create_app() -> Dash:
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="",  # pages register themselves manually; no folder-scan needed
        title="STIR & Rates",
        assets_folder=str(Path(__file__).parent / "assets"),
    )

    import infra.dashboard.layout  # noqa: F401  (import triggers dash.register_page)
    import infra.dashboard.rnd_layout  # noqa: F401  (import triggers dash.register_page)
    from infra.dashboard.callbacks import register_callbacks
    from infra.dashboard.rnd_callbacks import register_rnd_callbacks
    from infra.dashboard.shell import build_shell, register_shell_callbacks

    app.layout = build_shell
    register_shell_callbacks(app)
    register_callbacks(app)
    register_rnd_callbacks(app)
    return app
