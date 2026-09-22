"""The persistent app shell for a multi-page Dash app: nav + theme toggle, wrapping
``dash.page_container``. Registered pages (infra.dashboard.layout,
infra.dashboard.rnd_layout) supply only their own content - the theme control lives
here, once, so it persists across page navigation instead of being duplicated per page
(which would also collide on the "theme"/"root" component ids Dash Pages shares
across the whole app).
"""
from __future__ import annotations

import dash
from dash import Dash, Input, Output, dcc, html


def build_shell() -> html.Div:
    return html.Div(id="root", className="theme-light", children=[
        html.Header(className="bar", children=[
            html.Div(className="nav", children=[
                dcc.Link(page["name"], href=page["relative_path"], className="nav-link")
                for page in dash.page_registry.values()
            ]),
            dcc.RadioItems(
                id="theme", value="light", inline=True, className="theme-toggle",
                options=[{"label": "Light", "value": "light"}, {"label": "Dark", "value": "dark"}],
            ),
        ]),
        dash.page_container,
    ])


def register_shell_callbacks(app: Dash) -> None:
    @app.callback(Output("root", "className"), Input("theme", "value"))
    def set_theme_class(theme: str) -> str:
        return f"theme-{theme}"
