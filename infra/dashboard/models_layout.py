"""Static layout of the Models page (component tree only; behaviour in
models_callbacks.py): pick any stored series, a regression or PCA, its parameters and a
fit date, and see the fit, its statistics and the out-of-sample residuals - use case (b)
of infra/models/stats (root CLAUDE.md 25). Read-only: series come from disk through
infra.pipeline.series_panel and are never fetched.
"""
from __future__ import annotations

import dash
import pandas as pd
from dash import dcc, html

from infra.models.stats.config import PCA_METHODS, REGRESSION_METHODS
from infra.dashboard.models_callbacks import RESAMPLES, TRANSFORMS
from infra.pipeline.series_panel import example_ids

DEFAULT_SERIES = ["bond:US_BOND_10y", "bond:US_BOND_2y", "bond:US_BOND_30y"]
DEFAULT_REGIME_SERIES = [f"bond:{c}_BOND_{t}y" for c in ("US", "DE", "UK") for t in (2, 10, 30)]


def _num(id_, value=None, placeholder="", step="any"):
    return dcc.Input(id=id_, type="number", value=value, placeholder=placeholder, step=step, debounce=True)


def build_layout() -> html.Div:
    today = pd.Timestamp.now().normalize()
    return html.Div(className="page", children=[
        html.H2("Models: regression & PCA on any series"),
        html.P(
            "Fit a regression (first series = y, the rest = regressors) or a PCA on any stored "
            "series, using data up to the fit date only, then see the fit carried forward "
            "with its parameters frozen: everything in the shaded region is out of sample. "
            "Series ids are <source>:<key> (bond:, otr:, fut:, stir:, settle:, swap:, repo:, "
            "release:, bar:); anything not in the list can be typed in the extra box.",
            className="page-intro",
        ),
        html.Section(className="controls", children=[
            html.Label(["Model", dcc.RadioItems(id="mdl-kind", value="regression", inline=True,
                                                options=[{"label": "Regression", "value": "regression"},
                                                         {"label": "PCA", "value": "pca"},
                                                         {"label": "Regime PCA", "value": "regime_pca"}])]),
            html.Label(["Method", dcc.Dropdown(id="mdl-method", value="ols", clearable=False,
                                               options=list(REGRESSION_METHODS))]),
            html.Label(className="wide", children=["Series", dcc.Dropdown(
                id="mdl-series", multi=True, value=DEFAULT_SERIES, options=example_ids())]),
            html.Label(["Extra ids (comma separated)", dcc.Input(id="mdl-extra", type="text", debounce=True,
                                                                 placeholder="swap:USD:10y:NY1530")]),
        ]),
        html.Section(className="controls", children=[
            html.Label(["History", dcc.DatePickerRange(id="mdl-range", start_date=(today - pd.DateOffset(years=5)).date(),
                                                       end_date=today.date(), display_format="YYYY-MM-DD")]),
            html.Label(["Fit date (as of)", dcc.DatePickerSingle(id="mdl-asof",
                                                                 date=(today - pd.DateOffset(months=6)).date(),
                                                                 display_format="YYYY-MM-DD")]),
            html.Label(["Transform", dcc.Dropdown(id="mdl-transform", value="change, bp (x100)", clearable=False,
                                                  options=list(TRANSFORMS))]),
            html.Label(["Resample", dcc.Dropdown(id="mdl-resample", value="none", clearable=False,
                                                 options=list(RESAMPLES))]),
            html.Label(["Window (rows, blank = all)", _num("mdl-window", step=1)]),
            html.Label(["Half-life (rows, blank = equal)", _num("mdl-halflife")]),
            dcc.Checklist(id="mdl-flags", value=[], options=[
                {"label": " standardise (z-score)", "value": "zscore"},
                {"label": " diagnostics (slower)", "value": "diag"}]),
        ]),
        html.Section(className="controls", children=[
            html.Label(["Errors", dcc.Dropdown(id="mdl-cov", value="HAC", clearable=False,
                                               options=["nonrobust", "HC1", "HAC"])]),
            html.Label(["Alpha (blank = CV)", _num("mdl-alpha", placeholder="cv")]),
            html.Label(["L1 ratio", _num("mdl-l1", 0.5)]),
            html.Label(["Quantile tau", _num("mdl-tau", 0.5)]),
            html.Label(["Logit threshold", _num("mdl-threshold", 0)]),
            html.Label(["Knots", _num("mdl-knots", 1, step=1)]),
            html.Label(["Components", _num("mdl-ncomp", 3, step=1)]),
            html.Label(["PCA missing data", dcc.Dropdown(id="mdl-missing", value="em", clearable=False,
                                                         options=["em", "pairwise"])]),
            dcc.Checklist(id="mdl-pcascale", value=[], options=[{"label": " correlation PCA", "value": "scale"}]),
        ]),
        html.Section(className="controls", children=[
            html.Label(className="wide", children=["Regime PCA: regime series (N; the Series above are the K)",
                                                   dcc.Dropdown(id="mdl-rseries", multi=True, value=DEFAULT_REGIME_SERIES,
                                                                options=example_ids())]),
            html.Label(["Regimes", _num("mdl-nreg", 2, step=1)]),
            html.Label(["Sticky (pseudo-counts)", _num("mdl-sticky", 0)]),
            html.Label(["Combine", dcc.Dropdown(id="mdl-combine", value="covariance", clearable=False,
                                                options=["covariance", "residual", "robust"])]),
            html.Label(["Timing", dcc.Dropdown(id="mdl-timing", value="predicted", clearable=False,
                                               options=["predicted", "filtered"])]),
            html.Div(className="fetch", children=[html.Button("Fit", id="mdl-fit", n_clicks=0)]),
        ]),
        html.Div(id="mdl-status", className="status", role="status"),
        html.Section(id="mdl-kpis", className="kpis"),
        dcc.Loading(children=[
            dcc.Graph(id="mdl-g1", config={"displaylogo": False}),
            dcc.Graph(id="mdl-g2", config={"displaylogo": False}),
            html.Div(id="mdl-table", className="table-wrap"),
            dcc.Graph(id="mdl-g3", config={"displaylogo": False}),
            dcc.Graph(id="mdl-g4", config={"displaylogo": False}),
        ]),
        html.Div(className="note", children=[
            html.Strong("How to read it"),
            html.P("Parameters are estimated on rows up to the fit date (within the window), frozen, "
                   "and applied unchanged to every later row: the residual after the fit date is "
                   "what a live user of that fit would have seen. For a backtest that refits on a "
                   "schedule, use scripts/run_walk_forward.py (same models, same parameters)."),
            html.P("Regime PCA: a hidden Markov model on the regime series (N) gives each day's "
                   "regime probabilities; one PCA per regime on the Series (K), weighted by them. "
                   "After the fit date each day uses the PREDICTED probability (from the day "
                   "before), so a day's own move can't change the loadings that measure it. "
                   "Sticky > 0 forces persistent regimes (a regime you can't see coming is no use "
                   "for trading)."),
            html.P("Errors: HAC = Newey-West, the default here because daily rates changes and "
                   "especially levels are autocorrelated. Ridge reports errors conditional on its "
                   "alpha; lasso/elastic net report none. Hockey stick: coefficient errors are "
                   "conditional on the estimated knot. PCA in levels is dominated by trends - "
                   "prefer changes."),
        ]),
    ])


dash.register_page(__name__, path="/models", name="Models", layout=build_layout)
