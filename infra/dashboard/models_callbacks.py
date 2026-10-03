"""Callbacks of the Models page. ``run_model`` is plain Python (read the panel, build the
spec from the controls, fit at the fit date, predict the whole history with the frozen
fit, optional diagnostics) so it is testable without Dash; the callback only renders."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, html

from infra.dashboard import models_charts as mc
from infra.dashboard.charts import empty_figure
from infra.models.stats import diagnostics as dg
from infra.models.stats.common import pick
from infra.models.stats.config import PCA_METHODS, REGIME_MODELS, REGRESSION_METHODS
from infra.models.stats.pca import make_pca
from infra.models.stats.regime_pca import make_regime_pca
from infra.models.stats.regression import make_regression
from infra.pipeline.series_panel import read_panel

log = logging.getLogger(__name__)

# The page's prep choices -> infra.models.prep steps (here, not in the layout, so this module
# imports without registering the page).
TRANSFORMS = {"level": (), "change": ("diff",), "change, bp (x100)": ("diff", "mult:100"),
              "log change": ("logdiff",), "% change": ("pct",)}
RESAMPLES = {"none": None, "weekly (Fri)": "W-FRI", "month end": "ME"}


def prep_steps(transform: str, resample: str | None, zscore: bool) -> tuple[str, ...]:
    steps = (f"resample:{resample}",) if resample else ()
    steps += TRANSFORMS[transform]
    return steps + (("zscore",) if zscore else ())


def run_model(kind: str, method: str, series: list[str], start, end, as_of, *, prep: tuple[str, ...] = (),
              window=None, halflife=None, options: dict | None = None, diagnostics: bool = False,
              panel: pd.DataFrame | None = None) -> dict:
    """Fit ``kind`` (regression | pca) on ``series`` up to ``as_of`` and predict every row
    of ``[start, end]`` with the frozen fit. ``options``: spec fields. ``panel``: the data
    (default: read from disk). Returns the model, its predictions and diagnostics."""
    options = dict(options or {})
    if len(series) < 2:
        raise ValueError("pick at least two series (regression: y first, then regressors)")
    regime_series = list(dict(options.get("regime_overrides", ())).get("columns", ()))
    panel = read_panel(list(dict.fromkeys(series + regime_series)), start, end) if panel is None else panel
    if panel.empty:
        raise ValueError("no stored data for these series in this range")
    common = dict(prep=prep, window=int(window) if window else None, halflife=float(halflife) if halflife else None)

    def build():
        if kind == "regression":
            return make_regression(method, y=series[0], x=tuple(series[1:]), **common, **options)
        if kind == "regime_pca":
            return make_regime_pca(columns=tuple(series), regime=method, **common, **options)
        return make_pca(method, columns=tuple(series), **common, **options)
    model = build()
    data = model.prepare(panel)
    model.fit(data, as_of=as_of)
    out = model.predict(data)
    if kind == "regression":
        out = model.predict(data, contributions=True)
    res = {"model": model, "data": data, "panel": panel, "out": out}
    if diagnostics and kind != "regime_pca":
        factory = build
        first = data.index.min() + (data.index.max() - data.index.min()) / 3
        if kind == "regression":
            res["rolling"] = dg.rolling_fit(factory, panel, first, data.index.max(), refit="ME")
        else:
            res["factor_corr"] = dg.factor_correlations(model, data, freq="YE")
            res["stability"] = dg.eigenvector_stability(factory, panel, first, data.index.max(), refit="ME")
    return res


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    a = abs(v)
    return f"{v:.2e}" if (a and (a < 1e-3 or a >= 1e5)) else f"{v:.4g}"


def _table(df: pd.DataFrame) -> html.Table:
    head = html.Tr([html.Th(df.index.name or "")] + [html.Th(str(c)) for c in df.columns])
    rows = [html.Tr([html.Td(str(i))] + [html.Td(_fmt(v)) for v in r]) for i, r in zip(df.index, df.to_numpy())]
    return html.Table([html.Thead(head), html.Tbody(rows)])


def _kpis(pairs) -> list:
    return [html.Div(className="kpi", children=[html.Span(k), html.Strong(v)]) for k, v in pairs]


def _short(c: str) -> str:
    return c.split(":", 1)[-1]


def render_regression(res: dict, theme: str):
    m, out = res["model"], res["out"]
    f, st = m.fitted_, m.fitted_.stats
    binary = m.binary
    oos = out[~out["in_sample"]]
    kp = [("n (fit)", _fmt(st.get("n"))), ("as of", f.as_of.strftime("%Y-%m-%d"))]
    if binary:
        kp += [("pseudo R²", _fmt(st.get("pseudo_r2"))), ("AUC", _fmt(st.get("auc"))), ("LR p", _fmt(st.get("lr_p")))]
    else:
        kp += [("R²", _fmt(st.get("r2"))), ("adj R²", _fmt(st.get("adj_r2"))), ("F", _fmt(st.get("F"))),
               ("resid sd", _fmt(float(np.nanstd(out.loc[out["in_sample"], "residual"])))),
               ("half-life (rows)", _fmt(st.get("half_life"))), ("ADF t", _fmt(st.get("adf_t"))),
               ("DW", _fmt(st.get("dw")))]
    if len(oos):
        kp.append(("out-of-sample RMSE", _fmt(float(np.sqrt(np.nanmean(oos["residual"] ** 2))))))
        kp.append(("latest z", _fmt(float(out["resid_z"].dropna().iloc[-1])) if out["resid_z"].notna().any() else "–"))
    for k in ("alpha", "knot", "slope_left", "slope_right", "q", "share_downweighted"):
        if k in st:
            kp.append((k, _fmt(st[k])))
    label = _short(f.target)
    # daily changes overplot into noise: show them CUMULATED (the hedged P&L view)
    cumulative = not binary and any(s.split(":")[0] in ("diff", "logdiff", "pct") for s in m.spec.prep)
    shown = out[["y", "fitted"]].fillna(0.0).cumsum() if cumulative else out
    g1 = mc.fit_figure(shown, f.as_of, theme, label + (" (cumulated changes)" if cumulative else ""), binary=binary)
    g2 = mc.residual_figure(out, f.as_of, theme, label=label)
    table = f.table.drop(columns=[c for c in ("ci_low", "ci_high") if c in f.table]).rename(
        index=_short)
    table.index.name = "term"
    if any(c.startswith("beta:") for c in out.columns):
        betas = pick(out, "beta").rename(columns=_short)
        g3 = mc.lines_figure(betas.drop(columns=["const"], errors="ignore"), theme, "Filtered coefficients",
                             "Kalman: the coefficient as known at each row", as_of=f.as_of)
    elif len(f.regressors) == 1 or m.method == "hockey":
        kv = m.spec.knot_var or f.regressors[0]
        sample = out[out["in_sample"]]
        x = m.prep.inverse(res["data"].loc[sample.index, kv], kv) if kv in res["data"] else sample["y"] * np.nan
        knots = [st[k] for k in st if k == "knot" or (k.startswith("knot") and k[4:].isdigit())]
        g3 = mc.scatter_fit_figure(x, sample["y"], sample["fitted"], theme, _short(kv), label, knots)
    else:
        contrib = pick(out, "contrib").rename(columns=_short).drop(columns=["const"], errors="ignore")
        if cumulative:
            contrib = contrib.fillna(0.0).cumsum()
        g3 = mc.lines_figure(contrib, theme, "Contribution of each regressor to the fitted value"
                             + (" (cumulated)" if cumulative else ""),
                             "coefficient × regressor, in y's units", as_of=f.as_of)
    rolling = res.get("rolling")
    if rolling is not None and not rolling.empty:
        coefs = rolling[[c for c in rolling.columns if c.startswith("coef:") and c != "coef:const"]]
        g4 = mc.lines_figure(coefs.rename(columns=lambda c: _short(c.split(":", 1)[1])), theme,
                             "Coefficients refitted every month (stability)",
                             "same spec and window, each fit on data up to its own date")
    else:
        g4 = empty_figure("Tick 'diagnostics' for coefficient stability over time", theme)
    return kp, table, g1, g2, g3, g4


def render_pca(res: dict, theme: str):
    m, out = res["model"], res["out"]
    f, st = m.fitted_, m.fitted_.stats
    units = " ".join(m.spec.prep) or "levels"
    kp = [("n (fit)", _fmt(st.get("n"))), ("as of", f.as_of.strftime("%Y-%m-%d")),
          ("explained by k", f"{st['explained_k']:.1%}"), ("PCs above noise edge", _fmt(st.get("n_above_mp"))),
          ("eigen gap k/k+1", _fmt(st.get("gap_k")))]
    if "iterations" in st:
        kp += [("EM iterations", _fmt(int(st["iterations"]))), ("missing share", f"{st.get('missing_share', 0):.1%}")]
    g1 = mc.loadings_figure(m.loadings_units.rename(index=_short), theme, units)
    total = float(f.eigenvalues.sum())
    g2 = mc.explained_figure(m.explained, st["mp_edge"] / total if total else np.nan, theme)
    table = m.loadings_units.round(6)
    table.index = [_short(c) for c in table.index]
    table.index.name = "loading (1 sd)"
    res_z = pick(out, "resid_z").rename(columns=_short)
    scores = pick(out, "score")
    fc = res.get("factor_corr")
    if fc is not None:
        g3 = mc.factor_corr_figure(fc, theme)
    else:
        g3 = mc.lines_figure(scores, theme, "Factor scores", "projection of each row on the frozen loadings",
                             as_of=f.as_of)
    stab = res.get("stability")
    if stab is not None and not stab.empty:
        cos = stab.pivot_table(index="fit_as_of", columns="pc", values="cos_ref")
        g4 = mc.lines_figure(cos, theme, "Eigenvector stability: |cos| with the first monthly refit",
                             "1 = same direction; a drop with an eigen gap near 1 means the PC is not identified")
    else:
        g4 = mc.lines_figure(res_z, theme, "Residual z per series (x − reconstruction from k PCs)",
                             "rich/cheap vs the factor model; tick 'diagnostics' for stability", as_of=f.as_of)
    return kp, table, g1, g2, g3, g4


def render_regime_pca(res: dict, theme: str):
    m, out = res["model"], res["out"]
    f, st = m.fitted_, m.fitted_.stats
    R = m.regime.n_regimes
    probs = pick(out, "p")
    kp = [("n (fit)", _fmt(st.get("n"))), ("as of", f.as_of.strftime("%Y-%m-%d")),
          ("pooled explained", f"{st['explained_k_pooled']:.1%}")]
    for r in range(R):
        kp.append((f"R{r}: share / mean days", f"{st[f'occupancy_R{r}']:.0%} / {_fmt(st.get(f'regime.duration_R{r}'))}"))
    if len(probs):
        kp.append(("latest regime probs", " / ".join(f"{v:.0%}" for v in probs.iloc[-1])))
    g1 = mc.lines_figure(probs, theme, "Regime probabilities",
                         "smoothed up to the fit date; after it, the probability used that day "
                         f"({m.spec.timing})", as_of=f.as_of, height=300)
    z = pick(out, "resid_z").rename(columns=_short)
    g2 = mc.lines_figure(z, theme, "Residual z per series (regime-blended model)",
                         f"combine = {m.spec.combine}; z over the model-implied residual sd that day", as_of=f.as_of)
    lb = m.loadings_by_regime()
    table = lb.pivot_table(index=["regime", "pc"], columns="column", values="loading", sort=False).rename(columns=_short)
    table.index = [f"{r} {pc}" for r, pc in table.index]
    table.index.name = "loading (1 sd)"
    wide = lb.assign(series=lb["regime"] + " " + lb["pc"]).pivot_table(index="column", columns="series",
                                                                      values="loading", sort=False)
    wide.index = [_short(c) for c in wide.index]
    g3 = mc.loadings_figure(wide, theme, " ".join(m.spec.prep) or "levels")
    g3.update_layout(title=dict(text="Loadings by regime: a one-sd move of each PC in each regime"))
    scores = pick(out, "score")
    g4 = mc.lines_figure(scores, theme, "Factor scores (regime-blended loadings)", as_of=f.as_of)
    return kp, table, g1, g2, g3, g4


def register_models_callbacks(app: Dash) -> None:
    @app.callback(Output("mdl-method", "options"), Output("mdl-method", "value"), Input("mdl-kind", "value"))
    def methods(kind):
        opts = {"regression": list(REGRESSION_METHODS), "pca": list(PCA_METHODS),
                "regime_pca": list(REGIME_MODELS)}[kind]
        return opts, opts[0]

    @app.callback(
        Output("mdl-g1", "figure"), Output("mdl-g2", "figure"), Output("mdl-g3", "figure"),
        Output("mdl-g4", "figure"), Output("mdl-table", "children"), Output("mdl-kpis", "children"),
        Output("mdl-status", "children"),
        Input("mdl-fit", "n_clicks"), Input("theme", "value"),
        State("mdl-kind", "value"), State("mdl-method", "value"), State("mdl-series", "value"),
        State("mdl-extra", "value"), State("mdl-range", "start_date"), State("mdl-range", "end_date"),
        State("mdl-asof", "date"), State("mdl-transform", "value"), State("mdl-resample", "value"),
        State("mdl-window", "value"), State("mdl-halflife", "value"), State("mdl-flags", "value"),
        State("mdl-cov", "value"), State("mdl-alpha", "value"), State("mdl-l1", "value"), State("mdl-tau", "value"),
        State("mdl-threshold", "value"), State("mdl-knots", "value"), State("mdl-ncomp", "value"),
        State("mdl-missing", "value"), State("mdl-pcascale", "value"), State("mdl-rseries", "value"),
        State("mdl-nreg", "value"), State("mdl-sticky", "value"), State("mdl-combine", "value"),
        State("mdl-timing", "value"),
    )
    def fit(n_clicks, theme, kind, method, series, extra, start, end, as_of, transform, resample, window, halflife,
            flags, cov, alpha, l1, tau, threshold, knots, ncomp, missing, pcascale, rseries, nreg, sticky, combine,
            timing):
        blank = empty_figure("", theme)  # the defaults are fitted on load; Fit refits with the controls
        valid = {"regression": REGRESSION_METHODS, "pca": PCA_METHODS, "regime_pca": tuple(REGIME_MODELS)}[kind]
        if method not in valid:  # the method list may not have caught up with a kind switch yet
            method = valid[0]
        ids = list(series or []) + [s.strip() for s in (extra or "").split(",") if s.strip()]
        flags = flags or []
        prep = prep_steps(transform, RESAMPLES[resample], "zscore" in flags)
        if kind == "regression":
            options = {"cov": cov, "alpha": "cv" if alpha in (None, "") else float(alpha),
                       "l1_ratio": float(l1 if l1 is not None else 0.5), "tau": float(tau if tau is not None else 0.5),
                       "n_knots": int(knots or 1)}
            if method in ("logit", "probit"):
                options["threshold"] = float(threshold if threshold is not None else 0.0)
        elif kind == "regime_pca":
            options = {"n_components": int(ncomp or 2), "scale": "scale" in (pcascale or []), "combine": combine,
                       "timing": timing, "missing": "em" if missing == "em" else "drop",
                       "regime_overrides": (("columns", tuple(rseries or ())), ("n_regimes", int(nreg or 2)),
                                            ("sticky", float(sticky or 0.0)))}
        else:
            options = {"n_components": int(ncomp or 3), "scale": "scale" in (pcascale or [])}
            if method == "missing":
                options["missing"] = missing
            if method == "weighted" and not halflife:
                halflife = 126
        try:
            res = run_model(kind, method, ids, start, end, as_of, prep=prep, window=window, halflife=halflife,
                            options=options, diagnostics="diag" in flags)
            render = {"regression": render_regression, "pca": render_pca, "regime_pca": render_regime_pca}[kind]
            kp, table, g1, g2, g3, g4 = render(res, theme)
        except Exception as exc:  # a bad selection must not crash the page
            log.exception("model fit failed")
            msg = empty_figure(f"{type(exc).__name__}: {exc}", theme)
            return msg, blank, blank, blank, None, [], f"⚠ {type(exc).__name__}: {exc}"
        m = res["model"]
        status = (f"{type(m).__name__} on {len(res['data'])} rows ({res['data'].index.min():%Y-%m-%d} – "
                  f"{res['data'].index.max():%Y-%m-%d}), prep {', '.join(prep) or 'none'}; fit on rows up to "
                  f"{m.fitted_.as_of:%Y-%m-%d}.")
        return g1, g2, g3, g4, _table(table), _kpis(kp), status
