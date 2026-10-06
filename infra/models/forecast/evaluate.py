"""Out-of-sample evaluation of a directional-forecast spec, and FAMILIES of them with the
point-in-time false-discovery gate (``infra/models/forecast/CLAUDE.md``). Research only.

``evaluate(spec, panel, start, through)``: walk forward, then the named evaluations (swappable,
``EVALUATIONS``) on STAGGERED daily books (``infra.models.autocorr.evaluate.staggered_pnl``):
``benchmark`` (Sharpe of the forecast book, the drift rule and the own-momentum rule),
``clark_west`` (out-of-sample R2 vs the drift forecast and the Clark-West t), ``spanning`` (the
forecast book ~ drift + momentum books jointly: the intercept is what the regressors add),
``subperiods``, placebos ``permutation`` (each regressor column permuted in blocks) and
``time_shift`` (regressors a year back).

``run_family(members, panel_of, start, through)``: every member walked forward with its gates OFF
(fitted mode), its coefficient t per fit date collected, Benjamini-Hochberg across members per fit
date, and each member traded only where its own gates pass AND q <= ``fdr_q``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import norm

from infra.models.autocorr.evaluate import ANNUAL, staggered_pnl
from infra.models.autocorr.family import _spanning_joint_t
from infra.models.event_study.family import benjamini_hochberg
from infra.models.forecast.config import ForecastSpec, get_forecast_spec
from infra.models.forecast.model import ForecastModel, ols_hac


def walk(spec: ForecastSpec, panel: pd.DataFrame, start, through, refit: str = "ME"):
    from infra.models.walk_forward import walk_forward
    res = walk_forward(lambda: ForecastModel(spec), panel, start, through, refit=refit)
    return res, ForecastModel(spec).prepare(panel)


def books(res, prep, spec) -> dict[str, pd.Series]:
    p = res.predictions
    out = {k: staggered_pnl(p[c], prep, spec) for k, c in (("forecast", "signal"), ("drift", "drift_signal"),
                                                          ("momentum", "mom_signal"))}
    first = p.index.min()
    return {k: v[v.index > first] for k, v in out.items()}


def _sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * ANNUAL) if x.std() > 0 else 0.0


def ev_benchmark(ctx):
    b = ctx["books"]
    return {f"sharpe_{k}": _sharpe(v) for k, v in b.items()} | {
        "share_days_on": float((ctx["res"].predictions["signal"] != 0).mean()),
        "share_fits_passed": float(ctx["res"].predictions["passed"].mean())}


def ev_clark_west(ctx):
    res, prep, spec = ctx["res"], ctx["prep"], ctx["spec"]
    st = res.params[res.params["section"] == "stat"].pivot_table(index="fit_as_of", columns="row", values="value")
    p = res.predictions.join(prep[["fwd"]]).dropna(subset=["fwd", "forecast"])
    f1 = st.reindex(pd.to_datetime(p["fit_as_of"]))["drift"].to_numpy() if "drift" in st else np.zeros(len(p))
    f2 = f1 + p["forecast"].to_numpy()
    y = p["fwd"].to_numpy()
    ok = np.isfinite(f1) & np.isfinite(f2)
    y, f1, f2 = y[ok], f1[ok], f2[ok]
    e1, e2 = y - f1, y - f2
    adj = e1 ** 2 - (e2 ** 2 - (f1 - f2) ** 2)
    b, cov = ols_hac(adj, np.ones((len(adj), 1)), spec.horizon_days)
    return {"oos_r2_vs_drift": float(1 - (e2 ** 2).sum() / (e1 ** 2).sum()),
            "clark_west_t": float(b[0] / np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.nan, "n": int(len(y))}


def ev_spanning(ctx):
    b = ctx["books"]
    return {"alpha_t": _spanning_joint_t(b["forecast"], [b["drift"], b["momentum"]], ctx["spec"].horizon_days)}


def ev_subperiods(ctx):
    out = {}
    for lo, hi in (("2010", "2014"), ("2015", "2019"), ("2020", "2022"), ("2023", "2026")):
        v = ctx["books"]["forecast"].loc[lo:hi]
        if len(v) > 60:
            out[f"{lo}-{hi}"] = round(_sharpe(v), 2)
    return out


def _placebo(kind):
    def ev(ctx):
        spec, panel = ctx["spec"], ctx["panel"]
        real = ev_spanning(ctx)["alpha_t"]
        rng = np.random.default_rng(2)
        cols = ForecastModel(spec).names
        ts = []
        for _ in range(spec.n_placebo if kind == "permutation" else 1):
            fake = panel.copy()
            for c in cols:
                if kind == "permutation":
                    v = fake[c].to_numpy()
                    blocks = [v[s:s + spec.placebo_block_days] for s in range(0, len(v), spec.placebo_block_days)]
                    fake[c] = np.concatenate([blocks[o] for o in rng.permutation(len(blocks))])
                else:
                    fake[c] = fake[c].shift(252)
            res, prep = walk(spec, fake, ctx["start"], ctx["through"], ctx["refit"])
            if res.predictions.empty:
                continue
            b = books(res, prep, spec)
            ts.append(_spanning_joint_t(b["forecast"], [b["drift"], b["momentum"]], spec.horizon_days))
        ts = np.array([t for t in ts if np.isfinite(t)])
        if kind == "time_shift":
            return {"placebo_alpha_t": float(ts[0]) if len(ts) else np.nan, "real_alpha_t": real}
        return {"real_alpha_t": real, "p_placebo_ge_real": float((ts >= real).mean()) if len(ts) else np.nan,
                "n": int(len(ts))}
    return ev


EVALUATIONS = {"benchmark": ev_benchmark, "clark_west": ev_clark_west, "spanning": ev_spanning,
               "subperiods": ev_subperiods, "permutation": _placebo("permutation"), "time_shift": _placebo("time_shift")}


def evaluate(spec: ForecastSpec | str, panel: pd.DataFrame, start, through, *, refit: str = "ME", evaluations=None,
             **overrides) -> dict:
    spec = get_forecast_spec(spec, **overrides)
    res, prep = walk(spec, panel, start, through, refit)
    ctx = dict(spec=spec, panel=panel, res=res, prep=prep, books=books(res, prep, spec), start=start,
               through=through, refit=refit)
    out = {}
    for name in (evaluations or spec.evaluations):
        if name not in EVALUATIONS:
            raise KeyError(f"unknown evaluation {name!r}; known: {sorted(EVALUATIONS)}")
        out[name] = EVALUATIONS[name](ctx)
    out["_books"] = ctx["books"]
    return out


def run_family(members: dict[str, ForecastSpec], panel_of, start, through, *, refit: str = "ME",
               fdr_q: float = 0.10) -> dict:
    """``members``: name -> a FITTED single-regressor spec; ``panel_of(spec)`` -> its panel. Returns
    ``fits`` (fit_as_of, member, t, own_pass, q, passed) and ``summary`` per member."""
    fits, raw = [], {}
    for name, spec in members.items():
        free = replace(spec, gates=("min_obs",))                       # coefficients always computed
        res, prep = walk(free, panel_of(spec), start, through, refit)
        if res.predictions.empty:
            continue
        st = res.params[res.params["section"] == "stat"].pivot_table(index="fit_as_of", columns="row", values="value")
        c = ForecastModel(spec).names[0]
        th = dict(spec.thresholds)
        for fa, row in st.iterrows():
            t = row.get(f"t_{c}", np.nan)
            own = row.get("n", 0) >= th.get("min_obs", 150) and np.isfinite(t) and abs(t) >= th.get("coef_t", 2.0)
            fits.append({"fit_as_of": pd.Timestamp(fa), "member": name, "t": t, "own_pass": float(own)})
        raw[name] = (spec, res, prep)
    f = pd.DataFrame(fits)
    f["p"] = 2.0 * (1.0 - norm.cdf(f["t"].abs()))
    f["q"] = np.nan
    for _, g in f.groupby("fit_as_of"):
        f.loc[g.index, "q"] = benjamini_hochberg(g["p"].to_numpy())
    f["passed"] = ((f["own_pass"] == 1) & (f["q"] <= fdr_q)).astype(float)
    rows = []
    for name, (spec, res, prep) in raw.items():
        p = res.predictions
        gate = pd.to_datetime(p["fit_as_of"]).map(f[f["member"] == name].set_index("fit_as_of")["passed"]).fillna(0.0)
        cond = staggered_pnl(p["signal"] * gate.to_numpy(), prep, spec)
        b = books(res, prep, spec)
        cond = cond[cond.index > p.index.min()]
        rows.append({"member": name, "fdr_share": float(gate.mean()), "sharpe_fdr": _sharpe(cond),
                     "alpha_t_fdr": _spanning_joint_t(cond, [b["drift"], b["momentum"]], spec.horizon_days),
                     "sharpe_ungated": _sharpe(b["forecast"]),
                     "alpha_t_ungated": _spanning_joint_t(b["forecast"], [b["drift"], b["momentum"]], spec.horizon_days)})
    return {"fits": f, "summary": pd.DataFrame(rows).set_index("member")}
