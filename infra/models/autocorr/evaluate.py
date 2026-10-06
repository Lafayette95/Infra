"""Out-of-sample evaluation of a conditional-autocorrelation spec: does X add value beyond the
benchmark (the unconditional chase / fade rule)? Research only (``infra/models/autocorr/CLAUDE.md``).

``evaluate(spec, panel, start, through)`` runs the walk-forward, then each evaluation the spec
names (``EVALUATIONS``, swappable). P&L is a STAGGERED daily book (a day's signal holds 1/h of
the position over its own forward window, in units of the target's vol), so it's a real daily
series, not overlapping window sums:

* ``benchmark``: Sharpe of the conditional and the benchmark book, and the benchmark's mean chase;
* ``clark_west``: out-of-sample R2 of the conditional forecast of the chase P&L against the
  benchmark's, and the Clark-West t (nested models: plain MSE comparison is biased against the
  larger model);
* ``spanning``: conditional P&L ~ benchmark P&L (HAC): the intercept is what X adds;
* ``sharpe_diff``: block-bootstrap distribution of Sharpe(conditional) - Sharpe(benchmark);
* ``subperiods``: both Sharpes per sub-period;
* ``permutation`` / ``synthetic_ar1`` / ``time_shift``: PLACEBO X (X's raw series permuted in
  blocks / replaced by an AR(1) with X's persistence and scale / shifted a year back) through the
  same walk-forward: the share of placebos whose spanning t beats the real one is the p-value.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from infra.models.autocorr.config import AutocorrSpec, get_autocorr_spec
from infra.models.autocorr.model import ConditionalAutocorr, ols_hac

ANNUAL = np.sqrt(252.0)


def staggered_pnl(signal: pd.Series, prepared: pd.DataFrame, spec: AutocorrSpec) -> pd.Series:
    """Daily P&L (target-vol units) of holding each day's signal over its forward window."""
    h, g = spec.horizon_days, spec.gap_days
    w = (signal / prepared["vol"]).reindex(prepared.index).fillna(0.0)
    pos = w.rolling(h, min_periods=1).sum().shift(g + 1) / h
    return (pos * prepared["y"]).dropna()


def walk(spec: AutocorrSpec, panel: pd.DataFrame, start, through, refit: str = "ME"):
    from infra.models.walk_forward import walk_forward
    res = walk_forward(lambda: ConditionalAutocorr(spec), panel, start, through, refit=refit)
    prepared = ConditionalAutocorr(spec).prepare(panel)
    return res, prepared


def _sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * ANNUAL) if x.std() > 0 else np.nan


def _books(res, prepared, spec):
    p = res.predictions
    cond = staggered_pnl(p["signal"], prepared, spec)
    bench = staggered_pnl(p["bench_signal"], prepared, spec)
    first = p.index.min()
    return cond[cond.index > first], bench[bench.index > first]


def _spanning_t(cond, bench, lags) -> tuple[float, float, float]:
    """(alpha a year, its t, beta) of conditional ~ benchmark P&L; a conditional book that never
    traded (gates failed throughout) added nothing: alpha 0, t 0."""
    j = pd.concat([cond, bench], axis=1, keys=["c", "b"]).dropna()
    if j.empty or j["c"].abs().sum() == 0:
        return 0.0, 0.0, 0.0
    b, cov = ols_hac(j["c"].to_numpy(), np.c_[np.ones(len(j)), j["b"].to_numpy()], lags)
    return float(b[0] * 252), float(b[0] / np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.nan, float(b[1])


# ------------------------------------------------------------------ evaluations
def ev_benchmark(ctx):
    c, b = ctx["cond"], ctx["bench"]
    return {"sharpe_conditional": _sharpe(c), "sharpe_benchmark": _sharpe(b),
            "share_days_conditioned": float((ctx["res"].predictions["chase_or_fade"] != 0).mean()),
            "share_fits_passed": float(ctx["res"].predictions["passed"].mean())}


def ev_clark_west(ctx):
    res, prep, spec = ctx["res"], ctx["prepared"], ctx["spec"]
    fit = res.params
    stat = fit[fit["section"] == "stat"].pivot_table(index="fit_as_of", columns="row", values="value")
    p = res.predictions.join(prep[["chase"]], how="left").dropna(subset=["chase"])
    st = stat.reindex(pd.to_datetime(p["fit_as_of"])).to_numpy()
    cols = list(stat.columns)
    f1 = st[:, cols.index("bench_mean")] if "bench_mean" in cols else np.zeros(len(p))
    bm = {bk: st[:, cols.index(f"bucket_mean_{bk}")] for bk in (-1, 0, 1) if f"bucket_mean_{bk}" in cols}
    f2 = np.where(p["chase_or_fade"].to_numpy() != 0,
                  np.select([p["bucket"].to_numpy() == bk for bk in bm], [bm[bk] for bk in bm], default=np.nan), f1)
    y = p["chase"].to_numpy()
    ok = np.isfinite(f1) & np.isfinite(f2) & np.isfinite(y)
    y, f1, f2 = y[ok], f1[ok], f2[ok]
    e1, e2 = y - f1, y - f2
    adj = e1 ** 2 - (e2 ** 2 - (f1 - f2) ** 2)
    b, cov = ols_hac(adj, np.ones((len(adj), 1)), spec.horizon_days)
    return {"oos_r2_vs_benchmark": float(1 - (e2 ** 2).sum() / (e1 ** 2).sum()),
            "clark_west_t": float(b[0] / np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else np.nan, "n": int(len(y))}


def ev_spanning(ctx):
    ann, t, beta = _spanning_t(ctx["cond"], ctx["bench"], ctx["spec"].horizon_days)
    return {"alpha_per_year_volunits": ann, "alpha_t": t, "beta_to_benchmark": beta}


def ev_sharpe_diff(ctx, n_boot: int = 500, block: int = 63, seed: int = 0):
    j = pd.concat([ctx["cond"], ctx["bench"]], axis=1, keys=["c", "b"]).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n = len(j)
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=n // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        s = j[idx]
        diffs.append((s[:, 0].mean() / s[:, 0].std() - s[:, 1].mean() / s[:, 1].std()) * ANNUAL)
    diffs = np.array(diffs)
    real = _sharpe(ctx["cond"]) - _sharpe(ctx["bench"])
    return {"sharpe_diff": real, "p_le_zero": float((diffs <= 0).mean())}


def ev_subperiods(ctx):
    out = {}
    for lo, hi in (("2012", "2015"), ("2016", "2019"), ("2020", "2022"), ("2023", "2026")):
        c, b = ctx["cond"].loc[lo:hi], ctx["bench"].loc[lo:hi]
        if len(c) > 60:
            out[f"{lo}-{hi}"] = (round(_sharpe(c), 2), round(_sharpe(b), 2))
    return out


def _placebo(kind: str):
    def ev(ctx):
        spec, panel = ctx["spec"], ctx["panel"]
        real = _spanning_t(ctx["cond"], ctx["bench"], spec.horizon_days)[1]
        rng = np.random.default_rng(1)
        x = panel[spec.x]
        ts = []
        for i in range(spec.n_placebo if kind != "time_shift" else 1):
            fake = panel.copy()
            if kind == "permutation":
                v = x.to_numpy()
                blocks = [v[s:s + spec.placebo_block_days] for s in range(0, len(v), spec.placebo_block_days)]
                order = rng.permutation(len(blocks))
                fake[spec.x] = np.concatenate([blocks[o] for o in order])
            elif kind == "synthetic_ar1":
                v = x.dropna()
                phi = float(np.corrcoef(v.to_numpy()[1:], v.to_numpy()[:-1])[0, 1])
                e = rng.normal(0, v.std() * np.sqrt(max(1 - phi ** 2, 1e-6)), len(x))
                s = np.zeros(len(x))
                for k in range(1, len(x)):
                    s[k] = phi * s[k - 1] + e[k]
                fake[spec.x] = s + float(v.mean())
            elif kind == "time_shift":
                fake[spec.x] = x.shift(252)
            res, prep = walk(spec, fake, ctx["start"], ctx["through"], ctx["refit"])
            if res.predictions.empty:
                continue
            c, b = _books(res, prep, spec)
            ts.append(_spanning_t(c, b, spec.horizon_days)[1])
        ts = np.array([t for t in ts if np.isfinite(t)])
        if kind == "time_shift":
            return {"placebo_alpha_t": float(ts[0]) if len(ts) else np.nan, "real_alpha_t": real}
        return {"real_alpha_t": real, "placebo_median_t": float(np.median(ts)) if len(ts) else np.nan,
                "p_placebo_ge_real": float((ts >= real).mean()) if len(ts) else np.nan, "n": int(len(ts))}
    return ev


EVALUATIONS: dict[str, Callable] = {
    "benchmark": ev_benchmark, "clark_west": ev_clark_west, "spanning": ev_spanning, "sharpe_diff": ev_sharpe_diff,
    "subperiods": ev_subperiods, "permutation": _placebo("permutation"), "synthetic_ar1": _placebo("synthetic_ar1"),
    "time_shift": _placebo("time_shift"),
}


def evaluate(spec: AutocorrSpec | str, panel: pd.DataFrame, start, through, *, refit: str = "ME",
             evaluations=None, **overrides) -> dict:
    spec = get_autocorr_spec(spec, **overrides)
    res, prepared = walk(spec, panel, start, through, refit)
    cond, bench = _books(res, prepared, spec)
    ctx = dict(spec=spec, panel=panel, res=res, prepared=prepared, cond=cond, bench=bench, start=start,
               through=through, refit=refit)
    out = {}
    for name in (evaluations or spec.evaluations):
        if name not in EVALUATIONS:
            raise KeyError(f"unknown evaluation {name!r}; known: {sorted(EVALUATIONS)}")
        out[name] = EVALUATIONS[name](ctx)
    out["_books"] = (cond, bench)
    return out
