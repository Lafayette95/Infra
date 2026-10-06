"""Conditional autocorrelation (framework B1): does a third variable X decide whether an
instrument's price action continues (chase) or reverts (fade)? ``prepare`` -> ``fit`` ->
``predict`` (infra/models/CLAUDE.md 0); methodology ``infra/models/autocorr/CLAUDE.md``.

Per information day D (everything from data known at D):
* ``past`` = the target's k-day move through D / (vol x sqrt k); ``fwd`` = its h-day move over
  the window starting ``gap_days`` after D / (vol x sqrt h) - vol = EWMA of daily moves at D;
* ``chase = sign(past) x fwd``: the P&L of chasing the past move - the benchmark asks whether
  its mean is > 0 (momentum) or < 0 (reversal) unconditionally; B1 asks whether it depends on X;
* X's feature at D (``x_feature``) and its bucket -1 / 0 / +1 (``x_partition``, trailing);
* controls: the vol regime (vol z-scored against its trailing year) and |past|.

``fit(as_of)`` uses only windows whose forward move ENDED before ``as_of``'s day (its last move
is then public). Statistics (HAC errors, lags = h): the benchmark mean chase; chase ~ X (the
interaction ``d``); per-bucket means and the top-minus-bottom spread; X's t beyond the controls;
whether the two halves agree. The spec's GATES decide whether the conditioning is used; the
signal map is then chase (+1) / fade (-1) per bucket, by the sign of its mean; if a gate fails,
``fallback``: flat, or the benchmark's own rule. ``predict``: ``signal = sign(past) x map[bucket]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infra.models.autocorr.config import AutocorrSpec, get_autocorr_spec
from infra.models.autocorr.gates import run_gates
from infra.models.base import Model
from infra.models.stats.common import PARAM_COLUMNS
from infra.models.stats.inference import sandwich

PREP_COLUMNS = ["y", "vol", "past", "fwd", "fwd_bp", "fwd_end", "chase", "xf", "bucket", "pbucket", "vol_z",
                "abs_past"]
CELLS = [(xb, pb) for xb in (-1, 0, 1) for pb in (-1, 0, 1)]


def cell_key(xb, pb) -> str:
    return f"{int(xb)}|{int(pb)}"


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their Newey-West covariance."""
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    return beta, sandwich(X, y - X @ beta, bread, "HAC", lags=lags)


def _t(beta, cov, i) -> float:
    se = np.sqrt(cov[i, i])
    return float(beta[i] / se) if se > 0 else np.nan


@dataclass
class AutocorrFit:
    as_of: pd.Timestamp
    stats: dict
    gates: dict
    passed: bool
    signal_map: dict = field(default_factory=dict)   # bucket -> +1 chase / -1 fade / 0
    bench_sign: float = 0.0


class ConditionalAutocorr(Model):
    """See the module docstring."""

    def __init__(self, spec: AutocorrSpec | str | None = None, **overrides):
        self.spec = get_autocorr_spec(spec, **overrides)
        self.fitted_: AutocorrFit | None = None

    @property
    def is_fitted(self) -> bool:
        return self.fitted_ is not None

    # ------------------------------------------------------------------ data
    def read_panel(self, start, end) -> pd.DataFrame:
        from infra.pipeline.series_panel import read_panel
        return read_panel([self.spec.target, self.spec.x], start, end)

    LEGACY_FEATURES = ("move", "absmove", "level", "z")

    def x_grammar(self) -> tuple[str, str]:
        """(grammar, input kind) of X's feature in the central feature maker
        (``infra.processing.features``). Legacy names stay: ``move:K`` = ``chg:K|norm:vol:<vol_span>``,
        ``absmove:K`` = that ``|abs``, ``level`` = the raw values, ``z:W`` = ``lvl|norm:z:W``; any
        other string is the grammar itself."""
        sp = self.spec
        kind, *args = sp.x_feature.split(":")
        moves = "moves" if sp.x_is_moves else "level"
        if kind == "move":
            return f"chg:{args[0]}|norm:vol:{sp.vol_span}", moves
        if kind == "absmove":
            return f"chg:{args[0]}|norm:vol:{sp.vol_span}|abs", moves
        if kind == "level":
            return "lvl", "level"
        if kind == "z":
            return f"lvl|norm:z:{args[0]}", "level"
        return sp.x_feature, moves

    def _x_feature(self, xs: pd.Series) -> pd.Series:
        from infra.processing.features import apply
        grammar, kind = self.x_grammar()
        return apply(xs, grammar, kind)

    # ------------------------------------------------------------------ 1. prepare
    def prepare(self, raw: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = self.spec
        y = raw[sp.target].dropna()
        y.index = pd.DatetimeIndex(y.index).normalize()
        y = y[~y.index.duplicated(keep="last")].sort_index()
        xs = raw[sp.x]
        xs.index = pd.DatetimeIndex(xs.index).normalize()
        xs = xs[~xs.index.duplicated(keep="last")].sort_index()
        xs = xs.reindex(y.index) if sp.x_is_moves else xs.reindex(xs.index.union(y.index)).ffill().reindex(y.index)
        k, h, g = sp.past_days, sp.horizon_days, sp.gap_days
        from infra.processing.features import apply
        vol = y.ewm(span=sp.vol_span, min_periods=max(sp.vol_span // 2, 10)).std()
        past = apply(y, f"chg:{k}|norm:vol:{sp.vol_span}", "moves")     # the central maker
        fwd_sum = y.rolling(h, min_periods=h).sum().shift(-(g + h))
        fwd = fwd_sum / (vol * np.sqrt(h))
        days = pd.Series(y.index, index=y.index)
        fwd_end = days.shift(-(g + h))
        xf = self._x_feature(xs)
        from infra.processing.features import partition
        bucket = pd.Series(partition(xf, sp.x_partition), index=y.index)
        vol_z = (vol - vol.rolling(252, min_periods=60).mean()) / vol.rolling(252, min_periods=60).std()
        pbucket = pd.Series(partition(past, sp.past_partition), index=y.index)
        out = pd.DataFrame({"y": y, "vol": vol, "past": past, "fwd": fwd, "fwd_bp": fwd_sum, "fwd_end": fwd_end,
                            "chase": np.sign(past) * fwd, "xf": xf, "bucket": bucket, "pbucket": pbucket,
                            "vol_z": vol_z, "abs_past": past.abs()}, index=y.index)
        out.index.name = "timestamp"
        return out[PREP_COLUMNS]

    # ------------------------------------------------------------------ 2. fit
    def _sample(self, data: pd.DataFrame, as_of) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of).normalize()
        d = data[pd.to_datetime(data["fwd_end"]) < as_of]          # the forward move is public by as_of
        if self.spec.window:
            d = d[d.index > as_of - pd.Timedelta(self.spec.window)]
        return d.dropna(subset=["chase", "xf", "bucket", "past"])

    def fit(self, prepared: pd.DataFrame, as_of=None) -> "ConditionalAutocorr":
        sp = self.spec
        as_of = pd.Timestamp(prepared.index.max() if as_of is None else as_of)
        d = self._sample(prepared, as_of)
        n = len(d)
        lags = sp.hac_lags if sp.hac_lags is not None else sp.horizon_days
        stats: dict = {"n": float(n)}
        if n >= 20:
            c = d["chase"].to_numpy()
            ones = np.ones((n, 1))
            b, cov = ols_hac(c, ones, lags)
            stats.update(bench_mean=float(b[0]), bench_t=_t(b, cov, 0))
            xm, xsd = float(d["xf"].mean()), float(d["xf"].std())
            xz = ((d["xf"] - xm) / xsd).to_numpy() if xsd > 0 else np.zeros(n)
            stats.update(x_mean=xm, x_sd=xsd)
            b, cov = ols_hac(c, np.c_[ones, xz], lags)
            stats.update(x_coef=float(b[1]), x_t=_t(b, cov, 1))
            dp = (d["bucket"] == 1.0).to_numpy(dtype=float)
            dm = (d["bucket"] == -1.0).to_numpy(dtype=float)
            b, cov = ols_hac(c, np.c_[ones, dp, dm], lags)
            spread = float(b[1] - b[2])
            v = cov[1, 1] + cov[2, 2] - 2 * cov[1, 2]
            stats.update(spread=spread, spread_t=spread / np.sqrt(v) if v > 0 else np.nan)
            for bk in (-1, 0, 1):
                m = d["bucket"] == float(bk)
                stats[f"bucket_mean_{bk}"] = float(d.loc[m, "chase"].mean()) if m.any() else np.nan
                stats[f"bucket_n_{bk}"] = float(m.sum())
            ctrl = {"vol": d["vol_z"], "abs_past": d["abs_past"]}
            cols = [ctrl[k].fillna(0.0).to_numpy() for k in sp.controls]
            Xc = np.c_[ones, *[(c_ - c_.mean()) / (c_.std() or 1.0) for c_ in cols], xz] if cols else np.c_[ones, xz]
            b, cov = ols_hac(c, Xc, lags)
            stats["ctrl_t"] = _t(b, cov, Xc.shape[1] - 1)
            half = n // 2
            s1 = d.iloc[:half]
            s2 = d.iloc[half:]
            sp1 = s1.loc[s1.bucket == 1, "chase"].mean() - s1.loc[s1.bucket == -1, "chase"].mean()
            sp2 = s2.loc[s2.bucket == 1, "chase"].mean() - s2.loc[s2.bucket == -1, "chase"].mean()
            stats["halves"] = float(np.sign(sp1) == np.sign(sp2)) if np.isfinite(sp1) and np.isfinite(sp2) else 0.0
        if sp.fixed_map is not None:                    # a no-fit rule: report the statistics, impose the map
            if n >= 20:
                for xb in (-1, 0, 1):
                    m = d["bucket"] == float(xb)
                    stats[f"row_mean_{xb}"] = float(d.loc[m, "fwd"].mean()) if m.any() else np.nan
            self.fitted_ = AutocorrFit(as_of, stats, {}, True, {float(k): float(v) for k, v in sp.fixed_map},
                                       float(np.sign(stats.get("bench_mean", 0.0))))
            return self
        if n >= 20:
            for xb in (-1, 0, 1):                       # X's own direction: mean forward move per X bucket
                m = d["bucket"] == float(xb)
                stats[f"row_mean_{xb}"] = float(d.loc[m, "fwd"].mean()) if m.any() else np.nan
        if sp.mode == "cells" and n >= 20:
            self._fit_cells(d, stats, lags)
        gates, passed = run_gates(stats, sp.gates, dict(sp.thresholds)) if n >= 20 else ({}, False)
        bench_sign = float(np.sign(stats.get("bench_mean", 0.0)))
        if sp.mode == "cells":
            th = dict(sp.thresholds)
            smap = {}
            for xb, pb in CELLS:
                k = cell_key(xb, pb)
                t, cnt, mean = stats.get(f"cell_t_{k}", np.nan), stats.get(f"cell_n_{k}", 0.0), stats.get(f"cell_mean_{k}", np.nan)
                ok = passed and np.isfinite(t) and abs(t) >= th.get("cell_t", 2.0) and cnt >= th.get("cell_min_obs", 40.0)
                smap[k] = float(np.sign(mean)) if ok else 0.0
            passed = passed and any(v != 0 for v in smap.values())
            self.fitted_ = AutocorrFit(as_of, stats, gates, passed, smap, bench_sign)
            return self
        if passed:
            smap = {bk: float(np.sign(stats.get(f"bucket_mean_{bk}", 0.0) or 0.0)) for bk in (-1, 0, 1)}
        elif sp.fallback == "benchmark":
            smap = {bk: bench_sign for bk in (-1, 0, 1)}
        else:
            smap = {bk: 0.0 for bk in (-1, 0, 1)}
        self.fitted_ = AutocorrFit(as_of, stats, gates, passed, smap, bench_sign)
        return self

    def _fit_cells(self, d: pd.DataFrame, stats: dict, lags: int) -> None:
        """3x3 cells (X bucket x past-move bucket): the raw mean forward move (the traded sign),
        its EXCESS over the null (the unconditional mean, or the X row's with ``demean_by_x``), count,
        and the excess's HAC t from one dummy regression over the whole sample."""
        dd = d.dropna(subset=["pbucket", "fwd"])
        raw = dd["fwd"].to_numpy(dtype="float64")
        # the null is "this cell = the target's UNCONDITIONAL mean" (or its X row's, demean_by_x), never
        # "= 0": against 0, a target that merely trended makes every cell look significant (found
        # 2026-10-06 on the duration-flies cells: 52 'discoveries' a fit that added nothing out of sample)
        base = dd.groupby("bucket")["fwd"].transform("mean").to_numpy() if self.spec.demean_by_x else raw.mean()
        y = raw - base
        keys = [cell_key(xb, pb) for xb, pb in CELLS]
        D = np.column_stack([((dd["bucket"] == xb) & (dd["pbucket"] == pb)).to_numpy(dtype=float) for xb, pb in CELLS])
        have = D.sum(axis=0) > 0
        if have.sum() == 0:
            return
        b, cov = ols_hac(y, D[:, have], lags)
        j = 0
        for i, k in enumerate(keys):
            stats[f"cell_n_{k}"] = float(D[:, i].sum())
            if have[i]:
                m = D[:, i] == 1
                # traded sign = the cell's RAW mean forward move; significance = its excess over the null
                stats[f"cell_mean_{k}"], stats[f"cell_excess_{k}"], stats[f"cell_t_{k}"] = (
                    float(raw[m].mean()), float(b[j]), _t(b, cov, j))
                j += 1

    # ------------------------------------------------------------------ 3. predict
    def predict(self, prepared: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
        """Rows in ``(start, end]``: ``signal`` (+1 long / -1 short the target), the benchmark's
        own ``bench_signal``, ``chase_or_fade`` of the current bucket, ``expected_bp`` over the
        forward window (bucket mean x vol x sqrt h, signed)."""
        self.check_fitted()
        f, sp = self.fitted_, self.spec
        d = prepared
        if start is not None:
            d = d[d.index > pd.Timestamp(start)]
        if end is not None:
            d = d[d.index <= pd.Timestamp(end)]
        s_past = np.sign(d["past"]).fillna(0.0)
        xdir = d["bucket"].map({float(bk): float(np.sign(f.stats.get(f"row_mean_{bk}", 0.0) or 0.0))
                                for bk in (-1, 0, 1)}).fillna(0.0)
        if sp.mode == "cells":
            keys = [cell_key(x, p) if np.isfinite(x) and np.isfinite(p) else "" for x, p in zip(d["bucket"], d["pbucket"])]
            sig = pd.Series([f.signal_map.get(k, 0.0) for k in keys], index=d.index)
            mean_c = pd.Series([f.stats.get(f"cell_mean_{k}", np.nan) for k in keys], index=d.index)
            return pd.DataFrame({
                "past": d["past"], "bucket": d["bucket"], "pbucket": d["pbucket"], "chase_or_fade": sig,
                "signal": sig, "bench_signal": s_past * f.bench_sign, "xdir_signal": xdir,
                "expected_bp": (mean_c * d["vol"] * np.sqrt(sp.horizon_days)).where(sig != 0, 0.0),
                "passed": float(f.passed)}, index=d.index)
        cf = d["bucket"].map(f.signal_map).fillna(0.0)
        mean_b = d["bucket"].map({bk: f.stats.get(f"bucket_mean_{bk}", np.nan) for bk in (-1, 0, 1)})
        out = pd.DataFrame({
            "past": d["past"], "bucket": d["bucket"], "pbucket": d["pbucket"], "chase_or_fade": cf,
            "signal": s_past * cf, "bench_signal": s_past * f.bench_sign, "xdir_signal": xdir,
            "expected_bp": (s_past * mean_b * d["vol"] * np.sqrt(sp.horizon_days)).where(cf != 0, 0.0),
            "passed": float(f.passed)}, index=d.index)
        return out

    # ------------------------------------------------------------------ params
    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f = self.fitted_
        rows = [{"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan},
                {"section": "fit", "row": "passed", "col": "value", "value": float(f.passed)},
                {"section": "fit", "row": "bench_sign", "col": "value", "value": f.bench_sign}]
        rows += [{"section": "stat", "row": k, "col": "value", "value": float(v)} for k, v in f.stats.items()]
        rows += [{"section": "gate", "row": k, "col": "value", "value": float(v)} for k, v in f.gates.items()]
        rows += [{"section": "map", "row": k if isinstance(k, str) else str(int(k)), "col": "value", "value": float(v)}
                 for k, v in f.signal_map.items()]
        return pd.DataFrame(rows)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec: AutocorrSpec | str | None = None, **overrides):
        m = cls(spec, **overrides)
        get = lambda sec: params[params["section"] == sec]  # noqa: E731
        fit = get("fit").set_index("row")["value"]
        m.fitted_ = AutocorrFit(
            as_of=pd.Timestamp(get("meta").loc[lambda d: d["row"] == "as_of", "col"].iloc[0]),
            stats=get("stat").set_index("row")["value"].to_dict(),
            gates=get("gate").set_index("row")["value"].to_dict(),
            passed=bool(fit["passed"]),
            signal_map={(k if "|" in str(k) else float(k)): float(v)
                        for k, v in get("map").set_index("row")["value"].items()},
            bench_sign=float(fit["bench_sign"]))
        return m


def make_autocorr(spec: AutocorrSpec | str | None = None, **overrides) -> ConditionalAutocorr:
    return ConditionalAutocorr(spec, **overrides)
