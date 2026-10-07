"""Directional forecasts (framework A): do point-in-time features predict an instrument's forward
move? ``prepare`` -> ``fit`` -> ``predict`` (infra/models/CLAUDE.md 0); methodology
``infra/models/forecast/CLAUDE.md``.

Rows are the target's days D. The DECISION is taken on day D + ``gap_days`` at ``decision_time``
New York; every regressor is the feature maker's value AVAILABLE by then (``align``, with the
regressor's carry limit and fill); the forward move is the target's sum over the ``horizon_days``
closes after the decision day's close, / (EWMA vol at D x sqrt h).

``fit(as_of)`` uses rows whose forward window ended before ``as_of``'s day: OLS with HAC errors
(lags = h) of the forward move on [1, regressors] - always computed, for the statistics; the FIT MODE
decides the coefficients used: ``fitted`` (the estimate, if the gates pass, else flat), ``exante``
(the spec's weights - no fit), ``prior`` (ridge, then any coefficient on the wrong side of its
stated sign set to 0). The intercept (drift) never enters the forecast - it is the benchmark.

``predict``: ``forecast`` (vol units), ``signal`` = forecast / (2 x its in-sample sd) clipped to
[-1, 1], ``expected_bp``, and the benchmarks' signals: ``drift_signal`` (the sign of the fit
sample's mean forward move) and ``mom_signal`` (the sign of the target's own past h-day move).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infra.models.base import Model
from infra.models.forecast.config import ForecastSpec, get_forecast_spec
from infra.models.stats.common import PARAM_COLUMNS
from infra.models.stats.inference import sandwich

NY = "America/New_York"


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(coefficients, HAC covariance); ``ridge`` x n penalises every coefficient but the intercept."""
    n, k = X.shape
    pen = np.eye(k) * ridge * n
    pen[0, 0] = 0.0
    bread = np.linalg.pinv(X.T @ X + pen)
    beta = bread @ X.T @ y
    return beta, sandwich(X, y - X @ beta, bread, "HAC", lags=lags)


GATES = {
    "min_obs": lambda st, th: (st["n"], st["n"] >= th),
    "coef_t": lambda st, th: (min(abs(t) for t in st["t"]) if st["t"] else np.nan,
                              bool(st["t"]) and all(np.isfinite(t) and abs(t) >= th for t in st["t"])),
    "r2": lambda st, th: (st.get("r2", np.nan), st.get("r2", -1.0) >= th),
}


@dataclass
class ForecastFit:
    as_of: pd.Timestamp
    stats: dict
    passed: bool
    beta: np.ndarray = field(default_factory=lambda: np.zeros(0))   # regressor coefficients used (no intercept)
    scale: float = 1.0                                               # in-sample sd of the forecast
    drift_sign: float = 0.0


class ForecastModel(Model):
    """See the module docstring."""

    def __init__(self, spec: ForecastSpec | str | None = None, **overrides):
        self.spec = get_forecast_spec(spec, **overrides)
        self.fitted_: ForecastFit | None = None

    @property
    def is_fitted(self) -> bool:
        return self.fitted_ is not None

    @property
    def names(self) -> list[str]:
        return [r.name or f"r{i}" for i, r in enumerate(self.spec.regressors)]

    # ------------------------------------------------------------------ data
    def decision_instants(self, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Row D's decision instant: day D + gap (in the target's own days) at decision_time New York."""
        from infra.trading_calendar import snap_instants
        g = self.spec.gap_days
        dec_days = pd.Series(days, index=days).shift(-g)
        filled = dec_days.fillna(pd.Series(days, index=days) + pd.offsets.BDay(g))
        return pd.DatetimeIndex(snap_instants(pd.DatetimeIndex(filled.to_numpy()).normalize(), self.spec.decision_time, NY))

    def read_panel(self, start, end) -> pd.DataFrame:
        """``y`` (the target's daily moves) and one column per regressor, aligned onto each row's
        decision instant (point in time: the value available by then)."""
        from infra.pipeline.features import align, feature
        from infra.pipeline.series_panel import read_panel
        sp = self.spec
        y = read_panel([sp.target], start, end)[sp.target].dropna()
        y.index = pd.DatetimeIndex(y.index).normalize()
        y = y[~y.index.duplicated(keep="last")].sort_index()
        dec = self.decision_instants(y.index)
        out = pd.DataFrame({"y": y.to_numpy()}, index=y.index)
        lo = pd.Timestamp(start) - pd.Timedelta(days=1500)              # feature warm-up
        for name, r in zip(self.names, sp.regressors):
            f = feature(r.expr, lo, end)
            a = align(f, dec, limit=r.limit) if r.limit else align(f, dec)
            if r.fill is not None:
                a = a.fillna(r.fill)
            out[name] = a.to_numpy()
        out.index.name = "timestamp"
        return out

    # ------------------------------------------------------------------ 1. prepare
    def prepare(self, raw: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sp = self.spec
        h, g = sp.horizon_days, sp.gap_days
        y = raw["y"].astype("float64")
        vol = y.ewm(span=sp.vol_span, min_periods=max(sp.vol_span // 2, 10)).std()
        fwd_sum = y.rolling(h, min_periods=h).sum().shift(-(g + h))
        days = pd.Series(y.index, index=y.index)
        out = pd.DataFrame({"y": y, "vol": vol,
                            "past": y.rolling(h, min_periods=h).sum() / (vol * np.sqrt(h)),
                            "fwd": fwd_sum / (vol * np.sqrt(h)), "fwd_bp": fwd_sum,
                            "fwd_end": days.shift(-(g + h))}, index=y.index)
        for name in self.names:
            out[name] = raw[name].astype("float64")
        return out

    # ------------------------------------------------------------------ 2. fit
    def fit(self, prepared: pd.DataFrame, as_of=None) -> "ForecastModel":
        sp = self.spec
        as_of = pd.Timestamp(prepared.index.max() if as_of is None else as_of)
        cols = self.names
        d = prepared[pd.to_datetime(prepared["fwd_end"]) < as_of.normalize()]
        if sp.window:
            d = d[d.index > as_of - pd.Timedelta(sp.window)]
        d = d.dropna(subset=["fwd"] + cols)
        n = len(d)
        lags = sp.hac_lags if sp.hac_lags is not None else sp.horizon_days
        stats: dict = {"n": float(n), "t": []}
        w = np.array([r.weight for r in sp.regressors], dtype="float64")
        beta_used = np.zeros(len(cols))
        if n >= 20:
            y = d["fwd"].to_numpy()
            X = np.c_[np.ones(n), d[cols].to_numpy()]
            b, cov = ols_hac(y, X, lags, ridge=sp.ridge_lambda if sp.estimator == "ridge" else 0.0)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
            stats.update(drift=float(y.mean()), r2=float(1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
            stats["t"] = [float(b[i] / se[i]) if se[i] > 0 else np.nan for i in range(1, X.shape[1])]
            for i, c in enumerate(cols):
                stats[f"coef_{c}"], stats[f"t_{c}"] = float(b[i + 1]), stats["t"][i]
            if sp.fit_mode == "fitted":
                beta_used = b[1:]
            elif sp.fit_mode == "prior":
                bp, _ = ols_hac(y, X, lags, ridge=sp.prior_lambda)
                beta_used = np.where(np.sign(bp[1:]) == np.sign(w), bp[1:], 0.0)
            elif sp.fit_mode == "exante":
                beta_used = w
            else:
                raise ValueError(f"fit_mode {sp.fit_mode!r} (exante | fitted | prior)")
        elif sp.fit_mode == "exante":
            beta_used = w
        passed = True
        if sp.fit_mode != "exante":
            for g in sp.gates:
                th = sp.threshold(g)
                _, ok = GATES[g](stats, th) if n >= 20 else (np.nan, False)
                passed = passed and bool(ok)
            if not passed:
                beta_used = np.zeros(len(cols))
        fc = d[cols].to_numpy() @ beta_used if n else np.zeros(0)
        scale = float(np.std(fc)) if n and np.std(fc) > 0 else 1.0
        self.fitted_ = ForecastFit(as_of, {k: v for k, v in stats.items() if k != "t"} | {"t_list": stats["t"]},
                                   passed, beta_used, scale, float(np.sign(stats.get("drift", 0.0))))
        return self

    # ------------------------------------------------------------------ 3. predict
    def predict(self, prepared: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
        self.check_fitted()
        f, sp = self.fitted_, self.spec
        d = prepared
        if start is not None:
            d = d[d.index > pd.Timestamp(start)]
        if end is not None:
            d = d[d.index <= pd.Timestamp(end)]
        x = d[self.names].to_numpy()
        fc = np.where(np.isnan(x).any(axis=1), np.nan, np.nan_to_num(x) @ f.beta)
        sig = np.clip(fc / (2 * f.scale), -1, 1)
        return pd.DataFrame({"forecast": fc, "signal": np.nan_to_num(sig),
                             "expected_bp": fc * d["vol"].to_numpy() * np.sqrt(sp.horizon_days),
                             "drift_signal": f.drift_sign, "mom_signal": np.sign(d["past"]).fillna(0.0).to_numpy(),
                             "passed": float(f.passed)}, index=d.index)

    # ------------------------------------------------------------------ params
    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f = self.fitted_
        rows = [{"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan},
                {"section": "fit", "row": "passed", "col": "value", "value": float(f.passed)},
                {"section": "fit", "row": "scale", "col": "value", "value": f.scale},
                {"section": "fit", "row": "drift_sign", "col": "value", "value": f.drift_sign}]
        rows += [{"section": "beta", "row": c, "col": "value", "value": float(b)} for c, b in zip(self.names, f.beta)]
        rows += [{"section": "stat", "row": k, "col": "value", "value": float(v)} for k, v in f.stats.items()
                 if k != "t_list"]
        return pd.DataFrame(rows)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec: ForecastSpec | str | None = None, **overrides):
        m = cls(spec, **overrides)
        get = lambda sec: params[params["section"] == sec].set_index("row")["value"]  # noqa: E731
        fit, beta = get("fit"), get("beta")
        m.fitted_ = ForecastFit(as_of=pd.Timestamp(params.loc[params["section"] == "meta", "col"].iloc[0]),
                                stats=get("stat").to_dict(), passed=bool(fit["passed"]),
                                beta=beta.reindex(m.names).to_numpy(dtype="float64"), scale=float(fit["scale"]),
                                drift_sign=float(fit["drift_sign"]))
        return m


def make_forecast(spec: ForecastSpec | str | None = None, **overrides) -> ForecastModel:
    return ForecastModel(spec, **overrides)
