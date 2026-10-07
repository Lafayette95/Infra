"""Directional forecasts (framework A) parameters: named specs, toggled by name
(``infra/models/CLAUDE.md`` 0; methodology ``infra/models/forecast/CLAUDE.md``).

A spec: the TARGET (a daily-moves series id, bp, + = a long position made money), the forward
window (``horizon_days`` after a ``gap``), the REGRESSORS (feature-maker expressions, read point in
time and aligned onto the decision instants), the FIT MODE (``fit_mode``, ``infra.models.fit_modes``):

* ``exante``: no fit - forecast = sum_i weight_i x feature_i (a pre-stated direction: ANY feature
  string from the feature maker, signed); the fit only reports statistics;
* ``fitted``: walk-forward regression (``estimator``, default OLS with HAC errors), used if its gates
  pass;
* ``prior``: the sign of each coefficient is fixed ex ante (``weights``' signs); the fit only sizes
  it (ridge with ``prior_lambda``), a coefficient on the wrong side is set to 0.

Gates (in-sample, point in time) and evaluations (out of sample) are swappable registries, as in
the conditional-autocorrelation model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Regressor:
    expr: str                                   # feature-maker expression (infra.pipeline.features)
    weight: float = 1.0                         # exante: the weight; prior: its sign is the prior's
    limit: str | None = None                    # carry at most this long onto the decision instants (None =
                                                # the source's FEATURE_FFILL_LIMITS)
    fill: float | None = None                   # beyond the limit: this value (0 = "no news") or None = drop the row
    name: str | None = None                     # a short label for reports


@dataclass(frozen=True)
class ForecastSpec:
    name: str = "custom"
    description: str = ""
    target: str = "bmk:otr:US_BOND_10y"         # daily moves, bp, + = long gains
    regressors: tuple[Regressor, ...] = ()
    fit_mode: str = "fitted"                    # exante | fitted | prior (infra.models.fit_modes)
    horizon_days: int = 5
    gap_days: int = 1                           # decision on day D + gap at `decision_time` (New York); the
                                                # forward window starts at that day's close
    decision_time: str = "15:00"                # New York: before the 15:30 cash marks of that day
    vol_span: int = 60                          # EWMA of daily target moves: the forward move's normaliser
    estimator: str = "ols"                      # ols | ridge (fitted; prior is always ridge + sign constraint)
    ridge_lambda: float = 0.0                   # ridge penalty (x n) for estimator "ridge"
    prior_lambda: float = 0.1                   # prior mode: ridge penalty (x n)
    window: str | None = None                   # fit look back; None = all history
    hac_lags: int | None = None                 # None = horizon_days
    min_obs: int = 150
    gates: tuple[str, ...] = ("min_obs", "coef_t")
    thresholds: tuple[tuple[str, float], ...] = (("min_obs", 150.0), ("coef_t", 2.0))
    evaluations: tuple[str, ...] = ("benchmark", "clark_west", "spanning", "subperiods", "permutation",
                                    "time_shift")
    n_placebo: int = 50
    placebo_block_days: int = 63

    def __post_init__(self):
        from infra.models.fit_modes import check_fit_mode
        check_fit_mode(self.fit_mode, has_direction=True, what=f"forecast {self.name!r}")

    def threshold(self, gate: str, default: float = 0.0) -> float:
        return dict(self.thresholds).get(gate, default)


def surprise(ticker: str, weight: float = 1.0, *, window: int = 36, days: str = "4D") -> Regressor:
    """A release's surprise (actual - consensus), z-scored on its own trailing RMS (not demeaned:
    a surprise's mean is 0 by construction), clipped at 3, carried ``days`` after the release, then
    0 ("no news")."""
    return Regressor(f"surprise:{ticker} | lvl | norm:z0:{window} | clip:3", weight=weight, limit=days, fill=0.0,
                     name=ticker.replace(" Index", ""))


FORECAST_MODELS: dict[str, ForecastSpec] = {s.name: s for s in (
    ForecastSpec("default", "fitted OLS (HAC) on the spec's regressors"),
    ForecastSpec("exante", "no fit: the regressors' stated weights", fit_mode="exante", gates=()),
    ForecastSpec("prior", "sign fixed ex ante (the weights' signs), size fitted (ridge)", fit_mode="prior",
                 gates=("min_obs",)),
)}


def get_forecast_spec(spec: ForecastSpec | str | None = None, **overrides) -> ForecastSpec:
    base = FORECAST_MODELS["default"] if spec is None else (FORECAST_MODELS[spec] if isinstance(spec, str) else spec)
    if "mode" in overrides:                              # legacy name (runs stored before 2026-10-07)
        overrides["fit_mode"] = overrides.pop("mode")
    return replace(base, **overrides) if overrides else base
