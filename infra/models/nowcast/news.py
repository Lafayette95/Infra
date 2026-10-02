"""Nowcast of the target and its NEWS decomposition, in GDP units (the target's native
units, QoQ % SAAR). Pure numpy/pandas on a fitted ``DFM`` and standardized panels.

Nowcast of quarter q = E[target_q | data], mapped back to GDP units. Where the quarter's
GDP is already published, that IS the nowcast (the target is itself an observable).

News (Banbura & Modugno 2014, the NY Fed Staff Nowcast's "impact" table): going from an
old information set to a new one with the parameters held fixed,

    nowcast_new - nowcast_old
        = [revisions: old values re-published differently]
        + sum_j  weight_j * (actual_j - forecast_j)      (the new prints)

``forecast_j`` is the model's expectation of print j given the old data (revised);
``news_j = actual_j - forecast_j`` is the SURPRISE; ``weight_j`` is the gain the Kalman
smoother puts on that surprise in the target - how much of it the model reads as signal
about GDP rather than series-specific noise, net of what the other new prints in the same
batch already said. ``impact_j = weight_j * news_j`` sums EXACTLY to the change.

The weights are computed without any extra matrix algebra: E[target | old, I] is affine
in the new values I, and equals the old nowcast when I is set to its own expectation
(law of iterated expectations), so ``weight_j`` = the nowcast's response to a unit bump
of print j with every other new print at its expectation - one smoother run per print,
exact and order-independent (a joint decomposition, not a sequential one).

``signal_share_j`` - the same bump's effect on print j's OWN common component (lam.f):
the fraction of the surprise the model attributes to the common factors ("news") rather
than to the release's idiosyncratic component ("noise"). 1 = all signal, 0 = all noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from infra.models.nowcast import kalman
from infra.models.nowcast.dfm import DFM, QUARTERLY_WEIGHTS


def quarter_month(months: pd.DatetimeIndex, quarter) -> int:
    """Row of ``quarter``'s third month in ``months`` (quarter: "2026Q3" or a Period)."""
    q = pd.Period(quarter, freq="Q")
    month = q.asfreq("M", how="end").to_timestamp()
    loc = months.get_indexer([month])[0]
    if loc < 0:
        raise ValueError(f"{q} is not on the panel's month grid ({months[0].date()}..{months[-1].date()})")
    return int(loc)


def _target_col(model: DFM) -> int:
    return model.series.index(model.spec.target)


def common_loading_rows(model: DFM) -> np.ndarray:
    """(n, m): each series' COMMON component (lam . f, without its own idio)."""
    lay = model.layout()
    Z = model.state_space().Z.copy()
    for k in lay.idio_index.values():
        Z[:, k] = 0.0
    return Z


def target_value(model: DFM, X: np.ndarray, sm: kalman.Smoothed, t_q: int) -> float:
    """E[target at row t_q | X] in GDP units: the published value if there is one."""
    i = _target_col(model)
    x = X[t_q, i]
    if not np.isfinite(x):
        x = common_loading_rows(model)[i] @ sm.a[t_q]
    return float(model.mu[i] + model.sd[i] * x)


def factor_contributions(model: DFM, sm: kalman.Smoothed, t_q: int) -> pd.Series:
    """The target's COMMON component at ``t_q`` split by factor, in GDP units (pp), plus
    its mean (``mean``): mean + sum over factors = the model's own estimate of the quarter,
    whether or not GDP for it is already published."""
    i, lay = _target_col(model), model.layout()
    lam = model.params.lam[i]
    agg = sum(w * sm.a[t_q, lay.f(lag)] for lag, w in enumerate(QUARTERLY_WEIGHTS))
    out = pd.Series(model.sd[i] * lam * agg, index=list(model.structure.factors))
    return pd.concat([pd.Series({"mean": model.mu[i]}), out])


@dataclass
class News:
    quarter: pd.Period
    old: float  # nowcast on the old data
    revised: float  # ... on the old data with revisions applied
    new: float  # ... on the new data
    impacts: pd.DataFrame  # one row per new print

    @property
    def revision_impact(self) -> float:
        return self.revised - self.old

    @property
    def news_impact(self) -> float:
        return float(self.impacts["impact"].sum())

    def by(self, column: str = "ticker") -> pd.Series:
        return self.impacts.groupby(column, sort=False)["impact"].sum()


def decompose(model: DFM, X_old: np.ndarray, X_new: np.ndarray, months: pd.DatetimeIndex, quarter) -> News:
    """News from ``X_old`` to ``X_new`` (standardized, same month grid) for ``quarter``."""
    ss = model.state_space()
    t_q = quarter_month(months, quarter)
    sm_old = kalman.smooth(ss, X_old, lag=False)
    y_old = target_value(model, X_old, sm_old, t_q)

    both = np.isfinite(X_old) & np.isfinite(X_new)
    X_rev = np.where(both, X_new, X_old)
    X_rev[np.isfinite(X_old) & ~np.isfinite(X_new)] = np.nan  # retracted
    sm_rev = kalman.smooth(ss, X_rev, lag=False)
    y_rev = target_value(model, X_rev, sm_rev, t_q)

    new_cells = np.argwhere(np.isfinite(X_new) & ~np.isfinite(X_rev))
    Z, Zc = ss.Z, common_loading_rows(model)
    forecast = np.array([Z[i] @ sm_rev.a[t] for t, i in new_cells])
    actual = np.array([X_new[t, i] for t, i in new_cells])
    base = X_rev.copy()
    for (t, i), f in zip(new_cells, forecast):
        base[t, i] = f
    sm_base = kalman.smooth(ss, base, lag=False)
    y_base = target_value(model, base, sm_base, t_q)
    weight, signal = np.zeros(len(new_cells)), np.zeros(len(new_cells))
    for k, (t, i) in enumerate(new_cells):
        bumped = base.copy()
        bumped[t, i] += 1.0
        sm_k = kalman.smooth(ss, bumped, lag=False)
        weight[k] = target_value(model, bumped, sm_k, t_q) - y_base
        signal[k] = Zc[i] @ (sm_k.a[t] - sm_base.a[t])
    y_new = target_value(model, X_new, kalman.smooth(ss, X_new, lag=False), t_q)

    series = np.array(model.series)
    sd, mu = model.sd, model.mu
    cols = new_cells[:, 1] if len(new_cells) else np.array([], int)
    impacts = pd.DataFrame({
        "ticker": series[cols],
        "period": months[new_cells[:, 0]] if len(new_cells) else pd.DatetimeIndex([]),
        # in the release's model units (the table's transform), not standardized
        "actual": mu[cols] + sd[cols] * actual,
        "forecast": mu[cols] + sd[cols] * forecast,
        "news": sd[cols] * (actual - forecast),
        "weight": weight / sd[cols],  # GDP pp per model unit of news
        "impact": weight * (actual - forecast),
        "signal_share": signal,
    })
    impacts = impacts.sort_values("impact", key=np.abs, ascending=False).reset_index(drop=True)
    return News(pd.Period(quarter, freq="Q"), y_old, y_rev, y_new, impacts)


def prospective(model: DFM, X: np.ndarray, months: pd.DatetimeIndex, quarter, cells: list[tuple[int, int]]) -> pd.DataFrame:
    """For prints NOT yet published (``cells``: (row, column) of ``X``, missing): the
    model's forecast of each and the WEIGHT its surprise would carry in the nowcast of
    ``quarter`` - the same unit-bump response as ``decompose``, one print at a time given
    today's data. Columns ``row, col, forecast, weight`` (standardized units)."""
    ss = model.state_space()
    t_q = quarter_month(months, quarter)
    sm = kalman.smooth(ss, X, lag=False)
    Z = ss.Z
    out = []
    for t, i in cells:
        f = float(Z[i] @ sm.a[t])
        filled = X.copy()
        filled[t, i] = f
        y0 = target_value(model, filled, kalman.smooth(ss, filled, lag=False), t_q)
        filled[t, i] = f + 1.0
        y1 = target_value(model, filled, kalman.smooth(ss, filled, lag=False), t_q)
        out.append((t, i, f, y1 - y0))
    return pd.DataFrame(out, columns=["row", "col", "forecast", "weight"])
