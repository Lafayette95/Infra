"""Series transforms named in the release table (``MacroRelease.transform``): a string of
``;``-separated steps applied left to right, e.g. ``"deman_fix:50;ecdf_no_demean_exp:130"``.

Every step is TRAILING - a value only ever depends on its own and earlier observations -
so a transformed history is point-in-time: adding a new observation never changes an
earlier transformed value (a data REVISION can, and is treated as one downstream).

Steps (``W``/``N``/``K`` are the number after the colon):

* ``rolling_ma:N`` - mean of the last N observations (native frequency), NaN until N exist.
* ``deman_fix:K`` - subtract a FIXED level: 50 for a PMI (the expansion line), 100 for an
  index based at 100 - the "zero" that ``ecdf_no_demean`` then keeps.
* ``ecdf_exp:W`` - exponentially weighted empirical CDF, the robust analogue of a
  DEMEANED z-score: the value's weighted rank among its own history (itself included,
  ties half-counted), each past observation weighted ``0.5 ** (age / W)``, age in
  business days between observation periods (user, 2026-09-30: W is a half-life; 1305 =
  5y, 130 = 6m). Output ``2F - 1`` in (-1, 1): 0 = the series' own weighted median.
* ``ecdf_no_demean_exp:W`` - the analogue of a z-score WITHOUT demeaning (x / RMS around
  zero): rank the MAGNITUDE |x| among past magnitudes and keep the sign,
  ``sign(x) * F(|x|)``. Zero stays zero - so positive growth / a PMI above 50 stays
  positive whatever the series' own average. Equals ``ecdf_exp`` for a series symmetric
  around 0.
* ``gauss`` - map an ECDF output to a normal score: ``Phi^-1((1 + y) / 2)`` for
  ``ecdf_exp`` (so a standard normal), ``sign(y) * Phi^-1((1 + |y|) / 2)`` for
  ``ecdf_no_demean`` (a half-normal magnitude). Not in the table; the model can append
  it (``ModelSpec.gaussianize``) to suit its Gaussian likelihood.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# An ECDF value needs this many observations of history (itself included) - before that
# the rank of a value among a handful of others is noise, not information.
ECDF_MIN_OBS = 12
_GAUSS_CLIP = 1e-4  # keep Phi^-1 finite at the extremes


def parse(transform: str) -> list[tuple[str, float | None]]:
    steps = []
    for token in filter(None, (t.strip() for t in transform.split(";"))):
        name, _, arg = token.partition(":")
        if name not in STEPS:
            raise ValueError(f"unknown transform step {name!r} in {transform!r}")
        steps.append((name, float(arg) if arg else None))
    return steps


def rolling_ma(s: pd.Series, n: float) -> pd.Series:
    return s.rolling(int(n), min_periods=int(n)).mean()


def deman_fix(s: pd.Series, k: float) -> pd.Series:
    return s - k


def _weighted_rank(values: np.ndarray, dates: np.ndarray, half_life: float, min_obs: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    days = dates.astype("datetime64[D]")
    for t in range(len(values)):
        if not np.isfinite(values[t]):
            continue
        past = np.isfinite(values[: t + 1])
        if past.sum() < min_obs:
            continue
        v = values[: t + 1][past]
        age = np.busday_count(days[: t + 1][past], days[t])
        w = 0.5 ** (age / half_life)
        out[t] = (w * ((v < values[t]) + 0.5 * (v == values[t]))).sum() / w.sum()
    return out


def ecdf_exp(s: pd.Series, half_life: float, *, min_obs: int = ECDF_MIN_OBS) -> pd.Series:
    """``s`` indexed by observation period (datetime)."""
    u = _weighted_rank(s.to_numpy(dtype=float), s.index.to_numpy(), half_life, min_obs)
    return pd.Series(2.0 * u - 1.0, index=s.index)


def ecdf_no_demean_exp(s: pd.Series, half_life: float, *, min_obs: int = ECDF_MIN_OBS) -> pd.Series:
    x = s.to_numpy(dtype=float)
    u = _weighted_rank(np.abs(x), s.index.to_numpy(), half_life, min_obs)
    return pd.Series(np.sign(x) * u, index=s.index)


def gauss(s: pd.Series, _arg=None, *, demeaned: bool = True) -> pd.Series:
    y = s.to_numpy(dtype=float)
    if demeaned:
        return pd.Series(norm.ppf(np.clip((1.0 + y) / 2.0, _GAUSS_CLIP, 1 - _GAUSS_CLIP)), index=s.index)
    return pd.Series(np.sign(y) * norm.ppf(np.clip((1.0 + np.abs(y)) / 2.0, 0.5, 1 - _GAUSS_CLIP)), index=s.index)


STEPS = {"rolling_ma": rolling_ma, "deman_fix": deman_fix, "ecdf_exp": ecdf_exp,
         "ecdf_no_demean_exp": ecdf_no_demean_exp, "gauss": gauss}


def apply(s: pd.Series, transform: str, *, min_obs: int = ECDF_MIN_OBS) -> pd.Series:
    """Apply ``transform`` to ``s`` (index = observation period, sorted, native frequency)."""
    s = s.sort_index().astype(float)
    demeaned = True
    for name, arg in parse(transform):
        if name == "gauss":
            s = gauss(s, demeaned=demeaned)
        elif name.startswith("ecdf"):
            demeaned = name == "ecdf_exp"
            s = STEPS[name](s, arg, min_obs=min_obs)
        else:
            s = STEPS[name](s, arg)
    return s
