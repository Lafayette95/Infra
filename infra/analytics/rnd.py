"""Risk-neutral density extraction via Breeden-Litzenberger on a fitted smile.

See CLAUDE.md section 9 for methodology and known limitations (SR3 options are
American-exercise, not European - treated as European here, a documented
approximation; tick-floor tail truncation; the returned density is normalized over
its truncated strike range, not the true unconditional support).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.analytics.black76 import implied_vol
from infra.analytics.black76 import price as black76_price
from infra.analytics.forward import implied_forward_and_discount
from infra.analytics.smile import fit_smile


def extract_rnd(chain: pd.DataFrame, *, grid_points: int = 400) -> pd.DataFrame:
    """(forward, discount) -> OTM implied vols -> fitted smile -> dense synthetic call
    curve -> discrete butterfly second-difference -> normalized density.

    ``chain`` must be a single (underlying, expiry, day) slice - every row the same
    underlying/expiry/timestamp, both option types present - matching
    infra.pipeline.daily_options.load_daily_options's output shape.

    Uses the out-of-the-money side at each strike (puts below the forward, calls
    above) to compute implied vol: standard practice, since deep in-the-money options
    have near-zero vega and a poorly-conditioned vol inversion.

    Returns a DataFrame[strike, density] over a dense grid spanning the OBSERVED
    strike range only (no extrapolation). The density integrates to 1 over that
    truncated support, not the true unconditional density (which has mass outside the
    observed strikes too). Values are clipped at 0 before normalizing - numerical/
    model noise can otherwise push the raw second difference slightly negative.
    """
    underlyings, expiries, days = chain["underlying"].unique(), chain["expiry"].unique(), chain["timestamp"].unique()
    if len(underlyings) != 1 or len(expiries) != 1 or len(days) != 1:
        raise ValueError("chain must be a single (underlying, expiry, day) slice")
    expiry, day = pd.Timestamp(expiries[0]), pd.Timestamp(days[0])
    T = (expiry - day).days / 365.0
    if T <= 0:
        raise ValueError(f"option has already expired as of {day.date()} (expiry {expiry.date()})")

    forward, discount = implied_forward_and_discount(chain)

    piv = chain.pivot_table(index="strike", columns="option_type", values="settlement_price")
    if "C" not in piv.columns or "P" not in piv.columns:
        raise ValueError("chain must include both calls ('C') and puts ('P')")
    piv = piv.dropna(subset=["C", "P"], how="all")
    strikes = piv.index.to_numpy(dtype=float)
    is_otm_put = strikes < forward
    otm_price = np.where(is_otm_put, piv["P"].to_numpy(), piv["C"].to_numpy())
    otm_type = np.where(is_otm_put, "P", "C")

    vols = np.array([
        implied_vol(p, forward, k, T, discount, t)
        for p, k, t in zip(otm_price, strikes, otm_type)
    ])
    smile = fit_smile(strikes, vols)

    lo, hi = float(np.nanmin(strikes)), float(np.nanmax(strikes))
    h = (hi - lo) / grid_points  # finite-difference step; far finer than real tick spacing
    grid = np.linspace(lo + h, hi - h, grid_points - 1)

    def call_price(k: np.ndarray) -> np.ndarray:
        sigma = smile(k)
        return np.array([
            black76_price(forward, ki, T, si, discount, "C") if np.isfinite(si) else np.nan
            for ki, si in zip(k, sigma)
        ])

    c_minus, c_mid, c_plus = call_price(grid - h), call_price(grid), call_price(grid + h)
    density = (c_minus - 2 * c_mid + c_plus) / (h**2) / discount

    valid = np.isfinite(density)
    grid, density = grid[valid], density[valid]
    density = np.clip(density, 0.0, None)

    integral = float(np.sum((density[:-1] + density[1:]) / 2 * np.diff(grid)))
    if integral <= 0:
        raise ValueError("extracted density integrates to <= 0 - the fitted smile may be degenerate")
    density = density / integral

    return pd.DataFrame({"strike": grid, "density": density})


def percentiles(density: pd.DataFrame, probs: list[float]) -> list[float]:
    """Strike values at the given cumulative probabilities, via trapezoidal
    integration of ``extract_rnd``'s output - e.g. ``probs=[0.05, 0.95]`` for a 90%
    probability interval, or ``[0.5]`` for the median.
    """
    x = density["strike"].to_numpy(dtype=float)
    d = density["density"].to_numpy(dtype=float)
    cdf = np.concatenate([[0.0], np.cumsum((d[:-1] + d[1:]) / 2 * np.diff(x))])
    if cdf[-1] <= 0:
        raise ValueError("density does not integrate to a positive mass")
    cdf = cdf / cdf[-1]
    return [float(np.interp(p, cdf, x)) for p in probs]
