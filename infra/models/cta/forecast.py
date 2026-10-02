"""Expected signal, position and flow over the next h observations (UBS Q9).

UBS's method, kept as stated:
* the signal is path dependent and non-linear in price, so its expectation comes from a
  Monte Carlo: simulate price paths over the horizon, run the trend algorithm on each,
  average the terminal signals;
* paths are centred on the forward price (zero drift for a futures price), Gaussian, with
  the expected short-term volatility as their volatility;
* expected short-term vol = the current 2-month realised vol (``forecast_vol``);
* the trend normalisation (1y vol) and the portfolio vol scaling (3y) move too slowly
  to matter over the horizon: held at today's values.

Expected position = weight x liquidity x E[signal] x pvs / expected vol. Flow = expected
position - current position, split exactly into the part from the signal's expected
change and the part from the vol's:

    from_signal = k (E[s] - s) / vol         from_vol = k E[s] (1/E[vol] - 1/vol)

(k = weight x liquidity x pvs / position scale). The two sum to the flow before
clipping; positions are clipped to [-1, 1] only when ``spec.clip_position`` (an
out-of-sample record) - rare, and then ``flow`` is the clipped difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.models.cta.config import CTASpec
from infra.models.cta.state import AssetParams, AssetState, raw_position, score, to_position, to_signal


def simulate_signals(params: AssetParams, state: AssetState, spec: CTASpec,
                     horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Terminal signals per path at each horizon (antithetic Gaussian paths, zero drift)."""
    horizon = max(horizons)
    half = max(spec.mc_paths // 2, 1)
    rng = np.random.default_rng(spec.mc_seed)
    shocks = rng.standard_normal((half, horizon))
    shocks = np.vstack([shocks, -shocks]) * (state.forecast_vol / np.sqrt(spec.periods_per_year))
    level = np.full(len(shocks), state.level, dtype=float)
    emas = {n: np.full(len(shocks), state.emas[n], dtype=float) for n in spec.spans}
    weights = spec.weights()
    out = {}
    for h in range(1, horizon + 1):
        level = level + shocks[:, h - 1]
        for n in spec.spans:
            emas[n] = emas[n] + (level - emas[n]) / n
        if h in horizons:
            out[h] = to_signal(params, score(params, emas, state.norm_vol, spec.ewma_pairs, weights))
    return out


def expected_flows(params: AssetParams, state: AssetState, spec: CTASpec,
                   horizons: tuple[int, ...] | None = None) -> pd.DataFrame:
    """One row per horizon: expected signal / vol / position and the flow decomposition."""
    horizons = tuple(sorted(horizons or spec.flow_horizons))
    cols = ["timestamp", "asset", "horizon", "exp_signal", "exp_vol", "exp_position",
            "flow", "flow_from_signal", "flow_from_vol"]
    if not (np.isfinite(state.signal) and np.isfinite(state.vol) and np.isfinite(state.forecast_vol)):
        return pd.DataFrame(columns=cols)
    sims = simulate_signals(params, state, spec, horizons)
    k = params.class_weight * params.liquidity * state.pvs / params.pos_scale
    rows = []
    for h in horizons:
        e_sig = float(np.nanmean(sims[h]))
        e_vol = state.forecast_vol
        e_raw = raw_position(params, e_sig, state.pvs, e_vol)
        e_pos = float(to_position(params, e_raw, spec.clip_position))
        rows.append({
            "timestamp": state.timestamp, "asset": state.asset, "horizon": h,
            "exp_signal": e_sig, "exp_vol": e_vol, "exp_position": e_pos,
            "flow": e_pos - state.position,
            "flow_from_signal": k * (e_sig - state.signal) / state.vol,
            "flow_from_vol": k * e_sig * (1.0 / e_vol - 1.0 / state.vol),
        })
    return pd.DataFrame(rows, columns=cols)
