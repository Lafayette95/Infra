"""What a fitted CTA model holds per asset: frozen parameters, and the state at one
observation (enough to continue the recursion, forecast, or shock it)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from infra.models.cta import signal as sig


@dataclass(frozen=True)
class AssetParams:
    """Fitted, frozen until the next fit."""
    pair_scale: np.ndarray            # RMS of each pair's crossover over the fit sample
    response: Callable | None         # score -> signal; None = too little history
    pos_scale: float                  # max |raw position| over the lookback: position = raw / this
    class_weight: float
    liquidity: float


@dataclass(frozen=True)
class AssetState:
    """One asset at one observation."""
    asset: str
    timestamp: pd.Timestamp
    level: float
    emas: dict[int, float]
    norm_vol: float                   # per observation (trend normalisation, step ii)
    vol: float                        # annualised sizing vol (3m)
    forecast_vol: float               # annualised 2m vol, the forecast of ``vol``
    signal: float
    raw_position: float
    position: float
    pvs: float                        # portfolio vol scaling in force


def score(params: AssetParams, emas, norm_vol, pairs, weights):
    z = sig.crossover_z(emas, pairs, norm_vol)
    return sig.combined_score(z, params.pair_scale, weights)


def to_signal(params: AssetParams, scores):
    if params.response is None:
        return np.full(np.shape(scores), np.nan) if np.ndim(scores) else np.nan
    return params.response(scores)


def raw_position(params: AssetParams, signal, pvs, vol):
    return sig.risk_weight(params.class_weight, params.liquidity, pvs) * signal / vol


def to_position(params: AssetParams, raw, clip: bool):
    pos = np.asarray(raw, float) / params.pos_scale if params.pos_scale and np.isfinite(params.pos_scale) \
        else np.full(np.shape(raw), np.nan)
    return np.clip(pos, -1.0, 1.0) if clip else pos
