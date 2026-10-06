"""Volatility risk premium - pure. Realised vol from daily changes (any units: bp, futures
points), annualised by ``trading_days``; the premium is implied minus realised in the same
units, and the ratio implied / realised. Point in time: trailing and EWMA vols use changes
up to the day; the ex-post vol over an option's life is only known at its end."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_vol(changes: pd.Series, window: int, *, trading_days: int = 252, min_obs: int = 15) -> pd.Series:
    """Root-mean-square of the last ``window`` daily changes (no mean removed: daily drift is
    noise next to the vol), annualised."""
    return np.sqrt((changes ** 2).rolling(window, min_periods=min_obs).mean() * trading_days)


def ewma_vol(changes: pd.Series, lam: float, *, trading_days: int = 252, min_obs: int = 15) -> pd.Series:
    v = (changes ** 2).ewm(alpha=1.0 - lam, adjust=False).mean() * trading_days
    v[np.arange(len(v)) < min_obs - 1] = np.nan
    return np.sqrt(v)


def realised_over(path: pd.Series, *, trading_days: int = 252, min_obs: int = 15) -> float:
    """Annualised realised vol of a level path (values on consecutive observation days)."""
    d = path.dropna().diff().dropna()
    return float(np.sqrt((d ** 2).mean() * trading_days)) if len(d) >= min_obs else float("nan")
