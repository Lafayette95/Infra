"""Benchmark swap closes from clean DTCC par trades (infra.processing.dtcc_trades). Pure
functions, no I/O; every time is tz-naive UTC (the snap instant comes from
``infra.trading_calendar.snap_instants``, CLAUDE.md 7).

PURE method, per tenor: the trades within +-``pure_half_window_min`` of the snap (widened
to +-``pure_fallback_half_window_min`` below ``pure_min_trades``), each weighted by the
inverse of its expected squared error (``SwapCloseWeighting``: print noise plus unhedged
drift over its distance from the snap); a print further from a first weighted median
than ``outlier_k`` times the larger of the window's robust spread and its own expected
error is dropped; the close is the weighted median of the rest.

FUTURES-ADJUSTED method: the same estimator over +-``adjusted_half_window_min``, after
moving each print to the snap by its hedge future (``futures_move_bp`` = hedge ratio x
the future's mid change from the print to the snap, computed by the pipeline), weighted
with the adjusted variance (only the swap-vs-futures drift left, plus the hedge-ratio
error on that move). A print without a hedge move (no quote, no ratio) is left out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import SwapCloseSpec, SwapCloseWeighting

CLOSE_COLUMNS = ["timestamp", "close", "currency", "tenor", "method", "rate", "n_trades", "half_window_min",
                 "dispersion_bp", "se_bp"]
_MAD_TO_SD = 1.4826


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The value at which the cumulative weight first reaches half the total."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def _robust_sd_bp(rates: np.ndarray, weights: np.ndarray, centre: float) -> float:
    return _MAD_TO_SD * weighted_median(np.abs(rates - centre) * 100.0, weights)


def estimate(rates: np.ndarray, dt_hours: np.ndarray, weighting: SwapCloseWeighting, *, adjusted: bool = False,
             futures_move_bp: np.ndarray | None = None) -> dict | None:
    """One tenor's close from its window's prints (rates in percent). None if no prints."""
    if len(rates) == 0:
        return None
    moves = np.zeros_like(dt_hours) if futures_move_bp is None else futures_move_bp
    var = np.asarray(weighting.variance(np.abs(dt_hours), moves, adjusted=adjusted), dtype=float)
    w = 1.0 / var
    first = weighted_median(rates, w)
    # each print's own limit: the window's spread, or the print's OWN expected error if
    # larger - a print far from the snap is expected to have drifted, and the spread is
    # dominated by the heavily weighted prints near the snap
    limit = weighting.outlier_k * np.maximum(_robust_sd_bp(rates, w, first), np.sqrt(var))
    keep = np.abs(rates - first) * 100.0 <= limit
    rates, w = rates[keep], w[keep]
    close = weighted_median(rates, w)
    dispersion = _robust_sd_bp(rates, w, close)
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    # standard error of a median ~ 1.2533 sd / sqrt(n); the floor keeps a single print (or
    # identical ones) from claiming zero error
    se = 1.2533 * max(dispersion, weighting.trade_noise_bp) / np.sqrt(n_eff)
    return {"rate": close, "n_trades": int(keep.sum()), "dispersion_bp": dispersion, "se_bp": se}


def pure_closes(trades: pd.DataFrame, instant: pd.Timestamp, spec: SwapCloseSpec, weighting: SwapCloseWeighting,
                *, close_name: str, currency: str) -> pd.DataFrame:
    """Every tenor's PURE close at ``instant`` from one currency's par trades. A tenor with
    no print inside the fallback window gets no row (never a stale or invented value)."""
    rows = []
    dt_all = (trades["executed"] - instant).dt.total_seconds().to_numpy() / 3600.0 if len(trades) else np.array([])
    for tenor in sorted(trades["tenor"].unique()) if len(trades) else []:
        in_tenor = (trades["tenor"] == tenor).to_numpy()
        half = spec.pure_half_window_min
        window = in_tenor & (np.abs(dt_all) <= half / 60.0)
        if window.sum() < spec.pure_min_trades:
            half = spec.pure_fallback_half_window_min
            window = in_tenor & (np.abs(dt_all) <= half / 60.0)
        est = estimate(trades["rate"].to_numpy()[window], dt_all[window], weighting)
        if est is None:
            continue
        rows.append({"timestamp": instant, "close": close_name, "currency": currency, "tenor": int(tenor),
                     "method": "pure", "half_window_min": half, **est})
    if not rows:
        return pd.DataFrame(columns=CLOSE_COLUMNS)
    return pd.DataFrame(rows)[CLOSE_COLUMNS]


def adjusted_closes(trades: pd.DataFrame, instant: pd.Timestamp, spec: SwapCloseSpec, weighting: SwapCloseWeighting,
                    *, close_name: str, currency: str) -> pd.DataFrame:
    """Every tenor's FUTURES-ADJUSTED close at ``instant``. ``trades`` carries
    ``futures_move_bp`` per print (NaN = no hedge: left out)."""
    rows = []
    t = trades.dropna(subset=["futures_move_bp"]) if len(trades) else trades
    if len(t):
        dt_all = (t["executed"] - instant).dt.total_seconds().to_numpy() / 3600.0
        in_window = np.abs(dt_all) <= spec.adjusted_half_window_min / 60.0
        moved = t["rate"].to_numpy(dtype=float) + t["futures_move_bp"].to_numpy(dtype=float) / 100.0
        for tenor in sorted(t["tenor"].unique()):
            sel = in_window & (t["tenor"] == tenor).to_numpy()
            est = estimate(moved[sel], dt_all[sel], weighting, adjusted=True,
                           futures_move_bp=t["futures_move_bp"].to_numpy(dtype=float)[sel])
            if est is None:
                continue
            rows.append({"timestamp": instant, "close": close_name, "currency": currency, "tenor": int(tenor),
                         "method": "adjusted", "half_window_min": spec.adjusted_half_window_min, **est})
    if not rows:
        return pd.DataFrame(columns=CLOSE_COLUMNS)
    return pd.DataFrame(rows)[CLOSE_COLUMNS]
