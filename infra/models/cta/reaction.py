"""The CTA's SPOT reaction function: where its position would be after an instant shock
to one input, everything else equal.

A registry, so new axes plug in without touching the model (``REACTIONS[name] =
fn(params, state, spec, grid) -> DataFrame``). Each returns one row per grid point with
the shocked ``signal`` and ``position`` and ``d_position`` against the unshocked state.

* ``price``: the latest price moved by ``shock`` x one day's sizing vol. Every EWMA
  takes the move with its own weight 1/n (EWMA' = EWMA + shock/n): exactly what the
  algorithm would show had today's close been there. The vols are held (all else equal),
  so this is the pure trend response - the CTA's "gamma" curve; ``shock_level`` gives
  the move in price units.
* ``vol``: the sizing vol scaled by (1 + shock); with ``"norm"`` in
  ``spec.vol_shock_targets`` the trend normalisation is scaled too, so the signal moves.
* ``price_vol``: the two together on a grid (price shocks x vol shocks).
* ``path``: the same total move spread evenly over ``spec.reaction_path_days`` sessions
  (``shock`` x one day's vol x sqrt(days), i.e. a ``shock``-sigma move over the period),
  vols held. Where the instant shock is flat - a saturated signal needs a ~20-sigma
  one-day jump to react - this is the informative one: how much covering a steady
  rally against the position triggers within a week.

Ideas for later axes: ``time`` (price unchanged for h days: the decay of an unfed trend),
signal-flip levels (the price at which the position crosses 0).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from infra.models.cta.config import CTASpec
from infra.models.cta.state import AssetParams, AssetState, raw_position, score, to_position, to_signal


def _evaluate(params, state, spec, price_shock=0.0, vol_shock=0.0) -> tuple[float, float, float]:
    daily_vol = state.vol / np.sqrt(spec.periods_per_year)
    move = price_shock * daily_vol
    emas = {n: state.emas[n] + move / n for n in spec.spans}
    norm_vol = state.norm_vol * ((1.0 + vol_shock) if "norm" in spec.vol_shock_targets else 1.0)
    vol = state.vol * ((1.0 + vol_shock) if "sizing" in spec.vol_shock_targets else 1.0)
    s = float(to_signal(params, score(params, emas, norm_vol, spec.ewma_pairs, spec.weights())))
    pos = float(to_position(params, raw_position(params, s, state.pvs, vol), spec.clip_position))
    return move, s, pos


def _evaluate_path(params, state, spec, shock: float, days: int) -> tuple[float, float, float]:
    daily_vol = state.vol / np.sqrt(spec.periods_per_year)
    move = shock * daily_vol * np.sqrt(days)
    level, emas = state.level, dict(state.emas)
    for _ in range(days):
        level += move / days
        emas = {n: emas[n] + (level - emas[n]) / n for n in spec.spans}
    s = float(to_signal(params, score(params, emas, state.norm_vol, spec.ewma_pairs, spec.weights())))
    pos = float(to_position(params, raw_position(params, s, state.pvs, state.vol), spec.clip_position))
    return move, s, pos


def path_reaction(params: AssetParams, state: AssetState, spec: CTASpec, grid=None) -> pd.DataFrame:
    grid = spec.price_shock_grid if grid is None else grid
    days = spec.reaction_path_days
    base = _evaluate_path(params, state, spec, 0.0, days)[2]  # no move: the unfed trend's decay
    rows = []
    for g in grid:
        move, s, pos = _evaluate_path(params, state, spec, float(g), days)
        rows.append({"timestamp": state.timestamp, "asset": state.asset, "axis": "path",
                     "price_shock": float(g), "vol_shock": 0.0, "shock_level": move,
                     "level": state.level + move, "signal": s, "position": pos,
                     "d_position": pos - state.position, "d_position_vs_flat": pos - base})
    return pd.DataFrame(rows)


def _rows(params, state, spec, points, axis) -> pd.DataFrame:
    rows = []
    for p_shock, v_shock in points:
        move, s, pos = _evaluate(params, state, spec, p_shock, v_shock)
        rows.append({"timestamp": state.timestamp, "asset": state.asset, "axis": axis,
                     "price_shock": p_shock, "vol_shock": v_shock, "shock_level": move,
                     "level": state.level + move, "signal": s, "position": pos,
                     "d_position": pos - state.position})
    return pd.DataFrame(rows)


def price_reaction(params: AssetParams, state: AssetState, spec: CTASpec, grid=None) -> pd.DataFrame:
    grid = spec.price_shock_grid if grid is None else grid
    return _rows(params, state, spec, [(float(g), 0.0) for g in grid], "price")


def vol_reaction(params: AssetParams, state: AssetState, spec: CTASpec, grid=None) -> pd.DataFrame:
    grid = spec.vol_shock_grid if grid is None else grid
    return _rows(params, state, spec, [(0.0, float(g)) for g in grid], "vol")


def price_vol_reaction(params: AssetParams, state: AssetState, spec: CTASpec, grid=None) -> pd.DataFrame:
    prices, vols = (spec.price_shock_grid, spec.vol_shock_grid) if grid is None else grid
    return _rows(params, state, spec, [(float(p), float(v)) for p, v in itertools.product(prices, vols)],
                 "price_vol")


REACTIONS = {"price": price_reaction, "vol": vol_reaction, "price_vol": price_vol_reaction,
             "path": path_reaction}
