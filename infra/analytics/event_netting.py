"""Net an event study's richness drift of what FUNDING already carries - pure, no I/O
(infra/models/basis/CLAUDE.md 3g, add-on MS).

A special bond's spot price is rich by the value of the cheap financing still ahead of it;
as days pass that richness is consumed AS CARRY - a cheapening of its spot residual. The
basis models' funding layer already credits that specialness in each bond's forward (rate =
base + basis - specialness), so adding the raw event drift as well would count it twice.
One day of specialness ``s`` (bp a year) is worth s / 360 / 1e4 of price, i.e.
``s / 360 / modified duration`` bp of yield; ``specialness_carry_bp`` accumulates it along
each event path and ``net_paths`` subtracts it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def modified_duration(maturity_years, yield_pct, freq: int = 2) -> np.ndarray:
    """Par-bond modified duration (years) - accurate enough to turn bp of carry into bp of
    yield (the coupon effect on duration is second order here)."""
    m = np.asarray(maturity_years, dtype="float64")
    y = np.maximum(np.asarray(yield_pct, dtype="float64") / 100, 1e-4)
    return (1 - (1 + y / freq) ** (-freq * m)) / y


def specialness_carry_bp(paths: pd.DataFrame, specialness: pd.Series, durations: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """For each path row (``cusip, event_day, rel_day`` from ``event_study.event_paths``),
    the specialness carry accumulated from the path's anchor to that day, in bp of yield.
    ``specialness``: bp a year by (timestamp, cusip), missing = 0 (not special);
    ``durations``: modified duration by (timestamp, cusip); ``days``: the business-day
    index the paths were measured on."""
    sp = specialness.unstack("cusip").reindex(days).fillna(0.0)
    du = durations.unstack("cusip").reindex(days)
    daily = sp / 360.0 / du  # bp of yield per day
    out = np.full(len(paths), np.nan)
    pos = {d: i for i, d in enumerate(days)}
    for k, (c, ev, rel) in enumerate(zip(paths["cusip"], paths["event_day"], paths["rel_day"])):
        if c not in daily.columns:
            out[k] = 0.0
            continue
        i0 = days.searchsorted(pd.Timestamp(ev))
        anchor, i = i0 + int(paths["anchor_offset"].iloc[k]) if "anchor_offset" in paths else i0 - 1, i0 + int(rel)
        lo, hi = (anchor, i) if i >= anchor else (i, anchor)
        seg = daily[c].iloc[max(lo, 0):max(hi, 0)].fillna(0.0).sum()
        out[k] = seg if i >= anchor else -seg
    return pd.Series(out, index=paths.index)


def net_paths(paths: pd.DataFrame, carry_bp: pd.Series) -> pd.DataFrame:
    """``change_bp`` net of the specialness carry (``carry_bp``), as ``net_change_bp``."""
    return paths.assign(carry_bp=carry_bp.to_numpy(), net_change_bp=paths["change_bp"].to_numpy() - carry_bp.to_numpy())
