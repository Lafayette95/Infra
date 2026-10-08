"""Aggressor flow at DAILY horizons: multi-day "programs" (pure).

A program is a large participant working an order over several days (index extension, a
real-money rebalance, a hedge, a CTA's rebalance): it shows as same-sign net aggressive
flow on consecutive days, spread through the day, and - execution algorithms front-load -
continuing into the next morning. Inputs are the 1-minute signed bars
(``infra.pipeline.trades.read_signed_bars``) of ALL contracts of one root: a roll trade
done as two outright trades nets out across contracts, and the spread legs carry no
aggressor (``N``) at all.

Sessions (New York wall clock, trading day = CME's, root CLAUDE.md 6e):
``overnight`` 18:00-03:00 (Asia), ``london`` 03:00-08:20 (to the US cash open),
``ny`` 08:20-17:00.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.trading_calendar import local_wallclock, trading_day

SESSIONS = (("overnight", 18 * 60, 3 * 60), ("london", 3 * 60, 8 * 60 + 20), ("ny", 8 * 60 + 20, 17 * 60))


def session_of(instants: pd.DatetimeIndex) -> np.ndarray:
    """``overnight`` / ``london`` / ``ny`` / ``""`` (the 17:00-18:00 halt) per UTC instant."""
    local = local_wallclock(instants, "America/New_York")
    minute = local.hour * 60 + local.minute
    out = np.full(len(instants), "", dtype=object)
    for name, a, b in SESSIONS:
        inside = (minute >= a) & (minute < b) if a < b else (minute >= a) | (minute < b)
        out[inside] = name
    return out


def daily_flow(bars: pd.DataFrame, dataset: str = "GLBX.MDP3", bucket_minutes: int = 30) -> pd.DataFrame:
    """Per trading day: net signed volume in total and per session, aggressor volume, and
    ``evenness`` - the volume-weighted share of the day's ``bucket_minutes`` buckets whose
    net flow has the day's sign (1 = every bucket the same way: a steady program; ~0.5 =
    two-way trading; low = one burst against a background the other way), and
    ``first_last``: the day's first and last bar (UTC) for pricing."""
    b = bars.copy()
    idx = pd.DatetimeIndex(b["timestamp"])
    b["day"] = trading_day(idx, dataset)
    b["session"] = session_of(idx)
    b["bucket"] = idx.floor(f"{bucket_minutes}min")
    out = b.groupby("day").agg(net=("signed_volume", "sum"), buy=("buy_volume", "sum"), sell=("sell_volume", "sum"))
    for name, *_ in SESSIONS:
        out[f"net_{name}"] = b[b["session"] == name].groupby("day")["signed_volume"].sum()
    bk = b.groupby(["day", "bucket"])["signed_volume"].sum().rename("sv").reset_index()
    bk = bk.join(out["net"], on="day")
    bk["agree"] = np.sign(bk["sv"]) == np.sign(bk["net"])
    bk["w"] = bk["sv"].abs()
    out["evenness"] = bk.groupby("day").apply(lambda g: (g["w"] * g["agree"]).sum() / g["w"].sum() if g["w"].sum() else np.nan)
    out["imbalance"] = out["net"] / (out["buy"] + out["sell"])
    return out.fillna({f"net_{n}": 0 for n, *_ in SESSIONS})


def oi_flow_proxy(daily: pd.DataFrame) -> pd.Series:
    """The classic daily proxy from settlements alone: open-interest change x the sign of
    the price change (+ = positions ADDED in the direction prices moved: new longs on an up
    day, new shorts on a down day, read as net buying / selling pressure). ``daily``: one
    root's rows with ``oi`` (summed over contracts) and ``price_change`` (same-contract)."""
    return daily["oi"].diff() * np.sign(daily["price_change"])
