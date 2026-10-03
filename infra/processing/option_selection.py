"""Near-the-money option selection for implied vol (Rule 2.3's local filter step). Pure.

For a level-vol scaling only the at-the-money vol at a couple of expiries is needed, so
per day: the ``n_expiries`` nearest expiries at least ``min_days`` away, and around each
underlying's settlement the ATM strike plus ``n_strikes`` on each side, OUT-OF-THE-MONEY
side only (puts at/below the ATM strike, calls at/above - deep in-the-money options have
little vega and a poorly conditioned inversion, CLAUDE.md 9). Each selected instrument
gets the CONTIGUOUS range of days it is needed, so its statistics are fetched once.

Strikes are on whatever scale the definitions use; ``strike_scale`` infers the power of
ten between strikes and futures prices (SR3 options are quoted at 100x their future,
CLAUDE.md 9), so the same code serves any root without an assumed convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SELECTION_COLUMNS = ["timestamp", "instrument_id", "underlying", "option_type", "strike", "expiry", "atm_strike",
                     "underlying_price"]


def strike_scale(strikes: pd.Series, prices: pd.Series) -> float:
    """The power of ten s with strike ~= s x futures price (1, 100, ...)."""
    ratio = float(np.nanmedian(strikes)) / float(np.nanmedian(prices))
    return float(10 ** round(np.log10(ratio)))


def select_near_the_money(definitions: pd.DataFrame, settlements: pd.DataFrame, *, n_expiries: int = 2,
                          n_strikes: int = 2, min_days: int = 5) -> pd.DataFrame:
    """``definitions``: ``instrument_id, underlying, option_type, strike, expiry`` (the
    union of the snapshots covering the period); ``settlements``: ``timestamp, ticker,
    price`` (the underlying futures). One row per (day, selected option)."""
    if definitions.empty or settlements.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    # with validity windows (instrument ids are reused, infra.pipeline.futures_options_iv)
    # an id can appear once per window; without them, one definition per id
    windowed = {"valid_from", "valid_to"} <= set(definitions.columns)
    defs = definitions if windowed else definitions.drop_duplicates("instrument_id")
    px = settlements.dropna(subset=["price"])
    scale = strike_scale(defs["strike"], px["price"])
    by_und = {u: g for u, g in defs.groupby("underlying")}
    prices = px.set_index(["timestamp", "ticker"])["price"]
    rows = []
    for day in sorted(px["timestamp"].unique()):
        day = pd.Timestamp(day)
        live = defs[defs["expiry"] >= day + pd.Timedelta(days=min_days)]
        if windowed:
            live = live[(live["valid_from"] <= day) & (live["valid_to"] >= day)]
        for expiry in sorted(live["expiry"].unique())[:n_expiries]:
            e = live[live["expiry"] == expiry]
            for und, g in e.groupby("underlying"):
                if (day, und) not in prices.index or und not in by_und:
                    continue
                f = float(prices[(day, und)]) * scale
                strikes = np.sort(g["strike"].unique())
                atm = strikes[np.argmin(np.abs(strikes - f))]
                i = int(np.searchsorted(strikes, atm))
                band = strikes[max(0, i - n_strikes): i + n_strikes + 1]
                pick = g[g["strike"].isin(band) & (((g["option_type"] == "P") & (g["strike"] <= atm))
                                                   | ((g["option_type"] == "C") & (g["strike"] >= atm)))]
                for r in pick.itertuples():
                    rows.append((day, r.instrument_id, und, r.option_type, r.strike, r.expiry, atm, f / scale))
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)


def fetch_ranges(selection: pd.DataFrame) -> pd.DataFrame:
    """Per instrument, the first and last day it is selected (one fetch range each)."""
    if selection.empty:
        return pd.DataFrame(columns=["instrument_id", "start", "end", "days"])
    g = selection.groupby("instrument_id")["timestamp"]
    return pd.DataFrame({"start": g.min(), "end": g.max(), "days": g.size()}).reset_index()
