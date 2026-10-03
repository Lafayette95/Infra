"""At-the-money implied vol of options on a future, at a chosen horizon - pure, no I/O.

Built for the basis models' add-on IV (infra/models/basis/CLAUDE.md): a 1-factor scaling of
the level vol from the market's own forward-looking vol instead of an EWMA of past moves.

* ``implied_vols``: Black-76 implied vol per option (``infra.analytics.black76``), each
  option against its own underlying future's settlement, T in ACT/365 to the option's
  expiry. CBOT Treasury options are AMERICAN and pay their premium up front; both are
  ignored here (European, ``discount`` default 1): near the money and a few months out,
  the early-exercise premium is tiny, and discounting changes the vol by ~r x T (~0.5%
  relative) - documented approximations, not oversights.
* ``atm_vol``: per (day, expiry), the out-of-the-money options only (puts at/below the
  future, calls at/above), vol linearly interpolated in strike at the future's price.
* ``vol_at_horizon``: per day, the at-the-money vol at a horizon date, interpolated
  LINEARLY IN TOTAL VARIANCE (sigma^2 x T) between the two expiries around it, flat in
  vol outside them.
* ``normal_vol_points_per_day``: lognormal vol -> price vol in points per trading day
  (sigma x F / sqrt(252)); divide by a futures DV01 (points per bp) for bp of yield.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.analytics.black76 import implied_vol

TRADING_DAYS = 252.0


def implied_vols(options: pd.DataFrame, futures: pd.DataFrame, *, discount: float = 1.0) -> pd.DataFrame:
    """``options``: ``timestamp, underlying, option_type, strike, expiry, settlement_price``
    (strikes on the futures' own price scale); ``futures``: ``timestamp, ticker, price``.
    Adds ``future``, ``T`` and ``iv`` (NaN where no vol reproduces the price)."""
    px = futures.rename(columns={"ticker": "underlying", "price": "future"})[["timestamp", "underlying", "future"]]
    df = options.astype({"underlying": str, "option_type": str}).merge(px, on=["timestamp", "underlying"], how="inner")
    df = df.dropna(subset=["settlement_price", "future"])
    df["T"] = (pd.to_datetime(df["expiry"]) - pd.to_datetime(df["timestamp"])).dt.days / 365.0
    df = df[df["T"] > 0]
    df["iv"] = [implied_vol(p, f, k, t, discount, o) for p, f, k, t, o in
                zip(df["settlement_price"], df["future"], df["strike"], df["T"], df["option_type"])]
    return df.reset_index(drop=True)


def atm_vol(vols: pd.DataFrame, *, near_points: float = 1.5, band: tuple[float, float] = (2 / 3, 1.5)) -> pd.DataFrame:
    """Per (day, expiry): ``atm_iv`` interpolated in strike at the future, from the
    out-of-the-money options; ``future``, ``T``, ``n_options``. Options whose vol is
    outside ``band`` x the median vol of those within ``near_points`` of the future are
    dropped first: found 2026-10-03, some stored ZN settlements on quarter-point strikes
    are not premiums of those options at all (114.0, 39.7, 0.0003 next to clean ~1-3 point
    neighbours; another product or scale sharing the key - ``TOFIX.md``), inverting to
    0.4%..450% vols."""
    rows = []
    otm = vols[((vols["option_type"] == "P") & (vols["strike"] <= vols["future"]))
               | ((vols["option_type"] == "C") & (vols["strike"] >= vols["future"]))].dropna(subset=["iv"])
    for (day, expiry), g in otm.groupby(["timestamp", "expiry"]):
        f = float(g["future"].iloc[0])
        near = g.loc[(g["strike"] - f).abs() <= near_points, "iv"]
        if near.empty:
            continue
        med = float(near.median())
        g = g[g["iv"].between(band[0] * med, band[1] * med)]
        s = g.groupby("strike")["iv"].mean().sort_index()  # the ATM strike can carry both a call and a put
        if s.empty:
            continue
        below, above = s[s.index <= f], s[s.index >= f]
        if below.empty or above.empty:
            iv = float(s.iloc[np.argmin(np.abs(s.index.to_numpy() - f))])  # one-sided: nearest strike
        else:
            k0, k1 = below.index[-1], above.index[0]
            iv = float(below.iloc[-1]) if k1 == k0 else \
                float(below.iloc[-1] + (above.iloc[0] - below.iloc[-1]) * (f - k0) / (k1 - k0))
        rows.append((day, expiry, g["underlying"].iloc[0], f, float(g["T"].iloc[0]), iv, len(g)))
    return pd.DataFrame(rows, columns=["timestamp", "expiry", "underlying", "future", "T", "atm_iv", "n_options"])


def vol_at_horizon(atm: pd.DataFrame, horizon) -> pd.DataFrame:
    """Per day, the at-the-money vol at ``horizon`` - a date (the same for every day) or a
    Series of dates indexed by day (e.g. each day's delivery date)."""
    rows = []
    for day, g in atm.dropna(subset=["atm_iv"]).groupby("timestamp"):
        h = horizon.get(day, np.nan) if isinstance(horizon, pd.Series) else horizon
        if pd.isna(h):
            continue
        t = (pd.Timestamp(h) - pd.Timestamp(day)).days / 365.0
        g = g.sort_values("T")
        T, v = g["T"].to_numpy(), g["atm_iv"].to_numpy()
        if t <= T[0]:
            iv = v[0]
        elif t >= T[-1]:
            iv = v[-1]
        else:
            i = int(np.searchsorted(T, t))
            w = (t - T[i - 1]) / (T[i] - T[i - 1])
            iv = float(np.sqrt(((1 - w) * v[i - 1] ** 2 * T[i - 1] + w * v[i] ** 2 * T[i]) / t))
        rows.append((day, pd.Timestamp(h), t, iv, float(g["future"].iloc[0])))
    return pd.DataFrame(rows, columns=["timestamp", "horizon", "t", "iv", "future"])


def normal_vol_points_per_day(iv: float | np.ndarray, future: float | np.ndarray) -> np.ndarray:
    """Lognormal vol (annual) -> price vol in futures points per trading day."""
    return np.asarray(iv) * np.asarray(future) / np.sqrt(TRADING_DAYS)


def front_changes(settlements: pd.DataFrame, open_interest: pd.DataFrame) -> pd.Series:
    """Daily settlement changes (points) of the FRONT contract - the one with the most open
    interest the day before (known at the start of the day), same contract both days, so a
    roll never shows up as a move. Inputs wide: index day, columns contract."""
    prev = open_interest.shift(1).astype(float).dropna(how="all")
    front = prev.idxmax(axis=1)
    d1 = settlements - settlements.shift(1)
    return pd.Series([d1.at[d, c] if c in d1.columns else np.nan for d, c in front.items()], index=front.index).dropna()


def ewma_points(changes: pd.Series, lam: float) -> pd.Series:
    """EWMA vol (points per day) of daily changes, known at each day's close."""
    return np.sqrt((changes ** 2).ewm(alpha=1 - lam).mean())
