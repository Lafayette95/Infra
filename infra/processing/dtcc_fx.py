"""FX swaps from the DTCC FOREX archive - pure (no I/O): the implied 3-month rate of a currency
against USD, read off spot-start FX swaps (root CLAUDE.md 16).

An FX swap is disseminated one record per LEG, linked as a package (``Package indicator``):
same execution second, same notionals, different delivery dates. Pairing a swap's near leg
(delivery within ``NEAR_MAX_DAYS``) with its far leg (``FAR_DAYS`` out) gives the forward
points from ONE trade at ONE instant - no separate spot needed. Combined with the USD rate over
the same period (the SOFR curve), they imply the currency's own rate over it, basis included -
the rolling hedge's cost, (r + b), straight from the FX market:

    EURUSD-style (USD per unit):  growth_ccy = growth_usd x near / far
    USDJPY-style (units per USD): growth_ccy = growth_usd x far / near

``QUOTE`` is the market convention per currency; a leg whose notionals' ratio disagrees with it
(beyond 2%) is dropped (verified 2026-10-07: the convention matched 88-97% of swaps; the
``Exchange rate basis`` field is unreliable - "CAD/USD" and "USD/CAD" both next to 1.38).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUOTE = {"EUR": "usd_per", "GBP": "usd_per", "JPY": "ccy_per", "CAD": "ccy_per"}
NEAR_MAX_DAYS = 4
FAR_DAYS = (60, 120)
LEG_COLUMNS = ["ccy", "exec", "exp", "rate", "usd_notional", "usd_per_from_notionals", "platform"]
SWAP_COLUMNS = ["day", "ccy", "exec", "near_rate", "far_rate", "near_days", "far_days", "usd_notional", "platform",
                "orientation_ok"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def legs(raw: pd.DataFrame, day) -> pd.DataFrame:
    """The day's executed (``NEWT``) package legs of FX swaps / forwards on ``QUOTE``'s
    currencies against USD, executed on UTC ``day``."""
    day = pd.Timestamp(day).normalize()
    fisn = raw["UPI FISN"].astype(str)
    want = [f"NA/{p} {c} USD" for p in ("Swaps", "Fwd") for c in QUOTE]
    x = raw[fisn.isin(want) & (raw["Action type"].astype(str) == "NEWT") & (raw["Package indicator"].astype(str) == "True")]
    if x.empty:
        return pd.DataFrame(columns=LEG_COLUMNS)
    exec_ = pd.to_datetime(x["Execution Timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    n1, n2 = _num(x["Notional amount-Leg 1"]), _num(x["Notional amount-Leg 2"])
    c1, c2 = x["Notional currency-Leg 1"].astype(str), x["Notional currency-Leg 2"].astype(str)
    usd = np.where(c1 == "USD", n1, np.where(c2 == "USD", n2, np.nan))
    oth = np.where(c1 == "USD", n2, np.where(c2 == "USD", n1, np.nan))
    out = pd.DataFrame({"ccy": x["UPI FISN"].astype(str).str[-7:-4].to_numpy(), "exec": exec_.to_numpy(),
                        "exp": pd.to_datetime(x["Expiration Date"], errors="coerce").to_numpy(),
                        "rate": _num(x["Exchange rate"]).to_numpy(), "usd_notional": usd,
                        "usd_per_from_notionals": usd / oth, "platform": x["Platform identifier"].astype(str).to_numpy()})
    out = out[(pd.to_datetime(out["exec"]).dt.normalize() == day)].dropna(subset=["rate", "exp"])
    return out.reset_index(drop=True)


def spot_start_swaps(lg: pd.DataFrame, day) -> pd.DataFrame:
    """Package legs paired into swaps: per (currency, execution second) the nearest and the
    farthest delivery; kept when the near leg is spot-ish and the far one ~3 months out."""
    day = pd.Timestamp(day).normalize()
    rows = []
    for (c, t), g in lg.groupby(["ccy", "exec"]):
        g = g.sort_values("exp")
        if len(g) < 2 or g["exp"].nunique() < 2:
            continue
        near, far = g.iloc[0], g.iloc[-1]
        dn, dfar = (near["exp"] - day).days, (far["exp"] - day).days
        if dn > NEAR_MAX_DAYS or not (FAR_DAYS[0] <= dfar <= FAR_DAYS[1]):
            continue
        conv = QUOTE[c]
        usd_per = near["rate"] if conv == "usd_per" else 1.0 / near["rate"]
        ok = bool(np.isfinite(near["usd_per_from_notionals"]) and abs(near["usd_per_from_notionals"] / usd_per - 1) < 0.02)
        rows.append((day, c, t, float(near["rate"]), float(far["rate"]), int(dn), int(dfar),
                     float(pd.Series([near["usd_notional"], far["usd_notional"]]).max()), str(near["platform"]), ok))
    return pd.DataFrame(rows, columns=SWAP_COLUMNS)


def implied_growth(near_rate: float, far_rate: float, usd_growth: float, ccy: str) -> float:
    """The currency's gross growth over near -> far implied by the swap and USD's growth."""
    fs = far_rate / near_rate
    return usd_growth / fs if QUOTE[ccy] == "usd_per" else usd_growth * fs
