"""FX-hedged bond yields - pure (no I/O).

A bond in currency F hedged into base currency B with a cross-currency swap: receive B
floating (+ its basis), pay F floating (+ its basis), so

    hedged = y_F - (r_F + b_F) + (r_B + b_B)

with r the hedge-tenor OIS rate and b the cross-currency basis on that currency's leg
against USD (b_USD = 0; a negative basis = paying to swap into USD - a USD investor in JGBs
pays TONA + b_JPY < TONA, a pickup). Two hedges (``method``):

* ``rolling_3m`` (DEFAULT): roll 3-month hedges - r = the 3-month OIS rate, b = the 3-month
  basis; what most real-money investors do; the pickup earned this quarter.
* ``matched``: one swap to the bond's maturity - r = the T-year par OIS rate, b = the T-year
  basis; locked for the bond's life; the like-for-like comparison of sovereigns.

Everything is converted to SEMI-ANNUAL bond-equivalent first - the basis of every stored bond
curve (root CLAUDE.md 13); an OIS leg is annual ACT/360 (SOFR, EUR STR) or ACT/365 (SONIA, TONA;
CORRA semi-annual beyond 1y), a few bp apart at 4-5% rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("rolling_3m", "matched")


def annual_to_semiannual(r_pct: float) -> float:
    """An annually compounded rate (%) as its semi-annual equivalent (%)."""
    return 200.0 * (np.sqrt(1.0 + r_pct / 100.0) - 1.0)


def par_ois_to_semiannual(r_pct: float, *, day_basis: float, freq_months: int, tenor_years: float) -> float:
    """A par OIS fixed rate (its own day count and frequency) as a semi-annual ACT/365 rate."""
    r365 = r_pct * 365.0 / day_basis
    freq = 12 if tenor_years <= 1 else freq_months
    if freq == 6:
        return r365
    return annual_to_semiannual(r365)


def df_to_semiannual(df: float, t_years: float) -> float:
    """The semi-annual rate (%) that discounts to ``df`` over ``t_years`` (ACT/365)."""
    if not (df > 0 and t_years > 0):
        return float("nan")
    return 200.0 * (df ** (-1.0 / (2.0 * t_years)) - 1.0)


def basis_to_365(b_bp: float, day_basis: float) -> float:
    """A basis spread (bp, on a leg accruing ACT/``day_basis``) on the ACT/365 scale."""
    return b_bp * 365.0 / day_basis


def hedged_yield(y_f: float, r_f: float, b_f_bp: float, r_b: float, b_b_bp: float) -> float:
    """Hedged yield (%), every input semi-annual (%) / bp (module docstring)."""
    return y_f - (r_f + b_f_bp / 100.0) + (r_b + b_b_bp / 100.0)


def fill_basis(obs: pd.DataFrame, target_months: int, *, carry_days: int = 10) -> pd.DataFrame:
    """Per day, the basis (bp) at ``target_months`` and how it was obtained: ``obs`` = day x
    tenor (months) of observed closes. ``same_day`` > ``interpolated`` (same day, between the
    neighbouring observed tenors) > ``carried`` (the tenor's last close within ``carry_days``
    business days). Point in time: only that day and earlier."""
    days = obs.index.sort_values()
    out = []
    last_seen: tuple | None = None
    for d in days:
        row = obs.loc[d].dropna()
        val, how = np.nan, None
        if target_months in row.index:
            val, how = float(row[target_months]), "same_day"
        else:
            lo = row[row.index < target_months]
            hi = row[row.index > target_months]
            if len(lo) and len(hi):
                a, b = lo.index[-1], hi.index[0]
                val = float(lo.iloc[-1] + (hi.iloc[0] - lo.iloc[-1]) * (target_months - a) / (b - a))
                how = f"interpolated {a}m-{b}m"
            elif last_seen is not None and np.busday_count(last_seen[0].date(), d.date()) <= carry_days:
                val, how = last_seen[1], f"carried from {last_seen[0].date()}"
        if target_months in row.index:
            last_seen = (d, float(row[target_months]))
        out.append((d, val, how))
    return pd.DataFrame(out, columns=["timestamp", "basis_bp", "basis_source"]).set_index("timestamp")


def short_basis(obs: pd.DataFrame, *, carry_days: int = 10) -> pd.DataFrame:
    """The rolling hedge's 3-month basis per day: 3m that day > the last 3m within
    ``carry_days`` business days > 6m that day > 1y that day (each flagged)."""
    out = []
    last3: tuple | None = None
    for d in obs.index.sort_values():
        row = obs.loc[d].dropna()
        if 3 in row.index:
            val, how = float(row[3]), "same_day"
            last3 = (d, val)
        elif last3 is not None and np.busday_count(last3[0].date(), d.date()) <= carry_days:
            val, how = last3[1], f"carried from {last3[0].date()}"
        elif 6 in row.index:
            val, how = float(row[6]), "6m same day"
        elif 12 in row.index:
            val, how = float(row[12]), "1y same day"
        else:
            val, how = np.nan, None
        out.append((d, val, how))
    return pd.DataFrame(out, columns=["timestamp", "basis_bp", "basis_source"]).set_index("timestamp")
