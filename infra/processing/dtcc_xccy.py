"""DTCC cross-currency basis swaps (CFTC Part 43, RATES report) -> clean basis prints. Pure.

* The BASIS is the spread on the non-USD leg (in the leg whose ``Notional currency`` isn't
  USD); a trade with a spread on BOTH legs is ambiguous and dropped. A spread reported on
  the USD-labelled leg is NOT reliably the opposite side: some JPY reporters put the JPY
  basis there (-52bp on a "USD" leg among -50bp markets), others the true USD-leg spread.
  Its sign is chosen as the reading closer to the same day's prints reported on the non-USD
  leg in the same and neighbouring tenors; with no such reference it's dropped.
* Units by ``Spread notation`` (found 2026-10-07): 3 = DECIMAL (-0.0055 = -55bp), 4 = BASIS
  POINTS (-54); a "decimal" beyond +-0.01 (a 100bp basis) is in fact a PERCENTAGE (-0.86,
  -0.118 next to -55 / -12bp markets) and read as one. Before this, lone bad prints of
  -4,400 to -20,000bp reached the closes.
* A print further than ``NEIGHBOUR_BP`` from the same day's median of its own and the
  neighbouring tenors is dropped (a lone print can't be judged against its own tenor).
* Kept: executions (``NEWT``/``TRAD`` in their latest corrected state - ``dtcc_trades``'
  linking), spot-starting (effective within ``SPOT_MAX_DAYS`` of the trade date), no
  upfront, OIS-vs-OIS legs per the spec, a standard tenor in months (maturity within
  ``TENOR_TOLERANCE_DAYS`` of effective + N months); then prints far from their tenor's
  median that day dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.processing import dtcc_trades as dt

RAW_COLUMNS = {**dt.RAW_COLUMNS, "Notional currency-Leg 2": "currency_leg2", "Spread-Leg 1": "spread_leg1",
               "Spread-Leg 2": "spread_leg2", "Spread notation-Leg 1": "notation_leg1",
               "Spread notation-Leg 2": "notation_leg2"}
NEIGHBOUR_BP = 20.0
DECIMAL_AS_PERCENT_ABOVE = 0.01
SPOT_MAX_DAYS = 5
TENOR_TOLERANCE_DAYS = 4
TRADE_COLUMNS = ["executed", "tenor", "rate", "basis_bp", "notional", "platform", "trade_id"]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Report rows -> typed (``dtcc_trades.normalize`` plus the second currency and the
    two spreads, decimal)."""
    base = dt.normalize(raw)
    for src, dst in (("Notional currency-Leg 2", "currency_leg2"), ("Spread-Leg 1", "spread_leg1"),
                     ("Spread-Leg 2", "spread_leg2"), ("Spread notation-Leg 1", "notation_leg1"),
                     ("Spread notation-Leg 2", "notation_leg2")):
        base[dst] = raw[src].to_numpy() if src in raw else pd.NA
    for leg in ("1", "2"):
        v = pd.to_numeric(base[f"spread_leg{leg}"], errors="coerce")
        n = pd.to_numeric(base[f"notation_leg{leg}"], errors="coerce")
        bp = np.where(n == 4, v, np.where(v.abs() > DECIMAL_AS_PERCENT_ABOVE, v * 100.0, v * 1e4))
        base[f"spread_leg{leg}"] = pd.Series(bp, index=base.index)  # now in BASIS POINTS
    return base


def basis_bp(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(basis bp, ``on_usd_leg``): the spread on the non-USD leg (``normalize`` has put both
    legs in bp), or - flagged - the magnitude reported on the USD-labelled leg, its sign to
    be resolved (``resolve_usd_leg``); NaN where both legs carry a spread."""
    s1, s2 = df["spread_leg1"].fillna(0.0), df["spread_leg2"].fillna(0.0)
    usd1 = df["currency"].astype(str).eq("USD")
    nonusd, usd = np.where(usd1, s2, s1), np.where(usd1, s1, s2)
    on_usd = (nonusd == 0) & (usd != 0)
    out = np.where(usd == 0, nonusd, np.where(nonusd == 0, usd, np.nan))
    return pd.Series(out, index=df.index), pd.Series(on_usd, index=df.index)


def resolve_usd_leg(df: pd.DataFrame, tenors) -> pd.Series:
    """Sign of the USD-leg-reported prints: +x or -x, whichever is closer to the day's
    median of non-USD-leg prints in the same and neighbouring tenors; NaN without one."""
    out = df["basis_bp"].copy()
    tenors = sorted(tenors)
    day = df["executed"].dt.normalize()
    ref = df[~df["on_usd_leg"]]
    for i in df.index[df["on_usd_leg"]]:
        k = tenors.index(int(df.at[i, "tenor"]))
        near = ref[(day[ref.index] == day[i]) & ref["tenor"].isin(tenors[max(0, k - 1):k + 2])]["basis_bp"]
        if near.empty:
            out[i] = np.nan
            continue
        x, m = df.at[i, "basis_bp"], near.median()
        out[i] = x if abs(x - m) <= abs(-x - m) else -x
    return out


def tenor_months(effective: pd.Series, maturity: pd.Series, tenors) -> pd.Series:
    out = pd.Series(np.nan, index=effective.index)
    for n in tenors:
        target = effective + pd.DateOffset(months=int(n))
        out[(maturity - target).abs() <= pd.Timedelta(days=TENOR_TOLERANCE_DAYS)] = n
    return out


def basis_trades(df: pd.DataFrame, spec, *, as_of=None) -> pd.DataFrame:
    """Clean basis prints of one ``infra.config.XccyBasisSpec`` from normalized records."""
    rows = df[(df["fisn"] == spec.fisn) & df["underlier"].astype(str).str.contains(spec.underlier_pattern, regex=True)]
    if rows.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    cur = dt.current_trades(dt.trade_events(rows), as_of)
    if cur.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    lag = (cur["effective"] - cur["executed"].dt.normalize()).dt.days
    cur = cur[lag.between(0, SPOT_MAX_DAYS) & ~cur["upfront"] & ~cur["non_standard"]].copy()
    cur["tenor"] = tenor_months(cur["effective"], cur["maturity"], spec.tenors_months)
    cur["basis_bp"], cur["on_usd_leg"] = basis_bp(cur)
    cur = cur.dropna(subset=["tenor", "basis_bp"])
    if cur.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    cur["basis_bp"] = resolve_usd_leg(cur, spec.tenors_months)
    cur = cur.dropna(subset=["basis_bp"])
    if cur.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    cur["tenor"] = cur["tenor"].astype(int)
    day = cur["executed"].dt.normalize()
    med = cur.groupby([day, "tenor"])["basis_bp"].transform("median")
    mad = cur.groupby([day, "tenor"])["basis_bp"].transform(lambda s: (s - s.median()).abs().median())
    cur = cur[(cur["basis_bp"] - med).abs() <= np.maximum(spec.off_market_bp, 5 * 1.4826 * mad)]
    # against the day's prints in its own and the neighbouring tenors (catches a lone print)
    tenors = sorted(spec.tenors_months)
    keep = np.ones(len(cur), dtype=bool)
    for (d, t), idx in cur.groupby([cur["executed"].dt.normalize(), "tenor"]).groups.items():
        k = tenors.index(t)
        near = tenors[max(0, k - 1):k + 2]
        ref = cur[(cur["executed"].dt.normalize() == d) & cur["tenor"].isin(near)]["basis_bp"]
        if len(ref) >= 3:
            keep[cur.index.get_indexer(idx)] = (cur.loc[idx, "basis_bp"] - ref.median()).abs() <= NEIGHBOUR_BP
    cur = cur[keep]
    cur["rate"] = cur["basis_bp"] / 100.0  # percent: the close estimator's unit
    return cur[TRADE_COLUMNS].sort_values("executed").reset_index(drop=True)
