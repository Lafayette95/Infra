"""FX-hedged bond yields, 5 x 5 (root CLAUDE.md 16; maths ``infra.analytics.hedged_yields``):
every bond of ``HEDGED_YIELD_COUNTRIES`` hedged into every base currency at
``HEDGED_YIELD_TENORS``, by the rolling 3-month hedge - from FX swaps (``rolling_3m_fx``, the
default) or from OIS + basis (``rolling_3m``, v1) - and the maturity-matched one.
Disk only. Store ``Derived/HedgedYields`` (keys ``timestamp`` = day, ``bond`` =
``<country>_BOND_<t>y``, ``base`` currency, ``method``).

Inputs: bond yields (US CMT, DE our Bund curve - Svensson, UK BoE, JP MoF, CA BoC; all
semi-annual), the OIS curves (``HEDGED_YIELD_OIS``) - the 3-month rate from the curve's
discount factor, the T-year par rate in each currency's conventions - and the cross-currency
basis vs USD (``XccyBasisCloses``, clean; ``fill_basis`` / ``short_basis``, the source kept per
row). Each component is the day's own (each market's own close - levels, not synchronized
P&L). ``available_at`` = when the last input was public (the bond and swap availability rules,
the basis' 16:00 New York close, DTCC's posting of the FX swaps' day).

``rolling_3m_fx`` stores the FX-implied all-in cost (r + b) as ``rate_*`` with the bases 0 (the
basis is inside it); USD's leg is SOFR OIS 3m, as in v1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import hedged_yields as hy
from infra.config import (BUND_CURVES_DIR, DAILY_BONDS_DIR, HEDGED_YIELD_BASIS_CARRY_DAYS, HEDGED_YIELD_BASIS_PAIR,
                          HEDGED_YIELD_COUNTRIES, HEDGED_YIELD_FX_GAP_CARRY_DAYS, HEDGED_YIELD_OIS, HEDGED_YIELD_TENORS, HEDGED_YIELDS_DIR, OIS_CURVES,
                          FX_IMPLIED_DIR, OIS_CURVES_DIR, SWAP_CLOSES, SWAP_CURVES, XCCY_BASIS_CLOSES_DIR)
from infra.storage import parquet_store

KEYS = ["timestamp", "bond", "base", "method"]
COLUMNS = KEYS + ["tenor", "currency", "local_yield", "rate_bond_ccy", "rate_base_ccy", "basis_bond_bp", "basis_base_bp",
                  "hedged_yield", "base_yield", "pickup_bp", "basis_source", "available_at"]
_ONE_DAY = pd.Timedelta(days=1)
_FX_POSTED = pd.Timedelta(hours=24, minutes=30)   # DTCC posts UTC day D shortly after 00:00 on D+1


def _bond_yields(start, end, *, bonds_root: Path, bund_root: Path) -> pd.DataFrame:
    """Day x ``<country>_BOND_<t>y`` (%), each country from its source."""
    from infra.pipeline.bonds import read_bonds_from_disk
    from infra.pipeline.bund_curves import read_bund_curves
    tick = [f"{c}_BOND_{t}y" for c, (_, src) in HEDGED_YIELD_COUNTRIES.items() if src != "curve" for t in HEDGED_YIELD_TENORS]
    b = read_bonds_from_disk(tick, pd.Timestamp(start), pd.Timestamp(end) + _ONE_DAY, root=bonds_root)
    wide = b.pivot_table(index="timestamp", columns="ticker", values="par_yield") if len(b) else pd.DataFrame()
    wide.columns = [str(c) for c in wide.columns]
    de = read_bund_curves(start, end, method="svensson", root=bund_root)
    if len(de):
        d = de.set_index(pd.to_datetime(de["timestamp"]))[[f"par_{t}y" for t in HEDGED_YIELD_TENORS]]
        d.columns = [f"DE_BOND_{t}y" for t in HEDGED_YIELD_TENORS]
        wide = wide.join(d, how="outer") if len(wide) else d
    return wide.sort_index()


def _ois_rates(start, end, *, ois_root: Path) -> dict[str, pd.DataFrame]:
    """Per currency: day x {"3m", 2, 5, 10, 30} semi-annual rates (%)."""
    from infra.analytics import swap_curve as sc
    from infra.pipeline.ois_curves import pillar_schedule, read_ois_curves
    out = {}
    for ccy, curve in HEDGED_YIELD_OIS.items():
        df = read_ois_curves(start, pd.Timestamp(end) + _ONE_DAY, curve=curve, root=ois_root)
        conv = SWAP_CURVES[ccy]
        rows = {}
        for ts, g in df.groupby("timestamp"):
            day = pd.Timestamp(ts).normalize()
            c = sc.curve_from_nodes(g)
            t3 = ((day + pd.DateOffset(months=3)) - day).days / 365.0
            r = {"3m": hy.df_to_semiannual(float(c.discount([t3 * 365.0 / 365.25])[0]), t3)}
            for t in HEDGED_YIELD_TENORS:
                par = sc.par_rate(c, day, pillar_schedule(day, t, ccy))
                r[t] = hy.par_ois_to_semiannual(par, day_basis=conv.day_basis, freq_months=conv.fixed_freq_months,
                                                tenor_years=t)
            rows[day] = r
        out[ccy] = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    return out


def _basis(start, end, *, root: Path) -> dict[str, dict]:
    """Per non-USD currency: {"3m": frame, T: frame} of day -> basis_bp / basis_source,
    already on the ACT/365 scale."""
    from infra.pipeline.xccy_basis import read_closes
    lo = pd.Timestamp(start) - pd.Timedelta(days=30)        # history for the carry
    c = read_closes(lo, pd.Timestamp(end) + _ONE_DAY, clean=True, root=root)
    out = {}
    for ccy, pair in HEDGED_YIELD_BASIS_PAIR.items():
        g = c[c["pair"] == pair]
        if g.empty:
            continue
        obs = g.assign(day=pd.to_datetime(g["timestamp"]).dt.normalize()).pivot_table(index="day", columns="tenor",
                                                                                       values="basis_bp")
        scale = 365.0 / SWAP_CURVES[ccy].day_basis
        res = {"3m": hy.short_basis(obs, carry_days=HEDGED_YIELD_BASIS_CARRY_DAYS)}
        for t in HEDGED_YIELD_TENORS:
            res[t] = hy.fill_basis(obs, t * 12, carry_days=HEDGED_YIELD_BASIS_CARRY_DAYS)
        for k in res:
            res[k]["basis_bp"] = res[k]["basis_bp"] * scale
        out[ccy] = res
    return out


def _fx_costs(start, end, rates: dict, *, root: Path) -> dict[str, pd.DataFrame]:
    """Per non-USD currency: day -> ``cost`` (r + b, %) / ``source`` from FX swaps."""
    from infra.pipeline.fx_implied import read_fx_implied
    fx = read_fx_implied(pd.Timestamp(start) - pd.Timedelta(days=30), end, root=root)
    out = {}
    for ccy, g in fx.groupby("currency"):
        r = rates.get(ccy)
        if r is None or r.empty:
            continue
        out[ccy] = hy.fx_hedge_cost(g.set_index(pd.to_datetime(g["timestamp"]))["rate_pct"], r["3m"],
                                    carry_days=HEDGED_YIELD_FX_GAP_CARRY_DAYS)
    return out


class _Availability:
    """When each input of a day became public, computed once per series (``available_at`` of
    the bond and swap rules; the basis' 16:00 New York close)."""

    def __init__(self, days):
        from infra.pipeline.series_panel import available_at
        from infra.trading_calendar import snap_instants
        self.days = pd.DatetimeIndex(days)
        self._bond = {}
        self._avail = available_at
        self.ois = {}
        for ccy, curve in HEDGED_YIELD_OIS.items():   # its close's snap + 4h (the widest fallback window)
            sc = SWAP_CLOSES[OIS_CURVES[curve].close]
            self.ois[ccy] = pd.Series(snap_instants(self.days, sc.local_time, sc.timezone), index=self.days) + pd.Timedelta(hours=4)
        self.basis = pd.Series(snap_instants(self.days, "16:00", "America/New_York"), index=self.days)

    def bond(self, ticker: str) -> pd.Series:
        if ticker not in self._bond:
            self._bond[ticker] = pd.Series(self._avail(f"bond:{ticker}", self.days), index=self.days)
        return self._bond[ticker]

    def at(self, day, bond: str, base_bond: str, ccys) -> pd.Timestamp:
        return max([self.bond(bond)[day], self.bond(base_bond)[day], self.basis[day], *(self.ois[c][day] for c in ccys)])


def compute_hedged_yields(start, end, *, bonds_root: Path = DAILY_BONDS_DIR, bund_root: Path = BUND_CURVES_DIR,
                          ois_root: Path = OIS_CURVES_DIR, basis_root: Path = XCCY_BASIS_CLOSES_DIR,
                          fx_root: Path = FX_IMPLIED_DIR) -> pd.DataFrame:
    """Rows for days in ``[start, end]`` (module docstring). No writes."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    y = _bond_yields(start, end, bonds_root=bonds_root, bund_root=bund_root)
    if y.empty:
        return pd.DataFrame(columns=COLUMNS)
    y.index = pd.DatetimeIndex(y.index)
    rates = _ois_rates(start, end, ois_root=ois_root)
    basis = _basis(start, end, root=basis_root)
    fxc = _fx_costs(start - pd.Timedelta(days=30), end, rates, root=fx_root)
    days = y.index[(y.index >= start) & (y.index <= end)]
    avail = _Availability(days)
    rows = []
    for day in days:
        for fc, (fccy, _) in HEDGED_YIELD_COUNTRIES.items():
            for bc, (bccy, _) in HEDGED_YIELD_COUNTRIES.items():
                for t in HEDGED_YIELD_TENORS:
                    bond, base_bond = f"{fc}_BOND_{t}y", f"{bc}_BOND_{t}y"
                    yf = y.at[day, bond] if bond in y else np.nan
                    yb = y.at[day, base_bond] if base_bond in y else np.nan
                    if pd.isna(yf):
                        continue
                    for method in hy.METHODS:
                        key = t if method == "matched" else "3m"
                        def rate(ccy):
                            r = rates.get(ccy)
                            return r.at[day, key] if r is not None and day in r.index and key in r else np.nan
                        def bas(ccy):
                            if ccy == "USD":
                                return 0.0, None
                            b = basis.get(ccy, {}).get(key)
                            if b is None or day not in b.index:
                                return np.nan, None
                            return float(b.at[day, "basis_bp"]), b.at[day, "basis_source"]
                        def cost(ccy):
                            if ccy == "USD":
                                return rate("USD"), None
                            f = fxc.get(ccy)
                            if f is None or day not in f.index or pd.isna(f.at[day, "cost"]):
                                return np.nan, None
                            return float(f.at[day, "cost"]), f"fx {f.at[day, 'source']}"
                        if fc == bc:
                            h, rf, rb, bf, bb, src = yf, np.nan, np.nan, 0.0, 0.0, "same currency"
                        elif method == "rolling_3m_fx":
                            (rf, sf), (rb, sb) = cost(fccy), cost(bccy)
                            bf = bb = 0.0
                            h = hy.hedged_yield(yf, rf, 0.0, rb, 0.0)
                            src = "; ".join(f"{c} {s}" for c, s in ((fccy, sf), (bccy, sb)) if s)
                        else:
                            rf, rb = rate(fccy), rate(bccy)
                            (bf, sf), (bb, sb) = bas(fccy), bas(bccy)
                            h = hy.hedged_yield(yf, rf, bf, rb, bb)
                            src = "; ".join(f"{c} {s}" for c, s in ((fccy, sf), (bccy, sb)) if s)
                        rows.append({"timestamp": day, "bond": bond, "base": bccy, "method": method, "tenor": t,
                                     "currency": fccy, "local_yield": yf, "rate_bond_ccy": rf, "rate_base_ccy": rb,
                                     "basis_bond_bp": bf, "basis_base_bp": bb, "hedged_yield": h, "base_yield": yb,
                                     "pickup_bp": (h - yb) * 100.0 if pd.notna(h) and pd.notna(yb) else np.nan,
                                     "basis_source": src or None,
                                     "available_at": max(avail.at(day, bond, base_bond, {fccy, bccy}),
                                                         day + _FX_POSTED) if method == "rolling_3m_fx" and fc != bc
                                     else avail.at(day, bond, base_bond, {fccy, bccy})})
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    df = df.dropna(subset=["hedged_yield"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
    df["available_at"] = pd.to_datetime(df["available_at"]).astype("datetime64[ms]")
    for c in ("bond", "base", "method", "currency", "basis_source"):
        df[c] = df[c].astype("string")
    return df.reset_index(drop=True)


def store_hedged_yields(df: pd.DataFrame, start, end, *, root: Path = HEDGED_YIELDS_DIR) -> int:
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    if parquet_store.has_data(root):
        parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).ge(lo)
                                   & pd.to_datetime(part["timestamp"]).lt(hi))
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, KEYS)
    return len(df)


def read_hedged_yields(start=None, end=None, *, bond=None, base=None, method: str | None = None,
                       as_of=None, root: Path = HEDGED_YIELDS_DIR) -> pd.DataFrame:
    """Stored rows, optionally filtered; ``as_of`` keeps only rows public by then."""
    eq = {k: [v] for k, v in (("bond", bond), ("base", base), ("method", method)) if v is not None}
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + _ONE_DAY, equals_in=eq or None) \
        if parquet_store.has_data(root) else None
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    for c in ("bond", "base", "method", "currency", "basis_source"):
        df[c] = df[c].astype(str)
    if as_of is not None:
        df = df[pd.to_datetime(df["available_at"]) <= pd.Timestamp(as_of)]
    return df.sort_values(KEYS).reset_index(drop=True)
