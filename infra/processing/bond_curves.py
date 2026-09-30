"""Turn a source's curve (long ``timestamp, maturity, value``) into stored par-yield rows.

Pure functions, no I/O. The stored schema (CLAUDE.md 6a/6b applied to yields):
``timestamp`` (the curve's business date), ``ticker`` (``US_BOND_10y``, category on
decode), ``par_yield`` (percent, x10000 fixed-point, nullable ``Int32`` on disk - 0.01bp
resolution, far finer than any source publishes).

How par yields are obtained is a named, swappable method (``PAR_METHODS``, chosen per
curve by ``BondCurve.par_method``): ``None`` means the source already publishes par at
each tenor (US Treasury CMT); a source publishing a spot curve (the BoE) or a model's
parameters (the Bundesbank's Svensson fit, evaluated by ``svensson_curve``) derives par
from it - kept modular so the derivation can change without touching fetching/storage.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import pandas as pd

from infra.config import PRICE_SCALE, BondCurve, bond_ticker

BOND_COLUMNS = ["timestamp", "ticker", "par_yield"]
BOND_KEYS = ["timestamp", "ticker"]


def empty_bonds() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ms]"),
        "ticker": pd.Series(dtype="str"),
        "par_yield": pd.Series(dtype="float64"),
    })[BOND_COLUMNS]


def published_par(curve: pd.DataFrame, tenors: tuple[int, ...]) -> pd.DataFrame:
    """The source already publishes par yields: just the requested tenors.
    Returns long ``timestamp, tenor, par_yield``."""
    out = curve[curve["maturity"].isin([float(t) for t in tenors])]
    return pd.DataFrame({"timestamp": out["timestamp"].to_numpy(), "tenor": out["maturity"].astype(int).to_numpy(),
                         "par_yield": out["value"].astype("float64").to_numpy()})


# Maturity grid a model curve (Svensson) is evaluated on before par is derived - every
# semi-annual date, so annual- AND semi-annual-coupon par can both be read off it.
MODEL_GRID = tuple(np.round(np.arange(0.5, 40.0001, 0.5), 10))


def svensson_zero(maturity: np.ndarray, b0, b1, b2, b3, t1, t2) -> np.ndarray:
    """Svensson (1994) zero rate at ``maturity`` years, in the parameters' own units
    (percent for the Bundesbank). Each argument may be an array (one per day)."""
    x1, x2 = maturity / t1, maturity / t2
    f1, f2 = (1 - np.exp(-x1)) / x1, (1 - np.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))


def svensson_curve(params: pd.DataFrame, grid=MODEL_GRID) -> pd.DataFrame:
    """Daily Svensson parameters (``timestamp, B0, B1, B2, B3, T1, T2``) -> the model's
    zero curve on ``grid``, long ``timestamp, maturity, value`` - the same shape a
    spot-publishing source returns, so the same ``PAR_METHODS`` apply."""
    if params.empty:
        return pd.DataFrame(columns=["timestamp", "maturity", "value"])
    m = np.asarray(grid, dtype=float)[None, :]
    p = {k: params[k].to_numpy(dtype=float)[:, None] for k in ["B0", "B1", "B2", "B3", "T1", "T2"]}
    z = svensson_zero(m, p["B0"], p["B1"], p["B2"], p["B3"], p["T1"], p["T2"])
    return pd.DataFrame({
        "timestamp": np.repeat(params["timestamp"].to_numpy(), m.shape[1]),
        "maturity": np.tile(m[0], len(params)),
        "value": z.ravel(),
    })


def discount_factors(spot_pct: np.ndarray, maturity: np.ndarray, compounding: str) -> np.ndarray:
    z = spot_pct / 100.0
    if compounding == "annual":
        return (1.0 + z) ** (-maturity)
    if compounding == "semiannual":
        return (1.0 + z / 2.0) ** (-2.0 * maturity)
    if compounding == "continuous":
        return np.exp(-z * maturity)
    raise ValueError(f"unknown compounding {compounding!r}")


def par_from_spot(curve: pd.DataFrame, tenors: tuple[int, ...], *, compounding: str,
                  coupons_per_year: int = 2, flat_short_end: bool = True) -> pd.DataFrame:
    """Par yield of a bullet bond paying ``coupons_per_year`` coupons, priced off the spot
    curve: ``par = f x (1 - DF(T)) / sum_i DF(t_i)`` over the coupon dates ``t_i = 1/f ..
    T`` - each DF read straight from the spot grid (the BoE's grid is every 0.5y, so every
    semi-annual coupon date of an integer tenor is a grid point; no interpolation).
    ``flat_short_end``: coupon dates BELOW the day's shortest published maturity take that
    maturity's spot rate - the BoE's curve only starts at its shortest fitted gilt, which
    is past 0.5y on ~36% of days since 2016 (first point 0.25-0.92y). Measured 2026-09-30
    over 1,758 days that DO have 0.5y, filling it flat from 1y instead: par error mean
    0.00bp, sd 0.07bp, worst 0.31bp at 2y (0.22bp 3y, 0.14bp 5y). A hole anywhere else
    still gives NO value for that tenor rather than a guessed one. ``compounding`` is the
    SPOT rates' convention. Returns long ``timestamp, tenor,
    par_yield`` (percent)."""
    wide = curve.pivot_table(index="timestamp", columns="maturity", values="value", aggfunc="last")
    rows = []
    for tenor in tenors:
        dates = np.round(np.arange(1, tenor * coupons_per_year + 1) / coupons_per_year, 10)
        if not set(dates) <= set(np.round(wide.columns.to_numpy(dtype=float), 10)):
            continue  # the grid doesn't carry every coupon date at all
        block = wide[list(dates)]
        if flat_short_end:
            leading = ~block.notna().cummax(axis=1)  # before the day's first published point
            block = block.mask(leading, block.bfill(axis=1))
        spot = block.to_numpy(dtype=float)
        df = discount_factors(spot, dates[None, :], compounding)
        par = coupons_per_year * (1.0 - df[:, -1]) / df.sum(axis=1) * 100.0
        rows.append(pd.DataFrame({"timestamp": wide.index, "tenor": tenor, "par_yield": par}))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "tenor", "par_yield"])
    return pd.concat(rows, ignore_index=True).dropna(subset=["par_yield"])


# name -> fn(curve long frame, tenors) -> long timestamp/tenor/par_yield
PAR_METHODS: dict[str, Callable[[pd.DataFrame, tuple[int, ...]], pd.DataFrame]] = {
    # Semi-annual coupons (gilts pay semi-annually), spot read as semi-annually
    # compounded. Chosen 2026-09-30 as the closer of the two readings to the BoE's own
    # 5/10/20y par series (still off by up to a few bp - TOFIX.md).
    "semiannual_from_spot": partial(par_from_spot, compounding="semiannual"),
    "semiannual_from_continuous_spot": partial(par_from_spot, compounding="continuous"),
    # Annual coupons (Bunds pay annually) off an ANNUALLY compounded zero curve - the
    # Bundesbank's Svensson convention: reproduces its own published par curve within
    # +-0.45bp (infra/api/bundesbank_client.py). Same curve, semi-annual coupons:
    # ``semiannual_from_annual_spot`` (2.7-3.8bp lower, 2026-09-30).
    "annual_from_annual_spot": partial(par_from_spot, compounding="annual", coupons_per_year=1),
    "semiannual_from_annual_spot": partial(par_from_spot, compounding="annual", coupons_per_year=2),
}


def to_par_rows(curve: pd.DataFrame, spec: BondCurve) -> pd.DataFrame:
    """A source's curve -> canonical (decoded) rows for ``spec``'s tenors."""
    if curve.empty:
        return empty_bonds()
    par = published_par(curve, spec.tenors) if spec.par_method is None else PAR_METHODS[spec.par_method](curve, spec.tenors)
    if par.empty:
        return empty_bonds()
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(par["timestamp"]).dt.normalize().astype("datetime64[ms]"),
        "ticker": [bond_ticker(spec.country, int(t)) for t in par["tenor"]],
        "par_yield": par["par_yield"].astype("float64"),
    })
    return out.sort_values(BOND_KEYS).reset_index(drop=True)[BOND_COLUMNS]


def encode_bonds(df: pd.DataFrame) -> pd.DataFrame:
    out = df[BOND_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    out["ticker"] = out["ticker"].astype(str)
    out["par_yield"] = (out["par_yield"] * PRICE_SCALE).round().astype("Int32")
    return out


def decode_bonds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["par_yield"] = out["par_yield"].astype("float64") / float(PRICE_SCALE)
    out["ticker"] = out["ticker"].astype("category")
    return out
