"""Bond yield benchmarks (CLAUDE.md 13, 18): build / store / read the on-the-run yield
series, and ONE reader over every source of a ``<country>_BOND_<tenor>y`` ticker.

``read_bond_yields(tickers, start, end, source=...)`` returns the same columns whatever
the source, so a consumer that only knows ``US_BOND_10y`` (a backtest's pnl) switches
between the CMT par curve (``"cmt"``, Daily/Bonds), the on-the-run bond's yield
(``"otr"``, Derived/OTRYields) and our own fitted curve's par yield at the tenor (``"curve"``,
Derived/TreasuryCurves, spline method) by the source alone. Reads disk only.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.config import (
    BOND_YIELD_SOURCES,
    TREASURY_CURVES_DIR,
    DAILY_BONDS_DIR,
    DAILY_TREASURY_PRICES_DIR,
    OTR_YIELDS_DIR,
    TREASURY_OTR_DIR,
    TREASURY_PRICES_START,
)
from infra.pipeline.bonds import read_bonds_from_disk
from infra.pipeline.treasury_otr import read_otr
from infra.pipeline.treasury_prices import read_prices
from infra.processing import otr_yields as oy
from infra.storage import parquet_store

log = logging.getLogger(__name__)
YIELD_COLUMNS = ["timestamp", "ticker", "source", "yield"]
_ONE_DAY = pd.Timedelta(days=1)


def build_otr_yields(start=TREASURY_PRICES_START, end=None, *, root: Path = OTR_YIELDS_DIR,
                     otr_root: Path = TREASURY_OTR_DIR, prices_root: Path = DAILY_TREASURY_PRICES_DIR) -> int:
    """Compute and store the on-the-run yields for days in ``[start, end]`` (default
    through today), REPLACING those days (a re-fetched price or a revised map must not
    leave a stale row). Needs the OTR map and the prices on disk."""
    start = pd.Timestamp(start).normalize()
    end = (pd.Timestamp.now().normalize() if end is None else pd.Timestamp(end).normalize()) + _ONE_DAY
    df = oy.otr_yields(read_otr(start, end, rank=0, root=otr_root), read_prices(start, end, root=prices_root))
    parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).between(start, end, inclusive="left"))
    if df.empty:
        return 0
    parquet_store.write_partitioned(oy.encode(df), root, oy.OTR_YIELD_KEYS)
    log.info("OTR yields %s..%s: %d rows", start.date(), (end - _ONE_DAY).date(), len(df))
    return len(df)


def read_otr_yields(tickers, start, end, *, root: Path = OTR_YIELDS_DIR) -> pd.DataFrame:
    """Stored on-the-run rows (with the bond behind each) in ``[start, end)``."""
    raw = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end),
                                         equals_in={"ticker": list(tickers)})
    if raw is None or raw.empty:
        return pd.DataFrame(columns=oy.OTR_YIELD_COLUMNS)
    return oy.decode(raw[oy.OTR_YIELD_COLUMNS]).sort_values(oy.OTR_YIELD_KEYS).reset_index(drop=True)


def read_curve_yields(tickers, start, end, *, method: str = "spline", root: Path = TREASURY_CURVES_DIR) -> pd.DataFrame:
    """Our fitted curve's PAR yield at each ticker's tenor (``US_BOND_10y`` -> ``par_10y``), in
    ``[start, end)``: ``timestamp, ticker, yield``."""
    from infra.pipeline.treasury_curves import read_curves
    c = read_curves(pd.Timestamp(start), pd.Timestamp(end) - _ONE_DAY, method=method, root=root)
    rows = []
    for t in tickers:
        col = "par_" + str(t).rsplit("_", 1)[-1]
        if col in c.columns:
            rows.append(pd.DataFrame({"timestamp": pd.to_datetime(c["timestamp"]), "ticker": t,
                                      "yield": c[col].astype("float64")}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "ticker", "yield"])
    return out.dropna(subset=["yield"])


def read_bond_yields(tickers, start, end, *, source: str = "cmt", details: bool = False,
                     cmt_root: Path = DAILY_BONDS_DIR, otr_root: Path = OTR_YIELDS_DIR,
                     curve_root: Path = TREASURY_CURVES_DIR) -> pd.DataFrame:
    """``timestamp, ticker, source, yield`` (percent) in ``[start, end)`` from ``source``
    (BOND_YIELD_SOURCES). ``details=True`` adds the bond behind each ``"otr"`` row
    (``cusip``, ``coupon``, ``maturity_date``, ``price_eod``)."""
    if source not in BOND_YIELD_SOURCES:
        raise ValueError(f"unknown bond yield source {source!r}; one of {BOND_YIELD_SOURCES}")
    if source == "cmt":
        df = read_bonds_from_disk(list(tickers), pd.Timestamp(start), pd.Timestamp(end), root=cmt_root)
        df = df.rename(columns={"par_yield": "yield"})
    elif source == "curve":
        df = read_curve_yields(tickers, start, end, root=curve_root)
    else:
        df = read_otr_yields(tickers, start, end, root=otr_root)
    df = df.assign(source=source)
    cols = YIELD_COLUMNS + ([c for c in oy.OTR_YIELD_COLUMNS if c not in YIELD_COLUMNS] if details and source == "otr" else [])
    return df[cols].reset_index(drop=True)
