"""Canada's benchmark bonds - read / fetch / store, and the switch correction of the
``yield_boc`` P&L (pure parts in ``infra.processing.boc_benchmarks``).

* Snapshots ``Reference/Canada/Benchmarks`` (keys ``timestamp`` = the snapshot day, ``tenor``;
  ``maturity_date``, ``coupon``, ``effective``, ``source`` = ``page`` / ``wayback``): the BoC
  page fetched each run, plus archived captures from 2011 (``store_snapshots``).
* Zero curve ``RawData/BoC/ZeroCurve`` (keys ``timestamp``, ``maturity``; ``zero_pct`` x10000):
  fetched from the day after the last stored one at each run (published weekly, two weeks late).
* ``switches()``: every benchmark change seen in the snapshots - ``tenor``, ``effective`` and the
  new and previous bond.
* ``correct_switch_days(pnl)``: on a switch day, the benchmark change minus the new-minus-old
  yield spread priced off the zero curve of the latest day at least ``SPREAD_LAG_DAYS`` before -
  i.e. the move of the bond held the day before. Checked 2026-10-07 on switch days inferred
  from the zero curve 2001-2026: the predicted spread explains the jump with correlation 0.92-0.97
  (2/3/5/10y) and takes the switch-day error from 2.8-9.4bp to 1.3-2.4bp.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.api import boc_client
from infra.config import BOC_ZERO_DIR, CA_BENCHMARKS_DIR
from infra.processing import boc_benchmarks as pb
from infra.storage import parquet_store

FETCH_PAGE = boc_client.fetch_benchmark_page     # network hooks; tests stub them
FETCH_ZERO = boc_client.fetch_zero_curve
ZERO_START = "1986-01-01"
_ONE_DAY = pd.Timedelta(days=1)


# ------------------------------------------------------------------ snapshots
def store_snapshots(snaps: pd.DataFrame, *, root: Path = CA_BENCHMARKS_DIR) -> int:
    """``snaps``: ``timestamp, tenor, maturity_date, coupon, effective, source``; upserted."""
    if snaps.empty:
        return 0
    out = snaps.copy()
    for c in ("timestamp", "maturity_date", "effective"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    out["source"] = out["source"].astype("string")
    parquet_store.write_partitioned(out, root, ["timestamp", "tenor"])
    return len(out)


def snapshot_today(day=None, *, root: Path = CA_BENCHMARKS_DIR) -> int:
    day = pd.Timestamp.now().normalize() if day is None else pd.Timestamp(day).normalize()
    body = FETCH_PAGE()
    if body is None:
        return 0
    page = pb.parse_benchmark_page(body)
    return store_snapshots(page.assign(timestamp=day, source="page"), root=root)


def read_snapshots(*, root: Path = CA_BENCHMARKS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root) if parquet_store.has_data(root) else None
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "tenor", *pb.COLUMNS[1:], "source"])
    return df.sort_values(["timestamp", "tenor"]).reset_index(drop=True)


def switches(*, root: Path = CA_BENCHMARKS_DIR) -> pd.DataFrame:
    """``tenor, effective, maturity_date, coupon, prev_maturity, prev_coupon`` - one row per
    benchmark change seen in the snapshots (the first benchmark of each tenor has no prev)."""
    h = pb.benchmark_history(read_snapshots(root=root))
    if h.empty:
        return pd.DataFrame(columns=["tenor", "effective", "maturity_date", "coupon", "prev_maturity", "prev_coupon"])
    h = h.drop_duplicates(["tenor", "maturity_date", "coupon"], keep="first").sort_values(["tenor", "effective"])
    g = h.groupby("tenor")
    h["prev_maturity"], h["prev_coupon"] = g["maturity_date"].shift(), g["coupon"].shift()
    return h[["tenor", "effective", "maturity_date", "coupon", "prev_maturity", "prev_coupon"]].reset_index(drop=True)


# ------------------------------------------------------------------ zero curve
def update_zero_curve(*, root: Path = BOC_ZERO_DIR, now=None) -> int:
    """Fetch from the day after the last stored curve (the whole history the first time)."""
    now = pd.Timestamp.now().normalize() if now is None else pd.Timestamp(now)
    last = None
    if parquet_store.has_data(root):
        df = parquet_store.read_partitioned(root, start=now - pd.Timedelta(days=120))
        last = None if df is None or df.empty else pd.Timestamp(df["timestamp"].max())
    start = pd.Timestamp(ZERO_START) if last is None else last + _ONE_DAY
    if start > now:
        return 0
    z = FETCH_ZERO(start, now)
    if z is None or z.empty:
        return 0
    out = pd.DataFrame({"timestamp": pd.to_datetime(z["timestamp"]).astype("datetime64[ms]"),
                        "maturity": z["maturity"].astype("float64"),
                        "zero_pct": (z["zero"].astype("float64") * 100 * 10000).round().astype("Int32")})
    parquet_store.write_partitioned(out, root, ["timestamp", "maturity"])
    return len(out)


def zero_curve_before(day, *, lag_days: int = pb.SPREAD_LAG_DAYS, root: Path = BOC_ZERO_DIR):
    """``(curve day, maturities, zero decimals)`` of the latest stored curve at least
    ``lag_days`` before ``day``; None if none within 60 days of that."""
    cutoff = pd.Timestamp(day).normalize() - pd.Timedelta(days=lag_days)
    df = parquet_store.read_partitioned(root, start=cutoff - pd.Timedelta(days=60), end=cutoff + _ONE_DAY) \
        if parquet_store.has_data(root) else None
    if df is None or df.empty:
        return None
    d = df["timestamp"].max()
    c = df[df["timestamp"] == d].sort_values("maturity")
    return pd.Timestamp(d), c["maturity"].to_numpy(dtype="float64"), c["zero_pct"].astype("float64").to_numpy() / 1e6


# ------------------------------------------------------------------ the P&L correction
def bond_label(maturity, coupon) -> str:
    return f"CA {pd.Timestamp(maturity).date()} {float(coupon):g}%"


def correct_switch_days(pnl: pd.DataFrame, *, root: Path = CA_BENCHMARKS_DIR, zero_root: Path = BOC_ZERO_DIR) -> pd.DataFrame:
    """``yield_boc`` P&L rows with each benchmark switch day corrected (module docstring),
    and every row labelled with the bond behind it (``cusip`` / ``prev_cusip``) where the
    snapshots know it. A switch day with no zero curve keeps its raw change, labelled."""
    if pnl.empty:
        return pnl
    sw = switches(root=root)
    if sw.empty:
        return pnl
    out = pnl.copy()
    tenor = out["ticker"].astype(str).str.extract(r"CA_BOND_(\d+)y")[0].astype(float)
    for T, g in sw.groupby("tenor"):
        g = g.sort_values("effective")
        rows = tenor.eq(T)
        if not rows.any():
            continue
        days = pd.to_datetime(out.loc[rows, "timestamp"])
        prev_days = pd.to_datetime(out.loc[rows, "prev_timestamp"])
        idx = np.searchsorted(g["effective"].to_numpy(dtype="datetime64[ns]"), days.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        pidx = np.searchsorted(g["effective"].to_numpy(dtype="datetime64[ns]"), prev_days.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        labels = [bond_label(m, c) for m, c in zip(g["maturity_date"], g["coupon"])]
        out.loc[rows, "cusip"] = [labels[i] if i >= 0 else None for i in idx]
        out.loc[rows, "prev_cusip"] = [labels[i] if i >= 0 else None for i in pidx]
        for k, (r, i, j) in enumerate(zip(out.index[rows], idx, pidx)):
            if i < 0 or j < 0 or i == j or int(T) not in pb.CORRECTED_TENORS:
                continue
            new, old = g.iloc[i], g.iloc[j]
            curve = zero_curve_before(out.at[r, "timestamp"], root=zero_root)
            if curve is None:
                continue
            _, mats, zero = curve
            day = pd.Timestamp(out.at[r, "timestamp"])
            spread = pb.bond_yield_from_zero(new["coupon"], new["maturity_date"], day, mats, zero) \
                - pb.bond_yield_from_zero(old["coupon"], old["maturity_date"], day, mats, zero)
            if np.isfinite(spread):
                out.at[r, "pnl_per_dv01"] = out.at[r, "pnl_per_dv01"] + spread * 100.0
                out.at[r, "pnl"] = out.at[r, "pnl_per_dv01"]
    return out
