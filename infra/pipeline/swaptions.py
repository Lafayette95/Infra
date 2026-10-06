"""Swaptions from the DTCC archive (root CLAUDE.md 16; spec ``infra.config.SWAPTIONS``):
records, executed prints with implied normal vols, an ATM vol surface, and open interest.
Disk only - the archived RATES zips (``infra.pipeline.dtcc``) and the OIS curve
(``infra.pipeline.ois_curves``).

Stores (all ``~/Database/Derived``):
* ``SwaptionRecords`` - every record of the spec's product, typed, keys ``file_day``,
  ``diss_id`` (a recomputed file day replaces its rows). The ledger and the prints are
  built from it, never from the zips again.
* ``SwaptionPrints`` - one row per executed trade with a premium (not a package leg),
  keys ``timestamp`` (execution, UTC), ``trade_id``: forward and annuity on the day's OIS
  curve, moneyness, the OUT-of-the-money reading and its normal vol.
* ``SwaptionVols`` - ATM normal vol per day at the standard points (median of prints
  within ``atm_band_bp`` of the forward), keys ``timestamp`` (day), ``expiry``, ``tenor``.
* ``SwaptionOI`` - open interest per day by expiry x tenor bucket and origin, keys
  ``timestamp`` (day), ``expiry_bucket``, ``tenor_bucket``, ``origin``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import swap_curve as sc
from infra.analytics import swaptions as sw
from infra.config import (
    DTCC_DIR,
    OIS_CURVES,
    OIS_CURVES_DIR,
    SWAP_CLOSES,
    SWAPTION_OI_DIR,
    SWAPTION_PRINTS_DIR,
    SWAPTION_RECORDS_DIR,
    SWAPTION_VOLS_DIR,
    SWAPTIONS,
    SwaptionSpec,
)
from infra.pipeline import dtcc
from infra.pipeline.ois_curves import read_ois_curves
from infra.processing import dtcc_swaptions as ds
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants

log = logging.getLogger(__name__)
REPORT = "RATES"
_ONE_DAY = pd.Timedelta(days=1)
RECORD_KEYS = ["file_day", "diss_id"]
PRINT_KEYS = ["timestamp", "trade_id"]
VOL_KEYS = ["timestamp", "expiry", "tenor"]
OI_KEYS = ["timestamp", "expiry_bucket", "tenor_bucket", "origin"]
EXPIRY_BUCKETS = ((0, 1 / 12, "<1m"), (1 / 12, 0.25, "1-3m"), (0.25, 0.5, "3-6m"), (0.5, 1.0, "6m-1y"),
                  (1.0, 2.0, "1-2y"), (2.0, 5.0, "2-5y"), (5.0, 10.0, "5-10y"), (10.0, 99.0, ">10y"))
TENOR_BUCKETS = ((0, 1.5, "1y"), (1.5, 3.5, "2-3y"), (3.5, 7.5, "5-7y"), (7.5, 12.5, "10y"), (12.5, 22.5, "15-20y"),
                 (22.5, 99.0, "30y"))


def _between_days(root: Path, column: str, lo, hi) -> None:
    lo, hi = pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize() + _ONE_DAY
    parquet_store.delete_where(root, lambda p: pd.to_datetime(p[column]).ge(lo) & pd.to_datetime(p[column]).lt(hi))


# ------------------------------------------------------------------------- records
def parse_records(start, end, *, spec: SwaptionSpec = SWAPTIONS["USD_SOFR"], dtcc_root: Path = DTCC_DIR) -> pd.DataFrame:
    """Every archived file day in ``[start, end]`` -> the spec's swaption records."""
    have = dtcc.archived_days(REPORT, root=dtcc_root)
    cols = list(ds.RAW_COLUMNS)
    frames = [ds.normalize(dtcc.read_dtcc_day(REPORT, d, columns=cols, root=dtcc_root), d, spec.fisn_pattern)
              for d in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()) if d in have]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ds.RECORD_COLUMNS)


def store_records(df: pd.DataFrame, start, end, *, root: Path = SWAPTION_RECORDS_DIR) -> int:
    """Replace the file days ``[start, end]`` by ``df``. Partitioned by ``file_day``
    (stored under ``timestamp`` = the file day, the store layer's partition column)."""
    _between_days(root, "timestamp", start, end)
    if df.empty:
        return 0
    out = df.assign(timestamp=df["file_day"])
    parquet_store.write_partitioned(out, root, ["timestamp", "diss_id"])
    return len(out)


def read_records(start=None, end=None, *, root: Path = SWAPTION_RECORDS_DIR) -> pd.DataFrame:
    """Records with file day in ``[start, end)``."""
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end))
    if df is None or df.empty:
        return pd.DataFrame(columns=ds.RECORD_COLUMNS)
    for c in ("action", "event", "label", "platform"):
        df[c] = df[c].astype("string")
    return df[ds.RECORD_COLUMNS].reset_index(drop=True)


# -------------------------------------------------------------------------- prints
def price_prints(execs: pd.DataFrame, curves: dict, spec: SwaptionSpec) -> pd.DataFrame:
    """Executed trades -> prints with forward, annuity, moneyness and the OTM vol. A trade
    is priced on the OIS curve of its (UTC) execution day; trades without one, without a
    premium, package legs, garbage notionals and very short options / tails are skipped."""
    snap = OIS_CURVES[spec.curve].close
    e = execs[(~execs["package"]) & (execs["premium"] > 0) & (execs["notional"] > 0)
              & (execs["notional"] <= spec.max_notional) & execs["strike"].between(0.01, 25.0)
              & execs["expiry"].notna() & execs["maturity"].notna()]
    rows = []
    for r in e.itertuples(index=False):
        day = pd.Timestamp(r.executed).normalize()
        c = curves.get(day)
        if c is None or (r.expiry - day).days < spec.min_expiry_days:
            continue
        start = sc.add_business_days(r.expiry, 2)
        tenor = (r.maturity - start).days / 365.25
        if tenor < spec.min_tenor_years:
            continue
        f, a = sw.forward_annuity(c, day, start, r.maturity)
        t = (pd.Timestamp(r.expiry) - pd.Timestamp(r.executed)).total_seconds() / (365.25 * 86400)
        k = r.strike / 100.0
        payer = k >= f  # the out-of-the-money reading (the label doesn't say)
        vol = sw.implied_normal_vol(r.premium / r.notional / a, f, k, t, payer)
        snap_at = snap_instants([day], SWAP_CLOSES[snap].local_time, SWAP_CLOSES[snap].timezone)[0]
        rows.append({"timestamp": r.executed, "trade_id": r.trade_id, "expiry_date": r.expiry, "maturity": r.maturity,
                     "t_years": t, "tenor_years": tenor, "strike": r.strike, "forward": f * 100.0,
                     "moneyness_bp": (k - f) * 1e4, "annuity": a, "otm_side": "payer" if payer else "receiver",
                     "vol_bp": vol, "notional": r.notional, "capped": bool(r.capped), "premium": r.premium,
                     "label": r.label, "platform": r.platform,
                     "minutes_from_snap": abs((pd.Timestamp(r.executed) - snap_at).total_seconds()) / 60.0})
    out = pd.DataFrame(rows)
    if len(out):
        for col in ("timestamp", "expiry_date", "maturity"):
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ms]")
        out["trade_id"] = out["trade_id"].astype(str)
    return out


def _curves(start, end, curve: str, ois_root: Path) -> dict:
    n = read_ois_curves(start, pd.Timestamp(end) + _ONE_DAY, curve=curve, root=ois_root)
    if n.empty:
        return {}
    n["day"] = pd.to_datetime(n["timestamp"]).dt.normalize()
    return {d: sc.curve_from_nodes(g) for d, g in n.groupby("day")}


def compute_prints(start, end, *, spec: SwaptionSpec = SWAPTIONS["USD_SOFR"], records_root: Path = SWAPTION_RECORDS_DIR,
                   ois_root: Path = OIS_CURVES_DIR) -> tuple[pd.DataFrame, dict]:
    """Prints executed on days ``[start, end]``, with corrections disseminated up to
    ``correction_days`` later (as far as the archive goes)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    rec = read_records(start - pd.Timedelta(days=spec.fresh_days + 1),
                       end + pd.Timedelta(days=spec.correction_days + 1), root=records_root)
    if rec.empty:
        return pd.DataFrame(), {"days": [], "empty_days": {}}
    ex = ds.executions(rec, as_of=end + pd.Timedelta(days=spec.correction_days), fresh_days=spec.fresh_days)
    ex = ex[ex["executed"].dt.normalize().between(start, end)]
    curves = _curves(start, end, spec.curve, ois_root)
    pr = price_prints(ex, curves, spec)
    traded = sorted(set(ex["executed"].dt.normalize()))
    with_curve = [d for d in traded if d in curves]
    got = set(pd.to_datetime(pr["timestamp"]).dt.normalize()) if len(pr) else set()
    empty = {d: "executed swaptions and an OIS curve but no priced print" for d in with_curve
             if d not in got and d.dayofweek < 5}
    return pr, {"days": with_curve, "empty_days": empty}


def store_prints(df: pd.DataFrame, start, end, *, root: Path = SWAPTION_PRINTS_DIR) -> int:
    _between_days(root, "timestamp", start, end)
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, PRINT_KEYS)
    return len(df)


def read_prints(start=None, end=None, *, root: Path = SWAPTION_PRINTS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end))
    if df is None or df.empty:
        return pd.DataFrame()
    return df.sort_values(PRINT_KEYS).reset_index(drop=True)


# ------------------------------------------------------------------------- surface
def atm_surface(prints: pd.DataFrame, spec: SwaptionSpec = SWAPTIONS["USD_SOFR"]) -> pd.DataFrame:
    """Per day and standard point: the median OTM vol of prints within ``atm_band_bp`` of
    the forward, with their count, interquartile range and median distance to the snap."""
    if prints.empty:
        return pd.DataFrame(columns=VOL_KEYS)
    p = prints.dropna(subset=["vol_bp"])
    p = p[p["moneyness_bp"].abs() <= spec.atm_band_bp].assign(day=pd.to_datetime(p["timestamp"]).dt.normalize())
    p = p.assign(tenor=p["tenor_years"].round().astype(int))
    rows = []
    for name, lo, hi in spec.surface_expiries:
        if lo > spec.surface_max_expiry_years:
            continue
        b = p[p["t_years"].between(lo, hi) & p["tenor"].isin(spec.surface_tenors)]
        for (day, tenor), g in b.groupby(["day", "tenor"]):
            v = g["vol_bp"]
            rows.append({"timestamp": day, "expiry": name, "tenor": int(tenor), "vol_bp": float(v.median()),
                         "n_prints": int(len(v)), "iqr_bp": float(v.quantile(.75) - v.quantile(.25)),
                         "notional": float(g["notional"].sum()),
                         "median_minutes_from_snap": float(g["minutes_from_snap"].median())})
    out = pd.DataFrame(rows)
    if len(out):
        out["timestamp"] = out["timestamp"].astype("datetime64[ms]")
        out["tenor"] = out["tenor"].astype("int16")
        out["n_prints"] = out["n_prints"].astype("int32")
    return out


def store_days(df: pd.DataFrame, start, end, *, root: Path, keys) -> int:
    _between_days(root, "timestamp", start, end)
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, list(keys))
    return len(df)


def read_vols(start=None, end=None, *, root: Path = SWAPTION_VOLS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end))
    if df is None or df.empty:
        return pd.DataFrame(columns=VOL_KEYS)
    df["expiry"] = df["expiry"].astype(str)
    return df.sort_values(VOL_KEYS).reset_index(drop=True)


# ------------------------------------------------------------------- open interest
def _bucket(x: pd.Series, buckets) -> pd.Series:
    """``unknown`` where the value is missing or outside every bucket: 16% of USD open
    notional reports no underlying maturity (2026-10-05) - kept, so totals stay whole."""
    out = pd.Series("unknown", index=x.index, dtype="string")
    for lo, hi, name in buckets:
        out[(x >= lo) & (x < hi)] = name
    return out


def ledger(*, records_root: Path = SWAPTION_RECORDS_DIR, end=None) -> pd.DataFrame:
    """Every trade's versions by dissemination day (``infra.processing.dtcc_swaptions.
    ledger_versions``) from the stored records, up to file day ``end`` (None = all)."""
    rec = read_records(None, None if end is None else pd.Timestamp(end) + _ONE_DAY, root=records_root)
    return ds.ledger_versions(rec)


def open_interest(as_of, *, spec: SwaptionSpec = SWAPTIONS["USD_SOFR"], versions: pd.DataFrame | None = None,
                  records_root: Path = SWAPTION_RECORDS_DIR) -> pd.DataFrame:
    """Every swaption open after day ``as_of``'s file (point in time: only records
    disseminated by then), one row per trade with its terms, ``t_years`` / ``tenor_years``
    from ``as_of``, the buckets, ``capped`` and ``origin``. Notionals of capped trades are
    the floor. Pass ``versions`` (``ledger()``) to reuse one build across many days."""
    d = pd.Timestamp(as_of).normalize()
    v = ledger(records_root=records_root, end=d) if versions is None else versions
    o = ds.open_at(v, d, max_notional=spec.max_notional)
    o = o.assign(t_years=(o["expiry"] - d).dt.days / 365.25,
                 tenor_years=(o["maturity"] - o["expiry"]).dt.days / 365.25)
    o["expiry_bucket"] = _bucket(o["t_years"], EXPIRY_BUCKETS)
    o["tenor_bucket"] = _bucket(o["tenor_years"], TENOR_BUCKETS)
    return o.reset_index(drop=True)


def oi_summary(days, *, spec: SwaptionSpec = SWAPTIONS["USD_SOFR"], versions: pd.DataFrame) -> pd.DataFrame:
    """Per day: open notional (cap floors), trade count and capped share by expiry x
    tenor bucket and origin."""
    rows = []
    for d in days:
        o = open_interest(d, spec=spec, versions=versions)
        if o.empty:
            continue
        g = o.groupby(["expiry_bucket", "tenor_bucket", "origin"], observed=True)
        s = g.agg(notional=("notional", "sum"), n_trades=("notional", "size"),
                  capped_notional=("notional", lambda x: float(x[o.loc[x.index, "capped"]].sum()))).reset_index()
        s["timestamp"] = pd.Timestamp(d)
        rows.append(s)
    if not rows:
        return pd.DataFrame(columns=OI_KEYS)
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = out["timestamp"].astype("datetime64[ms]")
    out["n_trades"] = out["n_trades"].astype("int32")
    for c in ("expiry_bucket", "tenor_bucket", "origin"):
        out[c] = out[c].astype(str)
    return out[OI_KEYS + ["notional", "n_trades", "capped_notional"]]


def read_oi(start=None, end=None, *, root: Path = SWAPTION_OI_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end))
    if df is None or df.empty:
        return pd.DataFrame(columns=OI_KEYS)
    for c in ("expiry_bucket", "tenor_bucket", "origin"):
        df[c] = df[c].astype(str)
    return df.sort_values(OI_KEYS).reset_index(drop=True)
