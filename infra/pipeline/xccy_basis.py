"""Cross-currency OIS basis closes from the DTCC archive (root CLAUDE.md 16; spec
``infra.config.XCCY_BASIS``; pure handling ``infra.processing.dtcc_xccy``). Disk only.

Store ``Derived/XccyBasisCloses``, keys ``timestamp`` (the snap instant), ``close``,
``pair``, ``tenor`` (MONTHS), ``method``: the PURE estimator of the swap closes on the
basis (``rate`` in percent as the estimator works; ``basis_bp`` the same in bp). Day D
reads corrections in files D..D+``SWAP_CORRECTION_DAYS``; a recomputed day replaces its rows.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from infra.config import DTCC_DIR, SWAP_CLOSE_WEIGHTING, SWAP_CLOSES, SWAP_CORRECTION_DAYS, XCCY_BASIS, XCCY_BASIS_CLOSES_DIR
from infra.pipeline import dtcc
from infra.processing import dtcc_xccy as dx
from infra.processing.swap_closes import CLOSE_COLUMNS, pure_closes
from infra.storage import parquet_store
from infra.trading_calendar import snap_instants

REPORT = "RATES"
KEYS = ["timestamp", "close", "pair", "tenor", "method"]
_ONE_DAY = pd.Timedelta(days=1)


def day_records(day, *, as_of=None, dtcc_root: Path = DTCC_DIR, cache: dict | None = None) -> pd.DataFrame:
    """Normalized records of file ``day`` and the correction files after it."""
    day = pd.Timestamp(day).normalize()
    last = day + pd.Timedelta(days=SWAP_CORRECTION_DAYS)
    if as_of is not None:
        last = min(last, pd.Timestamp(as_of).normalize())
    frames = []
    for d in pd.date_range(day, last):
        if cache is not None and d in cache:
            f = cache[d]
        else:
            raw = dtcc.read_dtcc_day(REPORT, d, columns=list(dx.RAW_COLUMNS), root=dtcc_root)
            raw = raw[raw["UPI FISN"].astype(str).str.match(r"NA/Swap Flt Flt [A-Z]{3} USD$")] if len(raw) else raw
            f = dx.normalize(raw) if len(raw) else pd.DataFrame()
            if cache is not None:
                cache[d] = f
        if len(f):
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_closes(day, *, pairs=None, as_of=None, dtcc_root: Path = DTCC_DIR, cache: dict | None = None) -> pd.DataFrame:
    day = pd.Timestamp(day).normalize()
    rec = day_records(day, as_of=as_of, dtcc_root=dtcc_root, cache=cache)
    out = []
    for name in pairs or tuple(XCCY_BASIS):
        spec = XCCY_BASIS[name]
        tr = dx.basis_trades(rec, spec, as_of=as_of) if len(rec) else pd.DataFrame(columns=dx.TRADE_COLUMNS)
        tr = tr[(tr["executed"] >= day) & (tr["executed"] < day + _ONE_DAY)]
        for close in spec.closes:
            cs = dataclasses.replace(SWAP_CLOSES[close], pure_fallback_half_window_min=spec.fallback_half_window_min)
            inst = snap_instants([day], cs.local_time, cs.timezone)[0]
            out.append(pure_closes(tr, inst, cs, SWAP_CLOSE_WEIGHTING, close_name=close, currency=name))
    out = [o for o in out if not o.empty]
    if not out:
        return pd.DataFrame(columns=KEYS)
    df = pd.concat(out, ignore_index=True)[CLOSE_COLUMNS].rename(columns={"currency": "pair"})
    df["basis_bp"] = df["rate"] * 100.0
    return df.astype({"timestamp": "datetime64[ms]", "tenor": "int32", "n_trades": "int32", "half_window_min": "int32"})


def flag_suspect(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """``suspect`` per close: more than the pair's ``suspect_bp`` from the median of the
    previous ``suspect_window`` closes of its pair x close x tenor (``history``: stored rows
    before ``df``, so a recomputed window flags as a full build)."""
    if df.empty:
        return df.assign(suspect=pd.Series(dtype=bool))
    allrows = pd.concat([history[df.columns.intersection(history.columns)], df]) if history is not None and len(history) else df
    allrows = allrows.drop_duplicates(KEYS, keep="last").sort_values("timestamp")
    flags = {}
    for (pair, close, tenor), g in allrows.groupby(["pair", "close", "tenor"]):
        spec = XCCY_BASIS[pair]
        med = g["basis_bp"].rolling(spec.suspect_window, min_periods=3).median().shift(1)
        for k, s in zip(zip(g["timestamp"], g["pair"], g["close"], g["tenor"]), ((g["basis_bp"] - med).abs() > spec.suspect_bp)):
            flags[k] = bool(s)
    out = df.copy()
    out["suspect"] = [flags.get(k, False) for k in zip(out["timestamp"], out["pair"], out["close"], out["tenor"])]
    return out


def store_days(df: pd.DataFrame, start, end, *, root: Path = XCCY_BASIS_CLOSES_DIR) -> int:
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    parquet_store.delete_where(root, lambda p: pd.to_datetime(p["timestamp"]).ge(lo) & pd.to_datetime(p["timestamp"]).lt(hi))
    if df is None or df.empty:
        return 0
    parquet_store.write_partitioned(df, root, KEYS)
    return len(df)


def backfill_closes(start, end, *, root: Path = XCCY_BASIS_CLOSES_DIR, dtcc_root: Path = DTCC_DIR) -> dict:
    have = dtcc.archived_days(REPORT, root=dtcc_root)
    frames, cache, n = [], {}, 0
    for day in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()):
        for k in [k for k in cache if k < day]:
            del cache[k]
        if day in have:
            frames.append(compute_closes(day, dtcc_root=dtcc_root, cache=cache)); n += 1
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    hist = read_closes(pd.Timestamp(start) - pd.Timedelta(days=400), start, root=root)
    df = flag_suspect(df, hist) if len(df) else df
    return {"days": n, "rows": store_days(df, start, end, root=root)}


def read_closes(start=None, end=None, *, pair: str | None = None, clean: bool = False,
                root: Path = XCCY_BASIS_CLOSES_DIR) -> pd.DataFrame:
    """Stored closes in ``[start, end)``; ``clean`` drops the ``suspect`` ones."""
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end),
                                        equals_in={"pair": [pair]} if pair else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=KEYS)
    for c in ("close", "pair", "method"):
        df[c] = df[c].astype(str)
    if clean and "suspect" in df:
        df = df[~df["suspect"].fillna(False).astype(bool)]
    return df.sort_values(KEYS).reset_index(drop=True)
