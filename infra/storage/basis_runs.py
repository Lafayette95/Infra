"""Stored basis-model runs: ``BASIS_RUNS_DIR/<model>/{contracts,bonds}`` - hive year /
quarter, flat, ZSTD 5 (root CLAUDE.md 6a/6d), keys ``day`` + ``contract`` (+ ``cusip``,
``delivery_kind`` for bonds). Generic frames in and out: this module knows nothing about
the models (storage sits below ``infra/models``); ``scripts/run_basis.py --persist`` writes,
the basis dashboard page reads. A re-run of a day REPLACES that day's rows.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.config import BASIS_RUNS_DIR
from infra.storage import parquet_store

CONTRACT_KEYS = ["timestamp", "contract"]
BOND_KEYS = ["timestamp", "contract", "cusip", "delivery_kind"]


def _dir(model: str, kind: str, root: Path) -> Path:
    return Path(root) / model / kind


def save(model: str, contracts: pd.DataFrame, bonds: pd.DataFrame, *, root: Path = BASIS_RUNS_DIR) -> None:
    for kind, df, keys in (("contracts", contracts, CONTRACT_KEYS), ("bonds", bonds, BOND_KEYS)):
        if df is None or df.empty:
            continue
        out = df.rename(columns={"day": "timestamp"}).copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
        for c in out.columns:  # parquet needs plain types (no tz / object timestamps)
            if out[c].dtype == object and len(out) and isinstance(out[c].dropna().iloc[0] if out[c].notna().any() else None, pd.Timestamp):
                out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
        d = _dir(model, kind, root)
        if parquet_store.has_data(d):
            parquet_store.prune_rows(d, "timestamp", sorted(out["timestamp"].unique()))
        parquet_store.write_partitioned(out, d, keys)


def models(*, root: Path = BASIS_RUNS_DIR) -> list[str]:
    return sorted(p.name for p in Path(root).glob("*") if (p / "contracts").exists()) if Path(root).exists() else []


def read(model: str, kind: str, start=None, end=None, *, contracts=None, root: Path = BASIS_RUNS_DIR) -> pd.DataFrame:
    """``kind`` = "contracts" or "bonds"; optionally only ``contracts`` (tickers)."""
    d = _dir(model, kind, root)
    if not parquet_store.has_data(d):
        return pd.DataFrame()
    df = parquet_store.read_partitioned(d, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1),
                                        equals_in={"contract": list(contracts)} if contracts is not None else None)
    if df is None:
        return pd.DataFrame()
    for c in ("contract", "root", "cusip", "delivery_kind", "ctd"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df.rename(columns={"timestamp": "day"}).sort_values(["day", "contract"], ignore_index=True)
