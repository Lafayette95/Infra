"""Flat parquet master table of absolute futures contracts (root, ticker, expiry, ...)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from infra.processing.definitions import CONTRACT_COLUMNS

_KEYS = ["root", "ticker", "expiry"]


def read_contracts(path: Path, root: str | None = None) -> pd.DataFrame:
    """All known contracts (optionally one root), sorted by expiry. Empty if no file yet."""
    if not path.exists():
        return pd.DataFrame({
            "root": pd.Series(dtype="str"),
            "ticker": pd.Series(dtype="str"),
            "instrument_id": pd.Series(dtype="int64"),
            "expiry": pd.Series(dtype="datetime64[ms]"),
            "activation": pd.Series(dtype="datetime64[ms]"),
        })
    df = pd.read_parquet(path)
    if root is not None:
        df = df[df["root"] == root]
    return df.sort_values("expiry").reset_index(drop=True)


def write_contracts(path: Path, new: pd.DataFrame) -> None:
    """Merge ``new`` into the master table (dedupe on root/ticker/expiry), atomically."""
    if new.empty:
        return
    existing = read_contracts(path)
    frames = [f for f in (existing, new[CONTRACT_COLUMNS]) if not f.empty]
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=_KEYS, keep="last")
        .sort_values(["root", "expiry", "ticker"])
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    merged.to_parquet(tmp, engine="pyarrow", index=False, compression="zstd", compression_level=5)
    tmp.replace(path)


def dataset_lookup(path: Path) -> dict[str, str]:
    """Absolute ticker -> root, for routing absolute tickers to a dataset."""
    df = read_contracts(path)
    return dict(zip(df["ticker"], df["root"]))
