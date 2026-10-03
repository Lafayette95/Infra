"""Saved model runs (walk-forward params and predictions), one folder per run name under
``MODEL_RUNS_DIR``. Generic frames in, generic frames out: this module knows nothing about
the models (storage sits below ``infra/models``). Flat files, no row index, ZSTD level 5
(root CLAUDE.md 6a/6d); a run is small (one params frame per refit), so no partitioning.
A run is replaced whole - re-running a name overwrites it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from infra.config import MODEL_RUNS_DIR

_FILES = ("params", "predictions")


def _write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd", compression_level=5)
    os.replace(tmp, path)


def save_run(name: str, params: pd.DataFrame, predictions: pd.DataFrame, meta: dict | None = None, *,
             root: Path = MODEL_RUNS_DIR) -> Path:
    """Write ``params`` and ``predictions`` (timestamp index -> a ``timestamp`` column)
    plus ``meta`` (the spec, inputs, refit rule - JSON) to ``root/name``."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    _write(params.reset_index(drop=True), folder / "params.parquet")
    pred = predictions.copy()
    if isinstance(pred.index, pd.DatetimeIndex):
        pred = pred.rename_axis("timestamp").reset_index()
    _write(pred, folder / "predictions.parquet")
    (folder / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))
    return folder


def read_run(name: str, *, root: Path = MODEL_RUNS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(params, predictions indexed by timestamp, meta) of a saved run."""
    folder = root / name
    if not folder.exists():
        raise FileNotFoundError(f"no saved model run {name!r} under {root}")
    params = pd.read_parquet(folder / "params.parquet")
    pred = pd.read_parquet(folder / "predictions.parquet")
    if "timestamp" in pred.columns:
        pred = pred.set_index("timestamp")
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return params, pred, meta


def list_runs(*, root: Path = MODEL_RUNS_DIR) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "params.parquet").exists()) if root.exists() else []
