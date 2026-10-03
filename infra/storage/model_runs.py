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


# ----------------------------------------------------------------------------- incremental
# The scheduled operation (infra.models.runs): a run folder holds meta.json (the run's
# definition + bookkeeping), params.parquet (one block per fit date, column fit_as_of) and
# predictions.parquet (one row per timestamp, column fit_as_of = the fit behind it).
# Rebuilds are written to <name>/_rebuild/ and, when promoted, the replaced run is kept
# whole under <name>/_archive/<tag>/ (the as-traded record).


def read_meta(name: str, *, root: Path = MODEL_RUNS_DIR) -> dict:
    f = root / name / "meta.json"
    if not f.exists():
        raise FileNotFoundError(f"no model run {name!r} under {root} (create it first)")
    return json.loads(f.read_text())


def write_meta(name: str, meta: dict, *, root: Path = MODEL_RUNS_DIR) -> None:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, folder / "meta.json")


def read_params(name: str, *, root: Path = MODEL_RUNS_DIR) -> pd.DataFrame:
    f = root / name / "params.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["section", "row", "col", "value", "fit_as_of"])


def read_predictions(name: str, *, root: Path = MODEL_RUNS_DIR) -> pd.DataFrame:
    f = root / name / "predictions.parquet"
    if not f.exists():
        return pd.DataFrame()
    pred = pd.read_parquet(f)
    return pred.set_index("timestamp") if "timestamp" in pred.columns else pred


def append_params(name: str, new: pd.DataFrame, *, root: Path = MODEL_RUNS_DIR) -> int:
    """Add fits; a fit date already stored is replaced, every other stored fit kept as is."""
    if new is None or new.empty:
        return 0
    old = read_params(name, root=root)
    keep = old[~old["fit_as_of"].isin(new["fit_as_of"].unique())] if len(old) else old
    out = pd.concat([keep, new], ignore_index=True) if len(keep) else new.reset_index(drop=True)
    (root / name).mkdir(parents=True, exist_ok=True)
    _write(out.sort_values("fit_as_of", kind="stable").reset_index(drop=True), root / name / "params.parquet")
    return int(new["fit_as_of"].nunique())


def upsert_predictions(name: str, new: pd.DataFrame, *, root: Path = MODEL_RUNS_DIR) -> dict:
    """Write rows by timestamp (replace the stored row of the same timestamp). Returns how
    many rows were new, unchanged, and changed - a changed row is a data revision (or a
    fit appended after the row was first predicted)."""
    if new is None or new.empty:
        return {"new": 0, "unchanged": 0, "changed": 0}
    old = read_predictions(name, root=root)
    both = new.index.intersection(old.index) if len(old) else new.index[:0]
    changed = 0
    if len(both):
        cols = [c for c in new.columns if c in old.columns and pd.api.types.is_numeric_dtype(new[c])]
        a, b = old.loc[both, cols].astype("float64"), new.loc[both, cols].astype("float64")
        changed = int((((a - b).abs() > 1e-12 * (1 + a.abs())) | (a.isna() != b.isna())).any(axis=1).sum())
    keep = old.drop(index=both) if len(old) else old
    out = pd.concat([keep, new]).sort_index() if len(keep) else new.sort_index()
    out.index.name = "timestamp"
    (root / name).mkdir(parents=True, exist_ok=True)
    _write(out.reset_index(), root / name / "predictions.parquet")
    return {"new": int(len(new.index.difference(old.index)) if len(old) else len(new)),
            "unchanged": int(len(both)) - changed, "changed": changed}


def archive_run(name: str, tag: str, *, root: Path = MODEL_RUNS_DIR) -> Path:
    """Copy the run's files (not its _archive / _rebuild) to <name>/_archive/<tag>/."""
    import shutil
    src, dst = root / name, root / name / "_archive" / tag
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "params.parquet", "predictions.parquet"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    return dst
