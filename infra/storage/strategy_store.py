"""A strategy's stored outputs, one folder per strategy under ``STRATEGIES_DIR``: generic frames,
no strategy logic (storage sits below ``infra/strategies``). Flat files, no row index, ZSTD 5.

* ``signals.parquet`` / ``positions.parquet``: the FIRM series - one row per label (the
  decision time), upserted by label; ``positions_abs.parquet``: long ``label, instrument,
  contract, position``, upserted by (label, instrument);
* ``pnl.parquet``: per label gross / cost / net (and per instrument), upserted by label;
  ``pnl_contracts.parquet``: long per (label, contract) - target, executed, trade, mark,
  half-spread, gross, cost, net, the reason a trade waited; ``exec_checks.parquet``: the
  accounting checks, replaced whole each run (recomputed whole);
* ``plans.parquet``: the PLAN vintages - ``generated_at`` + the forward rows that day's run
  produced (signals and positions), replaced per ``generated_at``;
* ``meta.json``: the strategy's name, class, spec and bookkeeping.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from infra.config import STRATEGIES_DIR


LONG_KEYS = {"positions_abs": ["label", "instrument"], "pnl_contracts": ["label", "contract"]}


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd", compression_level=5)
    os.replace(tmp, path)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def read_meta(name: str, *, root: Path = STRATEGIES_DIR) -> dict:
    f = root / name / "meta.json"
    return json.loads(f.read_text()) if f.exists() else {}


def write_meta(name: str, meta: dict, *, root: Path = STRATEGIES_DIR) -> None:
    f = root / name / "meta.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, f)


def read_series(name: str, kind: str, *, root: Path = STRATEGIES_DIR) -> pd.DataFrame:
    """``kind``: ``signals`` | ``positions`` (wide, indexed by ``label``) | ``positions_abs`` (long)."""
    df = _read(root / name / f"{kind}.parquet")
    if df.empty or kind in LONG_KEYS:
        return df
    return df.set_index("label")


def upsert_series(name: str, kind: str, new: pd.DataFrame, *, root: Path = STRATEGIES_DIR) -> dict:
    """Write rows by key (``label``; ``label, instrument`` for ``positions_abs``), replacing stored
    rows with the same key. Returns new / unchanged / changed counts (a changed firm row is a
    revision: a data revision or a recomputation)."""
    if new is None or new.empty:
        return {"new": 0, "unchanged": 0, "changed": 0}
    keys = LONG_KEYS.get(kind, ["label"])
    incoming = new if kind in LONG_KEYS else new.rename_axis("label").reset_index()
    old = _read(root / name / f"{kind}.parquet")
    changed = unchanged = 0
    if len(old):
        m = old.merge(incoming, on=keys, how="inner", suffixes=("_o", "_n"))
        num = [c for c in incoming.columns if c not in keys and pd.api.types.is_numeric_dtype(incoming[c])]
        if len(m) and num:
            diff = pd.concat([((m[f"{c}_o"] - m[f"{c}_n"]).abs() > 1e-9) | (m[f"{c}_o"].isna() != m[f"{c}_n"].isna())
                              for c in num], axis=1).any(axis=1)
            changed, unchanged = int(diff.sum()), int((~diff).sum())
        keep = old.merge(incoming[keys], on=keys, how="left", indicator=True)
        old = old[(keep["_merge"] == "left_only").to_numpy()]
    out = pd.concat([old, incoming], ignore_index=True) if len(old) else incoming
    _write(out.sort_values(keys).reset_index(drop=True), root / name / f"{kind}.parquet")
    return {"new": len(incoming) - changed - unchanged, "unchanged": unchanged, "changed": changed}


def write_table(name: str, kind: str, df: pd.DataFrame, *, root: Path = STRATEGIES_DIR) -> int:
    """Replace a whole table (recomputed whole every run, e.g. ``exec_checks``)."""
    _write(df.reset_index(drop=True), root / name / f"{kind}.parquet")
    return len(df)


def write_plan(name: str, generated_at, plan: pd.DataFrame, *, root: Path = STRATEGIES_DIR) -> int:
    """Replace the plan vintage of ``generated_at`` (the forward rows of that day's run)."""
    old = _read(root / name / "plans.parquet")
    g = pd.Timestamp(generated_at)
    if len(old):
        old = old[pd.to_datetime(old["generated_at"]) != g]
    new = plan.rename_axis("label").reset_index().assign(generated_at=g) if len(plan) else plan
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(out):
        _write(out, root / name / "plans.parquet")
    return len(new)


def read_plans(name: str, *, root: Path = STRATEGIES_DIR) -> pd.DataFrame:
    return _read(root / name / "plans.parquet")


def archive(name: str, tag: str, *, root: Path = STRATEGIES_DIR) -> Path:
    import shutil
    src, dst = root / name, root / name / "_archive" / tag
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.parquet"):
        shutil.copy2(f, dst / f.name)
    if (src / "meta.json").exists():
        shutil.copy2(src / "meta.json", dst / "meta.json")
    return dst
