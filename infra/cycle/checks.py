"""Generic regression checks shared by every cycle step. Step-specific checks (presence
of today's data, custom/outlier checks) live next to their step."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from infra.cycle import vintage
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.storage import parquet_store

_ONE_DAY = pd.Timedelta(days=1)


def _equal(old: pd.Series, new: pd.Series, rtol: float, atol: float) -> np.ndarray:
    """Element-wise "not revised": NA == NA; numeric columns within tolerance (on-disk
    fixed-point integers compare exactly either way); and a value arriving where the
    vintage had NONE is a FILLED GAP, not a revision - e.g. open interest, which the
    exchange publishes the next session (CLAUDE.md 8), so the latest day's OI is always
    missing in a vintage and filled by the next run's trailing re-fetch. Counting that as a
    revision failed the first scheduled run (49 x Friday's OI arriving Monday) and would
    have failed every run. A value that CHANGES or DISAPPEARS is still a revision."""
    both_na = old.isna().to_numpy() & new.isna().to_numpy()
    filled = old.isna().to_numpy() & ~new.isna().to_numpy()
    if pd.api.types.is_numeric_dtype(old) and pd.api.types.is_numeric_dtype(new):
        a = pd.to_numeric(old, errors="coerce").astype("float64").to_numpy()
        b = pd.to_numeric(new, errors="coerce").astype("float64").to_numpy()
        with np.errstate(invalid="ignore"):
            same = np.isclose(a, b, rtol=rtol, atol=atol)
    else:
        same = (old.astype(str) == new.astype(str)).to_numpy()
    return both_na | filled | same


def compare_to_vintage(
    live_root: Path,
    vintage_root: Path,
    key_columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    equals_in: dict | None = None,
    rtol: float = 0.0,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Every key in ``[start, end]`` (inclusive) that the vintage holds and the live store
    either changed or no longer holds. One row per (key, column) difference; a deleted
    row shows ``column == "<row deleted>"``. Keys only in the live store are NEW history
    (e.g. a backfill), not revisions, and are ignored."""
    window = dict(start=start, end=end + _ONE_DAY, equals_in=equals_in)
    old = parquet_store.read_partitioned(vintage_root, **window)
    if old is None or old.empty:
        return pd.DataFrame(columns=[*key_columns, "column", "old", "new"])
    new = parquet_store.read_partitioned(live_root, **window)
    if new is None:
        new = old.iloc[0:0]
    for frame in (old, new):  # align non-datetime keys (category vs str would break the merge)
        for k in key_columns:
            if not pd.api.types.is_datetime64_any_dtype(frame[k]):
                frame[k] = frame[k].astype(str)
    merged = old.merge(new, on=key_columns, how="left", suffixes=("__old", "__new"), indicator=True)

    diffs = []
    deleted = merged[merged["_merge"] == "left_only"]
    if not deleted.empty:
        diffs.append(deleted[key_columns].assign(column="<row deleted>", old=None, new=None))
    both = merged[merged["_merge"] == "both"]
    value_columns = [c for c in old.columns if c not in key_columns and c in new.columns]
    for col in value_columns:
        o, n = both[f"{col}__old"], both[f"{col}__new"]
        changed = ~_equal(o, n, rtol, atol)
        if changed.any():
            diffs.append(both.loc[changed, key_columns].assign(
                column=col, old=o[changed].to_numpy(), new=n[changed].to_numpy()))
    if not diffs:
        return pd.DataFrame(columns=[*key_columns, "column", "old", "new"])
    return pd.concat(diffs, ignore_index=True)


def revision_check(
    store: Callable[[CyclePaths], Path],
    key_columns: list[str],
    *,
    name: str = "no_revisions",
    equals_in: Callable[[StepContext], dict | None] | None = None,
    severity: Severity = Severity.FAIL,
    rtol: float = 0.0,
    atol: float = 1e-12,
) -> Check:
    """Test (b): data for past days as of THIS run matches the same days as of the
    previous vintage - i.e. "T-1 as of T" == "T-1 as of T-1". Skipped (passes, says so)
    when no earlier vintage exists yet."""

    def fn(ctx: StepContext):
        if ctx.reference_vintage is None:
            return True, "no earlier vintage yet - nothing to compare against", None
        live = store(ctx.paths)
        diffs = compare_to_vintage(
            live, vintage.in_vintage(live, ctx.reference_vintage, ctx.paths), key_columns,
            ctx.start, ctx.end, equals_in=equals_in(ctx) if equals_in else None, rtol=rtol, atol=atol,
        )
        against = ctx.reference_vintage.name
        if diffs.empty:
            return True, f"no revisions vs vintage {against}", None
        return False, f"{len(diffs)} revised value(s) vs vintage {against}", diffs

    return Check(name, fn, severity)
