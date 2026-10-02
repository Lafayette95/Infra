"""Vintages: a full dated copy of the database taken as the LAST step of each run day's
cycle (``_vintages/YYYY-MM-DD``, dated by the day the run happened, not by the data it
covers). The next run's "no revisions" checks diff against the latest one before it.

Each file is a copy-on-write CLONE (APFS ``clonefile``), not a byte copy: instant and
free on disk until the live file changes (the database grew from ~3 MB to ~26 MB with the
raw inflation stores on 2026-09-30, most of it unchanged day to day). Chosen over hard
links: a clone is a genuinely separate file, so it stays correct even if some writer ever
edits a file in place, where a hard-linked vintage would silently change with it (every
writer here replaces files atomically today - a clone doesn't need to rely on that). Off
APFS (another OS or volume) it falls back to a plain copy.

Only the newest ``VINTAGES_KEPT`` vintages are kept (``prune``).
"""
from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path

import pandas as pd

from infra.config import VINTAGES_KEPT
from infra.cycle.paths import CyclePaths

_FORMAT = "%Y-%m-%d"


def _load_clonefile():
    try:
        fn = ctypes.CDLL(None, use_errno=True).clonefile  # macOS libSystem
    except (OSError, AttributeError):
        return None
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32], ctypes.c_int
    return fn


_CLONEFILE = _load_clonefile()


def clone_file(src, dst, *, follow_symlinks: bool = True):
    """``shutil.copy2`` drop-in: an APFS clone (metadata included) when possible, else a copy."""
    if _CLONEFILE is not None and _CLONEFILE(os.fsencode(src), os.fsencode(dst), 0) == 0:
        return dst
    return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)


def vintage_dir(run_day: pd.Timestamp, paths: CyclePaths) -> Path:
    return paths.vintage_root / pd.Timestamp(run_day).strftime(_FORMAT)


def snapshot(run_day: pd.Timestamp, paths: CyclePaths) -> Path:
    """Copy ``database_root`` (minus the vintages themselves and in-flight ``.tmp``
    files) to ``vintage_dir(run_day)``, replacing any earlier same-day vintage. Copies to
    a ``.partial`` sibling first, so a crash never leaves a half-written vintage that
    later looks valid."""
    dst = vintage_dir(run_day, paths)
    partial = dst.with_name(dst.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    paths.database_root.mkdir(parents=True, exist_ok=True)
    root = paths.database_root.resolve()
    vint = paths.vintage_root.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        skip = {n for n in names if n.endswith(".tmp")}
        here = Path(directory).resolve()
        skip |= {n for n in names if (here / n) == vint}
        return skip

    shutil.copytree(root, partial, ignore=ignore, copy_function=clone_file)
    if dst.exists():
        shutil.rmtree(dst)
    partial.rename(dst)
    return dst


def list_vintages(paths: CyclePaths) -> list[pd.Timestamp]:
    if not paths.vintage_root.exists():
        return []
    days = []
    for p in paths.vintage_root.iterdir():
        try:
            days.append(pd.Timestamp(pd.to_datetime(p.name, format=_FORMAT)))
        except ValueError:
            continue  # .partial leftovers, stray files
    return sorted(days)


def prune(paths: CyclePaths, keep: int = VINTAGES_KEPT, *, protect=None) -> list[pd.Timestamp]:
    """Delete all but the newest ``keep`` vintages - never ``protect`` (the one a run just
    wrote, which a backfill dated in the past would otherwise delete at once); returns the
    days deleted."""
    days = list_vintages(paths)
    dropped = [d for d in (days[:-keep] if keep > 0 else days)
               if protect is None or d != pd.Timestamp(protect).normalize()]
    for day in dropped:
        shutil.rmtree(vintage_dir(day, paths))
    return dropped


def latest_before(run_day: pd.Timestamp, paths: CyclePaths) -> Path | None:
    """The vintage a run on ``run_day`` compares against: the latest one taken strictly
    before it (a second run on the same day still compares against yesterday's)."""
    earlier = [d for d in list_vintages(paths) if d < pd.Timestamp(run_day).normalize()]
    return vintage_dir(earlier[-1], paths) if earlier else None


def in_vintage(store: Path, vintage: Path, paths: CyclePaths) -> Path:
    """Where a live store (anywhere under ``database_root``) lives inside a vintage."""
    return vintage / Path(store).resolve().relative_to(paths.database_root.resolve())
