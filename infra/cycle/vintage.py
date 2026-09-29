"""Vintages: a full dated copy of the database taken as the LAST step of each run day's
cycle (``_vintages/YYYY-MM-DD``, dated by the day the run happened, not by the data it
covers). The next run's "no revisions" checks diff against the latest one before it.

A plain copy is deliberate: the database is small (~2MB at introduction) and a copy can
never be mutated by a later write. Hard-linking would be cheaper on disk but relies on
every writer replacing files atomically rather than editing them in place - true today,
not something to silently depend on.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from infra.cycle.paths import CyclePaths

_FORMAT = "%Y-%m-%d"


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

    shutil.copytree(root, partial, ignore=ignore)
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


def latest_before(run_day: pd.Timestamp, paths: CyclePaths) -> Path | None:
    """The vintage a run on ``run_day`` compares against: the latest one taken strictly
    before it (a second run on the same day still compares against yesterday's)."""
    earlier = [d for d in list_vintages(paths) if d < pd.Timestamp(run_day).normalize()]
    return vintage_dir(earlier[-1], paths) if earlier else None


def in_vintage(store: Path, vintage: Path, paths: CyclePaths) -> Path:
    """Where a live store (anywhere under ``database_root``) lives inside a vintage."""
    return vintage / Path(store).resolve().relative_to(paths.database_root.resolve())
