"""Step 5 - ``backup_daily_vintage``: the LAST step of each run day's cycle, and only if
every other step came out ok - a failed day must not become the baseline tomorrow's
revision checks trust (tomorrow then compares against the last good vintage instead)."""
from __future__ import annotations

import pandas as pd

from infra.cycle import vintage
from infra.cycle.core import Check, Step, StepContext
from infra.cycle.paths import CyclePaths


def backup_daily_vintage(run_day, *, paths: CyclePaths | None = None) -> dict:
    paths = paths or CyclePaths.default()
    return {"vintage": vintage.snapshot(pd.Timestamp(run_day).normalize(), paths)}


def _files(root) -> set:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp")}


def _check_complete(ctx: StepContext):
    """The vintage holds exactly the live database's files (minus the vintages)."""
    dst = ctx.output["vintage"]
    root = ctx.paths.database_root
    live = {f for f in _files(root)
            if not (root / f).resolve().is_relative_to(ctx.paths.vintage_root.resolve())}
    copied = _files(dst)
    if live == copied:
        return True, f"{len(copied)} files copied to {dst.name}", None
    missing = sorted(map(str, live - copied))
    return False, f"vintage {dst.name} is missing {len(missing)} file(s)", pd.DataFrame({"missing": missing})


def _run(ctx: StepContext) -> dict:
    return backup_daily_vintage(ctx.run_day, paths=ctx.paths)


BACKUP_STEP = Step("backup", _run, checks=(Check("vintage_complete", _check_complete),))
