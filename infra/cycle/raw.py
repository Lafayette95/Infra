"""Step 1b - ``backfill_daily_raw_data``: non-price raw inputs (macro releases, EFFR,
calendars, ...). Nothing is registered yet; this is the slot new raw sources plug into.

A raw source is a function ``(start, end, *, paths, force_refetch) -> dict`` added to
``RAW_SOURCES``; each should also contribute its own checks to ``RAW_CHECKS``.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from infra.cycle.core import Check, Step, StepContext
from infra.cycle.paths import CyclePaths

RAW_SOURCES: dict[str, Callable[..., dict]] = {}
RAW_CHECKS: tuple[Check, ...] = ()


def backfill_daily_raw_data(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    sources: dict[str, Callable[..., dict]] | None = None,
) -> dict:
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    sources = RAW_SOURCES if sources is None else sources
    return {"sources": {name: fn(start, end, paths=paths, force_refetch=force_refetch)
                        for name, fn in sources.items()}}


def _run(ctx: StepContext) -> dict:
    return backfill_daily_raw_data(ctx.start, ctx.end, paths=ctx.paths, force_refetch=ctx.force_refetch,
                                   sources=ctx.options.get("raw_sources"))


RAW_STEP = Step("raw", _run, checks=RAW_CHECKS)
