"""Step 1b, inflation detail - ``backfill_daily_bulk``: snapshot every ``BULK_DATASETS`` file
(CPI, PPI by commodity and by industry, PCE by type of product) whose current version
isn't on disk yet, through ``infra.pipeline.bulk_series`` (no fetch/storage code here),
plus the step's checks.

Registered in ``infra.cycle.raw.RAW_SOURCES`` under ``"bulk"``. The window only scopes the
checks: a bulk file can't be asked for a past publication (the sources keep no vintages),
so every run ingests whatever version is current - stamped with its real publication day -
and ``force_refetch`` has nothing to re-ask (a version IS its Last-Modified time).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from infra.config import BULK_DATASETS, BulkDataset
from infra.cycle.checks import revision_check
from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import bulk_series as pbulk

log = logging.getLogger(__name__)

# Every dataset is monthly: a latest ingested version older than this (calendar days) is
# reported stale (warn) - a missed release, or a source that stopped publishing.
STALE_DAYS = 45


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def plan_daily_bulk(
    *, paths: CyclePaths | None = None, datasets: dict[str, BulkDataset] | None = None,
) -> dict[str, pd.Timestamp | None]:
    """``{dataset: latest ingested version}`` (None = never) - no network; the dry run's
    view (whether a newer version exists takes a HEAD request, made by the run itself)."""
    paths = paths or CyclePaths.default()
    datasets = BULK_DATASETS if datasets is None else datasets
    return {k: pbulk.latest_ingested(k, coverage_file=paths.bulk_coverage) for k in datasets}


def backfill_daily_bulk(
    start,
    end,
    *,
    paths: CyclePaths | None = None,
    force_refetch: bool = False,
    datasets: dict[str, BulkDataset] | None = None,
    sources=None,
    last_modified=None,
) -> dict:
    """Ingest each dataset's current file version if it's newer than what's on disk.
    ``start``/``end``/``force_refetch`` are unused (module doc) - kept for the raw-source
    signature: the scheduled run always passes ``force_refetch=True``, and re-downloading
    an already-ingested version (~190 MB) could add nothing. Per-dataset failures are
    collected, not raised."""
    paths = paths or CyclePaths.default()
    datasets = BULK_DATASETS if datasets is None else datasets
    last_modified = pbulk.LAST_MODIFIED if last_modified is None else last_modified
    unconfigured = sorted({d.source for d in datasets.values() if not pbulk.CONFIGURED.get(d.source, lambda: True)()})
    published, ingested, errors, rows = {}, [], {}, 0
    for key, d in datasets.items():
        if d.source in unconfigured:
            continue
        try:
            published[key] = last_modified[d.source](d)
            if not pbulk.needs_download(key, published[key], coverage_file=paths.bulk_coverage):
                continue
            values, catalog, version = pbulk.fetch_bulk_raw(d, sources=sources)
            rows += pbulk.store_bulk_raw(d, values, catalog, version, raw_root=paths.raw_data_root,
                                         coverage_file=paths.bulk_coverage, catalog_dir=paths.bulk_catalog_dir)
            ingested.append(key)
        except Exception as exc:
            errors[key] = _error(exc)
            log.warning("bulk fetch failed for %s: %s", key, errors[key])
    return {"datasets": datasets, "published": published, "ingested": ingested, "rows": rows,
            "fetch_errors": errors, "unconfigured": unconfigured}


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext) -> dict:
    return ctx.output.get("sources", {}).get("bulk", {})


def _datasets(ctx: StepContext) -> dict[str, BulkDataset]:
    return _out(ctx).get("datasets", {})


def _check_configured(ctx: StepContext):
    missing = _out(ctx).get("unconfigured", [])
    if not missing:
        return True, "every source configured", None
    return False, (f"source(s) not configured, NOT fetched: {', '.join(missing)} "
                   "(bls: set BLS_CONTACT_EMAIL in .env)"), None


def _check_fetch_ok(ctx: StepContext):
    out = _out(ctx)
    errors = out.get("fetch_errors", {})
    if not errors:
        new = ", ".join(out.get("ingested", [])) or "none"
        return True, f"new versions ingested: {new}; {out.get('rows', 0)} changed values", None
    details = pd.DataFrame({"dataset": list(errors), "error": list(errors.values())})
    return False, f"{len(errors)} dataset fetch error(s): {', '.join(errors)}", details


def _check_sane(ctx: StepContext):
    """Every value published in the window is finite, and no period lies after the month
    it was published in (a period in the future is a parsing error)."""
    bad = []
    one_per_store = {d.store: key for key, d in _datasets(ctx).items()}
    for store, key in sorted(one_per_store.items()):
        df = pbulk.read_bulk_from_disk(key, None, ctx.end, start=ctx.start, raw_root=ctx.paths.raw_data_root)
        wrong = df[~np.isfinite(df["value"]) | (df["period"] > df["timestamp"])]
        if len(wrong):
            bad.append(wrong.assign(store=store))
    if not bad:
        return True, "every value published in the window finite, no future period", None
    bad = pd.concat(bad, ignore_index=True)
    return False, f"{len(bad)} insane value(s)", bad


def _check_fresh(ctx: StepContext):
    """Warning: a dataset's latest ingested version is older than STALE_DAYS."""
    stale = []
    for key in _datasets(ctx):
        last = pbulk.latest_ingested(key, coverage_file=ctx.paths.bulk_coverage)
        if last is None or (ctx.end - last.normalize()).days > STALE_DAYS:
            stale.append((key, last))
    if not stale:
        return True, "every dataset has a version within its expected gap", None
    return False, f"{len(stale)} stale dataset(s): {', '.join(k for k, _ in stale)}", \
        pd.DataFrame(stale, columns=["dataset", "latest_version"])


def _no_revisions(store: str) -> Check:
    # A stored row is one PUBLISHED vintage: it should never change or disappear later.
    return revision_check(lambda p: p.raw_data_root / store, ["timestamp", "ticker", "period"],
                          name=f"bulk_{store.lower()}_no_revisions")


BULK_CHECKS = (
    Check("bulk_configured", _check_configured, severity=Severity.WARN),
    Check("bulk_fetch_ok", _check_fetch_ok),
    Check("bulk_sane", _check_sane),
    Check("bulk_fresh", _check_fresh, severity=Severity.WARN),
    *(_no_revisions(s) for s in sorted({d.store for d in BULK_DATASETS.values()})),
)
