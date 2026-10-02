"""Step 1b, CPI weights - ``backfill_daily_cpi_weights``: any due CPI relative-importance
year not on disk, through ``infra.pipeline.cpi_weights`` (no fetch/storage code here), plus
the step's checks. On most days the plan is empty and NO request is made.

Registered in ``infra.cycle.raw.RAW_SOURCES`` under ``"cpi_weights"`` - after
``"releases"``, whose ALFRED CPI vintages date each weight year's publication.
"""
from __future__ import annotations

import pandas as pd

from infra.cycle.core import Check, Severity, StepContext
from infra.cycle.paths import CyclePaths
from infra.pipeline import cpi_weights as pcw

# A due weight year still missing this long after it became due (see pcw.DUE_FROM) is
# reported (warn): the January CPI - and the weights with it - is out by late February.
LATE_AFTER = pd.Timedelta(days=45)


def backfill_daily_cpi_weights(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                               fetch=None) -> dict:
    """Fetch every due weight year not on disk, as of ``end``. ``start`` and
    ``force_refetch`` are unused: a published weight year never changes, and there is no
    window to re-ask (kept for the raw-source signature)."""
    paths = paths or CyclePaths.default()
    years = pcw.plan_weights_update(end, coverage_file=paths.cpi_weights_coverage)
    out = {"planned": years, "fetched": [], "rows": 0, "error": None, "unconfigured": False}
    if not years:
        return out
    if fetch is None and not pcw.CONFIGURED():
        return {**out, "unconfigured": True}
    try:
        fetched = pcw.fetch_weights_raw(years, fetch=fetch)
        out["rows"] = pcw.store_weights_raw(fetched, root=paths.cpi_weights_dir,
                                            coverage_file=paths.cpi_weights_coverage,
                                            releases_root=paths.releases_dir, catalog_dir=paths.bulk_catalog_dir)
        out["fetched"] = sorted(fetched)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    return out


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext) -> dict:
    return ctx.output.get("sources", {}).get("cpi_weights", {})


def _check_fetch_ok(ctx: StepContext):
    out = _out(ctx)
    if out.get("error"):
        return False, f"CPI weights fetch failed: {out['error']}", None
    if out.get("unconfigured"):
        return True, "CPI weights due but BLS_CONTACT_EMAIL not set - not fetched (see cpi_weights_fresh)", None
    if not out.get("planned"):
        return True, "every due weight year on disk - nothing requested", None
    return True, f"fetched weight years {out.get('fetched')}; not published yet: " \
                 f"{sorted(set(out['planned']) - set(out.get('fetched', [])))}", None


def _check_sane(ctx: StepContext):
    """Each weight year fetched this run: "All items" is 100 for CPI-U (and CPI-W), and
    every weight lies in [0, 100]."""
    bad = []
    for year in _out(ctx).get("fetched", []):
        df = pcw.read_cpi_weights(weight_year=year, root=ctx.paths.cpi_weights_dir)
        top = df[df["item_name"].str.lower().eq("all items")].head(1)
        if top.empty or not (top[["cpi_u", "cpi_w"]] == 100.0).all(axis=None):
            bad.append((year, "All items is not 100"))
        if not df["cpi_u"].between(0, 100).all():
            bad.append((year, "a CPI-U weight outside [0, 100]"))
    if not bad:
        return True, "every fetched weight year sane", None
    return False, f"{len(bad)} problem(s)", pd.DataFrame(bad, columns=["weight_year", "problem"])


def _check_fresh(ctx: StepContext):
    """Warning: a due weight year is still missing well after it became due."""
    missing = pcw.plan_weights_update(ctx.end - LATE_AFTER, coverage_file=ctx.paths.cpi_weights_coverage)
    if not missing:
        return True, "every weight year due by now is on disk", None
    return False, f"weight year(s) overdue: {missing}", None


CPI_WEIGHTS_CHECKS = (
    Check("cpi_weights_fetch_ok", _check_fetch_ok),
    Check("cpi_weights_sane", _check_sane),
    Check("cpi_weights_fresh", _check_fresh, severity=Severity.WARN),
)
