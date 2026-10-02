"""Step 0 - ``backfill_daily_ref``: reference data, FIRST in the cycle (CLAUDE.md 12, 18).

Sources, in order (each is a function ``(start, end, *, paths, force_refetch) -> dict``):
* ``tsy_auctions`` - the Treasury auctions (Fiscal Data), the nowcast session's source
  (``infra.cycle.raw_reference.backfill_daily_tsy_auctions``, unchanged; moved here from
  the raw step 2026-10-02 so everything below sees today's auctions);
* ``cme_tcf`` - CME's daily conversion-factor files, archived raw;
* ``treasury_ref`` - rebuild the securities table (from the auctions), the on/off-the-run
  map over the window, and the CME baskets (from the archive). Local, seconds.

``px`` runs after this step but does NOT depend on it: if Fiscal Data or CME's FTP is down,
prices still run on yesterday's reference - harmless, since a bond is auctioned days before
it has a price. Output nests under ``"sources"`` like the raw step's, so the auction checks
(which read ``ctx.output["sources"]["tsy_auctions"]``) work unchanged.
"""
from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from infra.config import TREASURY_BASKET_HISTORY, TREASURY_BASKET_RULES, TREASURY_OTR_TENORS
from infra.cycle.core import Check, Severity, Step, StepContext
from infra.cycle.paths import CyclePaths
from infra.cycle.raw_reference import AUCTION_CHECKS, backfill_daily_tsy_auctions
from infra.pipeline import futures_baskets as pfb
from infra.pipeline import treasury_otr as potr
from infra.pipeline import treasury_ref as pref
from infra.pipeline.tsy_auctions import read_auctions
from infra.processing import futures_baskets as fb

log = logging.getLogger(__name__)
_ONE_DAY = pd.Timedelta(days=1)
OTR_LOOKBACK_DAYS = 30


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def backfill_daily_cme_tcf(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False) -> dict:
    """Archive every CME conversion-factor file listed and not on disk yet (a file on disk
    is its own coverage; ``force_refetch`` is ignored - a published file never changes)."""
    paths = paths or CyclePaths.default()
    try:
        out = pfb.archive_tcf(root=paths.cme_tcf_dir)
        return {"archived": out["archived"], "errors": out["errors"]}
    except Exception as exc:
        log.warning("CME TCF listing failed: %s", exc)
        return {"archived": [], "errors": {"listing": _error(exc)}}


def backfill_daily_treasury_ref(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False) -> dict:
    """Rebuild the reference stores from what's on disk: securities (whole table), the
    OTR map for ``[start - OTR_LOOKBACK_DAYS, end]``, the CME baskets (whole archive).
    Never fetches."""
    paths = paths or CyclePaths.default()
    out = {"securities": 0, "otr_rows": 0, "basket_rows": 0, "errors": {}}
    for key, fn in (
        ("securities", lambda: pref.build_securities(auctions_root=paths.tsy_auctions_dir,
                                                     root=paths.treasury_securities_dir)),
        # 30 days back, not just the window: days a missed run never built get built (cheap)
        ("otr_rows", lambda: potr.build_otr(pd.Timestamp(start) - pd.Timedelta(days=OTR_LOOKBACK_DAYS), end,
                                            root=paths.treasury_otr_dir,
                                            securities_root=paths.treasury_securities_dir)),
        ("basket_rows", lambda: pfb.build_cme_baskets(archive_root=paths.cme_tcf_dir,
                                                      root=paths.treasury_baskets_dir)),
    ):
        try:
            out[key] = fn()
        except Exception as exc:
            out["errors"][key] = _error(exc)
            log.warning("treasury reference %s failed: %s", key, out["errors"][key])
    return out


REF_SOURCES: dict[str, Callable[..., dict]] = {
    "tsy_auctions": backfill_daily_tsy_auctions,  # first: the securities table is built from it
    "cme_tcf": backfill_daily_cme_tcf,
    "treasury_ref": backfill_daily_treasury_ref,
}


def backfill_daily_ref(start, end, *, paths: CyclePaths | None = None, force_refetch: bool = False,
                       sources: dict[str, Callable[..., dict]] | None = None) -> dict:
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    sources = REF_SOURCES if sources is None else sources
    return {"sources": {name: fn(start, end, paths=paths, force_refetch=force_refetch)
                        for name, fn in sources.items()}}


def _run(ctx: StepContext) -> dict:
    return backfill_daily_ref(ctx.start, ctx.end, paths=ctx.paths, force_refetch=ctx.force_refetch,
                              sources=ctx.options.get("ref_sources"))


# ------------------------------------------------------------------------ checks
def _out(ctx: StepContext, name: str) -> dict:
    return ctx.output.get("sources", {}).get(name, {})


def _check_cme_fetch_ok(ctx: StepContext):
    errors = _out(ctx, "cme_tcf").get("errors", {})
    if not errors:
        return True, f"CME files archived this run: {len(_out(ctx, 'cme_tcf').get('archived', []))}", None
    return False, f"{len(errors)} CME file(s) failed (retried next run)", \
        pd.DataFrame({"file": list(errors), "error": list(errors.values())})


def _check_cme_complete(ctx: StepContext):
    """Every business day in the window that CME must have posted by now is archived. CME
    posts day D's file ~06:30 Chicago ON day D, so the run day's own file may not be there
    yet: only days before the run day are judged."""
    have = pfb.archived_files(root=ctx.paths.cme_tcf_dir)
    last = min(ctx.end, ctx.run_day - _ONE_DAY)
    missing = [d for d in pd.bdate_range(ctx.start, last) if d not in have] if last >= ctx.start else []
    if not missing:
        return True, "every due CME file in the window archived", None
    return False, f"{len(missing)} business day(s) without a CME file (holiday, or not posted)", \
        pd.DataFrame({"day": missing})


def _check_ref_built(ctx: StepContext):
    errors = _out(ctx, "treasury_ref").get("errors", {})
    if not errors:
        o = _out(ctx, "treasury_ref")
        return True, (f"{o.get('securities', 0)} securities, {o.get('otr_rows', 0)} OTR rows, "
                      f"{o.get('basket_rows', 0)} CME basket rows"), None
    return False, f"reference build failed: {errors}", None


def _check_otr_complete(ctx: StepContext):
    """Every business day in the window has exactly one on-the-run issue per tenor
    (default convention), for every tenor that has an issue outstanding."""
    df = potr.read_otr(ctx.start, ctx.end + _ONE_DAY, rank=0, root=ctx.paths.treasury_otr_dir)
    days = pd.bdate_range(ctx.start, ctx.end)
    active = set(df["tenor"])
    counts = df.groupby(["timestamp", "tenor"]).size()
    bad = [(d, t, int(counts.get((d, t), 0))) for d in days for t in TREASURY_OTR_TENORS if t in active
           and counts.get((d, t), 0) != 1]
    if not bad:
        return True, f"{len(days)} day(s) x {len(active)} tenor(s): one on-the-run issue each", None
    return False, f"{len(bad)} day-tenor(s) without exactly one on-the-run issue", \
        pd.DataFrame(bad, columns=["timestamp", "tenor", "rank0_rows"])


def _check_baskets_match_rules(ctx: StepContext):
    """Warning: on the latest CME file in the window, CME's baskets equal the ones computed
    from TREASURY_BASKET_RULES - a mismatch means CME changed a rule, or a new issue slipped
    in late, and the computed (pre-2023) history may need revisiting."""
    have = {d: p for d, p in pfb.archived_files(root=ctx.paths.cme_tcf_dir).items() if ctx.start <= d <= ctx.end}
    if not have:
        return True, "no CME file in the window to compare", None
    day = max(have)
    cme = fb.parse_tcf(have[day].read_bytes(), day)
    sec = pref.read_securities(root=ctx.paths.treasury_securities_dir)
    terms = fb.issued_terms(read_auctions(nominal_only=False, root=ctx.paths.tsy_auctions_dir), day)
    diffs = []
    for (root, con), g in cme[cme["root"].isin(TREASURY_BASKET_HISTORY)].groupby(["root", "contract"]):
        comp = fb.computed_basket(sec, root, g["delivery_month"].iloc[0], TREASURY_BASKET_RULES[root], day, terms)
        a, b = set(g["cusip"]), set(comp["cusip"])
        diffs += [(day, con, c, "only CME") for c in a - b] + [(day, con, c, "only computed") for c in b - a]
    if not diffs:
        return True, f"CME baskets of {day.date()} = the rules' baskets", None
    return False, f"{len(diffs)} basket difference(s) vs the rules on {day.date()}", \
        pd.DataFrame(diffs, columns=["day", "contract", "cusip", "side"])


REF_CHECKS: tuple[Check, ...] = (
    *AUCTION_CHECKS,
    Check("cme_tcf_fetch_ok", _check_cme_fetch_ok, severity=Severity.WARN),
    Check("cme_tcf_complete", _check_cme_complete, severity=Severity.WARN),
    Check("treasury_ref_built", _check_ref_built),
    Check("treasury_otr_complete", _check_otr_complete),
    Check("baskets_match_rules", _check_baskets_match_rules, severity=Severity.WARN),
)

REF_STEP = Step("ref", _run, checks=REF_CHECKS)
