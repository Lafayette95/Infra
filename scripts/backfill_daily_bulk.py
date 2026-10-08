"""Bulk backfill of DAILY futures statistics (settlement / OI / cleared volume) for whole
roots: every outright contract in the root's expiry cycle, its whole life in the window.

Faster than the relative loaders' one-contract-at-a-time path (TOFIX "Daily statistics
requests are slow"): definition snapshots are fetched in parallel, then statistics go in
one request per (root, year) for every contract not yet covered there (Rule 2.1: covered
ranges are never re-asked; partly covered contracts keep the per-contract path for their
exact gaps). Network in parallel, file writes sequential.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/backfill_daily_bulk.py --roots ES NQ --start 2010-07-10 --dry-run
    $PY scripts/backfill_daily_bulk.py --macro --start 2010-07-10
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.config import FUTURES_ROOTS, MACRO_ROOTS, MAX_COST_USD  # noqa: E402
from infra.pipeline import contracts as contracts_pipe  # noqa: E402
from infra.pipeline import daily as dl  # noqa: E402
from infra.storage import contract_store, coverage_store  # noqa: E402

log = logging.getLogger("backfill_daily_bulk")


def definitions(roots, start, end, workers, dry_run, client, every_days: int) -> float:
    """Missing definition snapshots, thinned to one per ``every_days`` (on the standard
    30-day grid): enough whenever every contract is listed longer than that before it
    expires (true of the macro roots: index futures ~15 months ahead, FX years, energy and
    metals 2-9 years). The relative loaders never fetch the skipped ones (fetch_missing=False)."""
    step = max(1, every_days // contracts_pipe.DEFINITION_SNAPSHOT_DAYS)
    grid = contracts_pipe.snapshot_days(start, end)[::step]
    jobs = [(FUTURES_ROOTS[r], day) for r in roots
            for day, _ in contracts_pipe.missing_snapshots(FUTURES_ROOTS[r], start, end) if day in set(grid)]
    if dry_run:  # one snapshot per root priced and scaled (pricing ~3,000 one by one takes most of an hour)
        cost = 0.0
        for r in roots:
            mine = [day for cfg, day in jobs if cfg.root == r]
            if mine:
                cfg, day = FUTURES_ROOTS[r], mine[len(mine) // 2]
                cost += len(mine) * api.estimate_cost(cfg.dataset, api.SCHEMA_DEFINITION, [cfg.parent], day,
                                                      day + pd.Timedelta(days=1), "parent", client=client)
        log.info("definitions: %d snapshots missing, ~$%.2f (one per root priced, scaled)", len(jobs), cost)
        return cost
    def fetch(cfg, day):
        """None = nothing listed under the parent that day (e.g. RTY before 2017): a 422."""
        try:
            return api.fetch_definitions(cfg.dataset, [cfg.parent], day, max_cost_usd=MAX_COST_USD)
        except Exception as e:
            if "could be resolved" in str(e):
                return None
            raise

    done = failed = 0
    with ThreadPoolExecutor(workers) as pool:
        # one Databento client per request (fetch_definitions without client=): a requests
        # session is not documented thread-safe, so none is shared across workers
        # A failing job is logged and skipped: an exception escaping this loop would wait for
        # every queued job inside the pool's exit while storing nothing (found 2026-10-07).
        futs = {pool.submit(fetch, cfg, day): (cfg, day) for cfg, day in jobs}
        for f in as_completed(futs):
            cfg, day = futs[f]
            try:
                raw = f.result()
            except Exception as e:
                failed += 1
                log.warning("definitions %s %s failed, will be retried next run: %s", cfg.root, day.date(),
                            str(e).splitlines()[0][:160])
                continue
            if raw is None:   # nothing listed: covered, never asked again
                coverage_store.record_covered(contracts_pipe.FUTURES_DEFS_COVERAGE_FILE, f"defs:{cfg.root}",
                                              [(day, day + pd.Timedelta(days=1))])
            else:
                contracts_pipe.store_definitions_snapshot(cfg, day, raw)
            done += 1
            if done % 50 == 0:
                log.info("definitions: %d / %d", done, len(jobs))
    log.info("definitions: %d stored, %d failed", done, failed)
    return 0.0


def contracts_of(root: str, start, end) -> dict[str, list]:
    """Outright contracts of the root's expiry cycle alive in the window, each with its
    listed life(s) - activation to expiry (from the definitions)."""
    cfg = FUTURES_ROOTS[root]
    c = contract_store.read_contracts(contracts_pipe.FUTURES_CONTRACTS_FILE, root)
    if c.empty:
        return {}
    exp, act = pd.to_datetime(c["expiry"]), pd.to_datetime(c["activation"])
    keep = exp.dt.month.isin(cfg.expiry_months) & (exp >= pd.Timestamp(start))
    out: dict[str, list] = {}
    for t, a, e in zip(c.loc[keep, "ticker"].astype(str), act[keep], exp[keep]):
        out.setdefault(t, []).append((a.normalize(), e.normalize() + pd.Timedelta(days=1)))
    return out


def statistics(roots, start, end, workers, dry_run, client) -> float:
    # ONE request per (dataset, year) across ALL roots: Databento's time scales with the
    # date range scanned, not the symbol count (2026-10-07: ~300s for 8 ES contracts over a
    # year, ~230s for one SR3 contract over 15 months), so per-root requests multiplied it.
    by_piece: dict[tuple, list[str]] = {}
    singles = []
    for r in roots:
        cfg = FUTURES_ROOTS[r]
        lives = contracts_of(r, start, end)
        bulk, single = dl.plan_daily_bulk(sorted(lives), start, end, alive=lives)
        for piece, tickers in bulk:
            by_piece.setdefault((cfg.dataset, piece), []).extend(tickers)
        singles += [(cfg, t, gaps) for t, gaps in single]
    jobs = [(_Dataset(ds), piece, tickers) for (ds, piece), tickers in sorted(by_piece.items())]
    if dry_run:
        cost = sum(api.estimate_cost(cfg.dataset, api.SCHEMA_STATISTICS, tickers, *piece, "raw_symbol", client=client)
                   for cfg, piece, tickers in jobs)
        cost += sum(api.estimate_cost(cfg.dataset, api.SCHEMA_STATISTICS, [t], a, b, "raw_symbol", client=client)
                    for cfg, t, gaps in singles for a, b in gaps)
        log.info("statistics: %d bulk requests (%d contract-pieces), %d partial contracts, $%.2f",
                 len(jobs), sum(len(t) for *_, t in jobs), len(singles), cost)
        return cost
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(dl.fetch_daily_raw_bulk, tickers, piece, dataset=cfg.dataset):  # own client each
                (cfg, piece, tickers) for cfg, piece, tickers in jobs}
        for f in as_completed(futs):
            cfg, piece, tickers = futs[f]
            try:
                result = f.result()
            except Exception as e:
                log.warning("statistics %s %s..%s failed, will be retried next run: %s", cfg.dataset,
                            piece[0].date(), piece[1].date(), str(e).splitlines()[0][:160])
                continue
            for t, fetched in result.items():
                dl.store_daily_raw(t, fetched, dataset=cfg.dataset)
            done += 1
            log.info("statistics %s %s..%s: %d contracts stored (%d / %d requests)", cfg.dataset,
                     piece[0].date(), piece[1].date(), len(tickers), done, len(jobs))
    for cfg, t, gaps in singles:
        dl.fetch_and_store_daily(t, gaps, dataset=cfg.dataset, client=client)
    return 0.0


class _Dataset:
    """Stand-in for a root config where only the dataset matters (cross-root requests)."""
    def __init__(self, dataset: str):
        self.dataset = dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="*", default=[])
    parser.add_argument("--macro", action="store_true", help="all MACRO_ROOTS")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None, help="exclusive; default today")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--snapshot-every-days", type=int, default=180,
                        help="definition snapshot spacing (default 180: contracts listed > 6 months ahead)")
    parser.add_argument("--dry-run", action="store_true", help="price the plan (free), download nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("infra.pipeline.daily",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    roots = list(MACRO_ROOTS) if args.macro else args.roots
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    client = api.get_client()
    # bulk requests scan a whole year of the dataset: allow them far longer than the
    # default 300s (read at call time by databento_client)
    api.DATA_DEADLINE_S = 1800.0
    t = time.time()
    cost = definitions(roots, start, end, args.workers, args.dry_run, client, args.snapshot_every_days)
    if args.dry_run:
        log.info("statistics can only be priced exactly once the definitions are stored "
                 "(the contract list comes from them); pricing what is known now")
    cost += statistics(roots, start, end, args.workers, args.dry_run, client)
    log.info("%s in %.0fs%s", "priced" if args.dry_run else "done", time.time() - t,
             f": ${cost:.2f}" if args.dry_run else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
