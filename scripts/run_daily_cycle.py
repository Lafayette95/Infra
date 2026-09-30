"""Run the daily cycle once (CLAUDE.md section 12) - history backfill, or one scheduled-
style run by hand. Exits non-zero if the cycle fails.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/run_daily_cycle.py --start 2026-06-01 --end 2026-09-25 --dry-run   # free cost estimate
    $PY scripts/run_daily_cycle.py --start 2026-06-01 --end 2026-09-25             # history backfill
    $PY scripts/run_daily_cycle.py --start 2026-09-01 --end 2026-09-25 --steps px derived
    $PY scripts/run_daily_cycle.py --scheduled                                     # T-N..T, force_refetch
    $PY scripts/run_daily_cycle.py --scheduled --prefect                           # same, as a Prefect run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.config import BOND_CURVES, DAILY_BACKFILL, FUTURES_ROOTS  # noqa: E402
from infra.cycle.paths import CyclePaths  # noqa: E402
from infra.cycle.px import plan_daily_px_data  # noqa: E402
from infra.cycle.px_bonds import plan_daily_bond_px  # noqa: E402
from infra.cycle.runner import run_daily_cycle, run_scheduled_daily, scheduled_windows  # noqa: E402
from infra.cycle.universe import snapshot_grid_floor  # noqa: E402
from infra.pipeline import contracts as contracts_pipe  # noqa: E402
from infra.pipeline.daily import bounded_by_availability  # noqa: E402


def dry_run(start, end, force_refetch: bool) -> int:
    """Price every API call the cycle would make (px settlements + any missing definition
    snapshots) with the free metadata.get_cost endpoint. Downloads nothing."""
    paths, client = CyclePaths.default(), api.get_client()
    total = 0.0
    for root, spec in DAILY_BACKFILL.items():
        if not spec.enabled:
            continue
        cfg = FUTURES_ROOTS[root]
        for d0, d1 in contracts_pipe.missing_snapshots(cfg, snapshot_grid_floor(start), end + pd.Timedelta(days=1),
                                                        coverage_file=paths.defs_coverage):
            cost = api.estimate_cost(cfg.dataset, "definition", [cfg.parent], d0, d1, "parent", client)
            total += cost
            print(f"definitions {root:5s} {d0.date()}  est. ${cost:.4f}")
    plan, errors = plan_daily_px_data(start, end, force_refetch=force_refetch)
    for ticker, (m, gaps) in sorted(plan.items()):
        for g0, g1 in gaps:
            query_end, _ = bounded_by_availability(m.dataset, g1, client)  # same bound as the real fetch
            if query_end <= g0:
                print(f"settlement  {ticker:22s} {g0.date()} -> not queryable yet (dataset lag)")
                continue
            cost = api.estimate_cost(m.dataset, "statistics", [ticker], g0, query_end, "raw_symbol", client)
            total += cost
            print(f"settlement  {ticker:22s} {g0.date()} -> {query_end:%F %H:%M}  est. ${cost:.4f}")
    for root, err in errors.items():
        print(f"universe    {root}: {err}")
    for country, gaps in plan_daily_bond_px(start, end, force_refetch=force_refetch).items():
        for g0, g1 in gaps:
            print(f"bonds       {country:22s} {g0.date()} -> {g1.date()}  free ({BOND_CURVES[country].source})")
    print(f"{len(plan)} contracts to fetch; estimated total ${total:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="first day, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", help="last day, inclusive; default today")
    parser.add_argument("--steps", nargs="+", default=None, help="subset of steps (default: all)")
    parser.add_argument("--force-refetch", action="store_true", help="re-query covered days (history mode)")
    parser.add_argument("--scheduled", action="store_true", help="T-N..T per step with force_refetch")
    parser.add_argument("--today", default=None, help="T for --scheduled (default: today, UTC)")
    parser.add_argument("--prefect", action="store_true", help="run through the Prefect flow")
    parser.add_argument("--dry-run", action="store_true", help="estimate API cost only; download nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    today = pd.Timestamp(args.today) if args.today else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    if args.scheduled:
        if args.dry_run:
            start, end = scheduled_windows(today)["px"]
            return dry_run(start, end, force_refetch=True)
        if args.prefect:
            from infra.cycle.flows import daily_cycle_flow
            daily_cycle_flow(str(today.date()))
            return 0
        report = run_scheduled_daily(today)
    else:
        if not args.start:
            parser.error("--start is required unless --scheduled")
        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end) if args.end else today
        if args.dry_run:
            return dry_run(start, end, force_refetch=args.force_refetch)
        if args.prefect:
            from infra.cycle.flows import backfill_flow
            backfill_flow(str(start.date()), str(end.date()), args.steps, args.force_refetch)
            return 0
        report = run_daily_cycle(start, end, steps=args.steps, force_refetch=args.force_refetch)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
