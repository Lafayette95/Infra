"""Fetch and store repo rates (NY Fed SOFR/TGCR/BGCR, OFR's repo release, the DTCC GCF
history) and the NY Fed's Treasury securities lending results (CLAUDE.md 19) - free,
only what isn't covered yet.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_repo.py --start 1999-01-01             # full history, through yesterday
    $PY scripts/update_repo.py --start 2026-09-01 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.pipeline.repo import fetch_and_store_repo, plan_repo_update  # noqa: E402
from infra.pipeline.sec_lending import fetch_and_store_sec_lending, plan_sec_lending_update  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="first day, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", help="last day, inclusive (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true", help="list the requests that would be made")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    plan = plan_repo_update(pd.Timestamp(args.start), end)
    ranges = plan_sec_lending_update(pd.Timestamp(args.start), end)
    for key, (a, b) in plan.items():
        print(f"repo     {key:14s} {a.date()} -> {b.date()}")
    print(f"lending  {len(ranges)} request(s)" + (f", {ranges[0][0].date()} -> {ranges[-1][1].date()}" if ranges else ""))
    if args.dry_run:
        return 0
    t0 = time.time()
    rates = fetch_and_store_repo(plan) if plan else {"rows": {}, "errors": {}}
    lending = fetch_and_store_sec_lending(ranges) if ranges else {"rows": 0, "days": 0, "errors": {}}
    print(f"repo rows {sum(rates['rows'].values())}, lending rows {lending['rows']} ({lending['days']} days), "
          f"errors {len(rates['errors']) + len(lending['errors'])} in {time.time() - t0:.0f}s")
    return 1 if rates["errors"] or lending["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
