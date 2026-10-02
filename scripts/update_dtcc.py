"""Archive DTCC's daily swap-trade reports (CLAUDE.md section 16) outside the daily cycle -
e.g. to fill a gap by hand. Free; only days not on disk are requested.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_dtcc.py                                   # whole retention window
    $PY scripts/update_dtcc.py --start 2026-09-01 --end 2026-09-30
    $PY scripts/update_dtcc.py --dry-run                         # list what would be fetched
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.cycle.raw_dtcc import backfill_daily_dtcc, plan_daily_dtcc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", help="first day, inclusive (default: the retention lookback)")
    parser.add_argument("--end", help="last day, inclusive (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="list the days that would be requested")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    start = pd.Timestamp(args.start) if args.start else end
    if args.dry_run:
        for kind, days in plan_daily_dtcc(start, end).items():
            print(f"{kind}: {len(days)} day(s) to request" + (f", {days[0].date()} -> {days[-1].date()}" if days else ""))
        return 0
    failed = 0
    for kind, r in backfill_daily_dtcc(start, end)["reports"].items():
        print(f"{kind}: archived {len(r['archived'])}, not published {len(r['unpublished'])}, failed {len(r['errors'])}")
        for day, msg in r["errors"].items():
            print(f"  {day.date()}: {msg}")
        failed += len(r["errors"])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
