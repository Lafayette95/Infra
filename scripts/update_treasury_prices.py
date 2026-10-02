"""Fetch and store US Treasury prices per CUSIP from FedInvest (CLAUDE.md 18) - free,
only uncovered business days. Rebuilds the reference table first (yields need it).

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_treasury_prices.py --start 2016-01-04            # through yesterday
    $PY scripts/update_treasury_prices.py --start 2016-01-04 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.pipeline.treasury_prices import fetch_and_store_prices, plan_prices_update  # noqa: E402
from infra.pipeline.treasury_ref import build_securities  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="first day, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", help="last day, inclusive (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true", help="list the days that would be requested")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    days = plan_prices_update(pd.Timestamp(args.start), end)
    print(f"{len(days)} business day(s) to request" + (f", {days[0].date()} -> {days[-1].date()}" if days else ""))
    if args.dry_run or not days:
        return 0
    build_securities()
    t0 = time.time()
    out = fetch_and_store_prices(days)
    print(f"stored {len(out['stored'])}, holidays {len(out['holiday'])}, pending {len(out['pending'])}, "
          f"failed {len(out['errors'])} in {time.time() - t0:.0f}s")
    for day, msg in out["errors"].items():
        print(f"  {day.date()}: {msg}")
    return 1 if out["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
