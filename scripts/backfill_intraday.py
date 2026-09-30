"""Intraday cycle, run by hand until it is scheduled: fetch ZQ intraday px (bbo-1m quotes
+ ohlcv-1m bars, all 12 nearest contracts) for [start, end] - Rule 2.1, only
never-queried days - then recompute the 15-minute intraday WIRP over the same days.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/backfill_intraday.py --start 2026-08-31 --end 2026-09-29 --dry-run   # free cost estimate
    $PY scripts/backfill_intraday.py --start 2026-08-31 --end 2026-09-29
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.cycle.intraday import (  # noqa: E402
    INTRADAY_UNIVERSE,
    IntradayPaths,
    backfill_intraday_px,
    backfill_intraday_wirp,
    check_wirp_intraday,
    plan_intraday_px,
)
from infra.cycle.network import wait_for_network  # noqa: E402
from infra.cycle.paths import CyclePaths  # noqa: E402
from infra.cycle.universe import daily_universe  # noqa: E402


def dry_run(start: pd.Timestamp, end: pd.Timestamp) -> int:
    paths, ipaths, client = CyclePaths.default(), IntradayPaths.default(), api.get_client()
    members, errors = daily_universe(start, end, paths=paths, specs=INTRADAY_UNIVERSE, refresh_contracts=False)
    total = 0.0
    for (schema, ticker), gaps in sorted(plan_intraday_px(members, start, end, ipaths=ipaths).items()):
        for g0, g1 in gaps:
            cost = api.estimate_cost(members[ticker].dataset, schema, [ticker], g0, g1, "raw_symbol", client)
            total += cost
            print(f"{schema:9s} {ticker:6s} {g0.date()} -> {g1.date()}  est. ${cost:.4f}")
    for root, err in errors.items():
        print(f"universe {root}: {err}")
    print(f"estimated total ${total:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="first day, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="last day, inclusive")
    parser.add_argument("--dry-run", action="store_true", help="estimate API cost only; download nothing")
    parser.add_argument("--wirp-only", action="store_true", help="skip fetching; recompute intraday WIRP")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if args.dry_run:
        return dry_run(start, end)
    ok = True
    if not args.wirp_only:
        wait_for_network()
        px = backfill_intraday_px(start, end)
        print(f"px: {px['planned']} planned request(s), rows {px['rows']}, errors {px['fetch_errors'] or 'none'}")
        ok = not px["fetch_errors"] and not px["universe_errors"]
    wirp = backfill_intraday_wirp(start, end)
    problems = check_wirp_intraday(IntradayPaths.default(), start, end)
    print(f"intraday WIRP: {wirp['grid_times']} grid times, {wirp['rows']} rows; "
          f"checks: {'; '.join(problems) or 'ok'}")
    return 0 if ok and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
