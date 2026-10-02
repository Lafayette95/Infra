"""Staged bbo-1m backfill of a root's front contract (volume roll rule, <root>.v.0), plus
the other front-two contract only around each roll - infra/pipeline/front_quotes.py.

Run one stage at a time to watch the cost. --dry-run fetches only the cheap planning
inputs (definitions, daily cleared volume; ~$0.05/year for 6 roots) and PRICES the
quotes without buying them.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/backfill_front_quotes.py --roots ZT ZF ZN TN ZB UB --start 2015-01-01 --end 2016-01-01 --dry-run
    $PY scripts/backfill_front_quotes.py --roots ZT ZF ZN TN ZB UB --start 2015-01-01 --end 2016-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.config import MAX_COST_USD  # noqa: E402
from infra.cycle.network import wait_for_network  # noqa: E402
from infra.pipeline import front_quotes as fq  # noqa: E402
from infra.relative.rolls import roll_switches  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--start", required=True, help="UTC day, inclusive")
    parser.add_argument("--end", required=True, help="UTC day, exclusive")
    parser.add_argument("--pad-bdays", type=int, default=5, help="business days of the other contract around a roll")
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD, help="USD guardrail per request")
    parser.add_argument("--dry-run", action="store_true", help="plan + price the quotes; buy nothing but the planning inputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    wait_for_network()
    client, pad = api.get_client(), pd.offsets.BDay(args.pad_bdays)
    total_est, total_rows, failed = 0.0, 0, {}
    for root in args.roots:
        mapping, gaps = fq.plan_front_quotes(root, args.start, args.end, pad=pad, max_cost_usd=args.max_cost, client=client)
        rolls = ", ".join(f"{d:%m-%d} {a}->{b}" for d, a, b in roll_switches(mapping, 0))
        est = fq.price_front_quotes(root, gaps, client=client)
        total_est += est
        print(f"{root}: rolls [{rolls}] | {sum(len(v) for v in gaps.values())} window(s) over "
              f"{len(gaps)} contract(s), est. ${est:.3f}")
        if not args.dry_run and gaps:
            rows, errors = fq.fetch_front_quotes(root, gaps, max_cost_usd=args.max_cost, client=client)
            total_rows += rows
            failed.update({f"{root} {t}": e for t, e in errors.items()})
            print(f"   stored {rows:,} quotes" + (f", errors: {errors}" if errors else ""))
    print(f"TOTAL est. ${total_est:.3f}" + ("" if args.dry_run else f" | stored {total_rows:,} quotes | "
                                                                 f"errors: {failed or 'none'}"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
