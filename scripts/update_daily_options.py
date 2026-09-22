"""Pull daily settlement price / open interest for a filtered set of option contracts
(definitions -> filter -> statistics). Mirrors scripts/update_options.py's shape.

Example:
    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_daily_options.py --parent SR3.OPT --definitions-day 2025-03-12 \
        --start 2025-03-10 --end 2025-03-12 --dry-run
    $PY scripts/update_daily_options.py --parent SR3.OPT --definitions-day 2025-03-12 \
        --start 2025-03-10 --end 2025-03-12 --strike-min 95 --strike-max 96 \
        --expiry-min 2025-03-01 --expiry-max 2025-06-30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.config import MAX_COST_USD, OPTIONS_UNIVERSE  # noqa: E402
from infra.pipeline import daily_options as opt  # noqa: E402
from infra.processing import definitions as defs  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parent", default=next(iter(OPTIONS_UNIVERSE)))
    p.add_argument("--definitions-day", required=True, help="Day whose active contracts define the universe")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--underlyings", nargs="*")
    p.add_argument("--types", nargs="*", choices=["C", "P"])
    p.add_argument("--strike-min", type=float)
    p.add_argument("--strike-max", type=float)
    p.add_argument("--expiry-min")
    p.add_argument("--expiry-max")
    p.add_argument("--max-cost", type=float, default=MAX_COST_USD, help="USD guardrail per request")
    p.add_argument("--dry-run", action="store_true", help="Show contract count and estimated cost; download nothing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = OPTIONS_UNIVERSE[args.parent]
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    strikes = (args.strike_min, args.strike_max) if args.strike_min is not None and args.strike_max is not None else None
    expiries = (pd.Timestamp(args.expiry_min), pd.Timestamp(args.expiry_max)) if args.expiry_min and args.expiry_max else None
    client = api.get_client()

    all_defs = opt.load_definitions(args.parent, args.definitions_day, dataset=dataset, client=client)
    contracts = defs.filter_definitions(
        all_defs, underlyings=args.underlyings, option_types=args.types,
        strike_range=strikes, expiry_range=expiries,
    )
    print(f"{len(contracts):,} contract(s) match the filter (of {len(all_defs):,} total)")
    if contracts.empty:
        return 0

    if args.dry_run:
        plan = opt.plan_daily_options_update(contracts["instrument_id"].tolist(), start, end)
        total = 0.0
        for (g0, g1), ids in plan.items():
            for i in range(0, len(ids), api.MAX_SYMBOLS_PER_REQUEST):
                batch = ids[i:i + api.MAX_SYMBOLS_PER_REQUEST]
                cost = api.estimate_cost(dataset, "statistics", batch, g0, g1, "instrument_id", client)
                total += cost
                print(f"  {g0.date()} -> {g1.date()}: {len(batch)} contract(s)  est. ${cost:.4f}")
        print(f"Estimated total: ${total:.4f}")
        return 0

    rows = opt.fetch_and_store_daily_options(
        opt.plan_daily_options_update(contracts["instrument_id"].tolist(), start, end),
        contracts, dataset=dataset, max_cost_usd=args.max_cost, client=client,
    )
    print(f"{rows:,} daily settlement/OI rows saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
