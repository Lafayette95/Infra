"""Pull option bars for a filtered set of contracts (definitions -> filter -> bars).

Example:
    /opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python scripts/update_options.py --parent SR3.OPT \
        --definitions-day 2025-01-15 --start 2025-01-15 --end 2025-01-31 \
        --strike-min 95 --strike-max 96 --expiry-min 2025-03-01 --expiry-max 2025-06-30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import MAX_COST_USD, OPTIONS_UNIVERSE  # noqa: E402
from infra.pipeline import options as opt  # noqa: E402


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
    p.add_argument("--max-cost", type=float, default=MAX_COST_USD)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    strikes = (args.strike_min, args.strike_max) if args.strike_min is not None and args.strike_max is not None else None
    expiries = (pd.Timestamp(args.expiry_min), pd.Timestamp(args.expiry_max)) if args.expiry_min and args.expiry_max else None

    df = opt.load_options(
        args.parent, args.definitions_day, args.start, args.end,
        underlyings=args.underlyings, option_types=args.types,
        strike_range=strikes, expiry_range=expiries, max_cost_usd=args.max_cost,
    )
    print(f"{len(df):,} option bars on disk for the selection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
