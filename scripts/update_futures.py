"""Pull missing futures bars into the local database (disk first, API only for gaps).

Examples (always use the infra-env interpreter):
    /opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python scripts/update_futures.py --start 2025-01-01 --end 2025-03-01 --dry-run
    /opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python scripts/update_futures.py --tickers SR3.c.0 ZN.c.0 --start 2025-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.config import FUTURES_UNIVERSE, MAX_COST_USD, dataset_for_ticker  # noqa: E402
from infra.pipeline import futures as fut  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(FUTURES_UNIVERSE))
    parser.add_argument("--start", required=True, help="UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="UTC date, exclusive; default = today")
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD, help="USD guardrail per request")
    parser.add_argument("--dry-run", action="store_true", help="Show gaps and estimated cost; download nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None)

    client = api.get_client() if args.dry_run else None
    total = 0.0
    for ticker in args.tickers:
        gaps = fut.plan_futures_update(ticker, start, end)
        if not gaps:
            print(f"{ticker}: fully covered on disk - no API call")
            continue
        if args.dry_run:
            for g_start, g_end in gaps:
                cost = api.estimate_cost(dataset_for_ticker(ticker), "ohlcv-1m", [ticker], g_start, g_end,
                                         "continuous", client)
                total += cost
                print(f"{ticker}: would fetch {g_start.date()} -> {g_end.date()}  est. ${cost:.4f}")
        else:
            rows = fut.fetch_and_store_futures(ticker, gaps, max_cost_usd=args.max_cost)
            print(f"{ticker}: saved {rows:,} rows across {len(gaps)} range(s)")
    if args.dry_run:
        print(f"Estimated total: ${total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
