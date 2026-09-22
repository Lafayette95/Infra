"""Pull daily settlement price / open interest for ABSOLUTE futures contracts.

Disk first, API only for gaps (Database/Daily - a separate store from ohlcv-1m). Only
absolute tickers (e.g. "SRZ4") are accepted; resolve them from the futures side first
(e.g. via scripts/update_futures.py or the dashboard) so their dataset is on disk.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_daily.py --tickers SR3Z4 SR3H5 --start 2025-01-01 --dry-run
    $PY scripts/update_daily.py --tickers SR3Z4 --start 2025-01-01
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
from infra.pipeline import daily as dl  # noqa: E402
from infra.pipeline.series import dataset_for_absolute  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", required=True, help="absolute contracts, e.g. SR3Z4")
    parser.add_argument("--start", required=True, help="UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="UTC date, exclusive; default = today")
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD, help="USD guardrail per request")
    parser.add_argument("--dry-run", action="store_true", help="Show gaps and estimated cost; download nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None)
    client = api.get_client()

    total, failures = 0.0, 0
    for ticker in sorted(args.tickers):
        try:
            dataset = dataset_for_absolute(ticker)
        except KeyError as exc:
            failures += 1
            print(f"{ticker}: FAILED - {exc}")
            continue
        gaps = dl.plan_daily_update(ticker, start, end)
        if not gaps:
            print(f"{ticker}: covered on disk - no API call")
            continue
        try:
            if args.dry_run:
                for g0, g1 in gaps:
                    cost = api.estimate_cost(dataset, "statistics", [ticker], g0, g1, "raw_symbol", client)
                    total += cost
                    print(f"{ticker}: would fetch {g0.date()} -> {g1.date()}  est. ${cost:.4f}")
            else:
                rows = dl.fetch_and_store_daily(ticker, gaps, dataset=dataset,
                                                max_cost_usd=args.max_cost, client=client)
                print(f"{ticker}: saved {rows:,} daily rows across {len(gaps)} range(s)")
        except Exception as exc:  # one bad contract must not abort the rest
            failures += 1
            print(f"{ticker}: FAILED - {type(exc).__name__}: {str(exc).splitlines()[0]}")
    if args.dry_run:
        print(f"Estimated total: ${total:.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
