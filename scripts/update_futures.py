"""Pull missing futures data into the local database (disk first, API only for gaps).

Tickers may be relative (``SR3.c.0``, ``SR3.v.0``) or absolute (``SRZ4``). Either way only
ABSOLUTE contracts are queried and stored; relative tickers are resolved from definitions.
NOTE: even with --dry-run, missing ``definition`` snapshots are downloaded (tiny, ~$0.006 each)
because the plan cannot be computed without them. OHLCV data is never downloaded in a dry run.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_futures.py --tickers SR3.c.0 SR3.c.1 --start 2025-01-01 --end 2025-03-01 --dry-run
    $PY scripts/update_futures.py --tickers SR3.v.0 --start 2025-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.config import FUTURES_ROOTS, MAX_COST_USD  # noqa: E402
from infra.pipeline import contracts as contracts_pipe  # noqa: E402
from infra.pipeline import futures as fut  # noqa: E402
from infra.pipeline import relative as rel  # noqa: E402
from infra.pipeline.series import dataset_for_absolute, split_tickers  # noqa: E402


def _absolute_windows(specs, absolute, start, end, client):
    """{absolute ticker: (dataset, window)} for everything the request needs."""
    windows = {}
    for root, root_specs in rel.group_by_root(specs).items():
        cfg = FUTURES_ROOTS[root]
        contracts = contracts_pipe.ensure_contracts(cfg, start, end, fetch_missing=True, client=client)
        found, _ = rel.needed_contract_windows(root_specs, cfg, contracts, start, end)
        windows.update({t: (cfg.dataset, w) for t, w in found.items()})
    for ticker in absolute:
        windows[ticker] = (dataset_for_absolute(ticker), (start, end))
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", required=True,
                        help="relative (SR3.v.0) and/or absolute (SRZ4; quote symbols with spaces)")
    parser.add_argument("--start", required=True, help="UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="UTC date, exclusive; default = today")
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD, help="USD guardrail per request")
    parser.add_argument("--dry-run", action="store_true", help="Show gaps and estimated cost; download no bars")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None)

    client = api.get_client()
    specs, absolute = split_tickers(args.tickers)
    windows = _absolute_windows(specs, absolute, start, end, client)

    total, failures = 0.0, 0
    for ticker, (dataset, (w0, w1)) in sorted(windows.items()):
        gaps = fut.plan_futures_update(ticker, w0, w1)
        if not gaps:
            print(f"{ticker}: covered on disk - no API call")
            continue
        try:
            if args.dry_run:
                for g0, g1 in gaps:
                    cost = api.estimate_cost(dataset, "ohlcv-1m", [ticker], g0, g1, "raw_symbol", client)
                    total += cost
                    print(f"{ticker}: would fetch {g0.date()} -> {g1.date()}  est. ${cost:.4f}")
            else:
                rows = fut.fetch_and_store_futures(ticker, gaps, dataset=dataset,
                                                   max_cost_usd=args.max_cost, client=client)
                print(f"{ticker}: saved {rows:,} rows across {len(gaps)} range(s)")
        except Exception as exc:  # one bad contract must not abort the rest
            failures += 1
            print(f"{ticker}: FAILED - {type(exc).__name__}: {str(exc).splitlines()[0]}")
    if args.dry_run:
        print(f"Estimated total: ${total:.4f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
