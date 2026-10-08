"""Tick trades (aggressor side) for futures contracts, then their 1-minute signed bars.

Contracts: absolute tickers, or a ROOT (e.g. ZN) = every contract that was its front or
second (``v.0`` / ``v.1``) in the window - both contracts around each roll.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_trades.py --root ZN --start 2026-04-07 --end 2026-10-08 --dry-run
    $PY scripts/update_trades.py --root ZN --start 2026-04-07 --end 2026-10-08
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import databento_client as api  # noqa: E402
from infra.config import FUTURES_CONTRACTS_FILE, FUTURES_ROOTS, SCHEMA_TRADES  # noqa: E402
from infra.pipeline import trades as tp  # noqa: E402
from infra.pipeline.relative_daily import load_relative_daily  # noqa: E402
from infra.relative.symbology import parse_relative  # noqa: E402
from infra.storage import contract_store  # noqa: E402


def contracts_of(root: str, start, end) -> dict[str, tuple]:
    """Front and second contracts in the window, each with the part of the window inside
    its listed life (activation .. expiry, from the definitions): a request before a
    contract is listed can't resolve its symbol."""
    rel = load_relative_daily([parse_relative(f"{root}.v.0"), parse_relative(f"{root}.v.1")], start, end,
                              fetch_missing=False)
    if rel.empty:
        return {}
    table = contract_store.read_contracts(FUTURES_CONTRACTS_FILE, root)
    out = {}
    for t in sorted(rel["contract"].astype(str).unique()):
        life = table[table["ticker"].astype(str) == t]
        life = life[pd.to_datetime(life["expiry"]) >= start]
        a = max(start, pd.to_datetime(life["activation"]).min().normalize()) if len(life) else start
        b = min(end, pd.to_datetime(life["expiry"]).max().normalize() + pd.Timedelta(days=1)) if len(life) else end
        if a < b:
            out[t] = (a, b)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="e.g. ZN: its front + second contracts in the window")
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive")
    parser.add_argument("--dry-run", action="store_true", help="price what is missing, download nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    windows = {t: (start, end) for t in args.tickers}
    if args.root:
        windows |= contracts_of(args.root, start, end)
    dataset = FUTURES_ROOTS[args.root].dataset if args.root else "GLBX.MDP3"
    client = api.get_client()
    if args.dry_run:
        total = 0.0
        for t, (a0, b0) in windows.items():
            pieces = tp.month_pieces(tp.plan_trades_update(t, a0, b0))
            cost = sum(api.estimate_cost(dataset, SCHEMA_TRADES, [t], a, b, "raw_symbol", client=client)
                       for a, b in pieces)
            total += cost
            print(f"{t} {a0.date()}..{b0.date()}: {len(pieces)} monthly requests, ${cost:.2f}")
        print(f"TOTAL ${total:.2f}")
        return 0
    for t, (a0, b0) in windows.items():
        tp.load_trades([t], a0, b0, dataset=dataset, client=client, read=False)
    n = tp.build_signed_bars(list(windows), start, end)
    print(f"signed 1-minute bars written: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
