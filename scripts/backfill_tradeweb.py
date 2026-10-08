"""Backfill FTSE-Tradeweb closing prices (infra.pipeline.tradeweb_prices): discover the
securities alive on sampled days, then pull each one's history. Uses the user's InSite login
(.env); polite (one session, >= 2s between requests). Archived chunks are never asked again.

    $PY scripts/backfill_tradeweb.py --discover gilts --since 2017-07-24 --every 3
    $PY scripts/backfill_tradeweb.py --history --countries GB --types Conventional Index-linked
    $PY scripts/backfill_tradeweb.py --history --countries FR IT --since 2024-01-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infra.pipeline import tradeweb_prices as tw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", choices=["gilts", "iosco"], help="search-grid discovery (gilts / EuroGov)")
    ap.add_argument("--since", default=str(tw.HISTORY_START.date()))
    ap.add_argument("--every", type=int, default=3, help="months between discovery days")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--countries", nargs="*", default=["GB"])
    ap.add_argument("--types", nargs="*", default=["Conventional", "Index-linked"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if args.discover:
        days = pd.bdate_range(args.since, pd.Timestamp.now().normalize(), freq=f"{args.every}BMS")
        types = args.types if args.discover == "gilts" else ["all"]
        r = tw.discover(days, cp_type=args.discover, security_types=types)
        print("universe size:", r["universe"], "| failed days:", [d.date() for d in r["failed_days"]])
    if args.history:
        u = tw.read_universe()
        u = u[u["country"].isin(args.countries) & u["security_type"].isin(args.types)].sort_values("maturity_date")
        since = pd.Timestamp(args.since)
        for r in u.itertuples():
            lo = max(since, pd.Timestamp(r.first_seen) - pd.Timedelta(days=400))   # a little before first seen
            hi = min(pd.Timestamp(r.maturity_date) if pd.notna(r.maturity_date) else pd.Timestamp.now(), pd.Timestamp.now())
            res = tw.backfill_isin(r.isin, lo, hi)
            print(f"{r.isin} {r.name}: {res['rows']} rows, failed {len(res['failed_days'])}", flush=True)


if __name__ == "__main__":
    main()
