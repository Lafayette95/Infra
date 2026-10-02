"""Harvest Treasury auction TAILS (high yield - when-issued) from archived auction recaps
(ZeroHedge, ForexLive) into ~/Database/RawData/TsyAuctionTails - see
infra/pipeline/auction_tails.py. Needs the auctions first (scripts: infra.pipeline.
tsy_auctions.update_auctions). Resumable: every processed article URL is recorded and
never fetched again; one page per PAUSE_S - run it in the background.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/backfill_auction_tails.py --dry-run          # candidate articles per source
    $PY scripts/backfill_auction_tails.py                    # harvest
    $PY scripts/backfill_auction_tails.py --report           # coverage of the stored tails
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.pipeline import auction_tails as pt  # noqa: E402
from infra.pipeline.tsy_auctions import read_auctions  # noqa: E402
from infra.processing.auction_tails import SOURCES  # noqa: E402


def report() -> int:
    a = read_auctions()
    a = a[a["high_yield"].notna()]
    best = pt.best_tails()
    a["year"] = a["timestamp"].dt.year
    got = a.merge(best[["timestamp", "cusip", "source"]], on=["timestamp", "cusip"], how="left")
    out = got.groupby("year").agg(auctions=("cusip", "size"), with_tail=("source", lambda s: s.notna().sum()))
    out["share"] = (out["with_tail"] / out["auctions"]).map("{:.0%}".format)
    print(out.to_string())
    m = pt.read_manifest()
    print(m.groupby(["source", "outcome"]).size().to_string())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, action="append")
    parser.add_argument("--pause", type=float, default=pt.PAUSE_S)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.report:
        return report()
    sources = tuple(args.source or SOURCES)
    if args.dry_run:
        done = pt.read_manifest()
        for s in sources:
            c = pt.list_candidates(s)
            todo = c[~c["url_key"].isin(set(done.query("source == @s")["url_key"]))]
            print(f"{s}: {len(c)} candidate articles, {len(todo)} not yet processed "
                  f"({c['timestamp'].min():%Y-%m} .. {c['timestamp'].max():%Y-%m})")
        return 0
    stats = pt.harvest_tails(sources, pause_s=args.pause)
    for s, outcomes in stats.items():
        print(s, outcomes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
