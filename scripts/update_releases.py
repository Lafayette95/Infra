"""Fetch every published vintage of the nowcast's macro releases (MACRO_RELEASES) - the
same code the daily cycle's ``raw`` step runs, usable standalone. Disk first: a series
never fetched gets its whole vintage history, afterwards only what was published since.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_releases.py --dry-run          # what would be fetched
    $PY scripts/update_releases.py                    # fetch
    $PY scripts/update_releases.py --verify           # each FRED id's own metadata vs the table
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.api import fred_client  # noqa: E402
from infra.config import MACRO_RELEASES  # noqa: E402
from infra.cycle.raw_releases import backfill_daily_releases, plan_daily_releases  # noqa: E402


def verify() -> int:
    """Print FRED's own title/units/frequency/seasonal adjustment for every configured id,
    next to the table's name - the check that each id is the series the table means."""
    bad = 0
    for r in MACRO_RELEASES.values():
        if not (r.source or "").startswith("fred"):
            print(f"{r.ticker:16s} -> (no free source) {r.note}")
            continue
        try:
            info = fred_client.fetch_series_info(r.series_id)
            print(f"{r.ticker:16s} -> {r.series_id:20s} {info['frequency_short']:2s} {info['seasonal_adjustment_short']:5s}"
                  f" {info['units_short'][:28]:28s} {info['observation_start']}..{info['observation_end']}  "
                  f"{info['title'][:70]}")
        except fred_client.FredError as exc:
            bad += 1
            print(f"{r.ticker:16s} -> {r.series_id:20s} ERROR {exc}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="show the plan; download nothing")
    parser.add_argument("--verify", action="store_true", help="print FRED metadata for every configured id")
    parser.add_argument("--force-refetch", action="store_true", help="re-ask the last --days publication days")
    parser.add_argument("--days", type=int, default=3, help="window for --force-refetch (calendar days)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.verify:
        return verify()
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    start = end - pd.Timedelta(days=args.days)
    if args.dry_run:
        for sid, (source, ranges) in plan_daily_releases(start, end, force_refetch=args.force_refetch).items():
            for g0, g1 in ranges:
                print(f"{sid:22s} {g0.date()} -> {g1.date()}  ({source})")
        return 0
    out = backfill_daily_releases(start, end, force_refetch=args.force_refetch)
    for sid, err in out["fetch_errors"].items():
        print(f"FAILED {sid}: {err}")
    print(f"{len(out['planned'])} series fetched, {out['rows']} new values")
    return 1 if out["fetch_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
