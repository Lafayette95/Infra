"""Harvest economic-calendar history from archived MarketWatch calendar pages (Wayback
Machine) into ~/Database/RawData/EconCalendar - the free source for ISM, S&P Global
PMIs, Chicago PMI, Conference Board, NFIB and existing home sales (CALENDAR_PAGES,
infra/pipeline/econ_calendar.py). Resumable: covered capture days are never re-fetched,
coverage is recorded as it goes. One page per capture day, PAUSE_S apart (~3,000 pages
for 2009-2026, a few hours) - run it in the background.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/backfill_econ_calendar.py --dry-run                 # days still to process
    $PY scripts/backfill_econ_calendar.py --start 2024-09-01 --end 2024-10-01
    $PY scripts/backfill_econ_calendar.py                           # everything, 2009 -> yesterday
    $PY scripts/backfill_econ_calendar.py --names                   # report names on disk (mapping audit)
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import CALENDAR_PAGES, MACRO_RELEASES  # noqa: E402
from infra.pipeline import econ_calendar as pc  # noqa: E402
from infra.processing.econ_calendar import matches, normalize_report  # noqa: E402

# report names worth auditing: anything that looks like one of the calendar-sourced releases
_AUDIT = re.compile(r"ism|pmi|purchasing|confidence|nfib|small.business|existing|chicago|richmond|barometer", re.I)


def names() -> int:
    rows = pc.read_calendar_from_disk()
    cal = {t: r.calendar_pattern for t, r in MACRO_RELEASES.items() if r.calendar_pattern}
    seen = rows.groupby("report").agg(n=("timestamp", "size"), first=("timestamp", "min"), last=("timestamp", "max"))
    for name, r in seen.iterrows():
        if not _AUDIT.search(name):
            continue
        hit = [t for t, pat in cal.items() if matches(name, pat)]
        print(f"{'->' + ','.join(hit) if hit else '  UNMAPPED':28s} {r.n:5d}  {r['first'].date()}..{r['last'].date()}  "
              f"{name!r}  [{normalize_report(name)}]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--page", default="marketwatch", choices=list(CALENDAR_PAGES))
    parser.add_argument("--start", default=None, help="first capture day (default: the page's history start)")
    parser.add_argument("--end", default=None, help="capture day, exclusive (default: today)")
    parser.add_argument("--pause", type=float, default=pc.PAUSE_S, help="seconds between page fetches")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--names", action="store_true", help="audit stored report names against the patterns")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.names:
        return names()
    page = CALENDAR_PAGES[args.page]
    start = pd.Timestamp(args.start or page.history_start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    plan = pc.plan_harvest(page, start, end)
    print(f"{sum((b - a).days for a, b in plan)} capture day(s) to process in {len(plan)} range(s)")
    if args.dry_run:
        for a, b in plan:
            print(f"  {a.date()} -> {b.date()}")
        return 0
    stats = pc.harvest(args.page, start, end, pause_s=args.pause)
    print(f"done: {stats['pages']} pages, {stats['rows']} rows merged, {len(stats['failed'])} failed")
    for ts, err in stats["failed"][:20]:
        print(f"  FAILED {ts}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
