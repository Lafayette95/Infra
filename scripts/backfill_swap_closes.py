"""Compute and store the benchmark swap closes (CLAUDE.md 16) from the archived DTCC
reports - disk only, free. Each day replaces its stored rows.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/backfill_swap_closes.py --start 2024-09-30 --end 2026-09-30
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.pipeline.swap_closes import backfill_swap_closes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="first day, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", help="last day, inclusive (default: yesterday, UTC)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    t0 = time.time()
    out = backfill_swap_closes(pd.Timestamp(args.start), end)
    print(f"{out['days']} day(s), {out['rows']} row(s) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
