"""Snapshot the full-granularity inflation files (BULK_DATASETS: CPI, PPI by commodity and
by industry, PCE by type of product) and any due CPI weight year - the same code the daily
cycle's ``raw`` step runs, usable standalone. Downloads a file only if its current version
isn't on disk yet (the weights: only a due year not on disk).

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/update_bulk_series.py --dry-run            # latest version on disk vs published
    $PY scripts/update_bulk_series.py                      # fetch every dataset
    $PY scripts/update_bulk_series.py --datasets pce cpi
    $PY scripts/update_bulk_series.py --search cpi "shelter"   # find series ids in a catalog
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import BULK_DATASETS  # noqa: E402
from infra.cycle.raw_bulk import backfill_daily_bulk, plan_daily_bulk  # noqa: E402
from infra.cycle.raw_cpi_weights import backfill_daily_cpi_weights  # noqa: E402
from infra.pipeline import cpi_weights as pcw  # noqa: E402
from infra.pipeline import bulk_series as pbulk  # noqa: E402


def search(dataset: str, text: str) -> int:
    cat = pbulk.read_catalog(dataset)
    if cat.empty:
        print(f"no catalog for {dataset} yet - fetch it first")
        return 1
    text_cols = [c for c in cat.columns if cat[c].dtype == object]
    hit = cat[cat[text_cols].apply(lambda col: col.str.contains(text, case=False, na=False)).any(axis=1)]
    with pd.option_context("display.max_rows", 200, "display.width", 250, "display.max_colwidth", 90):
        print(hit.head(200).to_string(index=False))
    print(f"{len(hit)} match(es)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="compare versions (HEAD only); download nothing")
    parser.add_argument("--datasets", nargs="+", choices=sorted(BULK_DATASETS), help="default: all")
    parser.add_argument("--search", nargs=2, metavar=("DATASET", "TEXT"), help="search a stored catalog")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.search:
        return search(*args.search)
    datasets = {k: BULK_DATASETS[k] for k in (args.datasets or BULK_DATASETS)}
    if args.dry_run:
        for key, last in plan_daily_bulk(datasets=datasets).items():
            d = datasets[key]
            try:
                current = pbulk.LAST_MODIFIED[d.source](d) if pbulk.CONFIGURED[d.source]() else "(not configured)"
            except Exception as exc:  # noqa: BLE001 - a dry run reports, never raises
                current = f"ERROR {exc}"
            print(f"{key:15s} on disk: {last}  published: {current}")
        print(f"cpi_weights     weight years due, not on disk: {pcw.plan_weights_update() or 'none'}")
        return 0
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    out = backfill_daily_bulk(today, today, datasets=datasets)
    for key in out["unconfigured"]:
        print(f"NOT CONFIGURED {key} (bls: set BLS_CONTACT_EMAIL in .env)")
    for key, err in out["fetch_errors"].items():
        print(f"FAILED {key}: {err}")
    print(f"ingested: {', '.join(out['ingested']) or 'nothing new'}; {out['rows']} changed values")
    weights = backfill_daily_cpi_weights(today, today)
    print(f"cpi weights: fetched {weights['fetched'] or 'nothing'}"
          + (f" - FAILED {weights['error']}" if weights["error"] else ""))
    return 1 if out["fetch_errors"] or weights["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
