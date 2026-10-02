"""GDP nowcast from the stored release vintages (read-only: populate with
scripts/update_releases.py or the daily cycle first).

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/run_nowcast.py --version c                              # today, current quarter
    $PY scripts/run_nowcast.py --version d --tau 0.2 --as-of 2026-09-30 --news-from 2026-09-23
    $PY scripts/run_nowcast.py --version c --estimate-as-of 2026-06-30 --as-of 2026-09-30

Estimation uses the data as published by ``--estimate-as-of`` (default ``--as-of``); the
nowcast and news hold those parameters fixed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.models.nowcast.nowcast import estimate, load_raw, news, nowcast  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="c", choices=list("abcd"))
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; default today")
    parser.add_argument("--estimate-as-of", default=None, help="data vintage for estimation; default --as-of")
    parser.add_argument("--quarter", default=None, help="e.g. 2026Q3; default --as-of's quarter")
    parser.add_argument("--news-from", default=None, help="decompose the change since this day")
    parser.add_argument("--tau", type=float, default=None, help="version d: prior sd of off-block loadings")
    parser.add_argument("--global-factor", action="store_true", help="add a factor every release loads on")
    parser.add_argument("--gaussianize", action="store_true", help="map ECDF outputs to normal scores")
    parser.add_argument("--sample-start", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().normalize()
    est_as_of = pd.Timestamp(args.estimate_as_of) if args.estimate_as_of else as_of
    overrides = {k: v for k, v in (("tau", args.tau), ("sample_start", args.sample_start)) if v is not None}
    if args.global_factor:
        overrides["global_factor"] = True
    if args.gaussianize:
        overrides["gaussianize"] = True
    raw = load_raw(as_of)
    model = estimate(args.version, est_as_of, raw=raw[raw["timestamp"] <= est_as_of], **overrides)
    print(f"version {args.version}: factors {', '.join(model.structure.factors)}; "
          f"{len(model.series)} releases; {len(model.loglik)} EM iterations")
    nc = nowcast(model, as_of, args.quarter, raw=raw)
    state = "published" if nc.published else "nowcast"
    print(f"\n{nc.quarter} GDP ({state}, as of {as_of.date()}): {nc.value:.2f}% QoQ SAAR")
    print("model's common component by factor (pp):")
    print(nc.common.to_string(float_format=lambda v: f"{v:7.2f}"))
    if args.news_from:
        out = news(model, args.news_from, as_of, nc.quarter, raw=raw)
        print(f"\nnews {args.news_from} -> {as_of.date()}: {out.old:.2f} -> {out.new:.2f} "
              f"(revisions {out.revision_impact:+.2f}, news {out.news_impact:+.2f})")
        if not out.impacts.empty:
            cols = ["published", "ticker", "period", "actual_native", "actual", "forecast", "news", "weight",
                    "impact", "signal_share"]
            print(out.impacts[cols].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
