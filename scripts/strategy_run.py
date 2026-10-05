"""Operate event-family runs and strategies (root CLAUDE.md 27): thin CLI over
``infra.jobs.family_runs`` and ``infra.jobs.strategy_runs``. Reads stored data only.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    # an event family as runs (one run per code + the point-in-time FDR table):
    $PY scripts/strategy_run.py family-create nfp --family nfp_intraday --start 2020-01-03 --history 2016-01-01
    $PY scripts/strategy_run.py family-rebuild nfp --promote        # the history, one pass
    $PY scripts/strategy_run.py family-run nfp                      # daily
    # a strategy over family runs (its name in infra/strategies/config/<class>.py):
    $PY scripts/strategy_run.py run cevt_nfp [--update-families]    # firm series + plan vintage
    $PY scripts/strategy_run.py rebuild cevt_nfp [--promote]        # recompute + reconcile
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import MODEL_RUNS_DIR, STRATEGIES_DIR  # noqa: E402
from infra.jobs import family_runs, strategy_runs  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["family-create", "family-run", "family-rebuild", "run", "rebuild"])
    ap.add_argument("name")
    ap.add_argument("--family")
    ap.add_argument("--start")
    ap.add_argument("--history")
    ap.add_argument("--refit", default="W-FRI")
    ap.add_argument("--through", default=None, help="last day (default: yesterday)")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--update-families", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models-root", type=Path, default=MODEL_RUNS_DIR)
    ap.add_argument("--root", type=Path, default=STRATEGIES_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    through = pd.Timestamp(args.through) if args.through else pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    mr = args.models_root
    if args.command == "family-create":
        cs = family_runs.create(args.name, args.family, start=args.start, history_start=args.history,
                                refit=args.refit, root=mr, force=args.force)
        print(f"created {len(cs)} code runs under {mr / args.name}")
    elif args.command == "family-run":
        print(json.dumps(family_runs.run_daily(args.name, through, root=mr), indent=1, default=str))
    elif args.command == "family-rebuild":
        rep = family_runs.rebuild(args.name, through, root=mr, promote=args.promote)
        print(json.dumps({"codes": rep["codes"], "identical": rep["identical"]}, indent=1, default=str))
    elif args.command == "run":
        print(json.dumps(strategy_runs.run_daily(args.name, through, root=args.root, models_root=mr,
                                                 update_families=args.update_families), indent=1, default=str))
    else:
        print(json.dumps(strategy_runs.rebuild(args.name, through, root=args.root, models_root=mr,
                                               promote=args.promote), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
