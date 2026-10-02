"""CTA positioning model on stored futures (read-only: the history must be on disk).

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/run_cta.py                                   # ubs2022 on US bonds, latest day
    $PY scripts/run_cta.py --model ubs2022_cal --as-of 2026-09-30 --reaction price
    $PY scripts/run_cta.py --compare-paper --model ubs2022_cal  # vs UBS's 2022-09-02 numbers
    $PY scripts/run_cta.py --list                            # models and universes

Fits on everything up to ``--as-of`` (point in time) and prints, per asset: signal and
position in [-1, 1], past changes and expected flows, and optionally the spot reaction.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.models.cta.config import CTA_MODELS, CTA_UNIVERSES, get_universe  # noqa: E402
from infra.models.cta.inputs import universe_prices  # noqa: E402
from infra.models.cta.model import CTAModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="ubs2022", choices=sorted(CTA_MODELS))
    parser.add_argument("--universe", default="ubs_us_bonds", choices=sorted(CTA_UNIVERSES))
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; default the latest stored day")
    parser.add_argument("--start", default="2014-12-01", help="first day of history read")
    parser.add_argument("--reaction", choices=["price", "path", "vol", "price_vol"], default=None)
    parser.add_argument("--compare-paper", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    if args.list:
        for name, spec in CTA_MODELS.items():
            print(f"model    {name:16s} {spec.description}")
        for name, uni in CTA_UNIVERSES.items():
            print(f"universe {name:16s} {uni.description}")
        return 0
    if args.compare_paper:
        from infra.models.cta.compare import compare_with_paper
        print(compare_with_paper(args.model, args.universe).round(3).T.to_string())
        return 0

    uni = get_universe(args.universe)
    end = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().normalize()
    prices = universe_prices(uni, args.start, end)
    model = CTAModel(args.model)
    model.fit(model.prepare(prices, universe=uni), as_of=end)
    res = model.predict()
    cols = ["timestamp", "signal", "position"] + [c for c in res.frame.columns if c.startswith("chg_")] \
        + [c for c in res.latest().columns if c.startswith("flow_")]
    print(f"model {args.model}, fitted as of {res.fit_as_of.date()}")
    print(res.latest()[cols].round(3).drop(columns="timestamp").to_string())
    if args.reaction:
        print(model.reaction(res, args.reaction).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
