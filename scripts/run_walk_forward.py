"""Walk-forward (point-in-time refit) of a regression or PCA on any stored series.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    # 10y vs 2y & 30y hedge ratio on changes, refit every Friday on a 2y window:
    $PY scripts/run_walk_forward.py regression hedge_ratio \
        --series bond:US_BOND_10y bond:US_BOND_2y bond:US_BOND_30y \
        --start 2023-01-01 --end 2026-09-30 --refit W-FRI --save us10y_hedge
    # US curve PCA on bp changes, monthly refit, 2y window:
    $PY scripts/run_walk_forward.py pca curve_changes --series bond:US_BOND_2y bond:US_BOND_5y \
        bond:US_BOND_10y bond:US_BOND_30y --window 504 --refit ME --save us_curve_pca
    # US curve residuals, PCA weighted by 2 HMM regimes inferred from US/DE/UK curves:
    $PY scripts/run_walk_forward.py regime_pca regime_pca --series bond:US_BOND_2y bond:US_BOND_5y \
        bond:US_BOND_10y bond:US_BOND_30y --regime-series bond:US_BOND_2y bond:US_BOND_10y \
        bond:DE_BOND_2y bond:DE_BOND_10y bond:UK_BOND_2y bond:UK_BOND_10y --set n_components=2 \
        --regime-set sticky=200 --refit ME --save us_curve_regime_pca
    $PY scripts/run_walk_forward.py --list

For a regression the first series is y and the rest the regressors; for regime_pca
``--series`` are the K series and ``--regime-series`` the N the regimes come from
(``--regime-set`` overrides the regime spec). ``--set key=value``
overrides any spec field (``--set alpha=0.1 --set cov=HAC``). Reads disk only
(infra.pipeline.series_panel); ``--save NAME`` writes params + predictions to
``Database/Derived/ModelRuns/NAME`` (infra.storage.model_runs).
"""
from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.models.stats.common import pick  # noqa: E402
from infra.models.stats.config import PCA_MODELS, REGIME_MODELS, REGIME_PCA_MODELS, REGRESSION_MODELS  # noqa: E402
from infra.models.stats.pca import make_pca  # noqa: E402
from infra.models.stats.regime_pca import make_regime_pca  # noqa: E402
from infra.models.stats.regression import make_regression  # noqa: E402
from infra.models.walk_forward import walk_forward  # noqa: E402
from infra.pipeline.series_panel import read_panel  # noqa: E402
from infra.storage.model_runs import save_run  # noqa: E402


def _value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", nargs="?", choices=["regression", "pca", "regime_pca"])
    parser.add_argument("spec", nargs="?", help="a REGRESSION_MODELS / PCA_MODELS name, or a method")
    parser.add_argument("--series", nargs="+", default=[])
    parser.add_argument("--start", default="2020-01-01", help="first refit date")
    parser.add_argument("--end", default=None, help="last day predicted (default today)")
    parser.add_argument("--history", default=None, help="first day read (default: 3 years before --start)")
    parser.add_argument("--refit", default="W-FRI", help="pandas frequency or an int (every N rows)")
    parser.add_argument("--window", default=None, help="fit window: rows (int) or a Timedelta (730D)")
    parser.add_argument("--set", action="append", default=[], help="spec field override, key=value")
    parser.add_argument("--regime-series", nargs="+", default=[], help="regime_pca: the N series")
    parser.add_argument("--regime-set", action="append", default=[], help="regime_pca: regime spec override")
    parser.add_argument("--save", default=None, help="run name to save under Derived/ModelRuns")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pd.set_option("display.width", 220)
    if args.list or not args.kind:
        for name, s in REGRESSION_MODELS.items():
            print(f"regression {name:14s} {s.method:10s} {s.description}")
        for name, s in PCA_MODELS.items():
            print(f"pca        {name:14s} {s.method:10s} {s.description}")
        for name, s in REGIME_PCA_MODELS.items():
            print(f"regime_pca {name:14s} {'':10s} {s.description}")
        for name, s in REGIME_MODELS.items():
            print(f"  regimes  {name:14s} {s.method:10s} {s.description}")
        return 0

    overrides = dict(kv.split("=", 1) for kv in args.set)
    overrides = {k: _value(v) for k, v in overrides.items()}
    if args.window is not None:
        overrides["window"] = _value(args.window)
    if args.kind == "regression":
        overrides.update(y=args.series[0], x=tuple(args.series[1:]))
        factory = lambda: make_regression(args.spec, **overrides)  # noqa: E731
    elif args.kind == "regime_pca":
        regime_over = {k: _value(v) for k, v in (kv.split("=", 1) for kv in args.regime_set)}
        regime_over["columns"] = tuple(args.regime_series or args.series)
        overrides.update(columns=tuple(args.series), regime_overrides=tuple(regime_over.items()))
        factory = lambda: make_regime_pca(args.spec, **overrides)  # noqa: E731
    else:
        overrides.update(columns=tuple(args.series))
        factory = lambda: make_pca(args.spec, **overrides)  # noqa: E731
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    history = pd.Timestamp(args.history) if args.history else pd.Timestamp(args.start) - pd.DateOffset(years=3)
    panel = read_panel(list(dict.fromkeys(args.series + args.regime_series)), history, end)
    refit = int(args.refit) if args.refit.isdigit() else args.refit
    res = walk_forward(factory, panel, args.start, end, refit=refit)
    print(f"{len(res.fit_dates)} refits, {len(res.failures)} failed, {len(res.predictions)} predicted rows")
    for d, err in list(res.failures.items())[:5]:
        print(f"  failed {d.date()}: {err}")
    if args.kind == "regression":
        print(res.param_path("coef", "coef").tail(5).to_string())
        print(res.predictions[["y", "fitted", "residual", "resid_z"]].tail(5).to_string())
    elif args.kind == "regime_pca":
        print(res.param_path("stat", "value").filter(like="occupancy_R").tail(5).to_string())
        print(pick(res.predictions, "p").join(pick(res.predictions, "resid_z")).tail(5).round(3).to_string())
    else:
        print(res.param_path("stat", "value", "explained_k").tail(5).to_string())
        print(pick(res.predictions, "score").tail(5).to_string())
    if args.save:
        folder = save_run(args.save, res.params, res.predictions,
                          {"kind": args.kind, "spec": args.spec, "overrides": overrides, "series": args.series,
                           "start": args.start, "end": str(end.date()), "refit": args.refit})
        print(f"saved to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
