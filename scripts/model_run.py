"""Operate a model run on a schedule: weekly fit-append, daily predict-append, periodic
rebuild + reconciliation (infra/models/CLAUDE.md 0b). Reads stored data only.

    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    # define a run (nothing is fitted yet):
    $PY scripts/model_run.py create us_curve_pca --kind pca --spec curve_changes \\
        --series bond:US_BOND_2y bond:US_BOND_5y bond:US_BOND_10y bond:US_BOND_30y \\
        --start 2020-01-03 --history 2016-01-01 --refit W-FRI --set window=504 --set n_components=2
    $PY scripts/model_run.py run us_curve_pca          # daily: predict new rows, then any due fit
    $PY scripts/model_run.py fit us_curve_pca          # only the due fit(s) (weekly)
    $PY scripts/model_run.py predict us_curve_pca      # only the predictions (daily)
    $PY scripts/model_run.py rebuild us_curve_pca      # full walk-forward into _rebuild/ + reconcile report
    $PY scripts/model_run.py rebuild us_curve_pca --promote   # ... and replace (old run -> _archive/<today>)
    $PY scripts/model_run.py status us_curve_pca

``run`` predicts BEFORE fitting, so a Friday row uses last week's fit (the walk-forward's
convention), then re-predicts the rows after any fit it appended (a run that catches up
several days): the run is walk-forward-consistent whenever ``run`` returns. The first ``run``
of a new run fits every due date from ``--start`` (a long backfill: use ``rebuild --promote``
for that instead - same result, one pass).
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.config import MODEL_RUNS_DIR  # noqa: E402
from infra.models import runs  # noqa: E402
from infra.models.runs import RunConfig  # noqa: E402
from infra.jobs import model_runs as jobs  # noqa: E402
from infra.storage import model_runs as store  # noqa: E402

log = logging.getLogger("model_run")


def _value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _kv(items) -> dict:
    return {k: _value(v) for k, v in (kv.split("=", 1) for kv in items)}


def _load(name: str, root: Path) -> tuple[RunConfig, dict]:
    return jobs.load(name, root)


def cmd_create(args, root: Path) -> int:
    history = args.history or str((pd.Timestamp(args.start) - pd.DateOffset(years=3)).date())
    config = RunConfig(name=args.name, kind=args.kind, spec=args.spec, series=tuple(args.series),
                       regime_series=tuple(args.regime_series), start=args.start, history_start=history,
                       refit=int(args.refit) if args.refit.isdigit() else args.refit,
                       overrides=_kv(args.set), regime_overrides=_kv(args.regime_set))
    try:
        print(f"created {jobs.create(config, root=root, force=args.force)}")
    except FileExistsError as exc:
        print(exc)
        return 1
    return 0


def _prepared(config: RunConfig, through: pd.Timestamp):
    return config.make_model().prepare(runs.read_inputs(config, through))


def cmd_rebuild(args, root: Path, config: RunConfig, meta: dict, through) -> int:
    panel = runs.read_inputs(config, through)
    rep = jobs.rebuild(config, panel, through, root=root, promote=args.promote)
    print(json.dumps(rep, indent=2, default=str))
    return 0


def cmd_status(root: Path, config: RunConfig, meta: dict) -> int:
    params = store.read_params(config.name, root=root)
    pred = store.read_predictions(config.name, root=root)
    fits = runs.fit_dates(params)
    print(f"run {config.name}: {config.kind} {config.spec}  refit {config.refit}  from {config.start}")
    print(f"  fits: {len(fits)}  last {fits[-1].date() if fits else '-'}  failures {len(meta.get('failures', {}))}")
    print(f"  predictions: {len(pred)} rows  last {pred.index.max() if len(pred) else '-'}")
    print(f"  last rebuild: {meta.get('last_rebuild', '-')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["create", "fit", "predict", "run", "rebuild", "status"])
    parser.add_argument("name")
    parser.add_argument("--through", default=None, help="last day processed (default: today)")
    parser.add_argument("--root", default=None, help="runs folder (default Database/Derived/ModelRuns)")
    parser.add_argument("--promote", action="store_true", help="rebuild: replace the run, archiving the old one")
    # create
    parser.add_argument("--kind", choices=runs.KINDS)
    parser.add_argument("--spec", default=None)
    parser.add_argument("--series", nargs="+", default=[])
    parser.add_argument("--regime-series", nargs="+", default=[])
    parser.add_argument("--start")
    parser.add_argument("--history", default=None)
    parser.add_argument("--refit", default="W-FRI")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--regime-set", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(args.root).expanduser() if args.root else MODEL_RUNS_DIR
    if args.command == "create":
        return cmd_create(args, root)
    config, meta = _load(args.name, root)
    through = pd.Timestamp(args.through) if args.through else pd.Timestamp.now().normalize()
    if args.command == "status":
        return cmd_status(root, config, meta)
    if args.command == "rebuild":
        return cmd_rebuild(args, root, config, meta, through)
    prepared = _prepared(config, through)
    if args.command == "run":
        print(json.dumps(jobs.run_daily(config, prepared, through, root=root), indent=1, default=str))
    elif args.command == "predict":
        print(f"predict: {jobs.predict_append(config, prepared, through, root=root)}")
    else:
        print(f"fit: {jobs.fit_append(config, prepared, through, root=root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
