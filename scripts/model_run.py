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
    meta = store.read_meta(name, root=root)
    return RunConfig.from_json(meta), meta


def cmd_create(args, root: Path) -> int:
    if (root / args.name / "meta.json").exists() and not args.force:
        print(f"run {args.name!r} exists (use --force to redefine; its fits are NOT cleared)")
        return 1
    history = args.history or str((pd.Timestamp(args.start) - pd.DateOffset(years=3)).date())
    config = RunConfig(name=args.name, kind=args.kind, spec=args.spec, series=tuple(args.series),
                       regime_series=tuple(args.regime_series), start=args.start, history_start=history,
                       refit=int(args.refit) if args.refit.isdigit() else args.refit,
                       overrides=_kv(args.set), regime_overrides=_kv(args.regime_set))
    config.make_model()  # validates the spec now, not at the first scheduled run
    store.write_meta(args.name, {**config.to_json(), "failures": {}}, root=root)
    print(f"created {root / args.name}")
    return 0


def _prepared(config: RunConfig, through: pd.Timestamp):
    panel = runs.read_inputs(config, through)
    return config.make_model().prepare(panel)


def do_predict(config: RunConfig, prepared, through, root: Path) -> dict:
    params = store.read_params(config.name, root=root)
    after = runs.predict_window_after(params, store.read_predictions(config.name, root=root))
    rows = runs.predict_rows(config, prepared, params, after, through)
    res = store.upsert_predictions(config.name, rows, root=root)
    if res["changed"]:
        log.warning("%s: %d stored prediction row(s) changed on recompute (data revision, or a fit appended "
                    "after they were first predicted)", config.name, res["changed"])
    return res


def do_fit(config: RunConfig, meta: dict, prepared, through, root: Path) -> dict:
    params = store.read_params(config.name, root=root)
    fr = runs.fit_due(config, prepared, params, through, skip=set(pd.to_datetime(list(meta.get("failures", {})))))
    store.append_params(config.name, fr.params, root=root)
    if fr.failures:
        meta = {**meta, "failures": {**meta.get("failures", {}), **fr.failures}}
        store.write_meta(config.name, meta, root=root)
    return {"fitted": [d.date().isoformat() for d in fr.fitted], "failures": fr.failures}


def cmd_rebuild(args, root: Path, config: RunConfig, meta: dict, through) -> int:
    panel = runs.read_inputs(config, through)
    res = runs.rebuild(config, panel, through)
    stored = store.read_params(config.name, root=root)
    rep = runs.reconcile(stored, store.read_predictions(config.name, root=root), res.params, res.predictions)
    if stored.empty:
        rep = {"note": "nothing stored yet - this rebuild is the run's first build", "identical": None}
    rb = f"{config.name}/_rebuild"
    store.save_run(rb, res.params, res.predictions, {**config.to_json(), "failures":
                   {d.date().isoformat(): e for d, e in res.failures.items()}}, root=root)
    print(json.dumps(rep, indent=2))
    stamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H%M")
    meta = {**meta, "last_rebuild": {"at": stamp, "through": str(through.date()), "identical": rep["identical"]}}
    if args.promote:
        tag = store.archive_run(config.name, stamp, root=root)
        store.save_run(config.name, res.params, res.predictions,
                       {**meta, "failures": {d.date().isoformat(): e for d, e in res.failures.items()}}, root=root)
        print(f"promoted; previous run archived to {tag}")
    else:
        store.write_meta(config.name, meta, root=root)
        print(f"rebuild written to {root / rb} (not promoted)")
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
    if args.command in ("predict", "run"):
        print(f"predict: {do_predict(config, prepared, through, root)}")
    if args.command in ("fit", "run"):
        fitted = do_fit(config, meta, prepared, through, root)
        print(f"fit: {fitted}")
        if args.command == "run" and fitted["fitted"]:
            # rows after a fit just appended were predicted with the previous one: redo them now,
            # so the run is walk-forward-consistent when the command ends
            print(f"re-predict after new fit: {do_predict(config, prepared, through, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
