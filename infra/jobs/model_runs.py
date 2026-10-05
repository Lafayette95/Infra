"""Operating one model run: the daily job (predict-append, due fit-append, re-predict), the
rebuild + reconciliation, status - over ``infra.models.runs`` (computation) and
``infra.storage.model_runs`` (the run's folder). Moved here from ``scripts/model_run.py`` so
family runs and strategies use the same code (root CLAUDE.md 3b)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.config import MODEL_RUNS_DIR
from infra.models import runs
from infra.models.runs import RunConfig
from infra.storage import model_runs as store

log = logging.getLogger(__name__)


def load(name: str, root: Path = MODEL_RUNS_DIR) -> tuple[RunConfig, dict]:
    meta = store.read_meta(name, root=root)
    return RunConfig.from_json(meta), meta


def create(config: RunConfig, *, root: Path = MODEL_RUNS_DIR, force: bool = False) -> Path:
    if (root / config.name / "meta.json").exists() and not force:
        raise FileExistsError(f"run {config.name!r} exists (force=True to redefine; its fits are NOT cleared)")
    config.make_model()  # validates the spec now, not at the first scheduled run
    store.write_meta(config.name, {**config.to_json(), "failures": {}}, root=root)
    return root / config.name


def predict_append(config: RunConfig, prepared, through, *, root: Path = MODEL_RUNS_DIR) -> dict:
    params = store.read_params(config.name, root=root)
    meta = store.read_meta(config.name, root=root)
    after = runs.predict_window_after(params, store.read_predictions(config.name, root=root),
                                      last_through=meta.get("last_predict_through"))
    rows = runs.predict_rows(config, prepared, params, after, through)
    res = store.upsert_predictions(config.name, rows, root=root)
    store.write_meta(config.name, {**store.read_meta(config.name, root=root),
                                   "last_predict_through": str(pd.Timestamp(through).date())}, root=root)
    if res["changed"]:
        log.warning("%s: %d stored prediction row(s) changed on recompute (data revision, or a fit appended "
                    "after they were first predicted)", config.name, res["changed"])
    return res


def fit_append(config: RunConfig, prepared, through, *, root: Path = MODEL_RUNS_DIR) -> dict:
    meta = store.read_meta(config.name, root=root)
    params = store.read_params(config.name, root=root)
    fr = runs.fit_due(config, prepared, params, through, skip=set(pd.to_datetime(list(meta.get("failures", {})))))
    store.append_params(config.name, fr.params, root=root)
    if fr.failures:
        store.write_meta(config.name, {**meta, "failures": {**meta.get("failures", {}), **fr.failures}}, root=root)
    return {"fitted": [d.date().isoformat() for d in fr.fitted], "failures": fr.failures}


def run_daily(config: RunConfig, prepared, through, *, root: Path = MODEL_RUNS_DIR) -> dict:
    """Predict (a Friday row uses last week's fit), append due fits, re-predict the rows after
    any fit appended: walk-forward-consistent when it returns."""
    out = {"predict": predict_append(config, prepared, through, root=root)}
    out["fit"] = fit_append(config, prepared, through, root=root)
    if out["fit"]["fitted"]:
        out["repredict"] = predict_append(config, prepared, through, root=root)
    return out


def rebuild(config: RunConfig, panel, through, *, root: Path = MODEL_RUNS_DIR, promote: bool = False,
            prepare_kwargs: dict | None = None) -> dict:
    """The full walk-forward into ``<run>/_rebuild``, reconciled against the stored run;
    ``promote`` archives the stored run and replaces it."""
    through = pd.Timestamp(through)
    meta = store.read_meta(config.name, root=root)
    res = runs.rebuild(config, panel, through, prepare_kwargs=prepare_kwargs)
    stored = store.read_params(config.name, root=root)
    rep = runs.reconcile(stored, store.read_predictions(config.name, root=root), res.params, res.predictions)
    if stored.empty:
        rep = {"note": "nothing stored yet - this rebuild is the run's first build", "identical": None}
    failures = {d.date().isoformat(): e for d, e in res.failures.items()}
    store.save_run(f"{config.name}/_rebuild", res.params, res.predictions, {**config.to_json(), "failures": failures},
                   root=root)
    stamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H%M")
    meta = {**meta, "last_rebuild": {"at": stamp, "through": str(through.date()), "identical": rep["identical"]}}
    if promote:
        rep["archived_to"] = str(store.archive_run(config.name, stamp, root=root))
        store.save_run(config.name, res.params, res.predictions,
                       {**meta, "failures": failures, "last_predict_through": str(through.date())}, root=root)
    else:
        store.write_meta(config.name, meta, root=root)
    return rep
