"""Scheduled operation of a model run (infra/models/runs.py, infra/storage/model_runs.py):
a run built INCREMENTALLY - fits appended weekly, predictions appended daily, each day
seeing only the data known that day - must equal a full walk-forward exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from infra.models import runs
from infra.models.runs import RunConfig
from infra.models.walk_forward import refit_dates
from infra.storage import model_runs as store

sys.path.insert(0, str(Path(__file__).parent))
from test_regimes import regime_panel  # noqa: E402

CONFIGS = {
    "ols": dict(kind="regression", spec="ols", series=("a0", "a1", "a2"), overrides={"prep": ("diff",)}),
    "kalman": dict(kind="regression", spec="kalman", series=("a0", "a1"), overrides={"prep": ("diff",), "window": 300}),
    "pca": dict(kind="pca", spec="pca", series=("a0", "a1", "a2", "a3"), overrides={"prep": ("diff",), "n_components": 2}),
    "regime_pca": dict(kind="regime_pca", spec="regime_pca", series=("a0", "a1", "a2", "a3"),
                       regime_series=tuple(f"a{i}" for i in range(8)),
                       overrides={"prep": ("diff",), "n_components": 2, "min_obs": 200},
                       regime_overrides={"prep": ("diff",), "min_obs": 200}),
}


def _config(name, **kw):
    lvl, *_ = regime_panel(n=560)
    c = RunConfig(name=name, start=str(lvl.index[400].date()), history_start=str(lvl.index[0].date()),
                  refit="W-FRI", **{**CONFIGS[name], **kw})
    return c, lvl


@pytest.mark.parametrize("name", list(CONFIGS))
def test_incremental_run_equals_the_full_walk_forward(name, tmp_path):
    config, lvl = _config(name)
    days = lvl.index[400:]
    for day in days:  # the live schedule: every day sees only the data known that day
        live = lvl[lvl.index <= day]
        prepared = config.make_model().prepare(live)
        # the daily predict runs first (a Friday row uses last week's fit), then the weekly fit
        params = store.read_params(name, root=tmp_path)
        after = runs.predict_window_after(params, store.read_predictions(name, root=tmp_path))
        store.upsert_predictions(name, runs.predict_rows(config, prepared, params, after, day), root=tmp_path)
        fr = runs.fit_due(config, prepared, params, day)
        store.append_params(name, fr.params, root=tmp_path)
        assert not fr.failures
    # predictions for rows after the last fit, made before it existed, are redone by the next predict
    prepared = config.make_model().prepare(lvl)
    params = store.read_params(name, root=tmp_path)
    after = runs.predict_window_after(params, store.read_predictions(name, root=tmp_path))
    store.upsert_predictions(name, runs.predict_rows(config, prepared, params, after, days[-1]), root=tmp_path)

    full = runs.rebuild(config, lvl, days[-1])
    rep = runs.reconcile(store.read_params(name, root=tmp_path), store.read_predictions(name, root=tmp_path),
                         full.params, full.predictions)
    assert rep["identical"], rep


def test_catching_up_several_days_equals_the_walk_forward(tmp_path):
    """Runs skip days (a Mac asleep, a holiday week): each run predicts, appends every due
    fit, then re-predicts - the result must still be the walk-forward's."""
    config, lvl = _config("ols")
    days = lvl.index[400:]
    for day in list(days[::7]) + [days[-1]]:
        live = lvl[lvl.index <= day]
        prepared = config.make_model().prepare(live)
        for step in ("predict", "fit", "predict"):
            params = store.read_params("ols", root=tmp_path)
            if step == "fit":
                store.append_params("ols", runs.fit_due(config, prepared, params, day).params, root=tmp_path)
                continue
            after = runs.predict_window_after(params, store.read_predictions("ols", root=tmp_path))
            store.upsert_predictions("ols", runs.predict_rows(config, prepared, params, after, day), root=tmp_path)
    full = runs.rebuild(config, lvl, days[-1])
    rep = runs.reconcile(store.read_params("ols", root=tmp_path), store.read_predictions("ols", root=tmp_path),
                         full.params, full.predictions)
    assert rep["identical"], rep


def test_a_refit_waits_for_its_data():
    idx = pd.bdate_range("2024-01-01", "2024-01-25")  # Thursday 25th: Friday 26th not loaded yet
    d = refit_dates(idx, "2024-01-01", "2024-01-26", "W-FRI")
    assert pd.Timestamp("2024-01-25") not in d and d[-1] == pd.Timestamp("2024-01-19")
    holiday = idx.append(pd.DatetimeIndex(["2024-01-29"]))  # Friday missing, Monday loaded: snap to Thursday
    assert refit_dates(holiday, "2024-01-01", "2024-01-29", "W-FRI")[-1] == pd.Timestamp("2024-01-25")


def test_reconcile_finds_a_data_revision_and_its_first_date():
    config, lvl = _config("pca")
    end = lvl.index[-1]
    base = runs.rebuild(config, lvl, end)
    revised = lvl.copy()
    revised.loc[lvl.index[480], "a0"] += 5.0
    alt = runs.rebuild(config, revised, end)
    rep = runs.reconcile(base.params, base.predictions, alt.params, alt.predictions)
    assert not rep["identical"]
    assert pd.Timestamp(rep["params_first_differing_fit"]) >= lvl.index[480]
    assert pd.Timestamp(rep["prediction_first_differing_row"]) == lvl.index[480]


def test_failed_fits_are_recorded_and_not_retried():
    config, lvl = _config("ols", overrides={"prep": ("diff",), "min_obs": 10_000})
    prepared = config.make_model().prepare(lvl)
    fr = runs.fit_due(config, prepared, store.read_params("x", root=Path("/nonexistent")), lvl.index[-1])
    assert fr.fitted == [] and fr.failures
    again = runs.fit_due(config, prepared, fr.params, lvl.index[-1], skip=set(fr.failures))
    assert again.fitted == [] and again.failures == {}


def test_run_config_round_trips_through_json():
    config, _ = _config("regime_pca")
    back = RunConfig.from_json(config.to_json())
    assert back == config
    assert back.make_model().spec == config.make_model().spec


@pytest.mark.parametrize("combine", ["covariance", "robust"])
def test_regime_pca_rows_with_gaps_do_not_depend_on_the_batch(combine):
    """A row's output must not depend on which other rows are predicted with it."""
    config, lvl = _config("regime_pca", overrides={"prep": ("diff",), "n_components": 1 if combine == "robust" else 2,
                                                   "min_obs": 200, "combine": combine})
    gappy = lvl.copy()
    gappy.iloc[500:520:3, 0] = np.nan
    model = config.make_model()
    data = model.prepare(gappy)
    model.fit(data, as_of=lvl.index[450])
    whole = model.predict(data, start=lvl.index[450])
    one = model.predict(data, start=lvl.index[502], end=lvl.index[503])
    pd.testing.assert_frame_equal(one, whole.loc[one.index])
