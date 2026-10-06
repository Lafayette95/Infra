"""Operating a model run on a schedule: weekly fit-append, daily predict-append, periodic
full rebuild with reconciliation. The computation only; storage IO is
``infra.storage.model_runs`` and the wiring is ``scripts/model_run.py`` (models never write).

The invariant everything here keeps (infra/models/CLAUDE.md 0b): **a run built
incrementally - one refit appended per week, predictions appended daily - is IDENTICAL to a
full ``walk_forward`` over the same dates**, so a rebuild with unchanged data and code
reconciles to exactly zero, and any difference is a data revision or a bug.

* ``fit_due``: the refit dates that are due (``walk_forward.refit_dates``: a target is due
  once data exists on/after it) and not yet stored; each is fitted on data up to it, warm-
  started / aligned from the previous fit exactly as ``walk_forward`` does - the previous fit
  is rebuilt from its stored params (``from_params`` + ``restore_path`` for regime models,
  which recomputes its regime path exactly).
* ``predict_rows``: rows in ``(after, through]``, each predicted by the latest fit dated
  BEFORE it (the walk-forward's segments), each fit rebuilt from its params.
* ``predict_window_after``: where the daily predict must start: the earlier of the last
  stored prediction and the last fit, so rows a late fit should have covered are redone.
* ``rebuild``: the full walk-forward; ``reconcile``: what differs from the stored run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from infra.models.stats.pca import PCA, make_pca
from infra.models.stats.regime_pca import RegimePCA, make_regime_pca
from infra.models.stats.regression import Regression, make_regression
from infra.models.walk_forward import WalkForwardResult, refit_dates, walk_forward

log = logging.getLogger(__name__)

KINDS = ("regression", "pca", "regime_pca", "event_study", "autocorr", "forecast")


def _tuplify(x):
    """JSON lists back to the tuples the specs use (recursively)."""
    if isinstance(x, list):
        return tuple(_tuplify(v) for v in x)
    if isinstance(x, dict):
        return {k: _tuplify(v) for k, v in x.items()}
    return x


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines a run - stored as the run's ``meta.json``."""
    name: str
    kind: str                                   # regression | pca | regime_pca | event_study | autocorr | forecast
    spec: str | None
    series: tuple[str, ...]                     # regression: y first; pca / regime_pca: the K;
                                                # event_study: the instruments
    start: str                                  # first refit date
    history_start: str                          # first day read (prep always starts here)
    refit: str | int = "W-FRI"
    regime_series: tuple[str, ...] = ()         # regime_pca: the N
    overrides: dict = field(default_factory=dict)
    regime_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r}; one of {KINDS}")

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=str))

    @classmethod
    def from_json(cls, d: dict) -> "RunConfig":
        d = dict(d)
        for k in ("failures", "last_rebuild", "last_predict_through"):
            d.pop(k, None)
        d["series"], d["regime_series"] = tuple(d["series"]), tuple(d.get("regime_series", ()))
        d["overrides"] = _tuplify(d.get("overrides", {}))
        d["regime_overrides"] = _tuplify(d.get("regime_overrides", {}))
        d["refit"] = int(d["refit"]) if str(d["refit"]).isdigit() else d["refit"]
        return cls(**d)

    @property
    def all_series(self) -> list[str]:
        return list(dict.fromkeys(self.series + self.regime_series))

    def model_kwargs(self) -> dict:
        kw = dict(self.overrides)
        if self.kind == "event_study":
            kw["instruments"] = tuple(self.series)
            return kw
        if self.kind == "autocorr":
            kw.update(target=self.series[0], x=self.series[1])
            return kw
        if self.kind == "forecast":
            kw.update(target=self.series[0])          # regressors come from the named spec (not JSON-serialisable)
            return kw
        if self.kind == "regression":
            kw.update(y=self.series[0], x=tuple(self.series[1:]))
        else:
            kw.update(columns=tuple(self.series))
        if self.kind == "regime_pca":
            ro = dict(self.regime_overrides)
            if self.regime_series:
                ro["columns"] = tuple(self.regime_series)
            kw["regime_overrides"] = tuple(ro.items())
        return kw

    def make_model(self):
        from infra.models.autocorr.model import make_autocorr
        from infra.models.event_study.model import make_event_study
        from infra.models.forecast.model import make_forecast
        maker = {"regression": make_regression, "pca": make_pca, "regime_pca": make_regime_pca,
                 "event_study": make_event_study, "autocorr": make_autocorr, "forecast": make_forecast}[self.kind]
        return maker(self.spec, **self.model_kwargs())

    def from_params(self, params: pd.DataFrame):
        from infra.models.autocorr.model import ConditionalAutocorr
        from infra.models.event_study.model import EventStudy
        from infra.models.forecast.model import ForecastModel
        cls = {"regression": Regression, "pca": PCA, "regime_pca": RegimePCA, "event_study": EventStudy,
               "autocorr": ConditionalAutocorr, "forecast": ForecastModel}[self.kind]
        return cls.from_params(params, self.spec, **self.model_kwargs())


def read_inputs(config: RunConfig, through) -> pd.DataFrame:
    """The run's input panel from disk, ``history_start`` .. ``through`` (an event study:
    the grid step P&L of its instruments from its source, on its code's cycle)."""
    if config.kind in ("event_study", "forecast"):
        return config.make_model().read_panel(config.history_start, through)
    from infra.pipeline.series_panel import read_panel
    return read_panel(config.all_series, config.history_start, through)


def fit_dates(params: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(params["fit_as_of"].unique())) if len(params) else []


def params_at(params: pd.DataFrame, day) -> pd.DataFrame:
    return params[params["fit_as_of"] == pd.Timestamp(day)].drop(columns="fit_as_of").reset_index(drop=True)


def rebuild_fit(config: RunConfig, params: pd.DataFrame, day, prepared: pd.DataFrame | None = None):
    """The stored fit of ``day``, predict-ready; with ``prepared``, regime models also get
    their in-sample path back (needed to align / warm-start the next refit)."""
    model = config.from_params(params_at(params, day))
    if prepared is not None and hasattr(model, "restore_path"):
        model.restore_path(prepared)
    return model


@dataclass
class FitResult:
    params: pd.DataFrame                        # the NEW fits only (section, row, col, value, fit_as_of)
    fitted: list[pd.Timestamp]
    failures: dict[str, str]


def fit_due(config: RunConfig, prepared: pd.DataFrame, params: pd.DataFrame, through, *,
            skip: set | None = None) -> FitResult:
    """Fit every due refit date after the last stored (or skipped = previously failed) one,
    in order, exactly as ``walk_forward`` would have."""
    have = fit_dates(params)
    skip = {pd.Timestamp(d) for d in (skip or set())}
    last = max(have + list(skip)) if (have or skip) else None
    due = [d for d in refit_dates(prepared.index, config.start, through, config.refit) if last is None or d > last]
    prev = rebuild_fit(config, params, have[-1], prepared) if have else None
    out, fitted, failures = [], [], {}
    for d in due:
        model = config.make_model()
        if prev is not None and hasattr(model, "warm_start_from"):
            model.warm_start_from(prev)
        try:
            model.fit(prepared, as_of=d)
        except Exception as exc:  # recorded and skipped, as walk_forward does
            failures[d.date().isoformat()] = f"{type(exc).__name__}: {exc}"
            log.warning("fit %s at %s failed: %s", config.name, d.date(), failures[d.date().isoformat()])
            continue
        if prev is not None and hasattr(model, "align_to"):
            model.align_to(prev)
        out.append(model.params().assign(fit_as_of=d))
        fitted.append(d)
        prev = model
    new = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["section", "row", "col", "value",
                                                                             "fit_as_of"])
    return FitResult(params=new, fitted=fitted, failures=failures)


def predict_rows(config: RunConfig, prepared: pd.DataFrame, params: pd.DataFrame, after, through) -> pd.DataFrame:
    """Rows in ``(after, through]`` (``after`` None = from the first fit), each by the
    latest stored fit dated before it; ``fit_as_of`` says which."""
    fits = fit_dates(params)
    if not fits:
        return pd.DataFrame()
    through = pd.Timestamp(through)
    after = fits[0] if after is None else max(pd.Timestamp(after), fits[0])
    parts = []
    for i, d in enumerate(fits):
        seg_end = fits[i + 1] if i + 1 < len(fits) else through
        lo, hi = max(d, after), min(seg_end, through)
        if hi <= lo:
            continue
        model = rebuild_fit(config, params, d)
        out = model.predict(prepared, start=lo, end=hi)
        if len(out):
            parts.append(out.assign(fit_as_of=d))
    return pd.concat(parts).sort_index() if parts else pd.DataFrame()


def expected_fit(fits: list[pd.Timestamp], index: pd.DatetimeIndex) -> pd.Series:
    """For each row, the fit the walk-forward would use: the latest fit dated BEFORE it
    (NaT before the first fit)."""
    f = pd.DatetimeIndex(fits)
    pos = f.searchsorted(index, side="left") - 1
    return pd.Series([f[p] if p >= 0 else pd.NaT for p in pos], index=index)


def predict_upcoming(config: RunConfig, prepared: pd.DataFrame, params: pd.DataFrame, through,
                     horizon_days: int) -> pd.DataFrame:
    """Rows STARTING after ``through`` within ``horizon_days`` calendar days (events already
    scheduled, from ``prepared``), all with the LATEST stored fit - what the run expects now
    for the coming days. A plan, not a stored prediction: the daily predict records each row
    once its start has passed (with the fit then current)."""
    fits = fit_dates(params)
    if not fits:
        return pd.DataFrame()
    through = pd.Timestamp(through)
    model = rebuild_fit(config, params, fits[-1])
    out = model.predict(prepared, start=through, end=through.normalize() + pd.Timedelta(days=horizon_days + 1)
                        - pd.Timedelta(microseconds=1))
    return out.assign(fit_as_of=fits[-1]) if len(out) else out


def predict_window_after(params: pd.DataFrame, predictions: pd.DataFrame, last_through=None):
    """Where the daily predict starts (rows AFTER this are recomputed): the earliest of
    * the last fit - the current segment is always redone, so a revised day shows up as a
      changed row;
    * the last stored prediction - new rows;
    * the first stored row predicted by the WRONG fit (not the latest fit dated before it),
      from that row's correct fit on: a run catching up several days appends several fits
      after predicting those days with the old one (found 2026-10-03: catching up from 09-08
      to 09-18 left the 09-11 segment on the 09-04 fit when only the last new fit's rows were
      redone);
    * with ``last_through`` (the previous predict's cut-off) and an ``end`` column (event
      studies): the first stored row whose window ended after that cut-off - it was still
      open (P&L incomplete) when stored.
    None = from the first fit (nothing stored yet)."""
    fits = fit_dates(params)
    if not fits or predictions is None or predictions.empty:
        return None
    candidates = [pd.Timestamp(predictions.index.max()), fits[-1]]
    if "fit_as_of" in predictions.columns:
        exp = expected_fit(fits, predictions.index)
        wrong = exp.notna() & (pd.to_datetime(predictions["fit_as_of"]) != exp)
        if wrong.any():
            candidates.append(exp[wrong].min())
    if last_through is not None and "end" in predictions.columns:
        cut = pd.Timestamp(last_through)
        cut = cut + pd.Timedelta(days=1) if cut == cut.normalize() else cut
        still_open = pd.to_datetime(predictions["end"]) >= cut
        if still_open.any():
            candidates.append(predictions.index[still_open.to_numpy()].min() - pd.Timedelta(microseconds=1))
    return min(candidates)


def rebuild(config: RunConfig, panel: pd.DataFrame, through, *, prepare_kwargs: dict | None = None) -> WalkForwardResult:
    """The whole run from scratch (the periodic check)."""
    return walk_forward(config.make_model, panel, config.start, through, refit=config.refit,
                        prepare_kwargs=prepare_kwargs)


def reconcile(old_params: pd.DataFrame, old_pred: pd.DataFrame, new_params: pd.DataFrame, new_pred: pd.DataFrame,
              *, atol: float = 1e-9) -> dict:
    """What differs between a stored run and a rebuild: fit dates on one side only; per
    params section the largest difference and the first fit date that differs (a regime
    chain diverging shows as a first date and everything after it); prediction rows on one
    side only, per column the largest difference, and the first differing row."""
    report: dict = {}
    of, nf = set(fit_dates(old_params)), set(fit_dates(new_params))
    report["fits_only_stored"] = sorted(d.date().isoformat() for d in of - nf)
    report["fits_only_rebuilt"] = sorted(d.date().isoformat() for d in nf - of)
    keys = ["fit_as_of", "section", "row", "col"]
    m = old_params.merge(new_params, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["value_old"] - both["value_new"]).abs()
    nan_mismatch = both["value_old"].isna() != both["value_new"].isna()
    bad = both[(diff > atol * (1 + both["value_old"].abs().fillna(0))) | nan_mismatch]
    report["params_rows_compared"] = int(len(both))
    report["params_rows_differing"] = int(len(bad))
    report["params_keys_one_side"] = int((m["_merge"] != "both").sum())
    report["params_first_differing_fit"] = (pd.Timestamp(bad["fit_as_of"].min()).date().isoformat()
                                            if len(bad) else None)
    report["params_max_diff_by_section"] = {
        k: float(v) for k, v in both.assign(d=diff).groupby("section")["d"].max().items() if v > atol}
    if old_pred is not None and new_pred is not None and len(old_pred) and len(new_pred):
        idx = old_pred.index.intersection(new_pred.index)
        cols = [c for c in old_pred.columns if c in new_pred.columns and c != "fit_as_of"
                and pd.api.types.is_numeric_dtype(old_pred[c])]
        a = old_pred.loc[idx, cols].astype("float64")
        b = new_pred.loc[idx, cols].astype("float64")
        d = (a - b).abs()
        mismatch = (d > atol * (1 + a.abs())) | (a.isna() != b.isna())
        rows = mismatch.any(axis=1)
        report["prediction_rows_only_stored"] = int(len(old_pred.index.difference(new_pred.index)))
        report["prediction_rows_only_rebuilt"] = int(len(new_pred.index.difference(old_pred.index)))
        report["prediction_rows_compared"] = int(len(idx))
        report["prediction_rows_differing"] = int(rows.sum())
        report["prediction_first_differing_row"] = (pd.Timestamp(idx[rows.to_numpy()].min()).isoformat()
                                                    if rows.any() else None)
        report["prediction_max_diff_by_column"] = {c: float(v) for c, v in d.max().items() if v > atol}
    report["identical"] = (not report["fits_only_stored"] and not report["fits_only_rebuilt"]
                           and report["params_rows_differing"] == 0 and report["params_keys_one_side"] == 0
                           and report.get("prediction_rows_differing", 0) == 0
                           and report.get("prediction_rows_only_stored", 0) == 0
                           and report.get("prediction_rows_only_rebuilt", 0) == 0)
    return report
