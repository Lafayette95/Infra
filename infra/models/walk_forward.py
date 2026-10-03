"""Point-in-time walk-forward for any model: refit on a schedule, predict until the next refit.

This is use case (a) of the models layer (infra/models/CLAUDE.md 0a): a strategy backtest
that refits, say, every Friday on data up to that Friday only, stores the parameters as of
each refit, and uses that fit - parameters frozen - to produce the next week's fitted
values / residuals / factors. Stitching those out-of-sample segments gives the history
exactly as it would have been produced live: no row's output ever depends on data after
its own refit date.

    res = walk_forward(lambda: make_regression("hedge_ratio", y=..., x=...),
                       panel, start="2022-01-01", end="2026-09-30", refit="W-FRI")
    res.params        # one tidy params frame per refit, column fit_as_of
    res.predictions   # stitched out-of-sample rows, column fit_as_of = the fit behind each row

Contract a model must meet: ``prepare(raw)``, ``fit(prepared, as_of)``, ``params()`` and
``predict(prepared, start=, end=)`` returning a frame indexed by timestamp for rows
strictly after ``start`` up to ``end`` (``infra/models/stats`` does). The CTA model has its
own walk-forward (``infra.models.cta.model.walk_forward``), since its predict continues a
recursive state.

``raw`` may be a frame (prepared ONCE - every stateless prep step is trailing, so that is
point-in-time safe) or a callable ``as_of -> frame`` re-read at every refit: use that for
vintage data (``release:`` series), whose history itself changes with ``as_of``.
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    params: pd.DataFrame                       # section, row, col, value + fit_as_of
    predictions: pd.DataFrame                  # predict() rows + fit_as_of
    fit_dates: list[pd.Timestamp]
    failures: dict[pd.Timestamp, str] = field(default_factory=dict)
    models: dict[pd.Timestamp, object] | None = None

    def params_at(self, as_of) -> pd.DataFrame:
        """The params of the latest refit at or before ``as_of`` (to rebuild that fit with
        ``from_params``)."""
        dates = [d for d in self.fit_dates if d <= pd.Timestamp(as_of) and d not in self.failures]
        if not dates:
            raise KeyError(f"no successful fit at or before {as_of}")
        p = self.params[self.params["fit_as_of"] == dates[-1]]
        return p.drop(columns="fit_as_of").reset_index(drop=True)

    def param_path(self, section: str, col: str = "value", row: str | None = None) -> pd.DataFrame:
        """One parameter over time, wide ``fit_as_of x row`` (e.g. ``("coef", "coef")``:
        each coefficient's path; ``("stat", "value", "r2")``)."""
        p = self.params[(self.params["section"] == section) & (self.params["col"] == col)]
        if row is not None:
            p = p[p["row"] == row]
        return p.pivot_table(index="fit_as_of", columns="row", values="value", aggfunc="last")


def refit_dates(index: pd.DatetimeIndex, start, end, refit: str | int) -> list[pd.Timestamp]:
    """Refit dates in ``[start, end]``: a pandas frequency (``"W-FRI"``, ``"ME"``) snapped
    back to the last row on or before each target, or an int = every N rows. ``start`` is
    always the first refit (there must be a fit before the first prediction).

    A target AFTER the last row of ``index`` is not due yet: its data isn't in, and
    snapping it back (a not-yet-loaded Friday to Thursday) would differ from what a later
    run sees once Friday arrives - the incremental (append) run and a full rebuild must pick
    the same dates. A target on a genuine holiday is snapped back once a later row exists."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = index[(index >= start.normalize()) & (index <= end)]
    if rows.empty:
        return []
    if isinstance(refit, int):
        dates = list(rows[::refit])
    else:
        last = index.max()
        targets = [t for t in pd.date_range(start, end, freq=refit) if t <= last]
        pos = index.searchsorted(targets, side="right") - 1
        dates = [index[p] for p in pos if p >= 0 and index[p] >= rows[0]]
    dates = sorted(set([rows[0]] + dates))
    return [pd.Timestamp(d) for d in dates]


def walk_forward(make_model: Callable | object, raw: pd.DataFrame | Callable, start, end, *,
                 refit: str | int = "W-FRI", keep_models: bool = False, align: bool = True,
                 predict_kwargs: dict | None = None, prepare_kwargs: dict | None = None,
                 on_error: str = "skip") -> WalkForwardResult:
    """Fit at each refit date on data up to it, predict the rows after it up to the next
    refit, stitch. ``make_model``: a zero-argument factory or an unfitted model (deep-copied
    per refit). ``align``: if the model has ``align_to`` (PCA, regimes), each fit is aligned
    to the previous one so factor and regime identities don't flip; a model with
    ``warm_start_from`` (HMM regimes) starts its fit from the previous one's parameters. ``on_error``: ``"skip"`` (log, record
    in ``failures``, keep the previous fit's predictions out - those rows are just absent)
    or ``"raise"``."""
    factory = make_model if callable(make_model) and not hasattr(make_model, "fit") else (
        lambda: copy.deepcopy(make_model))
    predict_kwargs, prepare_kwargs = predict_kwargs or {}, prepare_kwargs or {}
    end = pd.Timestamp(end)
    static = None if callable(raw) else factory().prepare(raw, **prepare_kwargs)
    index = static.index if static is not None else raw(end).index
    dates = refit_dates(pd.DatetimeIndex(index), start, end, refit)
    params, preds, failures, models, prev = [], [], {}, {}, None
    for i, d in enumerate(dates):
        nxt = dates[i + 1] if i + 1 < len(dates) else end
        model = factory()
        data = static if static is not None else model.prepare(raw(d), **prepare_kwargs)
        if prev is not None and hasattr(model, "warm_start_from"):
            model.warm_start_from(prev)  # e.g. an HMM refit starts from the previous fit's parameters
        try:
            model.fit(data, as_of=d)
        except Exception as exc:  # a refit that can't run (too few rows) must not end the backtest
            if on_error == "raise":
                raise
            failures[d] = f"{type(exc).__name__}: {exc}"
            log.warning("walk_forward: fit at %s failed: %s", d.date(), failures[d])
            continue
        if align and prev is not None and hasattr(model, "align_to"):
            model.align_to(prev)
        prev = model
        params.append(model.params().assign(fit_as_of=d))
        future = data if static is not None else model.prepare(raw(nxt), **prepare_kwargs)
        out = model.predict(future, start=d, end=nxt, **predict_kwargs)
        preds.append(out.assign(fit_as_of=d))
        if keep_models:
            models[d] = model
    predictions = pd.concat(preds).sort_index() if preds else pd.DataFrame()
    return WalkForwardResult(params=pd.concat(params, ignore_index=True) if params else pd.DataFrame(),
                             predictions=predictions, fit_dates=dates, failures=failures,
                             models=models if keep_models else None)


def fit_path(make_model: Callable | object, raw: pd.DataFrame, dates, *, align: bool = True) -> dict:
    """Fits only (no predictions) at each of ``dates``: ``{date: fitted model}``. For
    stability diagnostics (eigenvector drift, coefficient paths)."""
    factory = make_model if callable(make_model) and not hasattr(make_model, "fit") else (
        lambda: copy.deepcopy(make_model))
    data = factory().prepare(raw)
    out, prev = {}, None
    for d in dates:
        m = factory()
        if prev is not None and hasattr(m, "warm_start_from"):
            m.warm_start_from(prev)
        try:
            m.fit(data, as_of=d)
        except ValueError as exc:
            log.info("fit_path: skip %s (%s)", pd.Timestamp(d).date(), exc)
            continue
        if align and prev is not None and hasattr(m, "align_to"):
            m.align_to(prev)
        out[pd.Timestamp(d)] = prev = m
    return out
