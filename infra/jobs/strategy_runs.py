"""Operating a strategy (root CLAUDE.md 27): the daily job writes its FIRM signals and positions
(relative and absolute tickers) and its PLAN vintage; the rebuild recomputes everything and
reconciles. Strategy classes compute (``infra.strategies``); this module reads their inputs and
writes ``Database/Strategies/<name>`` (``infra.storage.strategy_store``).

* FIRM series: every label up to ``through``, from the model runs' stored walk-forward
  predictions (each row carries the fit current at its start) and point-in-time inputs (vols as
  available at the label, the contract mapped at the label) - recomputed whole each run and
  upserted, so a changed past row is reported (a revision upstream), never silent.
* PLAN vintage: the labels after ``through`` within the horizon, from the events already
  scheduled (``family_runs.upcoming``, the latest fits, provisional regimes), stored under
  ``generated_at = through``. Research only (how our own forecast evolves): P&L uses the firm
  series.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from infra.config import MODEL_RUNS_DIR, STRATEGIES_DIR
from infra.jobs import family_runs
from infra.storage import strategy_store as store
from infra.strategies.base import contracts_at
from infra.strategies.cevt import CEVT
from infra.strategies.config.cevt import CEVT_STRATEGIES

log = logging.getLogger(__name__)

STRATEGY_GROUPS = {"cevt": (CEVT, CEVT_STRATEGIES)}


def make_strategy(name: str):
    for group, (cls, registry) in STRATEGY_GROUPS.items():
        if name in registry:
            return group, cls(registry[name])
    raise KeyError(f"unknown strategy {name!r}; known: {sorted(n for _, r in STRATEGY_GROUPS.values() for n in r)}")


def grid_labels(strategy, start, end) -> pd.DatetimeIndex:
    """The strategy's grid points (decision times) between two instants."""
    from infra.processing import event_windows as ew
    from infra.processing.schedule_rules import business_days
    from infra.reference.event_grid import resolve_cycle
    cyc = resolve_cycle(strategy.spec.cycle)
    g = ew.make_grid(cyc, business_days(pd.Timestamp(start).normalize() - pd.Timedelta(days=1),
                                        pd.Timestamp(end).normalize() + pd.Timedelta(days=1), cyc.calendar))
    pts = g.instants()
    return pts[(pts >= pd.Timestamp(start)) & (pts <= pd.Timestamp(end))]


def _inputs(strategy, through, models_root: Path):
    preds, fdr = {}, {}
    for f in strategy.spec.families:
        preds[f] = family_runs.predictions(f, root=models_root)
        fdr[f] = family_runs.read_fdr(f, root=models_root)
    return preds, fdr


def firm(strategy, through, *, models_root: Path = MODEL_RUNS_DIR, inputs=None, vols=None, contracts=None):
    """(views, positions, positions_abs) for every label up to the end of ``through``."""
    preds, fdr = inputs if inputs is not None else _inputs(strategy, through, models_root)
    starts = [p.index.min() for p in preds.values() if len(p)]
    if not starts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    end = pd.Timestamp(through).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    labels = grid_labels(strategy, min(starts), end)
    views = strategy.views(preds, fdr, labels)
    v = strategy.vols(labels, vols)
    pos = strategy.positions(strategy.signal(views), v)
    con = contracts or {inst: contracts_at(inst, labels) for inst in strategy.instruments}
    return views, pos, strategy.to_absolute(pos, con)


def plan(strategy, through, *, models_root: Path = MODEL_RUNS_DIR, upcoming=None, fdr=None, vols=None,
         contracts=None) -> pd.DataFrame:
    """The forward rows (labels after ``through`` within the horizon): views and positions."""
    t0 = pd.Timestamp(through).normalize() + pd.Timedelta(days=1)
    labels = grid_labels(strategy, t0, t0 + pd.Timedelta(days=strategy.spec.horizon_days))
    if upcoming is None:
        upcoming = {f: family_runs.upcoming(f, through, strategy.spec.horizon_days, root=models_root)
                    for f in strategy.spec.families}
    if fdr is None:
        fdr = {f: family_runs.read_fdr(f, root=models_root) for f in strategy.spec.families}
    views = strategy.views(upcoming, fdr, labels)
    pos = strategy.positions(strategy.signal(views), strategy.vols(labels, vols))
    return views.join(pos.add_prefix("pos:"))


def run_daily(name: str, through, *, root: Path = STRATEGIES_DIR, models_root: Path = MODEL_RUNS_DIR,
              update_families: bool = False, **inputs) -> dict:
    group, s = make_strategy(name)
    if update_families:
        for f in s.spec.families:
            family_runs.run_daily(f, through, root=models_root)
    views, pos, pabs = firm(s, through, models_root=models_root, inputs=inputs.get("inputs"),
                            vols=inputs.get("vols"), contracts=inputs.get("contracts"))
    out = {"signals": store.upsert_series(name, "signals", views, root=root),
           "positions": store.upsert_series(name, "positions", pos, root=root),
           "positions_abs": store.upsert_series(name, "positions_abs", pabs, root=root)}
    for k in ("signals", "positions"):
        if out[k]["changed"]:
            log.warning("%s: %d firm %s row(s) changed on recompute (an upstream revision)", name, out[k]["changed"], k)
    if s.spec.plan_vintages:
        pl = plan(s, through, models_root=models_root, upcoming=inputs.get("upcoming"),
                  fdr=(inputs.get("inputs") or (None, None))[1], vols=inputs.get("vols"))
        out["plan_rows"] = store.write_plan(name, pd.Timestamp(through).normalize(), pl, root=root)
    store.write_meta(name, {**store.read_meta(name, root=root), "name": name, "group": group,
                            "last_through": str(pd.Timestamp(through).date())}, root=root)
    return out


def rebuild(name: str, through, *, root: Path = STRATEGIES_DIR, models_root: Path = MODEL_RUNS_DIR,
            promote: bool = False, **inputs) -> dict:
    """Recompute the firm series into ``<name>/_rebuild`` and compare with the stored ones."""
    _, s = make_strategy(name)
    views, pos, pabs = firm(s, through, models_root=models_root, inputs=inputs.get("inputs"),
                            vols=inputs.get("vols"), contracts=inputs.get("contracts"))
    rep = {}
    for kind, new in (("signals", views), ("positions", pos)):
        old = store.read_series(name, kind, root=root)
        idx = old.index.intersection(new.index)
        cols = [c for c in new.columns if c in old.columns]
        diff = (old.loc[idx, cols] - new.loc[idx, cols]).abs().max().max() if len(idx) and cols else 0.0
        rep[kind] = {"stored": len(old), "rebuilt": len(new), "max_abs_diff": float(diff) if pd.notna(diff) else 0.0,
                     "only_stored": int(len(old.index.difference(new.index))),
                     "only_rebuilt": int(len(new.index.difference(old.index)))}
    rep["identical"] = all(r["max_abs_diff"] <= 1e-9 and r["only_stored"] == 0 and r["only_rebuilt"] == 0
                           for r in rep.values() if isinstance(r, dict))
    rb = f"{name}/_rebuild"
    store.upsert_series(rb, "signals", views, root=root)
    store.upsert_series(rb, "positions", pos, root=root)
    if promote:
        stamp = pd.Timestamp.now().strftime("%Y-%m-%dT%H%M")
        rep["archived_to"] = str(store.archive(name, stamp, root=root))
        for kind, new in (("signals", views), ("positions", pos)):
            (root / name / f"{kind}.parquet").unlink(missing_ok=True)
            store.upsert_series(name, kind, new, root=root)
        (root / name / "positions_abs.parquet").unlink(missing_ok=True)
        store.upsert_series(name, "positions_abs", pabs, root=root)
    return rep
