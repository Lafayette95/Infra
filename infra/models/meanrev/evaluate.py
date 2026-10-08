"""Out-of-sample evaluation of a mean-reversion spec (research only).

``evaluate(spec, panel, start, through)``: walk forward (``infra.models.walk_forward``), then the
NETTED book's daily P&L - positions decided on row t (its data is known by t + ``gap_days``) earn the
instruments' moves from t + 1 + ``gap_days`` - with its Sharpe, sub-periods, turnover and the fits'
statistics (share of residuals tradable, median half-life of the tradable ones).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.models.meanrev.config import MeanRevSpec, get_meanrev_spec
from infra.models.meanrev.model import MeanRevModel

ANNUAL = np.sqrt(252.0)


def walk(spec: MeanRevSpec, panel: pd.DataFrame, start, through, refit: str = "ME"):
    from infra.models.walk_forward import walk_forward
    return walk_forward(lambda: MeanRevModel(spec), panel, start, through, refit=refit)


def book_pnl(predictions: pd.DataFrame, moves: pd.DataFrame, spec: MeanRevSpec) -> pd.Series:
    """Daily P&L (in the moves' units x position units) of the netted positions."""
    cols = list(spec.columns)
    pos = predictions[[f"pos:{c}" for c in cols]].set_axis(cols, axis=1)
    held = pos.shift(1 + spec.gap_days)
    pnl = (held * moves[cols].reindex(held.index)).sum(axis=1, min_count=1)
    return pnl.dropna()


def _sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * ANNUAL) if len(x) > 20 and x.std() > 0 else np.nan


def evaluate(spec: MeanRevSpec | str, panel: pd.DataFrame, start, through, *, refit: str = "ME",
             **overrides) -> dict:
    spec = get_meanrev_spec(spec, **overrides)
    res = walk(spec, panel, start, through, refit)
    p = res.predictions
    pnl = book_pnl(p, panel, spec)
    cols = list(spec.columns)
    pos = p[[f"pos:{c}" for c in cols]]
    turnover = pos.diff().abs().sum(axis=1).mean() / max(pos.abs().sum(axis=1).mean(), 1e-12)
    ou_rows = res.params[res.params["section"] == "ou"].pivot_table(index=["fit_as_of", "row"], columns="col",
                                                                   values="value")
    trad = ou_rows[ou_rows["tradable"] == 1]
    sub = {f"{lo}-{hi}": round(_sharpe(pnl.loc[lo:hi]), 2)
           for lo, hi in (("2010", "2014"), ("2015", "2019"), ("2020", "2022"), ("2023", "2026"))
           if len(pnl.loc[lo:hi]) > 60}
    return {"sharpe": _sharpe(pnl), "subperiods": sub, "turnover": float(turnover),
            "share_tradable": float(ou_rows["tradable"].mean()) if len(ou_rows) else np.nan,
            "half_life_median": float(trad["half_life"].median()) if len(trad) else np.nan,
            "n_fits": int(len(res.params["fit_as_of"].unique())) if len(res.params) else 0,
            "failures": len(res.failures), "_pnl": pnl, "_result": res}
