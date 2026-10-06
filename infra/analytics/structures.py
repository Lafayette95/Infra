"""Curve structures and layered sizing: pure maths (root CLAUDE.md 28; definitions in
``infra.reference.structures``, data in ``infra.pipeline.structures``).

Conventions: a weight matrix ``W`` is structures x legs (DV01 per $1 of structure); a structure
position ``x`` ($ DV01 per bp, per structure) gives leg exposures ``e = W.T @ x`` ($/bp per leg);
contracts = e / contract DV01. With as many independent structures as legs, ``W`` is square and
invertible: any per-future exposure maps back to structure positions exactly
(``to_structures``) - the "reconstruct a future from the layers" property.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUAL = 252.0


def rolling_betas(y: pd.Series, X: pd.DataFrame, window: int, min_obs: int) -> pd.DataFrame:
    """OLS slopes (with an intercept) of ``y`` on ``X`` over the last ``window`` complete rows
    up to and INCLUDING each row (shift before use: a decision on day D uses row D-1)."""
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    cols = list(X.columns)
    out = np.full((len(df), len(cols)), np.nan)
    Y, Z = df["_y"].to_numpy(), np.c_[np.ones(len(df)), df[cols].to_numpy()]
    for i in range(len(df)):
        lo = max(0, i + 1 - window)
        if i + 1 - lo < min_obs:
            continue
        b, *_ = np.linalg.lstsq(Z[lo:i + 1], Y[lo:i + 1], rcond=None)
        out[i] = b[1:]
    return pd.DataFrame(out, index=df.index, columns=cols).reindex(y.index)


def hedged_weights(base: pd.Series, targets: pd.DataFrame, betas: pd.Series) -> pd.Series:
    """Leg weights of a hedged structure: its base legs minus sum_j beta_j x the legs of hedge
    target j (``targets``: targets x legs)."""
    w = base.reindex(targets.columns).fillna(0.0)
    return w - (targets.mul(betas.reindex(targets.index), axis=0)).sum(axis=0)


def to_legs(x: pd.Series, W: pd.DataFrame) -> pd.Series:
    """Structure positions -> leg exposures (``W.T @ x``)."""
    return W.T @ x.reindex(W.index).fillna(0.0)


def to_structures(e: pd.Series, W: pd.DataFrame) -> pd.Series:
    """Leg exposures -> the structure positions producing them exactly (``W`` square and
    invertible: the structures span the legs)."""
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"the structures ({W.shape[0]}) don't span the legs ({W.shape[1]}): no exact inverse")
    return pd.Series(np.linalg.solve(W.T.to_numpy(), e.reindex(W.columns).fillna(0.0).to_numpy()), index=W.index)


def basis_condition(W: pd.DataFrame) -> float:
    """Condition number of the structure basis (large = nearly redundant structures)."""
    return float(np.linalg.cond(W.to_numpy()))


@dataclass
class LayerSizing:
    contracts: pd.DataFrame        # labels x legs
    exposures: pd.DataFrame        # labels x legs, $/bp
    positions: pd.DataFrame        # labels x structures, $/bp
    diagnostics: pd.DataFrame      # labels: vol:<layer> (standalone, $/yr), vol_sum, vol_total, scale


def size_layers(views: pd.DataFrame, state_day: pd.Series, layers: dict[str, list[str]], budgets: dict[str, float],
                sigma: pd.DataFrame, weights: dict, cov: dict, dv01: pd.DataFrame,
                portfolio_cap: float | None) -> LayerSizing:
    """Views (labels x structures, in [-1, 1]) -> contracts per leg.

    Per structure s in layer L: x_s = view_s x budget_L / sqrt(n_L) / sigma_s ($/bp; sigma_s =
    the structure's annual bp vol), so a full-strength view in every structure of a layer
    uses that layer's budget if its structures were uncorrelated. Exposures e = W.T x, netted
    across structures; contracts = e / contract DV01. Portfolio vol sqrt(e' C e) (C = the legs'
    annual bp covariance) above ``portfolio_cap`` scales everything down. ``state_day`` maps
    each label to the decision day whose state (``sigma``, ``weights``, ``cov``, ``dv01``) it
    uses - known before the label."""
    labels = views.index
    structs = [s for L in layers.values() for s in L]
    unit = pd.Series({s: budgets[L] / np.sqrt(len(Ls)) for L, Ls in layers.items() for s in Ls})
    sig = sigma.reindex(index=pd.DatetimeIndex(state_day.to_numpy()), columns=structs).to_numpy()
    x = views.reindex(columns=structs).fillna(0.0).to_numpy() * unit.reindex(structs).to_numpy() / sig
    x = np.where(np.isfinite(x), x, np.nan)
    legs = list(dv01.columns)
    E = np.full((len(labels), len(legs)), np.nan)
    diag = {f"vol:{L}": np.full(len(labels), np.nan) for L in layers}
    diag.update(vol_sum=np.full(len(labels), np.nan), vol_total=np.full(len(labels), np.nan),
                scale=np.ones(len(labels)))
    days = pd.DatetimeIndex(state_day.to_numpy())
    for d in days.unique():
        rows = np.flatnonzero(days == d)
        if pd.isna(d) or d not in weights or d not in cov:
            continue
        W = weights[d].reindex(index=structs, columns=legs).fillna(0.0).to_numpy()
        C = cov[d].reindex(index=legs, columns=legs).to_numpy()
        X = x[rows]
        e = np.nan_to_num(X) @ W
        bad = np.isnan(X).any(axis=1) & (np.abs(views.iloc[rows].reindex(columns=structs).fillna(0).to_numpy())
                                         > 0).any(axis=1)
        e[bad] = np.nan
        total = np.sqrt(np.einsum("ij,jk,ik->i", e, C, e))
        vsum = np.zeros(len(rows))
        for L, Ls in layers.items():
            idx = [structs.index(s) for s in Ls]
            eL = np.nan_to_num(X[:, idx]) @ W[idx]
            vL = np.sqrt(np.einsum("ij,jk,ik->i", eL, C, eL))
            diag[f"vol:{L}"][rows] = vL
            vsum += vL
        scale = np.ones(len(rows))
        if portfolio_cap is not None:
            scale = np.where(total > portfolio_cap, portfolio_cap / np.where(total > 0, total, 1), 1.0)
        E[rows] = e * scale[:, None]
        diag["vol_sum"][rows], diag["vol_total"][rows], diag["scale"][rows] = vsum, total * scale, scale
    exposures = pd.DataFrame(E, index=labels, columns=legs)
    dv = dv01.reindex(index=days, columns=legs).to_numpy()
    contracts = pd.DataFrame(E / dv, index=labels, columns=legs)
    return LayerSizing(contracts, exposures, pd.DataFrame(x, index=labels, columns=structs),
                       pd.DataFrame(diag, index=labels))
