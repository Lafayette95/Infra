"""Stage 1, data preparation: any wide price frame -> the CTA model's input form.

The model is agnostic about what a series is: a back-adjusted futures price, a yield, an
FX rate or an index all go in the same way. What it needs per series is in ``CTAAsset``
(``returns``: measured as a difference, or as a log return; class weight and liquidity).

* Each series keeps ITS OWN calendar: a holiday in one market is not a zero-move day for
  the trend of another. Only the portfolio vol scaling, which needs one calendar, aligns
  them (``signal.portfolio_vol_scaling``).
* ``level`` is what the EWMAs run on (the price, or its log); ``ret`` its change between
  consecutive observations of that series.
* ``anchor`` continues the changes across a fit/predict boundary: the last fitted level
  per series, so the first new observation's change is measured against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infra.models.cta.config import CTAAsset, CTAUniverse


@dataclass(frozen=True)
class CTAData:
    levels: pd.DataFrame       # wide, timestamp x asset, NaN where a series has no row
    returns: pd.DataFrame      # same shape; first row of a series NaN unless anchored
    assets: dict[str, CTAAsset]
    class_weights: dict[str, float] = field(default_factory=dict)

    def class_weight(self, asset: str) -> float:
        return float(self.class_weights.get(self.assets[asset].asset_class, 1.0))

    def series(self, asset: str) -> tuple[pd.Series, pd.Series]:
        """One asset's (level, ret) on its own calendar."""
        level = self.levels[asset].dropna()
        return level, self.returns[asset].reindex(level.index)


def _to_level(prices: pd.Series, how: str) -> pd.Series:
    if how == "diff":
        return prices.astype("float64")
    if how == "log":
        if (prices <= 0).any():
            raise ValueError(f"{prices.name}: log returns need positive prices")
        return np.log(prices.astype("float64"))
    raise ValueError(f"{prices.name}: unknown returns kind {how!r} (diff | log)")


def prepare(prices: pd.DataFrame, *, universe: CTAUniverse | None = None,
            assets: dict[str, CTAAsset] | None = None, class_weights: dict[str, float] | None = None,
            anchor: dict[str, float] | None = None) -> CTAData:
    """Wide ``timestamp x asset`` prices -> ``CTAData``.

    Metadata comes from ``universe`` (matched by column name), else ``assets``, else a
    default ``CTAAsset`` per column (difference returns, weight 1, liquidity 1)."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices need a DatetimeIndex")
    if prices.index.tz is not None:
        raise ValueError("prices must be tz-naive UTC (CLAUDE.md 7)")
    if prices.index.has_duplicates:
        raise ValueError("prices have duplicate timestamps")
    prices = prices.sort_index()
    meta = dict(assets or {})
    if universe is not None:
        meta = {a.name: a for a in universe.assets} | meta
        class_weights = universe.class_weights if class_weights is None else class_weights
    meta = {c: meta.get(c, CTAAsset(name=str(c))) for c in prices.columns}
    levels, rets = {}, {}
    for col in prices.columns:
        px = prices[col].dropna()
        level = _to_level(px, meta[col].returns)
        prev = level.shift(1)
        if anchor is not None and col in anchor and len(level):
            prev.iloc[0] = anchor[col]
        levels[col], rets[col] = level, level - prev
    idx = prices.index
    return CTAData(
        levels=pd.DataFrame(levels).reindex(idx),
        returns=pd.DataFrame(rets).reindex(idx),
        assets=meta,
        class_weights=dict(class_weights or {}),
    )
