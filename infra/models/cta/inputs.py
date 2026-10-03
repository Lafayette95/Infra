"""Reading the CTA model's inputs from what the pipeline already stored (disk only).

The model itself takes any wide price frame (``prep.prepare``); this module is only the
convenience of building that frame from the database, one reader per source kind
(``SOURCES``, keyed by ``CTAAsset.source``). It never fetches (infra/models rule):
missing history has to be loaded first through the pipeline.

* ``futures``: a relative ticker (``ZN.v.0``) -> additive back-adjusted continuous
  settlement prices (``infra.processing.continuous``): each day's change measured on the
  contract held that day, so a roll gap is never read as a market move.
"""
from __future__ import annotations

import pandas as pd

from infra.models.cta.config import CTAUniverse
from infra.pipeline.series_panel import continuous_futures

# The reader lives below the models (infra.pipeline.series_panel) since 2026-10-03, shared
# with the statistical models; the name is kept for this module's callers.
futures_prices = continuous_futures


SOURCES = {"futures": futures_prices}


def universe_prices(universe: CTAUniverse, start, end) -> pd.DataFrame:
    """Wide prices for every asset of ``universe``, columns = asset names."""
    by_source: dict[str, list] = {}
    for a in universe.assets:
        by_source.setdefault(a.source, []).append(a)
    frames = []
    for source, assets in by_source.items():
        if source not in SOURCES:
            raise KeyError(f"no reader for source {source!r}; known: {sorted(SOURCES)}")
        wide = SOURCES[source]([a.ticker for a in assets], start, end)
        frames.append(wide.rename(columns={a.ticker: a.name for a in assets}))
    out = pd.concat(frames, axis=1).sort_index()
    return out.reindex(columns=[a.name for a in universe.assets])
