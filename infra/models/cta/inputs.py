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

import logging

import pandas as pd

from infra.models.cta.config import CTAUniverse
from infra.pipeline import daily as dl
from infra.pipeline.relative_daily import load_relative_daily
from infra.processing import continuous
from infra.relative.symbology import parse_relative

log = logging.getLogger(__name__)

# Absolute settlements are read this far before ``start`` so the first roll has its prior.
_PRIOR_DAYS = pd.Timedelta(days=10)


def futures_prices(tickers: list[str], start, end) -> pd.DataFrame:
    """Back-adjusted continuous settlement prices, wide ``timestamp x ticker``, over
    ``[start, end]`` (inclusive days) for relative futures tickers. Disk only."""
    specs = []
    for t in tickers:
        spec = parse_relative(t)
        if spec is None:
            raise ValueError(f"{t!r} is not a relative futures ticker (e.g. ZN.v.0)")
        specs.append(spec)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rel = load_relative_daily(specs, start, end + pd.Timedelta(days=1), fetch_missing=False)
    if rel.empty:
        return pd.DataFrame(columns=tickers, dtype="float64")
    contracts = sorted(rel["contract"].astype(str).unique())
    absolute = dl.read_daily_from_disk(contracts, start - _PRIOR_DAYS, end + pd.Timedelta(days=1))
    changes = continuous.same_contract_changes(rel, absolute)
    unknown = continuous.unknown_changes(changes)
    if len(unknown):
        log.warning("futures_prices: %d day(s) with no same-contract prior settlement (counted as no "
                    "move): %s", len(unknown),
                    ", ".join(f"{r.ticker} {r.timestamp.date()}" for r in unknown.head(10).itertuples()))
    return continuous.back_adjusted(changes).reindex(columns=tickers)


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
