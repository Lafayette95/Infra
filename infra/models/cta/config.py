"""CTA model parameters and input universes - kept apart from the code, so several
parametrised models can sit side by side and be toggled by name (``CTA_MODELS``).

Two separate things are configured here:

* ``CTASpec``: HOW the model turns a price series into a position (spans, windows,
  response function, risk targets, forecast settings). Nothing in it knows what the
  series is.
* ``CTAUniverse`` / ``CTAAsset``: WHICH series go in and the per-asset metadata the
  position formula needs (asset class, liquidity factor, how returns are measured).
  Only ``infra.models.cta.inputs`` reads the ``source``/``ticker`` fields; the model
  itself sees a plain price frame plus this metadata.

Source of the defaults: UBS Q-Series, "CTAs: How $375 bln Influences Global Assets?"
(2022-09-05), Part II (Q7, Q9). Where the note gives no number it says so below and the
default is the published reference it points to (Baz, Granger, Harvey, Le Roux, Rattray
2015, "Dissecting Investment Strategies in the Cross Section and Time Series", Man AHL).
See infra/models/cta/CLAUDE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class CTASpec:
    name: str
    description: str = ""
    # (short, long) EWMA time scales n, in observations, weight decay (n-1)/n, i.e. alpha
    # = 1/n. UBS: "3 to 5 frequencies", none given; these are Baz et al.'s three pairs.
    ewma_pairs: tuple[tuple[int, int], ...] = ((8, 24), (16, 48), (32, 96))
    # Weight of each pair in the combined score; None = equal.
    pair_weights: tuple[float, ...] | None = None
    # UBS step (ii): the crossover is normalised by the underlying's rolling volatility,
    # "usually a 1-year window": std of daily changes over this many observations.
    norm_vol_window: int = 252
    # UBS step (iii): the response function that makes the signal ~uniform on [-1, 1].
    # "ecdf" (fitted, exactly uniform in sample), "normal" (2*Phi(z)-1) or "tanh".
    response: str = "ecdf"
    # Steepness: the score is multiplied by this before the response. 1 = the response as
    # fitted (uniform for "ecdf"/"normal"); >1 saturates sooner, piling mass at +-1 - what
    # UBS's own signal charts show (Fig. 59/60: pinned at +-1 for months), whatever the
    # text says about uniformity. Applied in score space for "normal"/"tanh", and for
    # "ecdf" through the normal quantile of F (signal = 2*Phi(gain * Phi^-1(F)) - 1).
    response_gain: float = 1.0
    # Fit the response per asset ("asset") or on all assets' scores together ("pooled").
    response_pool: str = "asset"
    # Fewer fitted scores than this -> no signal (NaN) for that asset.
    response_min_obs: int = 252
    # UBS b): positions are scaled by the inverse of the 1-3 month realised volatility;
    # the worked examples use 3 months.
    sizing_vol_window: int = 63
    # UBS Q9 II): the forecast of short-term volatility is the 2-month realised vol.
    forecast_vol_window: int = 42
    # A rolling vol needs at least this many observations (all three windows above).
    vol_min_obs: int = 42
    # The first observations of a series carry the EWMAs' start-up bias: no signal before
    # this many (None = the longest EWMA time scale).
    warmup_obs: int | None = None
    # UBS d): portfolio volatility target (10% "very common") reached with a multiplier
    # estimated over a 3-year rolling window. None switches the scaling off.
    portfolio_vol_target: float | None = 0.10
    pvs_window: int = 756
    pvs_min_obs: int = 252
    # Positions are reported in [-1, 1]: divided by the largest absolute position over
    # this many observations up to the fit date (UBS reports "max abs position last 10y").
    position_scale_window: int = 2520
    clip_position: bool = True
    # Past changes and expected flows, in observations (trading days).
    change_horizons: tuple[int, ...] = (1, 3, 5, 30)
    flow_horizons: tuple[int, ...] = (1, 3, 5, 30)
    # UBS Q9: Monte Carlo of price paths "centred around the assets' forward prices" -
    # zero drift for a futures price. Antithetic pairs, fixed seed (reproducible).
    mc_paths: int = 4000
    mc_seed: int = 7
    # Reaction-function grids: price shocks in units of the DAILY sizing vol, vol shocks
    # as relative changes of the sizing vol.
    price_shock_grid: tuple[float, ...] = tuple(np.round(np.arange(-4.0, 4.01, 0.25), 2))
    vol_shock_grid: tuple[float, ...] = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    # Which volatilities a vol shock moves: "sizing" (positions only) and/or "norm"
    # (also the trend normalisation, so the signal moves too).
    vol_shock_targets: tuple[str, ...] = ("sizing",)
    # The "path" reaction spreads its move over this many observations (a week).
    reaction_path_days: int = 5
    periods_per_year: int = 252

    @property
    def spans(self) -> tuple[int, ...]:
        return tuple(sorted({n for pair in self.ewma_pairs for n in pair}))

    @property
    def warmup(self) -> int:
        return max(self.spans) if self.warmup_obs is None else self.warmup_obs

    @property
    def tail_length(self) -> int:
        """Returns kept at fit time so every rolling window continues exactly in predict."""
        return max(self.norm_vol_window, self.sizing_vol_window, self.forecast_vol_window)

    def weights(self) -> np.ndarray:
        w = np.ones(len(self.ewma_pairs)) if self.pair_weights is None else np.asarray(self.pair_weights, float)
        if len(w) != len(self.ewma_pairs):
            raise ValueError(f"{self.name}: {len(w)} pair weights for {len(self.ewma_pairs)} pairs")
        return w / w.sum()


_UBS = CTASpec(
    name="ubs2022",
    description="UBS Q-Series 2022 as described (Baz et al. spans, ECDF response per asset, "
                "3m sizing vol, 2m vol forecast, 10% target over 3y, 10y position scale).",
)

CTA_MODELS: dict[str, CTASpec] = {s.name: s for s in (
    _UBS,
    replace(_UBS, name="ubs2022_fast",
            description="Faster trend: pairs (4,12), (8,24), (16,48).",
            ewma_pairs=((4, 12), (8, 24), (16, 48))),
    replace(_UBS, name="ubs2022_slow",
            description="Slower trend: pairs (16,48), (32,96), (64,192).",
            ewma_pairs=((16, 48), (32, 96), (64, 192))),
    replace(_UBS, name="ubs2022_normal",
            description="Parametric response 2*Phi(z)-1 on the standardised score instead of the ECDF.",
            response="normal"),
    replace(_UBS, name="ubs2022_pooled",
            description="One ECDF response fitted on all assets' scores together.",
            response_pool="pooled"),
    # Calibrated 2026-10-02 to UBS's own published rates snapshot (2022-09-02, Fig. 63/102,
    # infra.models.cta.paper) over a grid of 4 speeds x 4 gains x 2 responses: the best fit
    # by far. ONE date and two parameters - see infra/models/cta/CLAUDE.md section 5.
    replace(_UBS, name="ubs2022_cal",
            description="Calibrated to UBS's 2022-09-02 snapshot: fast pairs, ECDF response gain 2.",
            ewma_pairs=((4, 12), (8, 24), (16, 48)), response_gain=2.0),
)}


def get_spec(name: str) -> CTASpec:
    try:
        return CTA_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown CTA model {name!r}; known: {sorted(CTA_MODELS)}") from None


@dataclass(frozen=True)
class CTAAsset:
    name: str                   # label in outputs, e.g. "US10Y"
    source: str = "frame"       # infra.models.cta.inputs.SOURCES key ("futures", ...)
    ticker: str = ""            # what that source reads, e.g. "ZN.v.0"
    asset_class: str = "default"
    liquidity: float = 1.0      # UBS liquidity factor (1..8, from ADV and open interest)
    # How a change is measured: "diff" (price points: futures, yields) or "log" (returns).
    returns: str = "diff"


@dataclass(frozen=True)
class CTAUniverse:
    name: str
    assets: tuple[CTAAsset, ...]
    # Risk weight per asset class (UBS: 30% to rates bond futures). Missing class -> 1.
    class_weights: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def asset(self, name: str) -> CTAAsset:
        return {a.name: a for a in self.assets}[name]


def _fut(name, ticker, liquidity, asset_class="rates_bonds"):
    return CTAAsset(name=name, source="futures", ticker=ticker, asset_class=asset_class,
                    liquidity=liquidity, returns="diff")


# UBS Fig. 63 liquidity factors. Label mapping is OUR reading of UBS's names: US2Y=TU (ZT),
# US5Y=FV (ZF), US10Y=TY (ZN), US20Y=US classic bond (ZB), US30Y=WN ultra bond (UB);
# EU2Y/5Y/10Y = Schatz/Bobl/Bund, IT10Y = BTP. TN (ultra 10y) is not in UBS's list.
_US_UBS = (
    _fut("US2Y", "ZT.v.0", 1), _fut("US5Y", "ZF.v.0", 4), _fut("US10Y", "ZN.v.0", 8),
    _fut("US20Y", "ZB.v.0", 4), _fut("US30Y", "UB.v.0", 4),
)
_EU_UBS = (
    _fut("EU2Y", "FGBS.v.0", 1), _fut("EU5Y", "FGBM.v.0", 4), _fut("EU10Y", "FGBL.v.0", 8),
    _fut("IT10Y", "FBTP.v.0", 2),
)

CTA_UNIVERSES: dict[str, CTAUniverse] = {u.name: u for u in (
    CTAUniverse("ubs_us_bonds", _US_UBS, {"rates_bonds": 0.30},
                "UBS's US bond futures, with its liquidity factors (history from 2014-12)."),
    CTAUniverse("ubs_bonds", _US_UBS + _EU_UBS, {"rates_bonds": 0.30},
                "UBS's US + Eurex bond futures we hold (Eurex history from 2025-07 only)."),
    CTAUniverse("us_bonds_all", _US_UBS + (_fut("US10Y_ULTRA", "TN.v.0", 2),), {"rates_bonds": 0.30},
                "US bond futures incl. TN (liquidity factor 2 is our assumption, not UBS's)."),
)}


def get_universe(name: str) -> CTAUniverse:
    try:
        return CTA_UNIVERSES[name]
    except KeyError:
        raise KeyError(f"unknown CTA universe {name!r}; known: {sorted(CTA_UNIVERSES)}") from None
