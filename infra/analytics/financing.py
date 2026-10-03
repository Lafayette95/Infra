"""The financing rate of a Treasury position (CLAUDE.md 20): three swappable layers, pure
functions, no I/O.

    rate(start, end) = base(start, end) + basis(start, end) - specialness(cusip, start, end)

all in percent, ACT/360, for rolling overnight funding over calendar days ``[start, end)``.
Each layer is a registry entry ``fn(inputs, spec, start, end, cusip) -> percent``; a model
(``infra.config.FINANCING_MODELS``) names one entry per layer, so a variant is a new entry
plus a new spec, and two models can be evaluated on the same inputs side by side.
``FinancingInputs`` carries everything already read for one ``as_of`` (point in time).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from infra.analytics import specialness as sa
from infra.analytics.sofr_curve import SofrPath


@dataclass(frozen=True)
class FinancingInputs:
    as_of: pd.Timestamp
    sofr_path: SofrPath | None = None  # layer 1 "sofr_futures"
    sofr_fixings: pd.DataFrame | None = None  # published NY Fed SOFR rows (rate, p75, ...), layer 2
    specialness_model: object | None = None  # layer 3 'lifecycle_decay': profile + half-life
    bond_states: dict | None = None  # layer 3: every tracked / recently lent bond's state
    extra: dict = field(default_factory=dict)  # room for later layers' inputs


@dataclass(frozen=True)
class FinancingQuote:
    model: str
    start: pd.Timestamp
    end: pd.Timestamp
    base: float
    basis: float
    specialness: float

    @property
    def rate(self) -> float:
        return self.base + self.basis - self.specialness


def client_basis_bp(fixings: pd.DataFrame, window: int) -> float:
    """SOFR p75 minus median (bp), median over the latest ``window`` published fixings."""
    tail = fixings.dropna(subset=["p75", "rate"]).sort_values("timestamp").tail(window)
    if tail.empty:
        return float("nan")
    return float(((tail["p75"] - tail["rate"]) * 100).median())


def _memo(inputs: FinancingInputs, key, compute):
    """One day's inputs are immutable, so a layer value depending only on them (and its
    arguments) is computed once per key - a basket of 50 bonds x 2 delivery days asks the
    same base path and client basis 100 times."""
    cache = inputs.extra.setdefault("_memo", {})
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def _base_sofr_futures(inputs: FinancingInputs, spec, start, end, cusip) -> float:
    if inputs.sofr_path is None:
        raise ValueError("base 'sofr_futures' needs a fitted SOFR path")
    return _memo(inputs, ("sofr_futures", start, end), lambda: inputs.sofr_path.compounded(start, end))


def _basis_sofr_p75(inputs: FinancingInputs, spec, start, end, cusip) -> float:
    if inputs.sofr_fixings is None:
        raise ValueError("basis 'sofr_p75' needs the published SOFR fixings")
    return _memo(inputs, ("sofr_p75", spec.basis_window),
                 lambda: client_basis_bp(inputs.sofr_fixings, spec.basis_window) / 100.0)


def _special_lifecycle_decay(inputs: FinancingInputs, spec, start, end, cusip) -> float:
    """Expected average specialness of ``cusip`` over the term (``infra.analytics.
    specialness``); 0 for a general-collateral position (no cusip) or an untracked bond."""
    if cusip is None:
        return 0.0
    if inputs.specialness_model is None or inputs.bond_states is None:
        raise ValueError("specialness 'lifecycle_decay' needs the specialness model and bond states")
    state = inputs.bond_states.get(str(cusip))
    if state is None:
        return 0.0
    m = inputs.specialness_model
    return sa.term_specialness(state, inputs.as_of, start, end, m.profile, m.half_life_bd) / 100.0


def _zero(inputs, spec, start, end, cusip) -> float:
    return 0.0


Layer = Callable[[FinancingInputs, object, pd.Timestamp, pd.Timestamp, str | None], float]
BASE_MODELS: dict[str, Layer] = {"sofr_futures": _base_sofr_futures}
BASIS_MODELS: dict[str, Layer] = {"none": _zero, "sofr_p75": _basis_sofr_p75}
SPECIALNESS_MODELS: dict[str, Layer] = {"none": _zero, "lifecycle_decay": _special_lifecycle_decay}
# what each layer needs read (infra.pipeline.financing gathers only that)
NEEDS = {"sofr_futures": {"sofr_path"}, "sofr_p75": {"sofr_fixings"}, "lifecycle_decay": {"specialness"},
         "none": set()}


def financing_rate(inputs: FinancingInputs, name: str, spec, start, end, cusip: str | None = None) -> FinancingQuote:
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return FinancingQuote(
        model=name, start=start, end=end,
        base=BASE_MODELS[spec.base](inputs, spec, start, end, cusip),
        basis=BASIS_MODELS[spec.basis](inputs, spec, start, end, cusip),
        specialness=SPECIALNESS_MODELS[spec.specialness](inputs, spec, start, end, cusip),
    )
