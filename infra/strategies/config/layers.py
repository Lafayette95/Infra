"""Layered sizing configurations (``infra.strategies.layered``): views per curve STRUCTURE
(``infra.reference.structures``) -> a risk budget per layer -> contracts per future, netted.
Named, picked by ``StrategySpec.layers``. Budgets are placeholders until a strategy is
calibrated; the front end's is smaller on purpose (user decision 2026-10-05: a jumpier factor,
a lower vol budget for now - a tail-based budget is the planned replacement, TOFIX)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class LayerSizingSpec:
    name: str
    structure_set: str = "ust_layers"           # infra.reference.structures.STRUCTURE_SETS
    budgets: dict = field(default_factory=lambda: {"macro": 1_000_000.0, "front": 300_000.0, "micro": 500_000.0})
    portfolio_cap: float | None = 1_500_000.0    # $ annual vol of the netted book (full covariance); None = no cap
    vol_span: int = 60                           # EWMA span of each structure's daily bp moves
    cov_span: int = 120                          # EWMA span of the legs' covariance (portfolio check)
    description: str = ""


LAYER_SIZINGS: dict[str, LayerSizingSpec] = {s.name: s for s in (
    LayerSizingSpec("ust_layers", description="front / macro / micro on US Treasury futures, TY duration"),
    LayerSizingSpec("ust_layers_uxy", structure_set="ust_layers_uxy", description="the same, duration in UXY"),
)}


def get_layer_sizing(spec: LayerSizingSpec | str, **overrides) -> LayerSizingSpec:
    base = LAYER_SIZINGS[spec] if isinstance(spec, str) else spec
    return replace(base, **overrides) if overrides else base
