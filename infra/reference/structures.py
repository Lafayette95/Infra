"""Curve STRUCTURES on the US Treasury futures: what each one is (static facts; root CLAUDE.md 28).

NAMES are ``KIND__LEG__LEG[__H]``, every token separated by a DOUBLE underscore (``CURVE__FV__WN``,
``FRONT__TU__H``; ``H`` = hedged): a single ``_`` lives inside instrument ids (``US_BOND_10y``,
the cash curve's tickers), so ``__`` is the only unambiguous separator (user decision 2026-10-05).
``parse_name`` splits one.

A structure is a vector of DV01 weights on relative futures tickers (``ZN.v.0``), named with the
user's Bloomberg roots (TU = ZT, FV = ZF, TY = ZN, UXY = TN, US = ZB, WN = UB). One unit of a
structure = $1 of DV01 on each leg times its weight, so its daily P&L in bp is the weighted sum
of the legs' bp moves (price-return per unit DV01, + = the leg rallied). Signs: a CURVE is long
the front leg, short the back one (a steepener); a FLY is long the belly, short 50/50 wings.

A HEDGED structure (``hedge_layer``) is its base legs minus point-in-time regression betas times
its set's structures of that layer (``infra.analytics.structures.hedged_weights``): what it contains beyond them.
Weights and betas are computed per decision day from data known by then
(``infra.pipeline.structures``). Layers and the default set were decided 2026-10-05 (user;
diagnostics in ``infra/strategies/CLAUDE.md`` 5):

* ``front``  - FRONT__TU__H: TU hedged on the macro layer. The front end is its own, jumpier factor
  (kurtosis 35 after hedging, 22% of its variance on 5 days), so it gets its own, smaller budget.
* ``macro``  - DUR__TY (duration; DUR__UXY an alternative), CURVE__FV__WN, FLY__FV__UXY__WN, all
  conventional DV01 weights (user decision: conventional; the fly keeps 13% level content).
* ``micro``  - MICRO__TY__FV__H, MICRO__US__WN__H, hedged on the macro layer (micro = residual to macro).
  TY-FV and TY-UXY are the SAME micro dimension once hedged (the macro set already uses FV, TY,
  UXY, WN), so only one of them is in the set.

Together the six span all six contracts (``STRUCTURE_SETS["ust_layers"]``), so per-future views
and structure views map into each other exactly (``infra.analytics.structures.basis_matrix``).
"""
from __future__ import annotations

from dataclasses import dataclass

BBG_FUTURES = {"TU": "ZT.v.0", "FV": "ZF.v.0", "TY": "ZN.v.0", "UXY": "TN.v.0", "US": "ZB.v.0", "WN": "UB.v.0"}


SEP = "__"


def parse_name(name: str) -> tuple[str, list[str], bool]:
    """``KIND__LEG__LEG[__H]`` -> (kind, legs, hedged)."""
    parts = name.split(SEP)
    hedged = parts[-1] == "H"
    return parts[0], parts[1:-1] if hedged else parts[1:], hedged


@dataclass(frozen=True)
class Structure:
    name: str
    layer: str                                   # front | macro | micro
    legs: tuple[tuple[str, float], ...]          # (Bloomberg root, DV01 weight)
    hedge_layer: str | None = None               # regressed out (point in time): the SET's structures of this layer
    description: str = ""

    def __post_init__(self):
        kind, legs, hedged = parse_name(self.name)
        if sorted(legs) != sorted(root for root, _ in self.legs) or hedged != (self.hedge_layer is not None):
            raise ValueError(f"structure name {self.name!r} doesn't match its legs / hedge (KIND__LEG__LEG[__H])")

    def base_weights(self) -> dict[str, float]:
        """DV01 weights on relative futures tickers, before any hedge."""
        return {BBG_FUTURES[root]: w for root, w in self.legs}


STRUCTURES: dict[str, Structure] = {s.name: s for s in (
    Structure("DUR__TY", "macro", (("TY", 1.0),), description="duration: long TY (default anchor; user 2026-10-05)"),
    Structure("DUR__UXY", "macro", (("UXY", 1.0),),
              description="duration in UXY (alternative anchor: 25% cheaper per bp, 3x thinner book, specialness)"),
    Structure("CURVE__FV__WN", "macro", (("FV", 1.0), ("WN", -1.0)), description="curve steepener: long FV, short WN"),
    Structure("FLY__FV__UXY__WN", "macro", (("UXY", 1.0), ("FV", -0.5), ("WN", -0.5)),
              description="fly: long UXY belly, short 50/50 FV / WN wings (conventional DV01 weights)"),
    Structure("FRONT__TU__H", "front", (("TU", 1.0),), hedge_layer="macro",
              description="front end: long TU, hedged on duration / curve / fly"),
    Structure("MICRO__TY__FV__H", "micro", (("TY", 1.0), ("FV", -1.0)), hedge_layer="macro",
              description="micro: TY vs FV, hedged on the macro layer (== TY vs UXY once hedged)"),
    Structure("MICRO__US__WN__H", "micro", (("US", 1.0), ("WN", -1.0)), hedge_layer="macro",
              description="micro: US vs WN, hedged on the macro layer"),
)}


@dataclass(frozen=True)
class StructureSet:
    name: str
    structures: tuple[str, ...]
    hedge_window: int = 250                      # daily rows behind each hedge regression
    hedge_min_obs: int = 120
    description: str = ""

    def layers(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for s in self.structures:
            out.setdefault(STRUCTURES[s].layer, []).append(s)
        return out

    def hedge_targets(self, structure: str) -> list[str]:
        layer = STRUCTURES[structure].hedge_layer
        return [] if layer is None else [s for s in self.structures if STRUCTURES[s].layer == layer
                                         and STRUCTURES[s].hedge_layer is None]

    def legs(self) -> list[str]:
        seen = []
        for s in self.structures:
            for leg in STRUCTURES[s].base_weights():
                if leg not in seen:
                    seen.append(leg)
        return seen


STRUCTURE_SETS: dict[str, StructureSet] = {s.name: s for s in (
    StructureSet("ust_layers", ("DUR__TY", "CURVE__FV__WN", "FLY__FV__UXY__WN", "FRONT__TU__H", "MICRO__TY__FV__H",
                                "MICRO__US__WN__H"),
                 description="US Treasury futures in layers: front (hedged TU) / macro / micro (user 2026-10-05)"),
    StructureSet("ust_layers_uxy", ("DUR__UXY", "CURVE__FV__WN", "FLY__FV__UXY__WN", "FRONT__TU__H", "MICRO__TY__FV__H",
                                    "MICRO__US__WN__H"), description="the same with duration in UXY"),
)}


# --------------------------------------------------------------------------- yield structures
YIELD_KIND_WEIGHTS = {"CURVE": (1.0, -1.0), "FLY": (-0.5, 1.0, -0.5)}


def yield_structure_weights(name: str) -> dict[str, float]:
    """A structure on YIELD tickers -> DV01 weights per ticker, legs listed front to back:
    ``CURVE__US_BOND_5y__US_BOND_30y`` is long the front leg and short the back (a steepener: P&L
    in bp = the change of back minus front yield); ``FLY__<front>__<belly>__<back>`` is long the
    belly against 50/50 wings. Each leg's benchmark P&L is -dy in bp, so the structure's P&L is
    the weighted sum (root CLAUDE.md 28-29). Futures structures are ``STRUCTURES`` instead."""
    kind, legs, hedged = parse_name(name)
    if hedged or kind not in YIELD_KIND_WEIGHTS or len(legs) != len(YIELD_KIND_WEIGHTS[kind]):
        raise ValueError(f"{name!r}: a yield structure is CURVE__<front>__<back> or FLY__<front>__<belly>__<back>")
    if not all("_BOND_" in leg for leg in legs):
        raise ValueError(f"{name!r}: legs must be yield tickers (<COUNTRY>_BOND_<t>y)")
    return dict(zip(legs, YIELD_KIND_WEIGHTS[kind]))
