"""The GATES deciding that X's conditioning is real (in sample, at each fit, point in time).

Each gate reads the fit's statistics (``model.AutocorrFit.stats``) and its threshold from the
spec, and returns ``(value, passed)``. A spec lists which gates apply (``AutocorrSpec.gates``) -
loosening = naming fewer gates or lower thresholds; a new gate = a new registry entry.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

Gate = Callable[[dict, float], "tuple[float, bool]"]


def _abs_at_least(key: str) -> Gate:
    def gate(stats: dict, th: float):
        v = stats.get(key, np.nan)
        return v, bool(np.isfinite(v) and abs(v) >= th)
    return gate


def _min_obs(stats: dict, th: float):
    return stats["n"], bool(stats["n"] >= th)


def _monotonic(stats: dict, th: float):
    m = [stats.get(f"bucket_mean_{b}", np.nan) for b in (-1, 0, 1)]
    ok = all(np.isfinite(m)) and (m[0] <= m[1] <= m[2] or m[0] >= m[1] >= m[2])
    return float(ok), ok


GATES: dict[str, Gate] = {
    "min_obs": _min_obs,                                # completed forward windows in the fit sample
    "x_t": _abs_at_least("x_t"),                        # chase ~ X: |t| of X (HAC) - the interaction d
    "bucket_spread_t": _abs_at_least("spread_t"),       # top-minus-bottom X bucket, mean chase P&L (HAC)
    "beats_controls": _abs_at_least("ctrl_t"),          # X's t with the vol / |past move| controls in
    "halves_agree": lambda s, th: (s.get("halves", np.nan), bool(s.get("halves", 0.0) >= th)),
    "monotonic": _monotonic,                            # the three bucket means ordered
}


def run_gates(stats: dict, names, thresholds: dict) -> tuple[dict, bool]:
    """``({gate: value}, all passed)``; an unknown gate name raises."""
    out, ok = {}, True
    for g in names:
        if g not in GATES:
            raise KeyError(f"unknown gate {g!r}; known: {sorted(GATES)}")
        v, p = GATES[g](stats, thresholds.get(g, 0.0))
        out[g], ok = v, ok and p
    return out, ok
