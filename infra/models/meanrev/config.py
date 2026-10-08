"""Mean reversion of factor-model residuals: named specs, toggled by name (``infra/models/CLAUDE.md`` 0;
methodology ``infra/models/meanrev/CLAUDE.md``).

A spec: the FACTOR model whose residuals are traded (``factor`` = ``pca`` - incl. the ``similarity`` and
``weighted`` methods - or ``regime_pca``, a spec name and overrides), the K instruments (``columns``) and
the regime model's N (``regime_columns``), the residual LEVEL window, the OU estimate (per residual or
pooled), the gates, the ``fit_mode`` (``infra.models.fit_modes``: the stated direction is REVERSION), the
signal rule and the trade-metric settings.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

GATES = ("min_obs", "reverting", "adf_t", "half_life", "halves")


@dataclass(frozen=True)
class MeanRevSpec:
    name: str = "custom"
    description: str = ""
    # the factor model (its residuals are the traded series)
    factor: str = "pca"                           # pca | regime_pca
    factor_spec: str = "curve_changes"            # a PCA_MODELS / REGIME_PCA_MODELS name
    factor_overrides: tuple[tuple[str, object], ...] = (("prep", ()), ("n_components", 3))
    columns: tuple[str, ...] = ()                 # the K instruments (daily moves, bp; + = long gains)
    regime_columns: tuple[str, ...] = ()          # regime_pca: the N series (also loaded for similarity states)
    # the residual level: each fit re-applies ITS factor fit to the last `window` rows and cumulates
    window: int = 120                             # rows
    # OU estimate
    pooling: str = "none"                         # none (per residual) | pooled (one b for all residuals)
    bias_correct: bool = True                     # Kendall's small-sample correction of b
    # decision (infra.models.fit_modes): exante = fade the level's z over the window, no OU, no gates;
    # fitted = OU s-score where every gate passes; prior = OU s-score where it reverts at all (b < 1)
    fit_mode: str = "fitted"
    gates: tuple[str, ...] = ("min_obs", "reverting", "adf_t", "half_life", "halves")
    thresholds: tuple[tuple[str, float], ...] = (("min_obs", 60.0), ("adf_t", 2.0), ("half_life_min", 1.0),
                                                 ("half_life_max", 30.0))
    # signal per residual (fading: short a high s-score): linear = -s / entry clipped to [-1, 1];
    # threshold = open at |s| >= entry, close at |s| <= exit (path from the fit date, flat at the start)
    signal_rule: str = "linear"
    entry: float = 1.5
    exit: float = 0.5
    cross_section: str = "none"                   # none | demean (s minus its mean over the tradable residuals)
    risk_scale: bool = True                       # size each residual by 1 / its daily innovation sd
    # trade metrics (per row, per residual)
    horizon_days: int = 5                         # expected move / sd / Sharpe horizon
    stop_width: float = 1.0                       # stop at |s| + this (s-score units), target at |s| = exit
    cost_bp: float = 0.0                          # round-trip cost per residual unit, bp (optimal entry)
    # evaluation
    gap_days: int = 1                             # rows from the decision row to the first traded move

    def __post_init__(self):
        from infra.models.fit_modes import check_fit_mode
        check_fit_mode(self.fit_mode, has_direction=True, what=f"meanrev {self.name!r}")   # reversion is stated
        if self.factor not in ("pca", "regime_pca"):
            raise ValueError(f"meanrev {self.name!r}: factor {self.factor!r} (pca | regime_pca)")
        if self.pooling not in ("none", "pooled") or self.signal_rule not in ("linear", "threshold") \
                or self.cross_section not in ("none", "demean"):
            raise ValueError(f"meanrev {self.name!r}: pooling / signal_rule / cross_section")
        bad = [g for g in self.gates if g not in GATES]
        if bad:
            raise ValueError(f"meanrev {self.name!r}: unknown gates {bad}; known {GATES}")

    def threshold(self, key: str, default: float = 0.0) -> float:
        return dict(self.thresholds).get(key, default)


MEANREV_MODELS: dict[str, MeanRevSpec] = {s.name: s for s in (
    MeanRevSpec("default", "plain PCA residuals, per-residual OU, every gate, linear fade"),
    MeanRevSpec("pooled", "one reversion speed for all residuals of the fit", pooling="pooled"),
    MeanRevSpec("prior", "reversion stated: trade every residual whose b < 1, sized by the OU s-score",
                fit_mode="prior", gates=("min_obs", "reverting")),
    MeanRevSpec("exante", "no OU: fade the level's z over the window", fit_mode="exante", gates=()),
    MeanRevSpec("threshold", "Avellaneda-Lee style: open at |s| >= 1.25, close at |s| <= 0.5",
                signal_rule="threshold", entry=1.25, exit=0.5),
)}


def get_meanrev_spec(spec: MeanRevSpec | str | None = None, **overrides) -> MeanRevSpec:
    base = MEANREV_MODELS["default"] if spec is None else (MEANREV_MODELS[spec] if isinstance(spec, str) else spec)
    if isinstance(overrides.get("factor_overrides"), dict):
        overrides["factor_overrides"] = tuple(overrides["factor_overrides"].items())
    return replace(base, **overrides) if overrides else base
