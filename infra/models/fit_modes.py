"""The three ways every directional framework model turns statistics into a decision
(``fit_mode``; user decision 2026-10-07, ``infra/models/CLAUDE.md`` section 0c):

* ``exante``: NO fit - the direction is stated in the spec (A: the regressors' weights; B1: the
  ``fixed_map``; D: ``expected_sign``); the fit only reports statistics; no gate applies. Honest
  only if the direction comes from outside the data.
* ``fitted``: estimated from the fit sample and used where the spec's gates pass.
* ``prior``: estimated and gated as ``fitted``, then anything AGAINST the stated sign is set to 0 -
  the data sizes or switches off a stated direction, never flips it.
"""
from __future__ import annotations

FIT_MODES = ("exante", "fitted", "prior")


def check_fit_mode(fit_mode: str, *, has_direction: bool, what: str) -> None:
    """A valid mode; ``exante`` / ``prior`` need a stated direction."""
    if fit_mode not in FIT_MODES:
        raise ValueError(f"{what}: fit_mode {fit_mode!r} (one of {FIT_MODES})")
    if fit_mode != "fitted" and not has_direction:
        raise ValueError(f"{what}: fit_mode {fit_mode!r} needs a stated direction")


def prior_keep(fitted_sign: float, stated_sign: float) -> float:
    """``prior``: the stated sign where the fitted one agrees with it, else 0."""
    return float(stated_sign) if stated_sign != 0 and fitted_sign == stated_sign else 0.0
