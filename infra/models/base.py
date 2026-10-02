"""The shape every model in ``infra/models`` follows: prepare -> fit -> predict.

1. ``prepare(raw)``: turn whatever the inputs are (prices, releases, ...) into the
   model's own aligned, cleaned, standardised form. Pure, deterministic, no fitting:
   the SAME call prepares the fit sample and the out-of-sample data, so the two can
   never be prepared differently.
2. ``fit(prepared, as_of=None)``: estimate the slow parameters from data up to
   ``as_of`` only (point in time, root CLAUDE.md 3) and keep them, plus whatever state
   ``predict`` needs to continue from the fit date. Returns ``self`` (sklearn style;
   fitted attributes end in ``_``). This is the heavy step, run on a schedule
   (daily/weekly).
3. ``predict(prepared, ...)``: the light step. Holds the fitted parameters fixed and
   runs on new data only - after the fit date, and possibly at a different granularity
   (an intraday price on a daily-fitted model).

A model instance is configured by a spec (its parameters, kept in a config registry so
several parametrisations can sit side by side and be toggled by name) and is otherwise
agnostic about where its inputs came from: reading stored data is a separate ``inputs``
module per model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class NotFittedError(RuntimeError):
    """``predict`` (or anything needing fitted parameters) called before ``fit``."""


class Model(ABC):
    """Base class: ``prepare`` -> ``fit`` -> ``predict``. See the module docstring."""

    @abstractmethod
    def prepare(self, raw: Any, **kwargs) -> Any:
        """Inputs -> the model's prepared form. Pure; no fitted state used or set."""

    @abstractmethod
    def fit(self, prepared: Any, as_of=None) -> "Model":
        """Estimate parameters from ``prepared`` up to ``as_of``; return ``self``."""

    @abstractmethod
    def predict(self, prepared: Any, **kwargs) -> Any:
        """Outputs on new data with the fitted parameters held fixed."""

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "fitted_", None) is not None

    def check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")
