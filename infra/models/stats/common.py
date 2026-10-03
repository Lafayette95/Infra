"""What regressions and PCAs share: spec + preprocessing, the point-in-time fit sample,
observation weights, the in-sample flag, and tidy parameter frames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.models.base import Model
from infra.models.prep import Preprocessor

PARAM_COLUMNS = ["section", "row", "col", "value"]


def cutoff_mask(index: pd.DatetimeIndex, as_of) -> np.ndarray:
    """Rows known by ``as_of``. A date (midnight) means the END of that day: everything
    stamped that day counts; an instant means rows at or before it."""
    as_of = pd.Timestamp(as_of)
    if pd.isna(as_of):
        return np.zeros(len(index), dtype=bool)
    if as_of == as_of.normalize():
        return np.asarray(index < as_of + pd.Timedelta(days=1))
    return np.asarray(index <= as_of)


def window_mask(index: pd.DatetimeIndex, as_of, window) -> np.ndarray:
    """``cutoff_mask`` restricted to the trailing ``window``: an int = that many rows
    (applied after the cutoff), a string = a Timedelta (``"730D"``) back from ``as_of``."""
    mask = cutoff_mask(index, as_of)
    if window is None:
        return mask
    if isinstance(window, (int, np.integer)):
        keep = np.flatnonzero(mask)[-int(window):]
        out = np.zeros(len(index), dtype=bool)
        out[keep] = True
        return out
    start = pd.Timestamp(as_of) - pd.Timedelta(window)
    return mask & np.asarray(index > start)


def decay_weights(n: int, halflife: float | None) -> np.ndarray | None:
    """Exponential weights, 1 on the latest row, 1/2 ``halflife`` rows earlier; None = equal."""
    if halflife is None:
        return None
    age = np.arange(n - 1, -1, -1, dtype="float64")
    return 0.5 ** (age / float(halflife))


def effective_n(w: np.ndarray | None, n: int) -> float:
    """Kish's effective sample size of weights (n when equal)."""
    if w is None:
        return float(n)
    return float(w.sum() ** 2 / (w ** 2).sum())


def params_frame(section: str, table: pd.DataFrame | pd.Series | dict) -> pd.DataFrame:
    """A table (rows x cols), a Series or a dict -> tidy ``section, row, col, value``."""
    if isinstance(table, dict):
        table = pd.Series(table, dtype="float64")
    if isinstance(table, pd.Series):
        return pd.DataFrame({"section": section, "row": table.index.astype(str), "col": "value",
                             "value": pd.to_numeric(table, errors="coerce").to_numpy(dtype="float64")})
    long = table.stack(future_stack=True).reset_index()
    long.columns = ["row", "col", "value"]
    long["section"] = section
    long["row"], long["col"] = long["row"].astype(str), long["col"].astype(str)
    long["value"] = pd.to_numeric(long["value"], errors="coerce").astype("float64")
    return long[PARAM_COLUMNS]


def section(params: pd.DataFrame, name: str) -> pd.DataFrame:
    """One section of a tidy params frame back to its rows x cols table."""
    sub = params[params["section"] == name]
    return sub.pivot_table(index="row", columns="col", values="value", aggfunc="last", sort=False)


def pick(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The ``<kind>:<name>`` columns of a predict frame, renamed to ``<name>``."""
    cols = [c for c in frame.columns if isinstance(c, str) and c.startswith(kind + ":")]
    return frame[cols].rename(columns=lambda c: c.split(":", 1)[1])


class StatModel(Model):
    """Base of ``Regression`` and ``PCA``: holds ``spec`` and a ``Preprocessor``."""

    spec_type: type = object

    def __init__(self, spec):
        self.spec = spec
        self.prep = Preprocessor(tuple(spec.prep), {c: tuple(s) for c, s in spec.prep_by_column})
        self.fitted_ = None

    # subclasses say which input columns they use
    def input_columns(self, raw: pd.DataFrame) -> list[str]:
        return list(raw.columns)

    def prepare(self, raw: pd.DataFrame, *, skip=()) -> pd.DataFrame:
        """Stateless prep of the model's input columns (``infra.models.prep``). ``skip``
        leaves steps out by name - ``skip=("resample",)`` runs new data at its own
        granularity through a model fitted on resampled data."""
        cols = self.input_columns(raw)
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            raise KeyError(f"{type(self).__name__}: input lacks columns {missing}")
        return self.prep.prepare(raw[cols], skip=skip)

    def fit_sample(self, prepared: pd.DataFrame, as_of=None) -> tuple[pd.DataFrame, pd.Timestamp]:
        """Rows up to ``as_of`` (default: the last row) inside the spec's ``window``."""
        as_of = prepared.index.max() if as_of is None else pd.Timestamp(as_of)
        return prepared[window_mask(prepared.index, as_of, self.spec.window)], as_of

    def in_sample(self, index: pd.DatetimeIndex) -> np.ndarray:
        return cutoff_mask(index, self.fitted_.as_of)

    def predict_window(self, prepared: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
        """Rows strictly after ``start`` and up to ``end`` (both point-in-time cutoffs;
        None = open)."""
        keep = np.ones(len(prepared), dtype=bool)
        if start is not None:
            keep &= ~cutoff_mask(prepared.index, start)
        if end is not None:
            keep &= cutoff_mask(prepared.index, end)
        return prepared[keep]
