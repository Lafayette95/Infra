"""Shared data preparation for any model: resample / diff / log / lag ... and fitted scaling.

A model's preparation is a chain of string steps (the nowcast's transform notation,
``name`` or ``name:arg[:arg]``), split by whether the step learns anything from data:

* **Stateless steps** (``STATELESS``) run in ``Model.prepare``: the same call prepares the
  fit sample and new data. Every one is TRAILING - a row only ever uses itself and
  earlier rows - so preparing a whole panel once, then fitting at ``as_of``, can never
  leak a later row into the fit (root CLAUDE.md 3).
* **Stateful steps** (``STATEFUL``: ``demean``, ``scale``, ``zscore``, ``winsor``) learn
  their parameters from the FIT sample in ``Model.fit`` and are frozen for ``predict``
  (as the nowcast freezes its mu/sigma). They must come after every stateless step. All
  are affine per column, so a model can map its outputs back (``Preprocessor.inverse``).

Steps:

=====================  =======================================================
``resample:FREQ[:how]``  to ``FREQ`` (pandas alias: ``W-FRI``, ``ME``, ``QE``,
                         ``1h``), ``how`` = last (default) / mean / sum / first /
                         max / min. Buckets are labelled and closed on the RIGHT, so
                         a bucket is never dated before its last input.
``log``                  natural log
``diff[:n]``             x_t - x_{t-n} (n rows; default 1)
``pct[:n]``              x_t / x_{t-n} - 1
``logdiff[:n]``          log x_t - log x_{t-n}
``lag[:n]``              x_{t-n} (a column's past as a regressor)
``mult:k`` / ``add:k``   x * k / x + k (units: ``mult:100`` for % -> bp)
``neg``                  -x
``ffill[:n]``            carry the last value forward at most n rows (a LEVEL
                         across another market's holiday; never use on changes)
``ewm_z:halflife``       (x - EWM mean) / EWM std, both trailing, in rows
``demean``               - fit-sample mean                          (stateful)
``scale``                / fit-sample std                           (stateful)
``zscore``               (x - mean) / std of the fit sample         (stateful)
``winsor:k``             clip at fit mean +- k std                  (stateful)
=====================  =======================================================

Every stateless step except ``resample`` runs through the central feature maker's LITERAL
transforms (``infra.processing.features.prep_grammar``, root CLAUDE.md 31; outputs identical -
tested): ``prep`` never knows input kinds, so a ``diff`` is always a difference.

Granularities: ``prepare(..., skip=("resample",))`` prepares new data WITHOUT the
fit's resampling - e.g. 1-minute levels run through a model fitted on daily levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STATELESS = ("resample", "log", "diff", "pct", "logdiff", "lag", "mult", "add", "neg", "ffill", "ewm_z")
STATEFUL = ("demean", "scale", "zscore", "winsor")
_HOW = ("last", "mean", "sum", "first", "max", "min")


def parse_step(step: str) -> tuple[str, list[str]]:
    name, *args = step.split(":")
    if name not in STATELESS + STATEFUL:
        raise ValueError(f"unknown prep step {step!r}; known: {STATELESS + STATEFUL}")
    return name, args


def _n(args, default=1) -> int:
    return int(args[0]) if args and args[0] != "" else default


def resample(df: pd.DataFrame, freq: str, how: str = "last") -> pd.DataFrame:
    """Right-labelled, right-closed buckets; empty buckets dropped."""
    if how not in _HOW:
        raise ValueError(f"resample how {how!r}; one of {_HOW}")
    r = df.resample(freq, label="right", closed="right")
    out = getattr(r, how)()
    return out.dropna(how="all")


def apply_stateless(df: pd.DataFrame, step: str) -> pd.DataFrame:
    """One stateless step on every column. ``resample`` is prep's own (it changes the index);
    every other step runs through the central feature maker's LITERAL transforms
    (``infra.processing.features.prep_grammar``; outputs identical - tested)."""
    name, args = parse_step(step)
    if name == "resample":
        if not args:
            raise ValueError("resample needs a frequency, e.g. resample:W-FRI")
        return resample(df, args[0], args[1] if len(args) > 1 else "last")
    if name in STATEFUL:
        raise ValueError(f"{step!r} is stateful; it runs in fit, not prepare")
    from infra.processing.features import apply, prep_grammar
    grammar = prep_grammar([step])
    return df.apply(lambda col: apply(col, grammar, "level")).astype("float64")


def run_stateless(df: pd.DataFrame, steps, *, skip=()) -> pd.DataFrame:
    """Apply the stateless part of ``steps`` in order (``skip``: step names to leave out)."""
    out = df
    for step in steps:
        name, _ = parse_step(step)
        if name in STATEFUL or name in skip:
            continue
        out = apply_stateless(out, step)
    return out


def split_steps(steps) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(stateless, stateful); raises if a stateful step comes before a stateless one."""
    names = [parse_step(s)[0] for s in steps]
    first_stateful = next((i for i, n in enumerate(names) if n in STATEFUL), len(names))
    if any(n in STATELESS for n in names[first_stateful:]):
        raise ValueError(f"stateful steps {STATEFUL} must come after every stateless step: {list(steps)}")
    return tuple(steps[:first_stateful]), tuple(steps[first_stateful:])


@dataclass
class ColumnScaling:
    """The frozen affine map of one column: ``x -> (clip(x) - shift) / scale``."""
    shift: float = 0.0
    scale: float = 1.0
    lo: float = -np.inf
    hi: float = np.inf


def _wmean_std(x: np.ndarray, w: np.ndarray | None) -> tuple[float, float]:
    ok = np.isfinite(x)
    x = x[ok]
    if not len(x):
        return np.nan, np.nan
    w = np.ones_like(x) if w is None else w[ok]
    m = float(np.average(x, weights=w))
    v = float(np.average((x - m) ** 2, weights=w)) * (w.sum() ** 2 / max(w.sum() ** 2 - (w ** 2).sum(), 1e-300))
    return m, float(np.sqrt(v))


@dataclass
class Preprocessor:
    """A step chain for a set of columns: ``steps`` for every column, ``by_column`` to
    override the chain of specific columns. ``prepare`` -> (model fits) -> ``fit`` the
    stateful steps on the fit sample -> ``transform`` / ``inverse``."""
    steps: tuple[str, ...] = ()
    by_column: dict[str, tuple[str, ...]] = field(default_factory=dict)
    scaling_: dict[str, ColumnScaling] | None = None

    def chain(self, column: str) -> tuple[str, ...]:
        return tuple(self.by_column.get(column, self.steps))

    def prepare(self, raw: pd.DataFrame, *, skip=()) -> pd.DataFrame:
        """Stateless steps, per column chain. Columns sharing a chain run together, so a
        resample happens once per chain; the outputs are outer-joined."""
        if not isinstance(raw.index, pd.DatetimeIndex):
            raise TypeError("prepare expects a DatetimeIndex (timestamp) frame")
        raw = raw.sort_index()
        groups: dict[tuple[str, ...], list[str]] = {}
        for c in raw.columns:
            stateless, _ = split_steps(self.chain(c))
            groups.setdefault(stateless, []).append(c)
        parts = [run_stateless(raw[cols], chain, skip=skip) for chain, cols in groups.items()]
        out = pd.concat(parts, axis=1) if parts else raw.iloc[:, :0]
        out = out.reindex(columns=list(raw.columns))
        out.index.name = raw.index.name or "timestamp"
        return out.replace([np.inf, -np.inf], np.nan)

    def fit(self, sample: pd.DataFrame, weights: np.ndarray | None = None) -> "Preprocessor":
        scaling = {}
        for c in sample.columns:
            _, stateful = split_steps(self.chain(c))
            sc = ColumnScaling()
            x = sample[c].to_numpy(dtype="float64")
            for step in stateful:
                name, args = parse_step(step)
                m, s = _wmean_std((np.clip(x, sc.lo, sc.hi) - sc.shift) / sc.scale, weights)
                if name == "winsor":
                    k = float(args[0]) if args else 4.0
                    # bounds in RAW units, before this column's later shift/scale
                    sc.lo, sc.hi = sc.shift + sc.scale * (m - k * s), sc.shift + sc.scale * (m + k * s)
                elif name in ("demean", "zscore"):
                    sc.shift += sc.scale * m
                if name in ("scale", "zscore"):
                    sc.scale *= s if s and np.isfinite(s) and s > 0 else 1.0
            scaling[c] = sc
        self.scaling_ = scaling
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaling_ is None:
            raise RuntimeError("Preprocessor.transform before fit")
        out = df.copy()
        for c in out.columns:
            sc = self.scaling_.get(c)
            if sc is not None:
                out[c] = (out[c].clip(sc.lo, sc.hi) - sc.shift) / sc.scale
        return out

    def inverse(self, values: pd.Series | np.ndarray, column: str, *, shift: bool = True):
        """Back to the column's prepared (pre-scaling) units; ``shift=False`` for a
        difference (a residual) - scale only."""
        sc = (self.scaling_ or {}).get(column, ColumnScaling())
        return values * sc.scale + (sc.shift if shift else 0.0)

    def params(self) -> pd.DataFrame:
        rows = []
        for c, sc in (self.scaling_ or {}).items():
            for k in ("shift", "scale", "lo", "hi"):
                rows.append({"section": "prep", "row": c, "col": k, "value": float(getattr(sc, k))})
        return pd.DataFrame(rows, columns=["section", "row", "col", "value"])

    def load_params(self, frame: pd.DataFrame) -> "Preprocessor":
        sub = frame[frame["section"] == "prep"]
        self.scaling_ = {c: ColumnScaling(**{r.col: float(r.value) for r in g.itertuples()})
                         for c, g in sub.groupby("row")}
        return self
