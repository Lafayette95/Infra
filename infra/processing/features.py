"""The central FEATURE MAKER, pure part: a feature grammar and its step functions
(spec: ``infra/processing/FEATURES.md``; root CLAUDE.md 31). Point in time by construction:
every step is TRAILING (a value at t uses rows up to t only), with explicit warm-up (NaN until
enough history). Anything needing a fit sample (a z-score on the fit window, a fitted beta) is
NOT a feature step - it belongs in a model's ``fit``.

A feature string: a KIND, then MODIFIERS, separated by ``|``::

    chg:20 | norm:vol:60 | abs | part:tercile:504
    x:5:20 | norm:z:252

``apply(x, spec, kind)``: ``x`` indexed by availability instant (or any ordered index), ``kind``
"level" (yields, positioning, indices) or "moves" (P&L, returns). A change is a DIFFERENCE of a
level and a SUM of moves; the level of moves is their cumulative sum.

Kinds: ``lvl``; ``gap:ewm:HL`` (level - EWMA, half-life HL); ``chg:N`` (N-observation change,
boxcar); ``chg:ewm:HL`` (EWMA of daily changes); ``x:F:S`` (EWMA(level, F) - EWMA(level, S));
``acc:N:M`` (chg:N now - chg:N M observations ago); ``accr:F:S`` (per-observation change over F
minus over S, F < S); ``range:N`` (position in the trailing range, 0..1); ``dd:N`` (drawdown from
the trailing high, <= 0).

Literal transforms (act on the CURRENT series, whatever the input kind - ``infra.models.prep``'s
stateless steps map one-to-one onto them): ``diff:N``, ``log``, ``pct:N``, ``logdiff:N``,
``mult:K``, ``add:K``, ``neg``, ``ffill:N``.

Modifiers: ``norm:vol:SPAN`` (/ EWMA vol of daily changes x the kind's scale: sqrt N for chg:N -
the user's choice, 2026-10-06: it assumes independent daily changes; the EWMA-variance factor for
chg:ewm; sqrt(1/F - 1/S) for accr; sqrt(2N) for acc; 1 otherwise), ``norm:z:W`` (demeaned),
``norm:z0:W`` (not demeaned: / RMS), ``norm:ewmz:HL`` (EWM mean / std), ``norm:rank:W`` (trailing percentile 0..1),
``norm:robust:W`` (median / MAD); ``abs``, ``sign``, ``clip:K``, ``pow:P`` (signed power),
``lag:N``; partitions ``part:tercile:W``, ``part:z:W:K``, ``part:sign``, ``part:fixed:A:B``
(-1 / 0 / +1).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

KINDS_OF_INPUT = ("level", "moves")


@dataclass
class _State:
    """What a step passes on: the series, the input's daily changes (for vol), the kind's scale."""
    x: pd.Series
    d: pd.Series
    scale: float = 1.0


def parse(spec: str) -> list[tuple[str, list[str]]]:
    """``"chg:20 | norm:vol:60"`` -> ``[("chg", ["20"]), ("norm", ["vol", "60"])]``."""
    out = []
    for part in [p.strip() for p in spec.split("|") if p.strip()]:
        name, *args = part.split(":")
        out.append((name, args))
    return out


def _level_and_changes(x: pd.Series, kind: str) -> tuple[pd.Series, pd.Series]:
    if kind not in KINDS_OF_INPUT:
        raise ValueError(f"input kind {kind!r}; one of {KINDS_OF_INPUT}")
    x = x.astype("float64")
    if kind == "moves":
        return x.cumsum(), x
    return x, x.diff()


def _ewm_factor(hl: float) -> float:
    lam = 0.5 ** (1.0 / hl)
    return float(np.sqrt((1 - lam) / (1 + lam)))


# --------------------------------------------------------------------------- kinds
def _k_lvl(x, kind, args):
    lvl, d = _level_and_changes(x, kind)
    return _State(lvl, d)


def _k_gap(x, kind, args):
    if args[0] != "ewm":
        raise ValueError("gap:ewm:HL")
    hl = float(args[1])
    lvl, d = _level_and_changes(x, kind)
    return _State(lvl - lvl.ewm(halflife=hl, min_periods=int(np.ceil(hl))).mean(), d)


def _k_chg(x, kind, args):
    lvl, d = _level_and_changes(x, kind)
    if args[0] == "ewm":
        hl = float(args[1])
        return _State(d.ewm(halflife=hl, min_periods=int(np.ceil(hl))).mean(), d, _ewm_factor(hl))
    n = int(args[0])
    c = d.rolling(n, min_periods=n).sum() if kind == "moves" else lvl - lvl.shift(n)
    return _State(c, d, float(np.sqrt(n)))


def _k_x(x, kind, args):
    f, s = float(args[0]), float(args[1])
    lvl, d = _level_and_changes(x, kind)
    return _State(lvl.ewm(halflife=f, min_periods=int(np.ceil(f))).mean()
                  - lvl.ewm(halflife=s, min_periods=int(np.ceil(s))).mean(), d)


def _k_acc(x, kind, args):
    n, m = int(args[0]), int(args[1])
    c = _k_chg(x, kind, [str(n)])
    return _State(c.x - c.x.shift(m), c.d, float(np.sqrt(2 * n)))


def _k_accr(x, kind, args):
    f, s = int(args[0]), int(args[1])
    if f >= s:
        raise ValueError("accr:F:S needs F < S")
    cf, cs = _k_chg(x, kind, [str(f)]), _k_chg(x, kind, [str(s)])
    return _State(cf.x / f - cs.x / s, cf.d, float(np.sqrt(1.0 / f - 1.0 / s)))


def _k_range(x, kind, args):
    n = int(args[0])
    lvl, d = _level_and_changes(x, kind)
    lo, hi = lvl.rolling(n, min_periods=n).min(), lvl.rolling(n, min_periods=n).max()
    return _State(((lvl - lo) / (hi - lo)).where(hi > lo), d)


def _k_dd(x, kind, args):
    n = int(args[0])
    lvl, d = _level_and_changes(x, kind)
    return _State(lvl - lvl.rolling(n, min_periods=n).max(), d)


KIND_STEPS: dict[str, Callable] = {"lvl": _k_lvl, "gap": _k_gap, "chg": _k_chg, "x": _k_x, "acc": _k_acc,
                                   "accr": _k_accr, "range": _k_range, "dd": _k_dd}


# --------------------------------------------------------------------------- timelines (moved from conditions)
def state_timeline(rows: pd.DataFrame, *, period_diff: bool = False) -> pd.DataFrame:
    """``available_at, label, value`` (one row per availability instant, the state then)."""
    if rows is None or rows.empty:
        return pd.DataFrame(columns=["available_at", "label", "value"])
    r = rows.sort_values(["available_at", "label"], kind="stable")
    known: dict = {}
    out = []
    for t, g in r.groupby("available_at", sort=True):
        for lab, v in zip(g["label"], g["value"]):
            known[pd.Timestamp(lab)] = v
        labels = sorted(known)
        last = labels[-1]
        val = known[last]
        if period_diff:
            val = known[last] - known[labels[-2]] if len(labels) > 1 else np.nan
        out.append((t, last, val))
    return pd.DataFrame(out, columns=["available_at", "label", "value"])


# --------------------------------------------------------------------------- partitions (moved from conditions)
def rolling_tercile(x: pd.Series, w: str) -> np.ndarray:
    w = int(w)
    q1 = x.rolling(w, min_periods=max(w // 2, 3)).quantile(1 / 3)
    q2 = x.rolling(w, min_periods=max(w // 2, 3)).quantile(2 / 3)
    out = np.where(x < q1, -1.0, np.where(x > q2, 1.0, 0.0))
    return np.where(q1.isna() | x.isna(), np.nan, out)


def zscore_bands(x: pd.Series, w: str, k: str = "1") -> np.ndarray:
    w, k = int(w), float(k)
    m = x.rolling(w, min_periods=max(w // 2, 3)).mean()
    sd = x.rolling(w, min_periods=max(w // 2, 3)).std()
    z = (x - m) / sd
    out = np.where(z < -k, -1.0, np.where(z > k, 1.0, 0.0))
    return np.where(z.isna(), np.nan, out)


def sign_bucket(x: pd.Series) -> np.ndarray:
    return np.where(x.isna(), np.nan, np.sign(x.to_numpy(dtype="float64")))


def fixed_bands(x: pd.Series, a: str, b: str) -> np.ndarray:
    a, b = float(a), float(b)
    out = np.where(x < a, -1.0, np.where(x > b, 1.0, 0.0))
    return np.where(x.isna(), np.nan, out)


PARTITIONS: dict[str, Callable] = {"tercile": rolling_tercile, "rolling_tercile": rolling_tercile,
                                   "z": zscore_bands, "zscore": zscore_bands, "sign": sign_bucket,
                                   "fixed": fixed_bands}


# --------------------------------------------------------------------------- modifiers
def _norm(st: _State, args):
    how = args[0]
    x = st.x
    if how == "vol":
        span = int(args[1])
        vol = st.d.ewm(span=span, min_periods=max(span // 2, 10)).std()
        return x / (vol * st.scale)
    w = int(args[1])
    mp = max(w // 2, 3)
    if how == "z":
        return (x - x.rolling(w, min_periods=mp).mean()) / x.rolling(w, min_periods=mp).std()
    if how == "z0":
        return x / np.sqrt((x ** 2).rolling(w, min_periods=mp).mean())
    if how == "ewmz":                            # EWM mean / std (prep's ewm_z, same warm-up)
        hl = float(args[1])
        m = x.ewm(halflife=hl, min_periods=max(2, int(hl))).mean()
        sd = x.ewm(halflife=hl, min_periods=max(2, int(hl))).std()
        return (x - m) / sd
    if how == "rank":
        return x.rolling(w, min_periods=mp).rank(pct=True)
    if how == "robust":
        med = x.rolling(w, min_periods=mp).median()
        mad = (x - med).abs().rolling(w, min_periods=mp).median()
        return (x - med) / (1.4826 * mad)
    raise ValueError(f"norm:{how} (vol | z | z0 | ewmz | rank | robust)")


def _modify(st: _State, name: str, args) -> pd.Series:
    x = st.x
    if name == "norm":
        return _norm(st, args)
    if name == "abs":
        return x.abs()
    if name == "sign":
        return np.sign(x)
    if name == "clip":
        k = float(args[0])
        return x.clip(-k, k)
    if name == "pow":
        p = float(args[0])
        return np.sign(x) * x.abs() ** p
    if name == "lag":
        return x.shift(int(args[0]))
    # LITERAL transforms of the current series, whatever the input kind (infra.models.prep's
    # stateless steps map one-to-one onto these - user decision 2026-10-06: prep stays literal)
    n = int(args[0]) if args and name in ("diff", "pct", "logdiff") else 1
    if name == "diff":
        return x - x.shift(n)
    if name == "log":
        return np.log(x.where(x > 0))
    if name == "pct":
        return x / x.shift(n) - 1.0
    if name == "logdiff":
        lg = np.log(x.where(x > 0))
        return lg - lg.shift(n)
    if name == "mult":
        return x * float(args[0])
    if name == "add":
        return x + float(args[0])
    if name == "neg":
        return -x
    if name == "ffill":
        return x.ffill(limit=int(args[0]) if args else None)
    if name == "part":
        rule, *rest = args
        if rule not in PARTITIONS:
            raise KeyError(f"unknown partition {rule!r}; known: {sorted(PARTITIONS)}")
        return pd.Series(PARTITIONS[rule](x, *rest), index=x.index)
    raise KeyError(f"unknown feature step {name!r}")


def apply(x: pd.Series, spec: str, kind: str = "level") -> pd.Series:
    """The feature ``spec`` of series ``x`` (see the module docstring). Output keeps ``x``'s index."""
    steps = parse(spec)
    if not steps or steps[0][0] not in KIND_STEPS:
        raise ValueError(f"feature {spec!r} must start with a kind: {sorted(KIND_STEPS)}")
    name, args = steps[0]
    st = KIND_STEPS[name](x, kind, args)
    for name, args in steps[1:]:
        st = _State(_modify(st, name, args), st.d, st.scale)
    out = st.x.astype("float64")
    out.attrs = {"feature": spec, "input_kind": kind}
    return out


def partition(x: pd.Series, rule: str) -> np.ndarray:
    """Legacy entry (``rolling_tercile:504``): the partition step on its own."""
    name, *args = rule.split(":")
    if name not in PARTITIONS:
        raise KeyError(f"unknown partition {rule!r}; known: {sorted(PARTITIONS)}")
    return PARTITIONS[name](x, *args)


PREP_ALIASES = {"diff": "diff", "log": "log", "pct": "pct", "logdiff": "logdiff", "lag": "lag", "mult": "mult",
                "add": "add", "neg": "neg", "ffill": "ffill", "ewm_z": "norm:ewmz"}


def prep_grammar(steps) -> str:
    """A chain of ``infra.models.prep`` stateless steps (``("diff", "mult:100")``) as the grammar,
    LITERAL: ``lvl | diff:1 | mult:100``, to be applied with input kind ``level`` (prep never knew
    kinds - a ``diff`` is a difference even of a P&L series; user decision 2026-10-06)."""
    out = ["lvl"]
    for step in steps:
        name, _, rest = step.partition(":")
        if name not in PREP_ALIASES:
            raise ValueError(f"prep step {step!r} has no feature alias (stateful steps and resample stay in prep)")
        alias = PREP_ALIASES[name]
        if name in ("diff", "pct", "logdiff", "lag") and not rest:
            rest = "1"
        out.append(f"{alias}:{rest}" if rest else alias)
    return " | ".join(out)
