"""CEVT: event-study views from one or more event-family runs (root CLAUDE.md 27).

Input: each family run's predictions (one row per code and event window, from the walk-forward:
each row carries the fit current at its start - ``fit_as_of`` - its per-instrument ``t:``,
``expected:``, ``passed:`` and, conditional, ``cond_passed:`` / ``cond_provisional``) and its
point-in-time FDR table. Output per label (a grid point = a decision time) and instrument:

* ``tsig:<inst>`` - the t-signal: each code's t mapped to [-1, 1] by
  ``tanh(a * clip(t, -cap, cap)) / tanh(a * cap)`` (full strength at the cap), aggregated;
* ``ev:<inst>`` - the raw expected move (the source's units, bp), aggregated the same way;
* ``n:<inst>`` - codes contributing; ``provisional:<inst>`` - 1 if any contributing view rests
  on a provisional regime (a conditional event not yet started).

A code's view is held over the labels its window covers: decision at the window start, held
to its end - labels ``start <= L < end`` (``base`` labelling). Codes failing the ``gate``
are left out (or count as zeros with ``include_failing``). Within a family, the views active at
a label combine by ``within`` (``AGGREGATORS``); across families, the mean over the families
active there (``pool_families``: one stage over all codes).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from infra.strategies.base import Strategy
from infra.strategies.config.cevt import CEVTSpec, get_cevt_spec


def t_signal(t, cap: float = 3.0, a: float = 0.5) -> np.ndarray:
    """``tanh(a * clip(t, -cap, cap)) / tanh(a * cap)``: concave, |1| at the cap."""
    t = np.clip(np.asarray(t, dtype="float64"), -cap, cap)
    return np.tanh(a * t) / np.tanh(a * cap)


# --------------------------------------------------------------------------- aggregators
def _parts(v: pd.Series, keys):
    """Per row, its group's: sum, sum |v|, count, agreement, and the largest |v| on the
    majority side (signed)."""
    s = v.groupby(keys).transform("sum")
    a = v.abs().groupby(keys).transform("sum")
    n = v.groupby(keys).transform("size")
    major = np.sign(s)
    smax = v.abs().where(np.sign(v) == major, 0.0).groupby(keys).transform("max") * major
    agree = (s.abs() / a).where(a > 0, 0.0)
    return s, n, agree, smax


def _smooth(v: pd.Series, keys) -> pd.Series:
    """agreement a = |sum v| / sum |v| (1 all agree, 0 evenly split); combined =
    a * (the largest |v| on the majority side, signed) + (1 - a) * mean(v)."""
    s, n, agree, smax = _parts(v, keys)
    return (agree * smax + (1 - agree) * (s / n)).groupby(keys).first()


def _mean(v, keys) -> pd.Series:
    return v.groupby(keys).mean()


def _absmax(v, keys) -> pd.Series:
    return v.groupby(keys).agg(lambda x: x.loc[x.abs().idxmax()] if len(x) else np.nan)


def _threshold(v, keys, cut: str = "0.8") -> pd.Series:
    """The largest |v| on the majority side if the agreement is at least ``cut``, else the mean."""
    s, n, agree, smax = _parts(v, keys)
    return smax.where(agree >= float(cut), s / n).groupby(keys).first()


AGGREGATORS: dict[str, Callable] = {"smooth": _smooth, "mean": _mean, "absmax": _absmax, "threshold": _threshold}


def aggregate(v: pd.Series, keys, method: str) -> pd.Series:
    name, *args = method.split(":")
    if name not in AGGREGATORS:
        raise KeyError(f"unknown aggregator {method!r}; known: {sorted(AGGREGATORS)}")
    return AGGREGATORS[name](v, keys, *args)


# --------------------------------------------------------------------------- the strategy
class CEVT(Strategy):
    """See the module docstring."""

    def __init__(self, spec: CEVTSpec | str, **overrides):
        super().__init__(get_cevt_spec(spec, **overrides))

    def code_views(self, predictions: pd.DataFrame, fdr: pd.DataFrame | None, family: str) -> pd.DataFrame:
        """One row per (code, event window, instrument) that counts: ``start, end, family,
        code, instrument, tsig, ev, provisional``."""
        sp = self.spec
        p = predictions[predictions["legal"] == 1.0]
        if p.empty:
            return pd.DataFrame(columns=["start", "end", "family", "code", "instrument", "tsig", "ev", "provisional"])
        rows = []
        conditional = "cond_bucket" in p.columns
        for inst in self.instruments:
            if f"t:{inst}" not in p.columns:
                continue
            own = p[f"cond_passed:{inst}"] if conditional else p[f"passed:{inst}"]
            if sp.gate == "fdr":
                if fdr is None or fdr.empty:
                    ok = pd.Series(False, index=p.index)
                else:
                    key = inst if not conditional else (inst + "|" + p["cond_bucket"].map(
                        lambda b: f"{int(b):+d}" if np.isfinite(b) else "nan"))
                    f = fdr.assign(fit_as_of=pd.to_datetime(fdr["fit_as_of"]))
                    look = pd.DataFrame({"fit_as_of": pd.to_datetime(p["fit_as_of"]).to_numpy(),
                                         "code": p["code"].to_numpy(),
                                         "row": key.to_numpy() if hasattr(key, "to_numpy") else key})
                    m = look.merge(f, on=["fit_as_of", "code", "row"], how="left")
                    ok = pd.Series((m["passed_fdr"] == 1.0).to_numpy(), index=p.index)
            elif sp.gate == "passed":
                ok = own == 1.0
            elif sp.gate == "none":
                ok = pd.Series(True, index=p.index)
            else:
                raise ValueError(f"gate {sp.gate!r} (fdr | passed | none)")
            tsig = pd.Series(t_signal(p[f"t:{inst}"], sp.t_cap, sp.t_curve), index=p.index)
            ev = p[f"expected:{inst}"]
            if sp.include_failing:
                tsig, ev = tsig.where(ok, 0.0), ev.where(ok, 0.0)
                keep = tsig.notna()
            else:
                keep = ok & tsig.notna()
            prov = p["cond_provisional"] if "cond_provisional" in p.columns else pd.Series(0.0, index=p.index)
            rows.append(pd.DataFrame({"start": p.index[keep.to_numpy()], "end": pd.to_datetime(p["end"][keep]).to_numpy(),
                                      "family": family, "code": p["code"][keep].to_numpy(), "instrument": inst,
                                      "tsig": tsig[keep].to_numpy(), "ev": ev[keep].to_numpy(),
                                      "provisional": prov[keep].astype(float).to_numpy()}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["start", "end", "family", "code", "instrument", "tsig", "ev", "provisional"])

    def views(self, predictions: dict[str, pd.DataFrame], fdr: dict[str, pd.DataFrame],
              labels: pd.DatetimeIndex) -> pd.DataFrame:
        """``predictions`` / ``fdr`` per family run -> per label: ``tsig:``, ``ev:``, ``n:``,
        ``provisional:`` per instrument (0 / NaN-free: no active view = 0, n = 0)."""
        sp = self.spec
        labels = pd.DatetimeIndex(labels).sort_values()
        cv = pd.concat([self.code_views(p, fdr.get(f), f) for f, p in predictions.items() if len(p)],
                       ignore_index=True) if predictions else pd.DataFrame()
        out = pd.DataFrame(index=labels)
        for inst in self.instruments:
            out[f"tsig:{inst}"], out[f"ev:{inst}"], out[f"n:{inst}"], out[f"provisional:{inst}"] = 0.0, 0.0, 0.0, 0.0
        if cv.empty:
            return out
        # expand each view onto the labels its window covers: start <= L < end
        i0 = labels.searchsorted(pd.DatetimeIndex(cv["start"]), side="left")
        i1 = labels.searchsorted(pd.DatetimeIndex(cv["end"]), side="left")
        span = np.maximum(i1 - i0, 0)
        rep = np.repeat(np.arange(len(cv)), span)
        offs = np.concatenate([np.arange(k) for k in span]) if span.sum() else np.array([], dtype=int)
        long = cv.iloc[rep].reset_index(drop=True)
        long["label"] = labels[np.repeat(i0, span) + offs]
        if long.empty:
            return out
        stage1 = ["label", "instrument"] if sp.pool_families else ["label", "instrument", "family"]
        keys = long[stage1]
        fam = pd.DataFrame({k: aggregate(long[k], [long[c] for c in stage1], sp.within) for k in ("tsig", "ev")})
        fam["n"] = long.groupby(stage1).size()
        fam["provisional"] = long.groupby(stage1)["provisional"].max()
        fam = fam.reset_index()
        if not sp.pool_families:
            fam = fam.groupby(["label", "instrument"]).agg(tsig=("tsig", "mean"), ev=("ev", "mean"), n=("n", "sum"),
                                                           provisional=("provisional", "max")).reset_index()
        for col in ("tsig", "ev", "n", "provisional"):
            wide = fam.pivot(index="label", columns="instrument", values=col)
            for inst in wide.columns:
                out.loc[wide.index, f"{col}:{inst}"] = wide[inst].to_numpy()
        del keys
        return out

    def signal(self, views: pd.DataFrame) -> pd.DataFrame:
        """The view positions are built from (``position_signal``), columns = instruments."""
        col = "tsig" if self.spec.position_signal == "t" else "ev"
        return pd.DataFrame({inst: views[f"{col}:{inst}"] for inst in self.instruments}, index=views.index)
