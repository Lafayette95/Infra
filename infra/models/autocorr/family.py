"""A FAMILY of conditional-autocorrelation tests, declared up front, with a point-in-time
false-discovery gate (``infra/models/autocorr/CLAUDE.md`` 4a).

A family = a grid of specs (target x X pairs x X features x past / forward horizons) over one
base spec. Every member is walked forward with its conditioning ALWAYS computed (its own gates
are re-applied afterwards from its stored statistics), so at each fit date the family has every
member's statistic. The ``fdr`` gate: at each fit date, Benjamini-Hochberg across the members'
p-values of ``fdr_stat`` (default ``x_t``, X's own HAC t) -> q; a member trades on that fit only
if its own gates pass AND q <= ``fdr_q``. In ``cells`` mode the tests are the CELLS: every cell
of every member is one p-value (its HAC t), and a prediction trades only if ITS cell passed.
Spanning t's are against both benchmarks jointly (own momentum rule, X's direction-only rule). Families are the unit of the correction: a new X is a
new family (it "resets"), so keep every family's p-values (``run_family`` returns them) for a
periodic global pass across families (``global_q``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

from infra.models.autocorr.config import AutocorrSpec, get_autocorr_spec
from infra.models.autocorr.evaluate import ANNUAL, _spanning_t, staggered_pnl, walk
from infra.models.autocorr.gates import run_gates
from infra.models.autocorr.model import CELLS, cell_key, ols_hac
from infra.models.event_study.family import benjamini_hochberg


@dataclass(frozen=True)
class AutocorrFamilySpec:
    name: str
    pairs: tuple[tuple[str, str], ...]                 # (target, x) series ids
    x_features: tuple[str, ...] = ("move:5", "move:20", "move:60")
    past_days: tuple[int, ...] = (1, 5, 20)
    horizon_days: tuple[int, ...] = (1, 5, 20)
    base: str = "default"                               # the spec whose gates / thresholds members keep
    fdr_q: float = 0.10
    fdr_stat: str = "x_t"                               # the member statistic whose p-value is corrected
    labels: tuple[tuple[str, str], ...] = ()            # short names for series ids (reporting)
    description: str = ""

    def members(self) -> list[tuple[str, AutocorrSpec]]:
        base = get_autocorr_spec(self.base)
        lab = dict(self.labels)
        out = []
        for (t, x), f, k, h in product(self.pairs, self.x_features, self.past_days, self.horizon_days):
            name = f"{lab.get(t, t)}|{lab.get(x, x)}|{f}|k{k}|h{h}"
            out.append((name, replace(base, name=name, target=t, x=x, x_feature=f, past_days=k, horizon_days=h)))
        return out


_DUR = "bmk:curve:US_BOND_10y"
_FLIES = {f"{a}s{b}s{c}": f"bmk:curve:FLY__US_BOND_{a}y__US_BOND_{b}y__US_BOND_{c}y"
          for a, b, c in ((2, 5, 10), (5, 7, 10), (2, 10, 30), (5, 10, 30))}
_DUR_OTR = "bmk:otr:US_BOND_10y"
_FLIES_OTR = {k: v.replace("bmk:curve:", "bmk:otr:") for k, v in _FLIES.items()}
_PAIRS_OTR = tuple((f, _DUR_OTR) for f in _FLIES_OTR.values()) + tuple((_DUR_OTR, f) for f in _FLIES_OTR.values())
_LABELS_OTR = ((_DUR_OTR, "10y"),) + tuple((v, k) for k, v in _FLIES_OTR.items())
AUTOCORR_FAMILIES: dict[str, AutocorrFamilySpec] = {f.name: f for f in (
    AutocorrFamilySpec(
        "duration_flies", pairs=tuple((fid, _DUR) for fid in _FLIES.values()) + tuple((_DUR, fid) for fid in _FLIES.values()),
        labels=(("bmk:curve:US_BOND_10y", "10y"),) + tuple((v, k) for k, v in _FLIES.items()),
        description="10y duration vs the 2s5s10 / 5s7s10 / 2s10s30 / 5s10s30 flies, both ways, our curve's P&L: "
                    "216 members (8 pairs x 3 X features x 3 past x 3 forward horizons)"),
    AutocorrFamilySpec(
        "duration_flies_cells", pairs=_PAIRS_OTR, labels=_LABELS_OTR, base="cells",
        x_features=("move:5", "move:20", "absmove:5", "absmove:20"),
        description="the user's quadrant spec: 3x3 cells (X bucket x own past-move bucket), signed and |X|, on-the-run "
                    "P&L: 288 members x 9 cells, FDR over cells"),
    AutocorrFamilySpec(
        "duration_flies_abs", pairs=_PAIRS_OTR, labels=_LABELS_OTR, base="default",
        x_features=("absmove:5", "absmove:20", "absmove:60"),
        description="chase / fade conditional on the SIZE of X's move (U-shaped effects), on-the-run P&L: 216 members"),
)}


def _p(t) -> np.ndarray:
    return 2.0 * (1.0 - norm.cdf(np.abs(np.asarray(t, dtype="float64"))))


def _spanning_joint_t(cond: pd.Series, benchmarks: list, lags: int) -> float:
    """t of the intercept of cond ~ every benchmark book jointly (HAC): what the conditional
    book adds beyond the target's own momentum rule AND X's direction-only rule. A book that
    never traded adds 0."""
    j = pd.concat([cond] + benchmarks, axis=1).dropna()
    if j.empty or j.iloc[:, 0].abs().sum() == 0:
        return 0.0
    X = np.c_[np.ones(len(j)), j.iloc[:, 1:].to_numpy()]
    b, cov = ols_hac(j.iloc[:, 0].to_numpy(), X, lags)
    return float(b[0] / np.sqrt(cov[0, 0])) if cov[0, 0] > 0 else 0.0


def run_family(family: AutocorrFamilySpec | str, panel: pd.DataFrame, start, through, *, refit: str = "ME",
               log=None) -> dict:
    """Walk every member forward, apply own gates + the per-fit-date FDR gate, and return:
    ``fits`` (fit_as_of, member, stat, p, own_pass, q, passed), ``books`` (member -> gated daily
    P&L, benchmark P&L) and ``summary`` (per member: gated Sharpe, spanning t, share of fits passed)."""
    fam = AUTOCORR_FAMILIES[family] if isinstance(family, str) else family
    fits, raw = [], {}
    for i, (name, spec) in enumerate(fam.members()):
        th = dict(spec.thresholds)
        # always compute the conditioning (cells: every cell's sign), gate afterwards from the stats
        free = replace(spec, gates=("min_obs",),
                       thresholds=tuple({**th, "cell_t": 0.0, "cell_min_obs": 1.0}.items()))
        res, prep = walk(free, panel[[spec.target, spec.x]], start, through, refit)
        if res.predictions.empty:
            continue
        stat = res.params[res.params["section"] == "stat"].pivot_table(index="fit_as_of", columns="row", values="value")
        for fa, row in stat.iterrows():
            st = row.dropna().to_dict()
            _, own = run_gates(st, spec.gates, th) if st.get("n", 0) >= 20 else ({}, False)
            if spec.mode == "cells":
                for xb, pb in CELLS:
                    k = cell_key(xb, pb)
                    t, cnt = st.get(f"cell_t_{k}", np.nan), st.get(f"cell_n_{k}", 0.0)
                    ok = own and np.isfinite(t) and abs(t) >= th.get("cell_t", 2.0) and cnt >= th.get("cell_min_obs", 40.0)
                    fits.append({"fit_as_of": pd.Timestamp(fa), "member": name, "cell": k, "stat": t, "own_pass": float(ok)})
            else:
                fits.append({"fit_as_of": pd.Timestamp(fa), "member": name, "cell": "", "stat": st.get(fam.fdr_stat, np.nan),
                             "own_pass": float(own)})
        raw[name] = (spec, res, prep)
        if log and i % 20 == 0:
            log(f"{i + 1} members")
    f = pd.DataFrame(fits)
    f["p"] = _p(f["stat"])
    f["q"] = np.nan
    for fa, g in f.groupby("fit_as_of"):
        f.loc[g.index, "q"] = benjamini_hochberg(g["p"].to_numpy())
    f["passed"] = ((f["own_pass"] == 1.0) & (f["q"] <= fam.fdr_q)).astype(float)
    books, rows = {}, []
    sh = lambda x: float(x.mean() / x.std() * ANNUAL) if x.std() > 0 else 0.0  # noqa: E731
    for name, (spec, res, prep) in raw.items():
        p = res.predictions
        fm = f[f["member"] == name]
        fa = pd.to_datetime(p["fit_as_of"])
        if spec.mode == "cells":
            key = [cell_key(x, q) if np.isfinite(x) and np.isfinite(q) else "" for x, q in zip(p["bucket"], p["pbucket"])]
            idx = pd.MultiIndex.from_arrays([fa, key])
            gate = fm.set_index(["fit_as_of", "cell"])["passed"].reindex(idx).fillna(0.0).to_numpy()
            own = fm.set_index(["fit_as_of", "cell"])["own_pass"].reindex(idx).fillna(0.0).to_numpy()
        else:
            gate = fa.map(fm.set_index("fit_as_of")["passed"]).fillna(0.0).to_numpy()
            own = fa.map(fm.set_index("fit_as_of")["own_pass"]).fillna(0.0).to_numpy()
        cond = staggered_pnl(p["signal"] * gate, prep, spec)
        naive = staggered_pnl(p["signal"] * own, prep, spec)
        bench = staggered_pnl(p["bench_signal"], prep, spec)
        xdir = staggered_pnl(p["xdir_signal"], prep, spec)
        first = p.index.min()
        cond, naive, bench, xdir = (x[x.index > first] for x in (cond, naive, bench, xdir))
        books[name] = (cond, bench, xdir)
        rows.append({"member": name, "fdr_share": float(gate.mean()), "own_share": float(own.mean()),
                     "sharpe_fdr": sh(cond), "sharpe_own_gates": sh(naive), "sharpe_benchmark": sh(bench),
                     "sharpe_xdir": sh(xdir),
                     "alpha_t_fdr": _spanning_joint_t(cond, [bench, xdir], spec.horizon_days),
                     "alpha_t_own_gates": _spanning_joint_t(naive, [bench, xdir], spec.horizon_days)})
    return {"fits": f, "books": books, "summary": pd.DataFrame(rows).set_index("member")}


def global_q(results: list[pd.DataFrame]) -> pd.DataFrame:
    """The cross-family check: BH over the fits of SEVERAL families together, per fit date."""
    f = pd.concat(results, ignore_index=True)
    f["q_global"] = np.nan
    for fa, g in f.groupby("fit_as_of"):
        f.loc[g.index, "q_global"] = benjamini_hochberg(g["p"].to_numpy())
    return f
