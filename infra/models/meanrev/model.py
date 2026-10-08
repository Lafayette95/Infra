"""Mean reversion of a factor model's residuals, cross-sectional (framework C's trading part):
``prepare`` -> ``fit`` -> ``predict`` (infra/models/CLAUDE.md 0); methodology
``infra/models/meanrev/CLAUDE.md``.

The model WRAPS a factor model (a PCA - plain, weighted, similarity - or a regime PCA over the K
instruments) and owns its fit. At each fit date:

1. the factor model is fitted on data up to the date;
2. its fit is RE-APPLIED to the last ``window`` rows (the residuals the CURRENT loadings give the
   recent past - point in time: the loadings are known at the fit date), and each residual is
   cumulated into a LEVEL from the window's first row;
3. each residual's portfolio weights in the K instruments are read off by regressing its daily
   residual on the instruments' moves over the window (exact for a constant-loading PCA; R2 < 1 for
   a regime blend whose loadings move by day - reported);
4. an OU (AR(1) on the level) per residual, or pooled (one b), with gates per residual.

``predict`` continues each level from the fit date: the level the fit stored at ``as_of``
(``level_end``) plus the out-of-sample residuals (the threshold rule continues from the stored
``state_end``) - so a prediction depends on the stored params and the data after the fit only,
never on the factor model re-deriving its in-sample path (a regime model rebuilt from its params
doesn't reproduce its in-sample regime path exactly; its out-of-sample one it does). It gives per residual
the level, s-score, signal and the OU trade metrics (``infra.analytics.ou``), and per instrument the
NETTED position: the K residual portfolios share only K - k independent directions, so they are
traded through one netted book, never as K separate bets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infra.analytics import ou
from infra.models.base import Model
from infra.models.meanrev.config import MeanRevSpec, get_meanrev_spec
from infra.models.stats.common import PARAM_COLUMNS, cutoff_mask

X_PREFIX = "X|"                     # the instruments' raw daily moves, kept next to the factor model's inputs
F_PREFIX = "factor:"                # the factor model's params sections
OU_FIELDS = ("a", "b", "se_b", "sd_e", "n", "adf_t", "kappa", "m", "sd_eq", "half_life", "b_h1", "b_h2",
             "tradable", "z_mean", "z_sd", "r2_proj", "opt_entry", "level_end", "state_end")


@dataclass
class MeanRevFit:
    as_of: pd.Timestamp
    window_start: pd.Timestamp
    ou: pd.DataFrame                          # residual x OU_FIELDS + gate:<name>
    weights: pd.DataFrame                     # residual x instrument (+ "const")
    stats: dict = field(default_factory=dict)


class MeanRevModel(Model):
    """See the module docstring."""

    def __init__(self, spec: MeanRevSpec | str | None = None, **overrides):
        self.spec = get_meanrev_spec(spec, **overrides)
        if not self.spec.columns:
            raise ValueError("MeanRevModel needs the K instruments (columns)")
        self.factor = self._make_factor()
        self.fitted_: MeanRevFit | None = None

    # ------------------------------------------------------------------ the factor model
    def _factor_kwargs(self) -> dict:
        sp = self.spec
        kw = dict(sp.factor_overrides)
        kw["columns"] = tuple(sp.columns)
        if sp.factor == "regime_pca":
            ro = dict(kw.pop("regime_overrides", ()))
            if sp.regime_columns:
                ro["columns"] = tuple(sp.regime_columns)
            kw["regime_overrides"] = tuple(ro.items())
        return kw

    def _make_factor(self):
        from infra.models.stats.pca import make_pca
        from infra.models.stats.regime_pca import make_regime_pca
        maker = make_pca if self.spec.factor == "pca" else make_regime_pca
        return maker(self.spec.factor_spec, **self._factor_kwargs())

    @property
    def is_fitted(self) -> bool:
        return self.fitted_ is not None

    @property
    def residuals(self) -> list[str]:
        return list(self.spec.columns)

    def warm_start_from(self, previous: "MeanRevModel") -> None:
        if hasattr(self.factor, "warm_start_from") and previous.is_fitted:
            self.factor.warm_start_from(previous.factor)

    def align_to(self, reference: "MeanRevModel") -> "MeanRevModel":
        """Factor / regime identities only: a residual doesn't depend on a factor's sign or order."""
        if hasattr(self.factor, "align_to") and reference.is_fitted:
            self.factor.align_to(reference.factor)
        return self

    def restore_path(self, prepared: pd.DataFrame) -> "MeanRevModel":
        if hasattr(self.factor, "restore_path"):
            self.factor.restore_path(self._fpart(prepared))
        return self

    # ------------------------------------------------------------------ 1. prepare
    def prepare(self, raw: pd.DataFrame, **kwargs) -> pd.DataFrame:
        f = self.factor.prepare(raw, **kwargs)
        x = raw[self.residuals].astype("float64").add_prefix(X_PREFIX)
        out = pd.concat([f, x.reindex(f.index)], axis=1)
        out.index.name = "timestamp"
        return out

    @staticmethod
    def _fpart(prepared: pd.DataFrame) -> pd.DataFrame:
        return prepared[[c for c in prepared.columns if not c.startswith(X_PREFIX)]]

    def _moves(self, prepared: pd.DataFrame) -> pd.DataFrame:
        return prepared[[X_PREFIX + c for c in self.residuals]].rename(columns=lambda c: c[len(X_PREFIX):])

    def _levels(self, prepared: pd.DataFrame, window_start: pd.Timestamp, end) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(daily residuals, cumulated levels) from ``window_start`` to ``end`` with the CURRENT factor fit."""
        idx = prepared.index
        before = idx[idx < window_start]
        start_excl = before[-1] if len(before) else None
        p = self.factor.predict(self._fpart(prepared), start=start_excl, end=end)
        res = pd.DataFrame({c: p[f"residual:{c}"] for c in self.residuals}, index=p.index)
        res = res[res.index >= window_start]
        return res, res.fillna(0.0).cumsum()

    # ------------------------------------------------------------------ 2. fit
    def fit(self, prepared: pd.DataFrame, as_of=None) -> "MeanRevModel":
        sp = self.spec
        as_of = pd.Timestamp(prepared.index.max() if as_of is None else as_of)
        self.factor.fit(self._fpart(prepared), as_of=as_of)
        known = prepared.index[cutoff_mask(prepared.index, as_of)]
        if len(known) < 3:
            raise ValueError(f"MeanRevModel: {len(known)} rows up to {as_of.date()}")
        window_start = known[-min(sp.window, len(known))]
        res, lev = self._levels(prepared, window_start, as_of)
        moves = self._moves(prepared).reindex(res.index)
        weights, r2 = self._weights(res, moves)
        fits = self._ou(lev)
        th = sp.threshold
        rows = {}
        for c in self.residuals:
            f = fits[c]
            p = ou.ou_params(f)
            x = lev[c].to_numpy()
            h1, h2 = ou.fit_ar1(x[: len(x) // 2]), ou.fit_ar1(x[len(x) // 2:])
            gates = {
                "min_obs": f.n >= th("min_obs", 60.0),
                "reverting": p.reverting,
                "adf_t": np.isfinite(f.adf_t) and f.adf_t <= -th("adf_t", 2.0),
                "half_life": p.reverting and th("half_life_min", 1.0) <= p.half_life <= th("half_life_max", 30.0),
                "halves": all(np.isfinite(h.b) and 0 < h.b < 1 for h in (h1, h2)),
            }
            if sp.fit_mode == "exante":
                tradable = f.n >= 2
            else:
                tradable = all(gates[g] for g in sp.gates) and p.reverting   # prior: reversion is the stated sign
            cost_s = sp.cost_bp / p.sd_eq if p.reverting and p.sd_eq > 0 else np.nan
            rows[c] = {"a": f.a, "b": f.b, "se_b": f.se_b, "sd_e": f.sd_e, "n": float(f.n), "adf_t": f.adf_t,
                       "kappa": p.kappa, "m": p.m, "sd_eq": p.sd_eq, "half_life": p.half_life,
                       "b_h1": h1.b, "b_h2": h2.b, "tradable": float(tradable),
                       "z_mean": float(np.nanmean(x)), "z_sd": float(np.nanstd(x)), "r2_proj": r2[c],
                       "opt_entry": ou.optimal_entry(round(float(cost_s), 2)) if np.isfinite(cost_s) else np.nan,
                       "level_end": float(x[-1])} | {f"gate:{g}": float(v) for g, v in gates.items()}
        table = pd.DataFrame(rows).T.astype("float64")
        table.index.name = "residual"
        if sp.signal_rule == "threshold":                         # the rule's state at the fit date
            S = self._scores(lev, table)
            table["state_end"] = self._signals(S).iloc[-1].reindex(table.index).fillna(0.0)
        else:
            table["state_end"] = 0.0
        self.fitted_ = MeanRevFit(as_of, pd.Timestamp(window_start), table, weights,
                                  {"n_tradable": float(table["tradable"].sum()), "window_rows": float(len(lev))})
        return self

    def _weights(self, res: pd.DataFrame, moves: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Each residual's portfolio in the instruments: least squares of its daily residual on the
        instruments' moves (with a constant) over the window."""
        ok = res.notna().all(axis=1) & moves.notna().all(axis=1)
        X = np.c_[np.ones(int(ok.sum())), moves[ok].to_numpy()]
        W, r2 = {}, {}
        for c in self.residuals:
            y = res.loc[ok, c].to_numpy()
            if len(y) <= X.shape[1]:
                W[c], r2[c] = np.full(X.shape[1], np.nan), np.nan
                continue
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = y - X @ beta
            ss = ((y - y.mean()) ** 2).sum()
            W[c], r2[c] = beta, float(1 - (e @ e) / ss) if ss > 0 else np.nan
        w = pd.DataFrame(W, index=["const"] + self.residuals).T
        w.index.name = "residual"
        return w, r2

    def _ou(self, lev: pd.DataFrame) -> dict[str, ou.AR1Fit]:
        sp = self.spec
        fits = {c: ou.fit_ar1(lev[c].to_numpy(), bias_correct=sp.bias_correct) for c in self.residuals}
        if sp.pooling == "none":
            return fits
        num = den = 0.0
        for c in self.residuals:                                  # one b across residuals, each its own mean
            x = lev[c].to_numpy()
            x0, x1 = x[:-1], x[1:]
            ok = np.isfinite(x0) & np.isfinite(x1)
            d0, d1 = x0[ok] - x0[ok].mean(), x1[ok] - x1[ok].mean()
            num, den = num + d0 @ d1, den + d0 @ d0
        if den <= 0:
            return fits
        b = num / den
        n = len(lev) - 1
        if sp.bias_correct and b < 1:
            b = min(b + (1 + 3 * b) / n, 0.999999)
        out, ss = {}, 0.0
        for c in self.residuals:
            x = lev[c].to_numpy()
            x0, x1 = x[:-1], x[1:]
            a = float(np.nanmean(x1) - b * np.nanmean(x0))
            e = x1 - a - b * x0
            sd_e = float(np.sqrt(np.nanmean(e ** 2) * n / max(n - 2, 1)))
            ss += float(np.nansum((x0 - np.nanmean(x0)) ** 2) / sd_e ** 2) if sd_e > 0 else 0.0
            out[c] = (a, sd_e, fits[c].n)
        se_b = 1.0 / np.sqrt(ss) if ss > 0 else np.nan
        return {c: ou.AR1Fit(a, float(b), se_b, sd_e, n_) for c, (a, sd_e, n_) in out.items()}

    # ------------------------------------------------------------------ 3. predict
    def predict(self, prepared: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
        """Rows in ``(start, end]``: per residual ``level:`` (bp), ``z:`` (vs the window), ``s:`` (OU
        s-score), ``signal:`` (fade, [-1, 1] x tradable), ``size:`` (signal / daily sd if
        ``risk_scale``), ``exp_move:`` / ``sd_h:`` / ``sharpe_h:`` (over ``horizon_days``, bp; + =
        the residual rises), ``fpt_median:`` / ``fpt_mean:`` (days to the mean), ``p_target:``
        (target |s| = exit before the stop), ``trade_days:``; per instrument ``pos:`` (the netted
        position); ``n_tradable``, ``in_sample``."""
        self.check_fitted()
        f, sp = self.fitted_, self.spec
        T = f.ou
        lev, state0 = self._continued_levels(prepared, start, end)
        idx = lev.index
        out = {}
        for c in self.residuals:
            r = T.loc[c]
            p = ou.OUParams(r["kappa"], r["m"], r["sd_eq"], r["sd_e"], r["b"])
            x = lev[c]
            s = (x - p.m) / p.sd_eq if p.reverting else pd.Series(np.nan, index=idx)
            z = (x - r["z_mean"]) / r["z_sd"] if r["z_sd"] > 0 else pd.Series(np.nan, index=idx)
            out[f"level:{c}"], out[f"s:{c}"], out[f"z:{c}"] = x, s, z
            if p.reverting:
                exp_move = ou.expected_level(x, p, sp.horizon_days) - x
                sd_h = float(ou.horizon_sd(p, sp.horizon_days))
                k = p.kappa
                out[f"exp_move:{c}"], out[f"sd_h:{c}"] = exp_move, pd.Series(sd_h, index=idx)
                out[f"sharpe_h:{c}"] = exp_move.abs() / sd_h
                out[f"fpt_median:{c}"] = pd.Series(ou.fpt_median(s.to_numpy()) / k, index=idx)
                out[f"fpt_mean:{c}"] = pd.Series(ou.mean_fpt_to_mean(s.to_numpy()) / k, index=idx)
                out[f"p_target:{c}"] = pd.Series(ou.p_target_before_stop(s.to_numpy(), sp.exit, sp.stop_width), index=idx)
                out[f"trade_days:{c}"] = pd.Series(ou.expected_trade_time(s.to_numpy(), sp.exit, sp.stop_width) / k,
                                                   index=idx)
        S = self._scores(lev, T)
        tradable = T["tradable"].reindex(self.residuals).fillna(0.0)
        sig = self._signals(S, state0) * tradable.to_numpy()
        size = sig.div(T["sd_e"].reindex(self.residuals), axis=1) if sp.risk_scale else sig
        size = size.fillna(0.0)
        for c in self.residuals:
            out[f"signal:{c}"], out[f"size:{c}"] = sig[c].fillna(0.0), size[c]
        W = f.weights.reindex(index=self.residuals, columns=self.residuals).fillna(0.0)
        pos = size.to_numpy() @ W.to_numpy()
        for j, c in enumerate(self.residuals):
            out[f"pos:{c}"] = pd.Series(pos[:, j], index=idx)
        frame = pd.DataFrame(out, index=idx)
        frame["n_tradable"] = float(tradable.sum())
        frame["in_sample"] = cutoff_mask(idx, f.as_of)
        keep = np.ones(len(frame), dtype=bool)
        if start is not None:
            keep &= ~cutoff_mask(idx, start)
        return frame[keep]

    def _continued_levels(self, prepared: pd.DataFrame, start, end) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Levels from the fit (``predict`` slices ``(start, end]``) and the threshold state they start
        from. Rows after the fit date: ``level_end`` + the out-of-sample residuals, starting from the
        stored ``state_end``. A request reaching back before the fit date (in sample) recomputes the
        window instead, the threshold path flat at the window start."""
        f = self.fitted_
        p = self.factor.predict(self._fpart(prepared), start=f.as_of, end=end)
        res = pd.DataFrame({c: p[f"residual:{c}"] for c in self.residuals}, index=p.index)
        lev_out = res.fillna(0.0).cumsum() + f.ou["level_end"].reindex(self.residuals).to_numpy()
        if start is not None and pd.Timestamp(start) >= f.as_of:
            return lev_out, f.ou["state_end"].reindex(self.residuals).fillna(0.0).to_numpy()
        _, lev_in = self._levels(prepared, f.window_start, f.as_of)
        return pd.concat([lev_in, lev_out]), None

    def _scores(self, lev: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
        """The score the signal fades: the OU s-score (fitted / prior) or the level's z over the window
        (exante), cross-sectionally demeaned over the tradable residuals if asked."""
        sp = self.spec
        S = {}
        for c in self.residuals:
            r = T.loc[c]
            if sp.fit_mode == "exante":
                S[c] = (lev[c] - r["z_mean"]) / r["z_sd"] if r["z_sd"] > 0 else lev[c] * np.nan
            else:
                ok = np.isfinite(r["kappa"]) and r["kappa"] > 0
                S[c] = (lev[c] - r["m"]) / r["sd_eq"] if ok else lev[c] * np.nan
        S = pd.DataFrame(S, index=lev.index)
        if sp.cross_section == "demean":
            live = T["tradable"].reindex(self.residuals).fillna(0.0)
            S = S.sub(S.loc[:, live[live > 0].index].mean(axis=1), axis=0)
        return S

    def _signals(self, S: pd.DataFrame, state0: np.ndarray | None = None) -> pd.DataFrame:
        sp = self.spec
        if sp.signal_rule == "linear":
            return (-S / sp.entry).clip(-1.0, 1.0)
        out = np.zeros(S.shape)                    # threshold: a path from state0 (flat at the window start)
        v = S.to_numpy()
        state = np.zeros(S.shape[1]) if state0 is None else np.asarray(state0, dtype="float64").copy()
        for t in range(len(v)):
            for j in range(S.shape[1]):
                x = v[t, j]
                if not np.isfinite(x):
                    state[j] = 0.0
                    continue
                if state[j] != 0 and (abs(x) <= sp.exit or np.sign(x) == state[j]):
                    state[j] = 0.0
                if state[j] == 0 and abs(x) >= sp.entry:
                    state[j] = -np.sign(x)
            out[t] = state
        return pd.DataFrame(out, index=S.index, columns=S.columns)

    # ------------------------------------------------------------------ params
    def params(self) -> pd.DataFrame:
        self.check_fitted()
        f = self.fitted_
        rows = [{"section": "meta", "row": "as_of", "col": f.as_of.isoformat(), "value": np.nan},
                {"section": "meta", "row": "window_start", "col": f.window_start.isoformat(), "value": np.nan}]
        rows += [{"section": "ou", "row": r, "col": c, "value": float(v)} for r, d in f.ou.iterrows()
                 for c, v in d.items()]
        rows += [{"section": "weight", "row": r, "col": c, "value": float(v)} for r, d in f.weights.iterrows()
                 for c, v in d.items()]
        rows += [{"section": "stat", "row": k, "col": "value", "value": float(v)} for k, v in f.stats.items()]
        fp = self.factor.params()
        fp = fp.assign(section=F_PREFIX + fp["section"].astype(str))
        return pd.concat([pd.DataFrame(rows), fp], ignore_index=True)[PARAM_COLUMNS]

    @classmethod
    def from_params(cls, params: pd.DataFrame, spec: MeanRevSpec | str | None = None, **overrides) -> "MeanRevModel":
        from infra.models.stats.pca import PCA
        from infra.models.stats.regime_pca import RegimePCA
        m = cls(spec, **overrides)
        sec = params["section"].astype(str)
        sub = params[sec.str.startswith(F_PREFIX)]
        sub = sub.assign(section=sub["section"].astype(str).str.slice(len(F_PREFIX)))
        fcls = PCA if m.spec.factor == "pca" else RegimePCA
        m.factor = fcls.from_params(sub.reset_index(drop=True), m.spec.factor_spec, **m._factor_kwargs())
        meta = params[sec == "meta"].set_index("row")["col"]
        piv = lambda s: params[sec == s].pivot(index="row", columns="col", values="value")  # noqa: E731
        table = piv("ou").reindex(index=m.residuals)
        weights = piv("weight").reindex(index=m.residuals, columns=["const"] + m.residuals)
        stats = params[sec == "stat"].set_index("row")["value"].to_dict()
        m.fitted_ = MeanRevFit(pd.Timestamp(meta["as_of"]), pd.Timestamp(meta["window_start"]),
                               table.astype("float64"), weights.astype("float64"), stats)
        return m


def make_meanrev(spec: MeanRevSpec | str | None = None, **overrides) -> MeanRevModel:
    return MeanRevModel(spec, **overrides)
