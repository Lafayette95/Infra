"""Basis models (infra/models/basis/CLAUDE.md), each a ``infra.models.base.Model``.

Output of ``predict`` for one day, every tier alike:
* ``contracts``: one row per contract - market and fair futures price, the CTD (cusip and
  delivery day), its net basis / implied repo / funding, the observed option value
  (fair - market, in 32nds), the model option value, futures DV01, the runner-up's gap;
* ``bonds``: one row per (contract, bond, delivery day) - forward, implied futures price,
  net basis, implied repo, and ``prob`` (the model's delivery probability).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.analytics.delivery_timing import eom_switch_value, wildcard_value, window_kinds
from infra.config import FOMC_MEETINGS
from infra.analytics.futures_basis import (bachelier_exchange, basis_table, clean_price_from_yield, forward_yield,
                                          futures_dv01, simulate_delivery)
from infra.analytics.sofr_curve import business_days
from infra.models.base import Model
from infra.models.basis.config import BASIS_MODELS, BasisSpec
from infra.models.basis.factors import change_panel, fit_factor_model, simulate_shocks
from infra.pipeline.futures_basis import BasisDay

TICKS = 32.0


class DeterministicBasis(Model):
    """M0: no uncertainty. Every bond's forward to the first and last delivery day, at the
    funding model's rate; the CTD is the (bond, day) with the lowest implied futures
    price, which is also the fair futures price. Delivery probability 1 on the CTD, option
    value 0 - so the OBSERVED option value (fair - market) is everything M0 leaves out."""

    def __init__(self, spec: BasisSpec = BASIS_MODELS["M0"]):
        self.spec = spec

    def prepare(self, raw: BasisDay, **kwargs) -> pd.DataFrame:
        if raw.bonds.empty:
            return pd.DataFrame()
        parts = []
        for contract, b in raw.bonds.groupby("contract"):
            f = raw.futures.set_index("contract").loc[contract]
            t = basis_table(b, float(f["futures"]) if pd.notna(f["futures"]) else None)
            parts.append(t.assign(futures=f["futures"], futures_source=f["futures_source"],
                                  settlement=f["settlement"], last_trading=f["last_trading"]))
        return pd.concat(parts, ignore_index=True).assign(day=raw.day)

    def fit(self, prepared, as_of=None) -> "DeterministicBasis":
        self.fitted_ = {"tier": "M0", "as_of": as_of}  # nothing to estimate
        return self

    def predict(self, prepared: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
        self.check_fitted()
        if prepared.empty:
            return {"contracts": pd.DataFrame(), "bonds": prepared}
        bonds = prepared.copy()
        bonds.attrs = {}  # inputs riding on the prepared frame (M1's level history) aren't outputs
        bonds["prob"] = 0.0
        rows = []
        for contract, g in bonds.groupby("contract"):
            ctd_idx = g["implied_futures"].idxmin()
            bonds.loc[ctd_idx, "prob"] = 1.0
            c = bonds.loc[ctd_idx]
            per_bond = g.groupby("cusip")["implied_futures"].min().sort_values()
            runner_gap = (per_bond.iloc[1] - per_bond.iloc[0]) * TICKS if len(per_bond) > 1 else np.nan
            market = c["futures"]
            rows.append({
                "day": c["day"], "root": c["root"], "contract": contract, "futures": market,
                "futures_source": c["futures_source"], "fair_futures": c["implied_futures"],
                "ctd": c["cusip"], "ctd_coupon": c["coupon"], "ctd_maturity": c["maturity"], "ctd_cf": c["cf"],
                "delivery": c["delivery"], "delivery_kind": c["delivery_kind"], "last_trading": c["last_trading"],
                "days_to_delivery": (c["delivery"] - c["settle"]).days,
                "ctd_net_basis_32": c.get("net_basis", np.nan) * TICKS,
                "ctd_irr": c.get("irr", np.nan), "ctd_repo": c["repo"], "ctd_repo_base": c["repo_base"],
                "option_value_obs_32": (c["implied_futures"] - market) * TICKS if pd.notna(market) else np.nan,
                # the street's convention (a DLV screen): against the 15:00 settlement - which
                # mismatches our 15:30 cash by half an hour, so noisier; for comparison only
                "settlement": c["settlement"],
                "option_value_obs_vs_settle_32": (c["implied_futures"] - c["settlement"]) * TICKS
                if pd.notna(c["settlement"]) else np.nan,
                "option_value_model_32": 0.0,
                "futures_dv01": futures_dv01(c["dv01"], c["repo"], c["settle"], c["delivery"], c["cf"]),
                "ctd_yield": c["yield_eod"], "runner_up_gap_32": runner_gap, "n_bonds": int(per_bond.size),
            })
        return {"contracts": pd.DataFrame(rows), "bonds": bonds}


def ewma_vol_bp(levels: pd.Series, lam: float) -> float:
    """EWMA std of daily changes (bp/day) of a yield series (%), seeded with the first
    20 changes' variance."""
    x = levels.dropna().diff().dropna().to_numpy() * 100.0
    if x.size < 20:
        return float("nan")
    var = float(np.mean(x[:20] ** 2))
    for v in x[20:]:
        var = lam * var + (1 - lam) * v * v
    return float(np.sqrt(var))


class OneFactorBasis(DeterministicBasis):
    """M1: one level factor. At each contract's M0 delivery day, every deliverable's
    forward yield gets the SAME normal shock, sd = sigma x sqrt(business days to delivery),
    sigma = the EWMA vol of the CMT yield at the contract's tenor (``fit``). Each bond's
    simulated price is centred on its forward; the short delivers the lowest P/CF per path.
    Fair futures = mean of the per-path minimum; option value = M0 fair - M1 fair (>= 0);
    delivery probabilities = each bond's share of paths; futures DV01 by a +-0.5bp bump on
    the same draws; a two-bond Bachelier/Margrabe check on the runner-up alongside.
    Timing stays M0's (deterministic per contract)."""

    def __init__(self, spec: BasisSpec = BASIS_MODELS["M1"]):
        super().__init__(spec)

    def prepare(self, raw: BasisDay, **kwargs) -> pd.DataFrame:
        out = super().prepare(raw)
        out.attrs["levels"] = raw.meta.get("levels")
        return out

    def fit(self, prepared, as_of=None) -> "OneFactorBasis":
        levels = prepared.attrs.get("levels") if prepared is not None else None
        if levels is None or levels.empty:
            raise ValueError("M1 needs the level history (meta['levels'] from infra.models.basis.inputs)")
        if as_of is not None:
            levels = levels[levels.index <= pd.Timestamp(as_of)]
        self.fitted_ = {"tier": "M1", "as_of": as_of,
                        "vol_bp_day": {r: ewma_vol_bp(levels[r], self.spec.vol_lambda) for r in levels.columns}}
        return self

    def predict(self, prepared: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
        base = super().predict(prepared)
        contracts, bonds = base["contracts"], base["bonds"]
        if contracts.empty:
            return base
        self._prepared_bonds = prepared
        rng = np.random.default_rng(self.spec.seed)
        half = rng.standard_normal(self.spec.n_paths // 2)
        z = np.concatenate([half, -half])  # antithetic: an exactly symmetric shock set
        bonds["prob"] = 0.0
        rows, new_rows = [], []
        for _, c in contracts.iterrows():
            g = bonds[(bonds["contract"] == c["contract"]) & (bonds["delivery_kind"] == c["delivery_kind"])]
            g = g.dropna(subset=["fwd"])
            sigma = self.fitted_["vol_bp_day"].get(c["root"], np.nan)
            day = pd.Timestamp(c["day"])
            n_bd = len(business_days(day + pd.Timedelta(days=1), self._quality_horizon_end(c)))
            out = c.to_dict() | {"fair_futures_m0": c["fair_futures"], "vol_bp_day": sigma, "horizon_bd": n_bd}
            if not np.isfinite(sigma) or len(g) == 0:
                rows.append(out)
                continue
            fy = np.array([forward_yield(r.fwd, r.coupon, r.maturity, r.delivery) for r in g.itertuples()])
            ok = np.isfinite(fy)
            g, fy = g[ok], fy[ok]
            extra = self._extra_bonds(c)  # deliverables not issued yet (M2+); empty for M1
            sim = pd.concat([g[["cusip", "fwd", "cf", "coupon", "maturity", "implied_futures"]],
                             extra[["cusip", "fwd", "cf", "coupon", "maturity", "implied_futures"]]], ignore_index=True)
            sim_fy = np.concatenate([fy, extra["fwd_yield"].to_numpy()]) if len(extra) else fy
            shocks = self._shocks(c, sim, n_bd, z)
            if shocks is None:
                rows.append(out)
                continue
            args = (sim["fwd"].to_numpy(), sim["cf"].to_numpy(), sim_fy, sim["coupon"].to_numpy(), list(sim["maturity"]),
                    c["delivery"], shocks)
            _, fair, share_all = simulate_delivery(*args)
            _, fair_dn, _ = simulate_delivery(*args, bump_bp=-0.5)
            _, fair_up, _ = simulate_delivery(*args, bump_bp=0.5)
            share = share_all[:len(g)]
            bonds.loc[g.index, "prob"] = share
            if len(extra):
                new_rows.append(extra.assign(day=c["day"], root=c["root"], contract=c["contract"],
                                             delivery=c["delivery"], delivery_kind=c["delivery_kind"],
                                             prob=share_all[len(g):]))
            # two-bond check: runner-up vs CTD, spread of implied futures normal
            order = np.argsort(g["implied_futures"].to_numpy())
            margrabe = np.nan
            if len(order) > 1:
                i, j = order[0], order[1]
                fdv = (g["dv01"].to_numpy() * (1 + g["repo"].to_numpy() / 100 * (c["delivery"] - g["settle"].iloc[0]).days / 360)
                       / g["cf"].to_numpy())
                s = self._pair_sd(c, g, fdv, i, j, n_bd)
                gap = g["implied_futures"].to_numpy()[j] - g["implied_futures"].to_numpy()[i]
                margrabe = bachelier_exchange(-gap, s) * TICKS
            top = sim["cusip"].to_numpy()[int(np.argmax(share_all))]
            out.update({
                "fair_futures": fair,
                "option_value_model_32": (c["fair_futures"] - fair) * TICKS,
                "futures_dv01": fair_dn - fair_up,
                "ctd_prob": float(share[list(g["cusip"]).index(c["ctd"])]) if c["ctd"] in set(g["cusip"]) else 0.0,
                "top_prob_bond": top, "top_prob": float(share_all.max()),
                "option_value_margrabe_32": margrabe,
                "prob_new_issues": float(share_all[len(g):].sum()), "n_new_issues": int(len(extra)),
            })
            if self.spec.timing_options:
                out.update(self._timing(c, z))
            rows.append(out)
        if new_rows:
            bonds = pd.concat([bonds, *new_rows], ignore_index=True)
        out = pd.DataFrame(rows)
        if self.spec.timing_options and "timing_32" in out:
            t = out["timing_32"].fillna(0.0)
            out["option_value_quality_32"] = out["option_value_model_32"]
            out["option_value_model_32"] = out["option_value_model_32"] + t
            out["fair_futures"] = out["fair_futures"] - t / TICKS
        return {"contracts": out, "bonds": bonds}

    def _quality_horizon_end(self, c) -> pd.Timestamp:
        """Where the quality-option simulation stops: the delivery day - or, with the timing
        options on and a LAST-day delivery, the last trading day: the switches after it
        (futures frozen) are the end-of-month option's, priced separately (no double count)."""
        if self.spec.timing_options and c["delivery_kind"] == "last":
            return min(pd.Timestamp(c["last_trading"]), pd.Timestamp(c["delivery"]))
        return pd.Timestamp(c["delivery"])

    def _timing(self, c, z: np.ndarray) -> dict:
        """The wild card and end-of-month options for contract ``c`` (futures 32nds), added
        to the model option value and taken off the fair futures price. EOM: the basket's
        forwards to the LAST delivery day, futures frozen at their lowest implied price,
        one level shock over the EOM business days (first order: bonds' DV01s differ while
        F stays put). Wild card: Bermudan over the intention days left (from the first
        intention day, 2 business days before the delivery month, to the last trading day),
        the EOM value as continuation."""
        sigma = self.fitted_["vol_bp_day"].get(c["root"], np.nan)
        b = self._prepared_bonds
        g = b[(b["contract"] == c["contract"]) & (b["delivery_kind"] == "last")].dropna(subset=["fwd", "dv01"])
        if not np.isfinite(sigma) or g.empty:
            return {"wildcard_32": np.nan, "eom_32": np.nan}
        bd = business_days(pd.Timestamp(c["day"]) - pd.Timedelta(days=10), c["delivery"] + pd.Timedelta(days=40))
        ltd, ld = pd.Timestamp(c["last_trading"]), g["delivery"].iloc[0]
        month_start = bd[bd >= ltd.replace(day=1)][0]
        first_intention = bd[bd < month_start][-2]
        if c["delivery_kind"] == "first":  # negative carry: deliver at once - one window, no EOM
            n_eom = 0
            window_days = bd[bd >= max(first_intention, pd.Timestamp(c["day"]))][:1]
        else:
            n_eom = int(((bd > ltd) & (bd <= ld)).sum())
            window_days = bd[(bd >= max(first_intention, pd.Timestamp(c["day"]))) & (bd <= ltd)]
        n_windows = len(window_days)
        implied = (g["fwd"] / g["cf"]).to_numpy()
        ctd = int(np.argmin(implied))
        cf = g["cf"].to_numpy()
        eom = eom_switch_value(g["fwd"].to_numpy(), cf, float(implied[ctd]), ctd, g["dv01"].to_numpy(),
                               sigma * np.sqrt(max(n_eom, 0)), z) if n_eom > 0 else 0.0
        share, mult = self._window_variance(c["root"])
        kinds = window_kinds(window_days, [pd.Timestamp(m.end_date) for m in FOMC_MEETINGS])
        window_sd = [sigma * g["dv01"].iloc[ctd] * np.sqrt(share * mult[k]) for k in kinds]
        wild = wildcard_value(cf[ctd], window_sd, continuation=eom)
        wild_32, eom_32 = wild / cf[ctd] * TICKS, eom / cf[ctd] * TICKS
        return {"wildcard_32": wild_32, "eom_32": eom_32, "wildcard_windows": n_windows, "eom_days": n_eom,
                "wildcard_event_windows": sum(k != "ordinary" for k in kinds), "timing_32": wild_32 + eom_32}

    def _window_variance(self, root: str) -> tuple[float, dict]:
        """The root's ordinary-day window share and day-kind multipliers (spec)."""
        for r, share, fomc, qe, me in self.spec.wildcard_window:
            if r == root:
                return share, {"ordinary": 1.0, "fomc": fomc, "quarter_end": qe, "month_end": me}
        return self.spec.wildcard_var_share, {"ordinary": 1.0, "fomc": 1.0, "quarter_end": 1.0, "month_end": 1.0}

    def _extra_bonds(self, c) -> pd.DataFrame:
        """Deliverables not issued yet, to simulate with the basket (M1: none)."""
        return pd.DataFrame(columns=["cusip", "fwd", "cf", "coupon", "maturity", "implied_futures", "fwd_yield"])

    def _shocks(self, c, g: pd.DataFrame, n_bd: int, z: np.ndarray):
        """Yield shocks (bp) at delivery: one per path, shared by the basket."""
        return self.fitted_["vol_bp_day"][c["root"]] * np.sqrt(max(n_bd, 0)) * z

    def _pair_sd(self, c, g: pd.DataFrame, fdv: np.ndarray, i: int, j: int, n_bd: int) -> float:
        """Sd (futures points) of the implied-futures spread of bonds i and j at delivery."""
        return abs(fdv[i] - fdv[j]) * self.fitted_["vol_bp_day"][c["root"]] * np.sqrt(max(n_bd, 0))


class FactorBasis(OneFactorBasis):
    """M2: per contract, a factor model of the basket's yields (level random walk + spread
    covariance estimated AT the delivery horizon - PCA + idiosyncratic,
    ``infra.models.basis.factors``), fitted point in time on the yields published by the
    day; correlated shocks for every deliverable at the M0 delivery day, otherwise as M1.
    The level shock reuses M1's draws, so M1 and M2 differ only by the relative structure."""

    def __init__(self, spec: BasisSpec = BASIS_MODELS["M2"]):
        super().__init__(spec)

    def prepare(self, raw: BasisDay, **kwargs) -> pd.DataFrame:
        out = super().prepare(raw)
        for k in ("yields", "predecessor", "maturity", "future_issues"):
            out.attrs[k] = raw.meta.get(k)
        return out

    def fit(self, prepared, as_of=None) -> "FactorBasis":
        super().fit(prepared, as_of)
        yields = prepared.attrs.get("yields")
        if yields is None or yields.empty:
            raise ValueError("M2 needs the yield panel (meta['yields'] from infra.models.basis.inputs)")
        if as_of is not None:
            yields = yields[yields.index <= pd.Timestamp(as_of)]
        models, errors = {}, {}
        if prepared.empty:  # a day with no deliverable priced: nothing to fit
            self.fitted_ = dict(self.fitted_, tier="M2", factor_models=models, factor_errors=errors)
            return self
        day = pd.Timestamp(prepared["day"].iloc[0])
        future = prepared.attrs.get("future_issues")
        self._future = {}
        for contract, g in prepared.groupby("contract"):
            cusips = sorted(set(g["cusip"]))
            best = g.loc[g["implied_futures"].idxmin()]  # M0's choice: the horizon
            delivery = best["delivery"]
            horizon = len(business_days(day + pd.Timedelta(days=1), self._quality_horizon_end(best)))
            predecessor, maturity = dict(prepared.attrs["predecessor"]), dict(prepared.attrs["maturity"])
            if self.spec.future_issues and future is not None and len(future):
                fut = future[(future["contract"] == contract) & (future["issue_date"] <= delivery)]
                self._future[contract] = fut
                cusips += list(fut["cusip"])
                predecessor.update(zip(fut["cusip"], fut["predecessor"]))
                maturity.update(zip(fut["cusip"], fut["maturity"]))
            try:
                dy, src = change_panel(yields, cusips, predecessor, maturity, self.spec.factor_window)
                models[contract] = fit_factor_model(dy, horizon, k=self.spec.n_spread_factors,
                                                    lam_slow=self.spec.factor_lambda, lam_fast=self.spec.vol_lambda,
                                                    backfilled=src)
            except (ValueError, KeyError) as exc:
                errors[contract] = str(exc)
        self.fitted_ = dict(self.fitted_, tier="M2", factor_models=models, factor_errors=errors)
        return self

    def predict(self, prepared: pd.DataFrame, **kwargs) -> dict[str, pd.DataFrame]:
        out = super().predict(prepared)
        c = out["contracts"]
        if len(c):
            fm = self.fitted_["factor_models"]
            c["level_vol_bp_day"] = [fm[k].sigma_level if k in fm else np.nan for k in c["contract"]]
            for part in ("level", "spread_factors", "idio"):
                c[f"var_share_{part}"] = [fm[k].explained[part] if k in fm else np.nan for k in c["contract"]]
            c["n_backfilled"] = [len(fm[k].backfilled) if k in fm else np.nan for k in c["contract"]]
            c["spread_variance_ratio"] = [fm[k].variance_ratio if k in fm else np.nan for k in c["contract"]]
        return out

    def _extra_bonds(self, c) -> pd.DataFrame:
        """Expected new issues for this contract, priced on the delivery day at their
        reference yield (the latest same-tenor issue's, today) - no carry: nobody owns them
        before issue."""
        fut = getattr(self, "_future", {}).get(c["contract"])
        if fut is None or fut.empty:
            return super()._extra_bonds(c)
        fwd = [float(clean_price_from_yield(r.ref_yield, r.coupon, r.maturity, c["delivery"])) for r in fut.itertuples()]
        return pd.DataFrame({"cusip": fut["cusip"].to_numpy(), "fwd": fwd, "cf": fut["cf"].to_numpy(),
                             "coupon": fut["coupon"].to_numpy(), "maturity": list(fut["maturity"]),
                             "implied_futures": np.array(fwd) / fut["cf"].to_numpy(),
                             "fwd_yield": fut["ref_yield"].to_numpy()})

    def _shocks(self, c, g: pd.DataFrame, n_bd: int, z: np.ndarray):
        fm = self.fitted_["factor_models"].get(c["contract"])
        if fm is None:
            return None
        idx = [fm.cusips.index(x) for x in g["cusip"]]
        rng = np.random.default_rng(self.spec.seed + 1)
        shocks = simulate_shocks(fm, n_bd, z, rng, spread_df=self.spec.spread_df, idio_scale=self.spec.idio_scale)
        return shocks[:, idx]

    def _pair_sd(self, c, g: pd.DataFrame, fdv: np.ndarray, i: int, j: int, n_bd: int) -> float:
        fm = self.fitted_["factor_models"].get(c["contract"])
        if fm is None:
            return float("nan")
        a, b = fm.cusips.index(g["cusip"].iloc[i]), fm.cusips.index(g["cusip"].iloc[j])
        cov = fm.sigma_level ** 2 * max(n_bd, 0) + fm.phi @ fm.phi.T + np.diag(fm.psi)  # level + spreads at horizon
        var = fdv[i] ** 2 * cov[a, a] + fdv[j] ** 2 * cov[b, b] - 2 * fdv[i] * fdv[j] * cov[a, b]
        return float(np.sqrt(max(var, 0.0)))


TIERS = {"M0": DeterministicBasis, "M1": OneFactorBasis, "M2": FactorBasis}


def make_model(name: str) -> Model:
    spec = BASIS_MODELS[name]
    return TIERS[spec.tier](spec)
