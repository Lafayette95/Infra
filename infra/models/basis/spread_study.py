"""The explanatory basis-spread STUDY (infra/models/basis/CLAUDE.md) - not part of any
pricing tier (user decision 2026-10-03): what explains the basis the delivery models leave
unexplained, and a CALIBRATION of that residual so other add-ons (IV) can be tested
against a cleaner target.

Target, per contract-day: ``residual_32`` = observed option value (M0 fair - market,
32nds) - the model's option value (M2T by default: quality + timing). Positive = futures
CHEAP to the model; negative = RICH.

Drivers (``drivers``), all point in time - known by the close of the day:
* ``carry_bp``      - the CTD's yield minus its funding rate (bp); ``neg_carry`` its sign.
* ``client_bp``     - SOFR's 75th percentile minus its median (the hedge-fund funding
                      premium over the median, funding v1's client basis); ``tail_bp`` the
                      99th minus the 75th. SOFR day D is published D+1: the latest by D-1.
* ``term_bp``       - OFR DVP >30-day minus overnight (the term repo premium).
* ``lev_net``/``am_net`` - leveraged funds' / asset managers' net position, % of open
                      interest, in the contract's own TFF market, as RELEASED by the day.
* ``dealer_tsy``    - primary dealers' net Treasury positions ($bn), as released, MINUS
                      their own trailing 52-week mean: the level trended from ~$100bn to
                      ~$470bn over 2019-2026, and a raw level made the out-of-sample fit
                      extrapolate (R^2 -11.8, found 2026-10-03).
* ``qe_window``     - 1 within 10 business days before a quarter end.
* ``days_to_delivery``, and root fixed effects.

``fit`` is OLS with root dummies (standard errors clustered by date); ``oos_fitted`` refits
on all years before each year and predicts that year (expanding window) - the calibration
series: ``residual_32 - oos_fitted`` is what's left once the explainable part is removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TFF_MARKETS = {"ZT": "042601", "ZF": "044601", "ZN": "043602", "TN": "043607", "ZB": "020601", "UB": "020604"}
DRIVERS = ["carry_bp", "neg_carry", "client_bp", "tail_bp", "term_bp", "lev_net", "am_net", "dealer_tsy",
           "qe_window", "days_to_delivery"]


def _asof_join(left: pd.DataFrame, right: pd.DataFrame, on_left: str, on_right: str, by=None) -> pd.DataFrame:
    l = left.assign(_k=pd.to_datetime(left[on_left]).astype("datetime64[ns]")).sort_values("_k")
    r = right.assign(_k=pd.to_datetime(right[on_right]).astype("datetime64[ns]")).sort_values("_k").drop(columns=[on_right])
    out = pd.merge_asof(l, r, on="_k", by=by, direction="backward")
    return out.drop(columns="_k")


def drivers(contracts: pd.DataFrame) -> pd.DataFrame:
    """``contracts`` (a model run's contract rows: day, root, contract, ctd_yield,
    ctd_repo, days_to_delivery, option_value_obs_32, option_value_model_32) + the
    drivers and ``residual_32``. Reads the stores (repo, TFF, primary dealers)."""
    from infra.pipeline.cftc_tff import read_tff
    from infra.pipeline.primary_dealer import read_primary_dealer
    from infra.pipeline.repo import read_repo
    c = contracts.dropna(subset=["option_value_obs_32", "option_value_model_32"]).copy()
    c["day"] = pd.to_datetime(c["day"])
    c["residual_32"] = c["option_value_obs_32"] - c["option_value_model_32"]
    c["carry_bp"] = (c["ctd_yield"] - c["ctd_repo"]) * 100
    c["neg_carry"] = (c["carry_bp"] < 0).astype(float)
    lo, hi = c["day"].min() - pd.Timedelta(days=60), c["day"].max() + pd.Timedelta(days=2)
    # repo: day D published D+1 -> known from D+1
    r = read_repo(lo, hi, series=["SOFR", "OFR_DVP_OO", "OFR_DVP_G30"])
    sofr = r[r["series"] == "SOFR"].assign(client_bp=lambda x: (x["p75"] - x["rate"]) * 100,
                                          tail_bp=lambda x: (x["p99"] - x["p75"]) * 100)
    sofr = sofr.assign(known=sofr["timestamp"] + pd.Timedelta(days=1))[["known", "client_bp", "tail_bp"]]
    w = r[r["series"].isin(["OFR_DVP_OO", "OFR_DVP_G30"])].pivot_table(index="timestamp", columns="series", values="rate")
    term = ((w["OFR_DVP_G30"] - w["OFR_DVP_OO"]) * 100).rename("term_bp").reset_index()
    term = term.assign(known=term["timestamp"] + pd.Timedelta(days=1))[["known", "term_bp"]]
    c = _asof_join(c, sofr, "day", "known")
    c = _asof_join(c, term, "day", "known")
    # TFF by the contract's own market, as released
    tff = read_tff(lo - pd.Timedelta(days=30), hi, markets=list(TFF_MARKETS.values()))
    oi = tff["open_interest_all"].astype(float)
    tff = tff.assign(lev_net=(tff["lev_money_positions_long"] - tff["lev_money_positions_short"]).astype(float) / oi * 100,
                     am_net=(tff["asset_mgr_positions_long"] - tff["asset_mgr_positions_short"]).astype(float) / oi * 100,
                     root=tff["market_code"].astype(str).map({v: k for k, v in TFF_MARKETS.items()}))
    tff = tff.dropna(subset=["root"])[["known_from", "root", "lev_net", "am_net"]]
    c = _asof_join(c, tff, "day", "known_from", by="root")
    pdl = read_primary_dealer(["PDPOSGST-TOT"], lo - pd.Timedelta(days=30), hi)
    pdl = pdl.sort_values("timestamp")
    lvl = pdl["value"] / 1000
    pdl = pdl.assign(dealer_tsy=lvl - lvl.rolling(52, min_periods=26).mean())[["known_from", "dealer_tsy"]]
    c = _asof_join(c, pdl, "day", "known_from")
    bd = pd.bdate_range(lo, hi + pd.Timedelta(days=120))
    qe = bd[(bd + pd.offsets.BDay(1)).quarter != bd.quarter]  # last business day of each quarter
    nxt = qe[qe.searchsorted(c["day"].to_numpy())]
    c["qe_window"] = ((nxt - c["day"].to_numpy()).days <= 14).astype(float)
    return c.sort_values(["day", "root"], ignore_index=True)


def _design(df: pd.DataFrame, drivers_=DRIVERS, roots=None) -> tuple[np.ndarray, list[str]]:
    roots = sorted(df["root"].unique()) if roots is None else roots
    cols = list(drivers_) + [f"root_{r}" for r in roots]
    X = np.column_stack([df[d].to_numpy(dtype=float) for d in drivers_]
                        + [(df["root"] == r).to_numpy(dtype=float) for r in roots])
    return X, cols


def fit(df: pd.DataFrame, drivers_=DRIVERS) -> pd.DataFrame:
    """OLS of ``residual_32`` on the drivers + root dummies (no common intercept), drivers
    standardised (coefficients = 32nds per 1 sd); standard errors clustered by DATE (the
    roots of one day share every market driver). Returns coef, se, t; R^2 in attrs."""
    d = df.dropna(subset=["residual_32", *drivers_]).copy()
    z = {k: (d[k].mean(), d[k].std() or 1.0) for k in drivers_}
    for k, (m, s) in z.items():
        d[k] = (d[k] - m) / s
    X, cols = _design(d, drivers_)
    y = d["residual_32"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in d.groupby("day").indices.items():
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    out = pd.DataFrame({"coef": beta, "se": se, "t": beta / se}, index=cols)
    root_means = d.groupby("root")["residual_32"].transform("mean").to_numpy()
    out.attrs.update({"n": len(d), "r2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                      "r2_within": 1 - (e ** 2).sum() / ((y - root_means) ** 2).sum()})
    return out


def oos_fitted(df: pd.DataFrame, drivers_=DRIVERS, min_years: int = 2) -> pd.Series:
    """Expanding-window out-of-sample fitted residual: each year predicted by a fit on
    all EARLIER years (raw drivers; NaN for the first ``min_years``)."""
    d = df.dropna(subset=["residual_32", *drivers_])
    years = sorted(d["day"].dt.year.unique())
    out = pd.Series(np.nan, index=df.index)
    for y in years[min_years:]:
        train, test = d[d["day"].dt.year < y].copy(), d[d["day"].dt.year == y].copy()
        for k in drivers_:  # standardise on the TRAINING window only
            m, s = train[k].mean(), train[k].std() or 1.0
            train[k], test[k] = (train[k] - m) / s, (test[k] - m) / s
        roots = sorted(set(train["root"]) | set(test["root"]))
        Xtr, _ = _design(train, drivers_, roots)
        beta, *_ = np.linalg.lstsq(Xtr, train["residual_32"].to_numpy(), rcond=None)
        Xte, _ = _design(test, drivers_, roots)
        out.loc[test.index] = Xte @ beta
    return out


def fit_by_root(df: pd.DataFrame, drivers_=DRIVERS) -> pd.DataFrame:
    """``fit`` per root (a constant instead of root dummies): t-statistics by root and
    driver, with each root's n and R^2."""
    rows = {}
    for root, g in df.groupby("root"):
        g = g.assign(root="_")
        f = fit(g, drivers_)
        rows[root] = pd.concat([f["t"].drop("root__"), pd.Series({"n": f.attrs["n"], "r2": f.attrs["r2_within"]})])
    return pd.DataFrame(rows)
