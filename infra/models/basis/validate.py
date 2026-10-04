"""The validation bench every basis tier is scored on (infra/models/basis/CLAUDE.md):

1. OPTION VALUE: the model's option value against the observed one (fair minus market
   futures, 32nds), by root, year and time to delivery. For M0 the model value is 0, so
   this measures what the delivery options (+ any futures richness) are worth.
2. CTD CALIBRATION: the model's delivery probabilities against the REALISED CTD - the
   bond with the lowest implied futures price at the contract's last trading day (M0 at
   that day: forwards only, no futures price needed). Brier score per (day, contract):
   sum over bonds of (p - realised)^2, 0 = perfect, 2 = sure and wrong.
3. TRACKING: daily change of the market futures price against the model's fair price
   (same contract both days), and the futures DV01 against the market's realised
   sensitivity to the CTD's yield.
Pure functions over ``predict`` outputs, plus ``run`` (loops a model over stored days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.models.basis.config import BASIS_MODELS
from infra.models.basis.inputs import basis_days
from infra.models.basis.model import TIERS, make_model  # noqa: F401 (make_model: callers)


# per-bond output kept from a run (the basis dashboard reads these): identity, the basis
# metrics, the model's delivery probability
BOND_COLUMNS = ["day", "root", "contract", "cusip", "delivery_kind", "delivery", "coupon", "maturity", "cf",
                "price", "price_bid", "yield_eod", "dv01", "repo", "carry", "fwd", "implied_futures", "gross_basis",
                "net_basis", "irr", "futures", "prob"]


def run(name, start, end, *, contracts: str = "quoted", log_every: int = 0, days=None) -> dict[str, pd.DataFrame]:
    """Fit (point in time, per day) and predict ``name`` (a registered spec name, or a
    ``BasisSpec``) over ``[start, end]`` - every day, or only ``days`` (a sample)."""
    spec = BASIS_MODELS[name] if isinstance(name, str) else name
    model, out_c, out_b = TIERS[spec.tier](spec), [], []
    for i, raw in enumerate(basis_days(spec, start, end, contracts=contracts, days=days)):
        prepared = model.prepare(raw)
        pred = model.fit(prepared, as_of=raw.day).predict(prepared)
        out_c.append(pred["contracts"])
        b = pred["bonds"]
        out_b.append(b[[c for c in BOND_COLUMNS if c in b.columns]] if len(b) else pd.DataFrame())
        if log_every and i % log_every == 0:
            print(f"{raw.day.date()}", flush=True)
    if not out_c:
        return {"contracts": pd.DataFrame(), "bonds": pd.DataFrame()}
    return {"contracts": pd.concat(out_c, ignore_index=True), "bonds": pd.concat(out_b, ignore_index=True)}


def option_value_bench(contracts: pd.DataFrame) -> pd.DataFrame:
    """Observed vs model option value (32nds) by root: median, p5, share < 0, error."""
    c = contracts.dropna(subset=["option_value_obs_32"])
    err = c["option_value_obs_32"] - c["option_value_model_32"]
    return pd.DataFrame({
        "n": c.groupby("root").size(),
        "obs_median": c.groupby("root")["option_value_obs_32"].median(),
        "obs_p5": c.groupby("root")["option_value_obs_32"].quantile(0.05),
        "obs_share_neg": c.groupby("root")["option_value_obs_32"].apply(lambda s: (s < 0).mean()),
        "model_median": c.groupby("root")["option_value_model_32"].median(),
        "mae_32": err.abs().groupby(c["root"]).mean(),
        "bias_32": err.groupby(c["root"]).mean(),
    })


def realised_ctd(contracts: pd.DataFrame, name: str = "M0") -> pd.DataFrame:
    """The realised CTD of each contract seen in ``contracts`` that has traded out - where
    the shorts actually delivered, by M0 on the day that decides it:
    1. on the delivery month's FIRST INTENTION day (2 business days before its first
       business day), M0 picks early vs late delivery by carry; if EARLY (negative carry),
       its CTD is the realised one - the shorts deliver that week;
    2. otherwise M0's CTD on the LAST TRADING day.
    Scoring every contract at its last trading day (the first version) was wrong whenever
    carry was negative: for ZT, whose last trading day is the delivery month's END, every
    2022-2025 contract "missed" - the low-coupon note was correctly the early-June CTD, and
    the bond cheapest a month later was beside the point. A fact about the market, the same
    for every tier, so computed once and reused."""
    from infra.analytics.futures_basis import delivery_window
    from infra.analytics.sofr_curve import business_days
    last_day = pd.Timestamp(contracts["day"].max())
    meta = contracts.drop_duplicates("contract")[["contract", "root", "last_trading"]].copy()
    bd = business_days(pd.Timestamp(contracts["day"].min()) - pd.Timedelta(days=30), last_day + pd.Timedelta(days=200))
    meta["first_intention"] = [
        bd[bd < delivery_window(r, pd.Timestamp(ltd).replace(day=1), bd)["first_delivery"]][-2]
        for r, ltd in zip(meta["root"], meta["last_trading"])]
    out = {}
    for fid, g in meta.groupby("first_intention"):
        if pd.Timestamp(fid) > last_day:
            continue
        res = run(name, fid, fid, contracts="all")["contracts"]
        if len(res):
            for _, r in res[res["contract"].isin(set(g["contract"])) & (res["delivery_kind"] == "first")].iterrows():
                out[r["contract"]] = (r["ctd"], "first", pd.Timestamp(fid))
    for ltd, g in meta[~meta["contract"].isin(out)].groupby("last_trading"):
        if pd.Timestamp(ltd) > last_day:
            continue
        res = run(name, ltd, ltd, contracts="all")["contracts"]
        if len(res):
            for _, r in res[res["contract"].isin(set(g["contract"]))].iterrows():
                out[r["contract"]] = (r["ctd"], r["delivery_kind"], pd.Timestamp(ltd))
    return pd.DataFrame([(k, *v) for k, v in out.items()],
                        columns=["contract", "realised_ctd", "realised_kind", "decided_on"])


def resolve_expected_issues(cusips: pd.Series, securities: pd.DataFrame) -> pd.Series:
    """Placeholder ``NEW:<tenor>:<issue date>`` -> the actual CUSIP of that tenor issued
    within 5 days of it (SCORING ONLY - it looks ahead; the model never sees it). Other
    cusips pass through; an expected issue that never came stays a placeholder."""
    from infra.config import TREASURY_OTR_TENORS
    out = cusips.astype(str).copy()
    new = out.str.startswith("NEW:")
    if not new.any():
        return out
    for i in out.index[new]:
        _, tenor, day = out[i].split(":")
        stype, term = TREASURY_OTR_TENORS[tenor]
        s = securities[(securities["security_type"] == stype) & (securities["original_term"] == term)]
        d = (pd.to_datetime(s["issue_date"]) - pd.Timestamp(day)).abs()
        if len(s) and d.min() <= pd.Timedelta(days=5):
            out[i] = str(s.loc[d.idxmin(), "cusip"])
    return out


def ctd_calibration(bonds: pd.DataFrame, realised: pd.DataFrame, contracts: pd.DataFrame,
                    securities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Brier score and hit rate of the delivery probabilities, by root and days to the
    last trading day. Expected-issue placeholders count as the actual issue they became
    (``securities``: the reference table; read if not given)."""
    if bonds["cusip"].astype(str).str.startswith("NEW:").any():
        if securities is None:
            from infra.pipeline.treasury_ref import read_securities
            securities = read_securities()
        bonds = bonds.assign(cusip=resolve_expected_issues(bonds["cusip"], securities))
    p = bonds.groupby(["day", "contract", "cusip"])["prob"].sum().reset_index()
    p = p.merge(realised[["contract", "realised_ctd"]], on="contract")
    p["hit"] = (p["cusip"] == p["realised_ctd"]).astype(float)
    brier = p.assign(sq=(p["prob"] - p["hit"]) ** 2).groupby(["day", "contract"])["sq"].sum()
    # a realised CTD the model gave no row at all (not in its basket that day) still counts
    covered = p.groupby(["day", "contract"])["hit"].sum()
    brier = brier + (1 - covered.clip(upper=1))
    top = p.loc[p.groupby(["day", "contract"])["prob"].idxmax()].set_index(["day", "contract"])["hit"]
    meta = contracts.set_index(["day", "contract"])[["root", "last_trading"]]
    df = pd.DataFrame({"brier": brier, "hit": top}).join(meta, how="inner").reset_index()
    df["days_to_ltd"] = (pd.to_datetime(df["last_trading"]) - pd.to_datetime(df["day"])).dt.days
    df["bucket"] = pd.cut(df["days_to_ltd"], [-1, 10, 30, 60, 90, 400])
    return df.groupby(["root", "bucket"], observed=True).agg(n=("brier", "size"), brier=("brier", "mean"),
                                                             hit_rate=("hit", "mean"))


def tracking_bench(contracts: pd.DataFrame) -> pd.DataFrame:
    """Daily changes (same contract and source both days): std of market minus fair
    (32nds), and the regression slope of the market change on -DV01 x CTD yield change
    (1 = the model's DV01 is right on average)."""
    c = contracts.sort_values(["contract", "day"]).copy()
    g = c.groupby("contract")
    same = (g["futures_source"].shift() == c["futures_source"]) & (g["ctd"].shift() == c["ctd"])
    d_mkt = g["futures"].diff()
    d_fair = g["fair_futures"].diff()
    pred = -g["futures_dv01"].shift() * g["ctd_yield"].diff() * 100
    c = c.assign(d_mkt=d_mkt, d_fair=d_fair, pred=pred)[same].dropna(subset=["d_mkt", "d_fair", "pred"])

    def stats(x):
        slope = (x["pred"] * x["d_mkt"]).sum() / (x["pred"] ** 2).sum()
        return pd.Series({"n": len(x), "mkt_minus_fair_sd_32": ((x["d_mkt"] - x["d_fair"]) * 32).std(),
                          "mkt_sd_32": (x["d_mkt"] * 32).std(), "dv01_slope": slope,
                          "dv01_r2": 1 - ((x["d_mkt"] - x["pred"]) ** 2).sum() / ((x["d_mkt"] - x["d_mkt"].mean()) ** 2).sum()})
    return c.groupby("root").apply(stats)
