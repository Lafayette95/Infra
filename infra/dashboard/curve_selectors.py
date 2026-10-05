"""Data for the bond-curve page (``/curve``): one day's fitted Treasury curve, every note and
bond on it with its relative-value metrics, on-the-run status, futures-basket membership and
any stored CTD - all READ from stores (our curve, ``Derived/TreasuryCurves`` / ``TreasuryRV``;
the OTR map; the delivery baskets; stored basis runs). Never fits or runs a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.analytics import treasury_curve as tc
from infra.pipeline.futures_baskets import read_baskets
from infra.pipeline.treasury_curves import read_curves, read_rv
from infra.pipeline.treasury_otr import read_otr
from infra.pipeline.treasury_ref import read_securities
from infra.storage import basis_runs

ROOT_ORDER = ["ZT", "Z3N", "ZF", "ZN", "TN", "TWE", "ZB", "UB"]


def curve_days(method: str = "spline") -> list[pd.Timestamp]:
    c = read_curves(method=method)
    return sorted(pd.to_datetime(c["timestamp"].unique()), reverse=True) if not c.empty else []


def curve_object(row: pd.Series, method: str):
    """The fitted curve rebuilt from a stored parameter row."""
    params = np.array([row[f"p{k}"] for k in range(12) if f"p{k}" in row.index and pd.notna(row[f"p{k}"])])
    return tc.SplineCurve(params) if method == "spline" else tc.SvenssonCurve(params[:6])


def par_line(curve, max_years: float = 30.0, step: float = 0.25) -> pd.DataFrame:
    m = np.arange(0.5, max_years + 1e-9, step)
    return pd.DataFrame({"maturity_years": m, "par_yield": [tc.par_yield(curve, x) for x in m]})


def _ctds(day: pd.Timestamp) -> pd.DataFrame:
    """Stored CTDs that day (M2T preferred, else M2, else none): each contract's MOST LIKELY
    deliverable under the model (``top_prob_bond`` - not M0's deterministic CTD, which can
    carry little probability under the simulation), with that probability."""
    for model in ("M2T", "M2"):
        if model in basis_runs.models():
            c = basis_runs.read(model, "contracts", day, day)
            if not c.empty:
                c = c.dropna(subset=["top_prob_bond"])
                return c[["contract", "root", "top_prob_bond", "top_prob"]].rename(
                    columns={"top_prob_bond": "cusip", "top_prob": "ctd_prob"}).assign(model=model)
    return pd.DataFrame(columns=["contract", "root", "cusip", "ctd_prob", "model"])


def curve_view(day, method: str = "spline") -> dict:
    """Everything the page draws for one day: ``line`` (the par curve), ``bonds`` (one row
    per note / bond), ``sectors`` (per futures root: the maturity span of its front
    contract's deliverables), ``ctds``, ``fit`` (n bonds, fit error)."""
    day = pd.Timestamp(day)
    crow = read_curves(day, day, method=method)
    if crow.empty:
        return {"line": pd.DataFrame(), "bonds": pd.DataFrame(), "sectors": pd.DataFrame(), "ctds": pd.DataFrame(), "fit": {}}
    curve = curve_object(crow.iloc[0], method)
    rv = read_rv(day, day, method=method)
    sec = read_securities().drop_duplicates("cusip")
    sec["cusip"] = sec["cusip"].astype(str)
    b = rv.merge(sec[["cusip", "coupon", "issue_date", "maturity_date", "original_term", "security_type"]], on="cusip", how="left")
    otr = read_otr(day, day + pd.Timedelta(days=1))  # end is exclusive
    otr = otr[otr["convention"] == "issue"][["cusip", "tenor", "rank"]].astype({"cusip": str})
    b = b.merge(otr, on="cusip", how="left")
    bk = read_baskets(day, day + pd.Timedelta(days=1))
    bk["cusip"] = bk["cusip"].astype(str)
    if not bk.empty:
        member = bk.groupby("cusip").apply(lambda g: ", ".join(f"{r.contract} (CF {r.conversion_factor:.4f})"
                                                               for r in g.sort_values("contract").itertuples()))
        b["baskets"] = b["cusip"].map(member).fillna("")
        # each root's FRONT contract (earliest delivery month) - its deliverables' maturity span
        front = bk.sort_values("delivery_month").groupby("root")["contract"].first()
        fb = bk[bk["contract"].isin(front.values)].merge(b[["cusip", "maturity_years"]], on="cusip")
        sectors = fb.groupby(["root", "contract"])["maturity_years"].agg(["min", "max", "size"]).reset_index()
        sectors["order"] = sectors["root"].map({r: i for i, r in enumerate(ROOT_ORDER)})
        sectors = sectors.sort_values("order").drop(columns="order")
    else:
        b["baskets"] = ""
        sectors = pd.DataFrame(columns=["root", "contract", "min", "max", "size"])
    ctds = _ctds(day)
    if not ctds.empty:
        lab = ctds.groupby("cusip").apply(lambda g: ", ".join(f"{r.contract} {r.ctd_prob:.0%}" for r in g.itertuples()))
        b["ctd_of"] = b["cusip"].map(lab).fillna("")
    else:
        b["ctd_of"] = ""
    b["status"] = np.select([b["rank"] == 0, b["rank"].between(1, 5)], ["on-the-run", "old"], "other")
    b["curve_yield"] = b["ytm"] - b["zspread_bp"] / 100.0  # roughly where the curve prices it
    return {"line": par_line(curve), "bonds": b.sort_values("maturity_years", ignore_index=True), "sectors": sectors,
            "ctds": ctds, "fit": {"n_fit": int(crow["n_fit"].iloc[0]), "rmse_bp": float(crow["rmse_bp"].iloc[0])}}
