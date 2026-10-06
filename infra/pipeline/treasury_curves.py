"""Our US Treasury zero curve, fitted daily from FedInvest's END OF DAY per-CUSIP prices
(``infra.analytics.treasury_curve``; methodology: infra/models/curves/CLAUDE.md). Reads
disk only; writes ``Derived/TreasuryCurves`` and ``Derived/TreasuryRV``.

Per day: the FIT universe = notes and bonds with > ``CURVE_FIT_MIN_YEARS`` left, minus the
on-the-run and first off-the-run of each tenor (``CURVE_FIT_EXCLUDE_RANKS``, issue
convention - their liquidity premium would bend the curve; the Fed's GSW curve excludes
them too). Every note and bond still gets its metrics. Prices: END OF DAY is a 15:30 BID
(CLAUDE.md 18); dirty = clean + accrued at the T+1 settlement. Point in time: a day's
curve uses that day's prices only (Svensson warm-starts from the previous day's fit -
a starting point, not information).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import treasury_curve as tc
from infra.config import (CURVE_FIT_EXCLUDE_RANKS, CURVE_FIT_MIN_YEARS, CURVE_GRID_YEARS, CURVE_HORIZON_DAYS,
                          TREASURY_CURVES_DIR, TREASURY_RV_DIR)
from infra.pipeline.treasury_otr import read_otr
from infra.pipeline.treasury_prices import read_prices
from infra.pipeline.treasury_ref import read_securities
from infra.processing.treasury_prices import settlement_day
from infra.storage import parquet_store

CURVE_KEYS = ["timestamp", "method"]


def callable_cusips(*, auctions_root: Path | None = None) -> set[str]:
    """CUSIPs auctioned as CALLABLE (Fiscal Data's ``callable`` field; 11 bonds, 1979-1984 -
    the Treasury stopped issuing callables in 1985). Out of the fit and the metrics: priced
    to their call, read as bullets they 'yield' 10-13% - found 2026-10-04, they put our
    2008-09 curve 150-260bp off the Fed's GSW (fit error 117-124bp) until they matured or
    were called (the last in 2014)."""
    import json
    from infra.pipeline.tsy_auctions import read_auctions
    a = read_auctions() if auctions_root is None else read_auctions(root=auctions_root)
    flag = a["raw_json"].map(lambda s: json.loads(s).get("callable") == "Yes")
    return set(a.loc[flag, "cusip"].astype(str))
RV_KEYS = ["timestamp", "cusip", "method"]


def fit_day(day, prices: pd.DataFrame, sec: pd.DataFrame, excluded: set[str], *, svensson_start=None,
            horizon_days: int = CURVE_HORIZON_DAYS) -> tuple[list[dict], list[dict], np.ndarray | None]:
    """One day: (curve rows, bond rows, Svensson params for the next warm start)."""
    settle = settlement_day(day)
    p = prices.dropna(subset=["price_eod", "accrued"]).merge(sec, on="cusip")
    p = p[(p["price_eod"] > 0) & (pd.to_datetime(p["maturity_date"]) > settle + pd.Timedelta(days=30))]
    bonds, dirty, info = [], [], []
    for r in p.itertuples(index=False):
        t, a = tc.cash_flows(float(r.coupon), r.maturity_date, settle, int(r.coupons_per_year or 2))
        if t.size == 0:
            continue
        d = float(r.price_eod) + float(r.accrued)
        try:
            y = tc.ytm(d, t, a, guess=float(r.yield_eod) / 100 if pd.notna(getattr(r, "yield_eod", np.nan)) else 0.04)
        except ValueError:
            continue
        dur = tc.modified_duration(d, t, a, y)
        bonds.append((t, a)); dirty.append(d)
        info.append((r.cusip, float(t[-1]), dur, y, float(r.accrued)))
    if len(bonds) < 20:
        return [], [], svensson_start
    dirty = np.array(dirty)
    cusip = np.array([i[0] for i in info]); mat = np.array([i[1] for i in info]); dur = np.array([i[2] for i in info])
    fit = (mat >= CURVE_FIT_MIN_YEARS) & ~np.isin(cusip, list(excluded))
    w = 1.0 / (dirty * dur) ** 2
    fb = [b for b, f in zip(bonds, fit) if f]
    curves, rows = [], []
    sp, sp_res, h = tc.fit_spline(fb, dirty[fit], w[fit])
    sv, sv_res = tc.fit_svensson(fb, dirty[fit], w[fit], start=svensson_start)
    for name, curve, res, params in (("spline", sp, sp_res, sp.beta), ("svensson", sv, sv_res, sv.params)):
        rmse_bp = float(np.sqrt(np.mean((res / (dirty[fit] * dur[fit]) * 1e4) ** 2)))
        row = {"timestamp": pd.Timestamp(day), "method": name, "n_fit": int(fit.sum()), "rmse_bp": rmse_bp,
               **{f"p{k}": float(v) for k, v in enumerate(params)}}
        for m in CURVE_GRID_YEARS:
            row[f"par_{m}y"] = tc.par_yield(curve, m)
            row[f"zero_{m}y"] = float(tc.zero_rate(curve, m)[0])
        curves.append(row)
        loo = np.full(len(bonds), np.nan)
        if name == "spline":
            loo[np.where(fit)[0]] = 1.0 / np.maximum(1.0 - h, 1e-6)  # closed-form leave-one-out scaling
        for k, ((t, a), d, (c, m, du, y, acc)) in enumerate(zip(bonds, dirty, info)):
            try:
                mtr = tc.bond_metrics(curve, d, acc, t, a, horizon_days=horizon_days)
            except ValueError:
                continue
            rows.append({"timestamp": pd.Timestamp(day), "cusip": c, "method": name, "in_fit": bool(fit[k]),
                         "maturity_years": m, "ytm": mtr["ytm"], "zspread_bp": mtr["zspread_bp"],
                         "zspread_loo_bp": mtr["zspread_bp"] * (loo[k] if np.isfinite(loo[k]) else 1.0)
                         if name == "spline" else np.nan,
                         "carry_curve_bp": mtr["carry_curve_bp"], "rolldown_bp": mtr["rolldown_bp"]})
    return curves, rows, sv.params


def stored_svensson_seed(before, *, curves_root: Path = TREASURY_CURVES_DIR) -> np.ndarray | None:
    """The Svensson parameters STORED for the last day before ``before`` - the warm start that
    makes an incremental run (the daily cycle's window) identical to a full sequential build:
    each day's fit starts from the previous day's fitted parameters either way. None if
    nothing is stored before it (a cold start)."""
    if not parquet_store.has_data(curves_root):
        return None
    c = read_curves(pd.Timestamp(before) - pd.Timedelta(days=30), pd.Timestamp(before) - pd.Timedelta(days=1),
                    method="svensson", root=curves_root)
    if c.empty:
        return None
    row = c.iloc[-1]
    return np.array([row[f"p{k}"] for k in range(6)], dtype="float64")


def compute_curves(start, end, *, prices_root: Path | None = None, securities_root: Path | None = None,
                   otr_root: Path | None = None, auctions_root: Path | None = None,
                   curves_root: Path = TREASURY_CURVES_DIR, log=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every business day with prices in ``[start, end]`` - no writing. Returns (curve
    rows, bond rows, diagnostics: ``days`` with prices, ``empty_days`` {day: reason}). Inputs
    read from the given roots (the daily cycle's ``CyclePaths``) or the defaults. Svensson is
    seeded from the stored fit of the day before ``start`` (``stored_svensson_seed``)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    kw = lambda root: {} if root is None else {"root": root}
    prices = read_prices(start, end + pd.Timedelta(days=1), **kw(prices_root))
    prices = prices[prices["security_type"].astype(str).isin(["Note", "Bond"])]
    prices["cusip"] = prices["cusip"].astype(str)
    sec = read_securities(**kw(securities_root)).drop_duplicates("cusip")[["cusip", "coupon", "maturity_date", "coupons_per_year"]]
    sec["cusip"] = sec["cusip"].astype(str)
    sec = sec[~sec["cusip"].isin(callable_cusips(auctions_root=auctions_root))]
    # end is EXCLUSIVE in read_otr: +1 day, or the window's last day fits WITH its on-the-runs
    # (found 2026-10-05: every 31 December of the first build)
    otr = read_otr(start - pd.Timedelta(days=5), end + pd.Timedelta(days=1), **kw(otr_root))
    otr = otr[(otr["convention"] == "issue") & (otr["rank"] < CURVE_FIT_EXCLUDE_RANKS)]
    excl = otr.groupby("timestamp")["cusip"].apply(lambda s: set(s.astype(str)))
    start_params = stored_svensson_seed(start, curves_root=curves_root)
    all_c, all_r, days, empty = [], [], [], {}
    for i, (day, g) in enumerate(prices.groupby("timestamp")):
        if not g["price_eod"].gt(0).any():
            continue  # END OF DAY not posted yet (FedInvest posts day D ~10:00 New York on D+1): not an input day
        days.append(pd.Timestamp(day))
        ex = excl.get(day, set())
        c, r, start_params = fit_day(day, g, sec, ex, svensson_start=start_params)
        if not c:
            empty[pd.Timestamp(day)] = f"too few priced notes / bonds to fit ({g['price_eod'].gt(0).sum()} with a price)"
        all_c += c; all_r += r
        if log and i % 50 == 0:
            log(f"{pd.Timestamp(day).date()} n_fit={c[0]['n_fit'] if c else 0}")
    cdf, rdf = pd.DataFrame(all_c), pd.DataFrame(all_r)
    for df in (cdf, rdf):
        if not df.empty:
            df["timestamp"] = df["timestamp"].astype("datetime64[ms]")
    return cdf, rdf, {"days": days, "empty_days": empty}


def build_curves(start, end, *, curves_root: Path = TREASURY_CURVES_DIR, rv_root: Path = TREASURY_RV_DIR,
                 log=None) -> dict:
    """``compute_curves`` + write: a rebuilt day replaces its rows in both stores."""
    cdf, rdf, _ = compute_curves(start, end, curves_root=curves_root, log=log)
    if not cdf.empty:
        days = sorted(cdf["timestamp"].unique())
        parquet_store.prune_rows(curves_root, "timestamp", days) if parquet_store.has_data(curves_root) else None
        parquet_store.prune_rows(rv_root, "timestamp", days) if parquet_store.has_data(rv_root) else None
        parquet_store.write_partitioned(cdf, curves_root, CURVE_KEYS)
        parquet_store.write_partitioned(rdf, rv_root, RV_KEYS)
    return {"days": int(cdf["timestamp"].nunique()) if not cdf.empty else 0, "bond_rows": len(rdf)}


def read_curves(start=None, end=None, *, method: str | None = None, root: Path = TREASURY_CURVES_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1))
    if df is None:
        return pd.DataFrame(columns=CURVE_KEYS)
    df["method"] = df["method"].astype(str)
    return (df if method is None else df[df["method"] == method]).sort_values(CURVE_KEYS, ignore_index=True)


def read_rv(start=None, end=None, *, method: str | None = "spline", cusips=None, root: Path = TREASURY_RV_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1),
                                        equals_in={"cusip": list(cusips)} if cusips is not None else None)
    if df is None:
        return pd.DataFrame(columns=RV_KEYS)
    df["method"] = df["method"].astype(str); df["cusip"] = df["cusip"].astype(str)
    return (df if method is None else df[df["method"] == method]).sort_values(RV_KEYS, ignore_index=True)
