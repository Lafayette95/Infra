"""Our German Federal zero curve, fitted daily to the Bundesbank's per-ISIN prices - the US
curve's own fit (``infra.pipeline.treasury_curves.fit_day``: a spline on the discount
function and a Svensson fit, plus per-bond z-spread, curve carry and rolldown) on the Bund
universe. Reads disk only; writes ``Derived/BundCurves`` (keys ``timestamp``, ``method``) and
``Derived/BundRV`` (keys ``timestamp``, ``cusip`` = the ISIN, ``method``).

Per day: prices are the Bundesbank's DIRTY price (KDP) and its accrued = dirty - clean
(published from 2022-06, so the curve starts at ``BUND_CURVES_START``: German first coupons
are often long or short and the issuance file doesn't give the dates, while the Bundesbank's
accrued has them); cash flows on a regular annual schedule from maturity, settlement T+2
weekdays. The FIT universe: conventional Schatz / Bobls / Bunds over ``CURVE_FIT_MIN_YEARS``,
minus the on-the-run and first off-the-run of each tenor (``CURVE_FIT_EXCLUDE_RANKS``, as for
the US), Green and inflation-linked bonds, and bonds still in their FIRST coupon period. Those
get their own cash flows (``first_periods`` + ``infra.processing.bunds.first_period_flows``:
interest from the start implied by the published accrued, a long first coupon unless the data
show a short one) and their metrics; every conventional bond gets metrics, flagged ``in_fit``. Par yields with
SEMI-ANNUAL coupons, the convention of every stored curve (CLAUDE.md 13).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.config import (BUND_CURVE_KNOTS, BUND_CURVES_DIR, BUND_CURVES_START, BUND_FIT_EXCLUDE_RANKS,
                          BUND_IRREGULAR_DAYS, BUND_RV_DIR, BUND_SETTLEMENT_DAYS, DAILY_BUND_PRICES_DIR, DE_AUCTIONS_DIR)
from infra.pipeline import bunds
from infra.processing import bunds as pb
from infra.pipeline.treasury_curves import CURVE_KEYS, RV_KEYS, fit_day, read_curves, read_rv, stored_svensson_seed
from infra.processing.treasury_prices import coupon_dates
from infra.storage import parquet_store

CONVENTIONAL = ("Schatz", "Bobl", "Bund")


class _Row:
    """The attributes ``flows`` reads, for a security outside the price row."""
    def __init__(self, isin, b):
        self.cusip, self.coupon, self.maturity_date = isin, b["coupon"], b["maturity_date"]


def settlement(day) -> pd.Timestamp:
    return pd.Timestamp(day).normalize() + pd.offsets.BDay(BUND_SETTLEMENT_DAYS)


def first_periods(prices: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """Per ISIN in an irregular (first) coupon period somewhere in ``prices``: the interest
    start implied by the Bundesbank's accrued (median over those days) and whether its first
    coupon was SHORT (the accrued reset on the first anniversary). ``isin, commencement,
    short_first``. A day is irregular when its implied start is more than
    ``BUND_IRREGULAR_DAYS`` from the regular schedule's last anniversary."""
    s = sec.set_index("isin")
    rows = []
    for isin, g in prices.dropna(subset=["price_dirty"]).groupby("isin"):
        if isin not in s.index or s.loc[isin, "coupon"] <= 0:
            continue
        b = s.loc[isin]
        g = g.sort_values("timestamp")
        acc = (g["price_dirty"] - g["price_clean"]).to_numpy()
        starts = []
        for day, x in zip(g["timestamp"], acc):
            settle = settlement(day)
            if settle >= b["maturity_date"]:
                continue
            prev = coupon_dates(b["maturity_date"], settle, 1)[0]
            c = pb.implied_commencement(x, b["coupon"], settle, b["maturity_date"])
            if pd.notna(c) and abs((c - prev).days) > BUND_IRREGULAR_DAYS:
                starts.append(c)
        if not starts:
            continue
        c0 = pd.Series(starts).median().normalize()
        # a first period starts at the bond's own first issuance: within a month after its first
        # auction, or up to ~13 months before it (a Green twin dated from its conventional twin).
        # Anything else is noise in the published accrued (a 2005 Bund "starting" in late 2022).
        if not (b["first_auction"] - pd.Timedelta(days=400) <= c0 <= b["first_auction"] + pd.Timedelta(days=30)):
            continue
        a1 = coupon_dates(b["maturity_date"], c0, 1)[1]
        before = g[g["timestamp"].map(settlement) <= a1].tail(1)
        after = g[g["timestamp"].map(settlement) > a1].head(1)
        short = bool(len(before) and len(after) and
                     float((after["price_dirty"] - after["price_clean"]).iloc[0])
                     < float((before["price_dirty"] - before["price_clean"]).iloc[0]))
        rows.append({"isin": isin, "commencement": c0, "short_first": short})
    return pd.DataFrame(rows, columns=["isin", "commencement", "short_first"])


def settlement_mismatch(g: pd.DataFrame, sec_ix: pd.DataFrame, settle: pd.Timestamp, fp: pd.DataFrame) -> set[str]:
    """Bonds (outside their first coupon period) whose published accrued is off our regular
    schedule at ``settle`` by more than ``BUND_IRREGULAR_DAYS`` of accrual - around a coupon
    date the Bundesbank's settlement day sometimes differs from T+2 (found 2026-10-07: a
    Schatz paying on our settlement day still carried ~a whole coupon as accrued, an 85bp fit
    error that day), so the dirty price and the cash flows disagree. Left out that day."""
    out = set()
    for r in g.itertuples(index=False):
        b = sec_ix.loc[r.isin]
        if b["coupon"] <= 0 or settle >= b["maturity_date"]:
            continue
        if r.isin in fp.index and pb.first_period_flows(b["coupon"], b["maturity_date"], settle,
                                                        fp.loc[r.isin, "commencement"],
                                                        bool(fp.loc[r.isin, "short_first"])) is not None:
            continue    # in its first coupon period: its own cash flows
        prev, nxt = coupon_dates(b["maturity_date"], settle, 1)[:2]
        ours = b["coupon"] * (settle - prev).days / (nxt - prev).days
        if abs(r.accrued - ours) > b["coupon"] * BUND_IRREGULAR_DAYS / 365.0:
            out.add(r.isin)
    return out


def compute_curves(start, end, *, prices_root: Path = DAILY_BUND_PRICES_DIR, auctions_root: Path = DE_AUCTIONS_DIR,
                   curves_root: Path = BUND_CURVES_DIR, log=None, knots=BUND_CURVE_KNOTS,
                   exclude_ranks: int = BUND_FIT_EXCLUDE_RANKS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every day with dirty prices in ``[start, end]`` - no writing. Returns (curve rows,
    bond rows with ``in_fit``, diagnostics ``days`` / ``empty_days``). Svensson is seeded from
    the stored fit of the day before ``start``."""
    start = max(pd.Timestamp(start).normalize(), pd.Timestamp(BUND_CURVES_START))
    end = pd.Timestamp(end).normalize()
    if end < start:
        return pd.DataFrame(), pd.DataFrame(), {"days": [], "empty_days": {}}
    sec = bunds.read_securities(end, root=auctions_root)
    sec = sec[sec["type"].isin(CONVENTIONAL + ("Green",))]
    green = set(sec.loc[sec["type"] == "Green", "isin"])
    sec = sec[sec["type"].isin(CONVENTIONAL)]
    px = bunds.read_bund_prices(start, end + pd.Timedelta(days=1), root=prices_root)
    px = px[px["isin"].isin(sec["isin"])].dropna(subset=["price_clean", "price_dirty"])
    otr = bunds.otr_map(start - pd.Timedelta(days=5), end, depth=max(exclude_ranks - 1, 0), root=auctions_root)
    otr = otr[otr["rank"] < exclude_ranks] if len(otr) else otr
    excl_otr = otr.groupby("timestamp")["isin"].apply(set) if len(otr) else pd.Series(dtype=object)
    s_fit = sec.assign(coupons_per_year=1)[["isin", "coupon", "maturity_date", "coupons_per_year"]]
    # each bond's first coupon period from its whole published history (2022-06 on), so a
    # window sees the same schedule as a full build
    hist = bunds.read_bund_prices(BUND_CURVES_START, end + pd.Timedelta(days=1), isins=sorted(set(px["isin"])),
                                  root=prices_root)
    fp = first_periods(hist, sec).set_index("isin")

    def flows(r, settle):
        if r.cusip not in fp.index:          # fit_day names the id column ``cusip``
            return None
        f = fp.loc[r.cusip]
        return pb.first_period_flows(float(r.coupon), r.maturity_date, settle, f["commencement"], bool(f["short_first"]))

    seed = stored_svensson_seed(start, curves_root=curves_root)
    sec_ix = sec.set_index("isin")
    all_c, all_r, days, empty = [], [], [], {}
    for i, (day, g) in enumerate(px.groupby("timestamp")):
        day = pd.Timestamp(day)
        days.append(day)
        settle = settlement(day)
        g = g.assign(price_eod=g["price_clean"], accrued=g["price_dirty"] - g["price_clean"], yield_eod=g["yield"])
        g = g[~g["isin"].isin(settlement_mismatch(g, sec_ix, settle, fp))]
        in_first = {i for i in g["isin"] if i in fp.index and flows(_Row(i, sec_ix.loc[i]), settle) is not None}
        ex = excl_otr.get(day, set()) | green | in_first
        c, r, seed = fit_day(day, g, s_fit, ex, svensson_start=seed, settle=settle, id_column="isin", flows=flows,
                             knots=knots)
        if not c:
            empty[day] = f"too few priced bonds to fit ({len(g)} with a price)"
        all_c += c; all_r += r
        if log and i % 100 == 0:
            log(f"{day.date()} n_fit={c[0]['n_fit'] if c else 0}")
    cdf, rdf = pd.DataFrame(all_c), pd.DataFrame(all_r)
    for df in (cdf, rdf):
        if not df.empty:
            df["timestamp"] = df["timestamp"].astype("datetime64[ms]")
    return cdf, rdf, {"days": days, "empty_days": empty}


def build_curves(start=BUND_CURVES_START, end=None, *, curves_root: Path = BUND_CURVES_DIR, rv_root: Path = BUND_RV_DIR,
                 log=None) -> dict:
    """``compute_curves`` + write: a rebuilt day replaces its rows in both stores."""
    end = pd.Timestamp.now().normalize() if end is None else end
    cdf, rdf, diag = compute_curves(start, end, curves_root=curves_root, log=log)
    if not cdf.empty:
        days = sorted(cdf["timestamp"].unique())
        for root in (curves_root, rv_root):
            if parquet_store.has_data(root):
                parquet_store.prune_rows(root, "timestamp", days)
        parquet_store.write_partitioned(cdf, curves_root, CURVE_KEYS)
        parquet_store.write_partitioned(rdf, rv_root, RV_KEYS)
    return {"days": int(cdf["timestamp"].nunique()) if not cdf.empty else 0, "bond_rows": len(rdf),
            "empty_days": diag["empty_days"]}


def read_bund_curves(start=None, end=None, *, method: str | None = None, root: Path = BUND_CURVES_DIR) -> pd.DataFrame:
    return read_curves(start, end, method=method, root=root)


def read_bund_rv(start=None, end=None, *, method: str | None = "spline", isins=None, root: Path = BUND_RV_DIR) -> pd.DataFrame:
    return read_rv(start, end, method=method, cusips=isins, root=root)
