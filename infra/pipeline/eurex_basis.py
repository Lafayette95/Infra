"""Eurex German government-bond futures (FGBS / FGBM / FGBL / FGBX): delivery baskets with
conversion factors, and the daily basis - read / compute / store, disk only except the
deliverables file (``infra.api.eurex_client``). Pure maths ``infra.processing.eurex_baskets``.

* Eurex's own deliverables CSV is archived raw per day, ``RawData/EUREX_CF/<year>/
  eurex_cf_<day>.csv`` (a file = that day's coverage; it lists only the contracts listed then).
* Baskets (``Reference/Bunds/FuturesBaskets``, keys ``timestamp``, ``root``, ``contract``,
  ``isin``, ``source``): per day and listed contract (``listed_contracts``: the next three
  quarterly months, named as Databento's raw symbol), every deliverable with its CF.
  ``source = "eurex"`` on days with an archived file, ``"computed"`` from the rules
  (``EUREX_BOND_FUTURES``) otherwise - checked equal on every German contract in Eurex's
  file, 42 CFs exact (2026-10-07).
* Basis (``Derived/EurexBasis``, keys ``timestamp``, ``contract``, ``isin``): for the front two
  contracts each day, each deliverable's dirty price (Bundesbank, 11:15 Frankfurt, T+2), the
  futures mid at the SAME instant (``bbo-1m``; the settlement if no quote), gross basis,
  implied repo (ACT/360) and the CTD = highest implied repo, with the futures DV01 = the CTD's
  DV01 / CF (spot DV01, price points per 1bp per 100 nominal).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import treasury_curve as tc
from infra.api import eurex_client
from infra.config import (BBO_FUTURES_DIR, BUND_PRICES_LOCAL_TIME, DAILY_BUND_PRICES_DIR, DAILY_FUTURES_DIR,
                          DE_AUCTIONS_DIR, EUREX_BASIS_DIR, EUREX_BASKETS_DIR, EUREX_BOND_FUTURES, EUREX_CF_DIR)
from infra.pipeline import bunds
from infra.processing import bunds as pb
from infra.processing import eurex_baskets as eb
from infra.storage import parquet_store

log = logging.getLogger(__name__)
FETCH = eurex_client.fetch_deliverables_csv   # network hook; tests stub it
BASKET_KEYS = ["timestamp", "root", "contract", "isin", "source"]
BASIS_KEYS = ["timestamp", "contract", "isin"]
_ONE_DAY = pd.Timedelta(days=1)
MAX_SPREAD_POINTS = {"FGBS": 0.05, "FGBM": 0.05, "FGBL": 0.05, "FGBX": 0.10}   # ~5 ticks (Buxl 5 x 0.02)


# ------------------------------------------------------------------ Eurex's file
def _cf_path(day, root: Path) -> Path:
    d = pd.Timestamp(day)
    return root / f"{d.year}" / f"eurex_cf_{d:%Y-%m-%d}.csv"


def archive_deliverables(day=None, *, root: Path = EUREX_CF_DIR) -> Path | None:
    """Fetch and archive today's file (once per day; an archived day is never re-fetched)."""
    day = pd.Timestamp.now().normalize() if day is None else pd.Timestamp(day).normalize()
    path = _cf_path(day, root)
    if path.exists():
        return path
    body = FETCH()
    if body is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return path


def read_deliverables(day, *, root: Path = EUREX_CF_DIR) -> pd.DataFrame:
    """The archived file of ``day`` (German roots only): ``root, contract, isin, cf``."""
    path = _cf_path(day, root)
    if not path.exists():
        return pd.DataFrame(columns=["root", "contract", "isin", "cf"])
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = ["contract", "isin", "coupon", "maturity", "cf"]
    df["root"] = df["contract"].str.split().str[0]
    return df[df["root"].isin(list(EUREX_BOND_FUTURES))][["root", "contract", "isin", "cf"]].reset_index(drop=True)


# ------------------------------------------------------------------ baskets
def _first_periods(prices_root: Path, auctions_root: Path) -> pd.DataFrame:
    from infra.pipeline.bund_curves import first_periods
    sec = bunds.read_securities(root=auctions_root)
    hist = bunds.read_bund_prices("2022-06-01", pd.Timestamp.now().normalize() + _ONE_DAY, root=prices_root)
    return first_periods(hist, sec).set_index("isin")


def compute_baskets(start, end, *, auctions_root: Path = DE_AUCTIONS_DIR, prices_root: Path = DAILY_BUND_PRICES_DIR,
                    cf_root: Path = EUREX_CF_DIR) -> pd.DataFrame:
    """Basket rows for business days in ``[start, end]``. Point in time: securities first
    auctioned by the day and issued by it; Eurex's own list wherever archived for that day."""
    days = pd.bdate_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    auctions = bunds.read_auctions(root=auctions_root)
    fp = _first_periods(prices_root, auctions_root)
    out = []
    for day in days:
        a = auctions[auctions["timestamp"] <= day]
        sec = pb.securities(a)
        vol = a.groupby("isin")["volume_m"].sum()
        six = sec.set_index("isin")
        ex = read_deliverables(day, root=cf_root)
        for root, spec in EUREX_BOND_FUTURES.items():
            for contract, delivery, ltd in eb.listed_contracts(root, day):
                theirs = ex[ex["contract"] == contract]
                if len(theirs):
                    rows = theirs.assign(source="eurex")[["isin", "cf", "source"]]
                else:
                    b = eb.basket(sec, vol, delivery, spec, as_of=day)
                    cfs = []
                    for isin in b["isin"]:
                        s = six.loc[isin]
                        c0 = fp.loc[isin, "commencement"] if isin in fp.index else s["issue_date"]
                        sf = bool(fp.loc[isin, "short_first"]) if isin in fp.index else False
                        cfs.append(eb.conversion_factor(s["coupon"], s["maturity_date"], delivery, spec.notional_pct,
                                                        commencement=c0, short_first=sf))
                    rows = pd.DataFrame({"isin": b["isin"].to_numpy(), "cf": cfs, "source": "computed"})
                if rows.empty:
                    continue
                out.append(rows.assign(timestamp=day, root=root, contract=contract, delivery=delivery,
                                       last_trading_day=ltd))
    if not out:
        return pd.DataFrame(columns=BASKET_KEYS + ["cf", "delivery", "last_trading_day"])
    df = pd.concat(out, ignore_index=True)
    for c in ("timestamp", "delivery", "last_trading_day"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ms]")
    return df[BASKET_KEYS + ["cf", "delivery", "last_trading_day"]]


def store_days(df: pd.DataFrame, start, end, root: Path, keys) -> int:
    """Replace the business days ``[start, end]`` in ``root`` by ``df``."""
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    if parquet_store.has_data(root):
        parquet_store.delete_where(root, lambda part: pd.to_datetime(part["timestamp"]).ge(lo)
                                   & pd.to_datetime(part["timestamp"]).lt(hi))
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, keys)
    return len(df)


def read_baskets(start, end, *, root: Path = EUREX_BASKETS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end) + _ONE_DAY)
    if df is None or df.empty:
        return pd.DataFrame(columns=BASKET_KEYS + ["cf", "delivery", "last_trading_day"])
    for c in ("root", "contract", "isin", "source"):
        df[c] = df[c].astype(str)
    return df.sort_values(BASKET_KEYS).reset_index(drop=True)


# ------------------------------------------------------------------ basis
def _futures_marks(contracts, start, end, *, bbo_root: Path = BBO_FUTURES_DIR, daily_root: Path = DAILY_FUTURES_DIR) -> pd.DataFrame:
    """Per day and contract: the bbo-1m mid at the Bundesbank price time (11:15 Frankfurt,
    the last two-sided quote in the 5 minutes before it), else the day's settlement."""
    from infra.pipeline.bbo import read_bbo_from_disk
    from infra.pipeline.daily import read_daily_from_disk
    from infra.trading_calendar import snap_instants
    days = pd.bdate_range(start, end)
    inst = pd.Series(snap_instants(days, *BUND_PRICES_LOCAL_TIME), index=days)
    q = read_bbo_from_disk(list(contracts), pd.Timestamp(start), pd.Timestamp(end) + _ONE_DAY, root=bbo_root).dropna(subset=["bid", "ask"])
    q = q.assign(ticker=q["ticker"].astype(str), day=q["timestamp"].dt.normalize())
    q = q[(q["timestamp"] <= q["day"].map(inst)) & (q["timestamp"] > q["day"].map(inst) - pd.Timedelta(minutes=5))]
    # a mid is a price only on a tight book: deferred contracts quote 0.6-8 points wide (found
    # 2026-10-07: FGBX Dec-25 117 / 125 on 2025-08-07 gave an 11% implied repo); else settlement
    width = q["ticker"].str[:4].map(MAX_SPREAD_POINTS).fillna(0.05)
    q = q[(q["ask"] - q["bid"]) <= width + 1e-9]
    mid = q.sort_values("timestamp").groupby(["day", "ticker"])["mid"].last().rename("futures").reset_index()
    mid["futures_source"] = "bbo_1115"
    st = read_daily_from_disk(list(contracts), pd.Timestamp(start), pd.Timestamp(end) + _ONE_DAY, root=daily_root)
    st = st.dropna(subset=["settlement_price"]).assign(ticker=lambda d: d["ticker"].astype(str))
    st = st.rename(columns={"timestamp": "day", "settlement_price": "futures"})[["day", "ticker", "futures"]]
    st["futures_source"] = "settlement"
    both = pd.concat([mid, st], ignore_index=True)
    both["rank"] = (both["futures_source"] != "bbo_1115").astype(int)
    return both.sort_values("rank").drop_duplicates(["day", "ticker"]).drop(columns="rank")


def compute_basis(start, end, *, baskets_root: Path = EUREX_BASKETS_DIR, prices_root: Path = DAILY_BUND_PRICES_DIR,
                  auctions_root: Path = DE_AUCTIONS_DIR, front: int = 2, bbo_root: Path = BBO_FUTURES_DIR,
                  daily_root: Path = DAILY_FUTURES_DIR) -> pd.DataFrame:
    """Basis rows for days in ``[start, end]`` (module docstring)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    bk = read_baskets(start, end, root=baskets_root)
    if bk.empty:
        return pd.DataFrame()
    order = bk.drop_duplicates(["timestamp", "contract"]).sort_values(["timestamp", "root", "delivery"])
    order["n"] = order.groupby(["timestamp", "root"]).cumcount()
    bk = bk.merge(order[["timestamp", "contract", "n"]], on=["timestamp", "contract"])
    bk = bk[bk["n"] < front]
    px = bunds.read_bund_prices(start, end + _ONE_DAY, isins=sorted(set(bk["isin"])), root=prices_root)
    px = px.rename(columns={"yield": "bbk_yield"})      # ``yield`` can't be a tuple field
    sec = bunds.read_securities(end, root=auctions_root).set_index("isin")
    fp = _first_periods(prices_root, auctions_root)
    fut = _futures_marks(sorted(set(bk["contract"])), start, end, bbo_root=bbo_root, daily_root=daily_root)
    df = bk.merge(px, on=["timestamp", "isin"], how="inner").merge(
        fut.rename(columns={"day": "timestamp", "ticker": "contract"}), on=["timestamp", "contract"], how="inner")
    rows = []
    for r in df.itertuples(index=False):
        s = sec.loc[r.isin]
        first = {"commencement": fp.loc[r.isin, "commencement"], "short_first": bool(fp.loc[r.isin, "short_first"])} \
            if r.isin in fp.index else {}
        settle = pd.Timestamp(r.timestamp) + pd.offsets.BDay(2)
        acc_s = eb.accrued_at(s["coupon"], s["maturity_date"], settle, **first)
        dirty = r.price_dirty if pd.notna(r.price_dirty) else r.price_clean + acc_s
        amount, tau, pay = eb.next_payment(s["coupon"], s["maturity_date"], settle, **first)
        coupons = amount if pay <= pd.Timestamp(r.delivery) else 0.0
        acc_d = eb.accrued_at(s["coupon"], s["maturity_date"], r.delivery, **first)
        irr = eb.implied_repo(dirty, settle, r.delivery, r.futures, r.cf, acc_d, coupons)
        own = pb.first_period_flows(s["coupon"], s["maturity_date"], settle, first["commencement"],
                                    first["short_first"]) if first else None
        t, a = own if own is not None else tc.cash_flows(float(s["coupon"]), s["maturity_date"], settle, 1)
        try:
            y = tc.ytm(dirty, t, a, 1, guess=float(r.bbk_yield) / 100 if pd.notna(r.bbk_yield) else 0.03)
            dv01 = tc.modified_duration(dirty, t, a, y, 1) * dirty / 10000.0   # price points per 1bp per 100 nominal
        except (ValueError, ZeroDivisionError):
            dv01 = np.nan
        rows.append({"timestamp": r.timestamp, "root": r.root, "contract": r.contract, "isin": r.isin,
                     "delivery": r.delivery, "cf": r.cf, "basket_source": r.source, "price_clean": r.price_clean,
                     "dirty_settle": dirty, "futures": r.futures, "futures_source": r.futures_source,
                     "gross_basis": r.price_clean - r.futures * r.cf, "implied_repo": irr, "bond_dv01": dv01,
                     "futures_dv01": dv01 / r.cf})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best = out.groupby(["timestamp", "contract"])["implied_repo"].transform("max")
    out["is_ctd"] = out["implied_repo"].eq(best)
    for c in ("timestamp", "delivery"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    return out.sort_values(BASIS_KEYS).reset_index(drop=True)


def read_basis(start, end, *, ctd_only: bool = False, root: Path = EUREX_BASIS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=pd.Timestamp(start), end=pd.Timestamp(end) + _ONE_DAY)
    if df is None or df.empty:
        return pd.DataFrame()
    for c in ("root", "contract", "isin", "basket_source", "futures_source"):
        df[c] = df[c].astype(str)
    df = df[df["is_ctd"]] if ctd_only else df
    return df.sort_values(BASIS_KEYS).reset_index(drop=True)


def futures_dv01(start, end, tickers=None, *, root: Path = EUREX_BASIS_DIR) -> pd.DataFrame:
    """The bmk hook's shape (``infra.cycle.bmk.EUREX_FUTURES_DV01``): ``timestamp, ticker,
    futures_dv01, ctd`` - each contract's CTD futures DV01 (price points per 1bp)."""
    b = read_basis(start, end, ctd_only=True, root=root)
    if b.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "futures_dv01", "ctd"])
    b = b.rename(columns={"contract": "ticker", "isin": "ctd"})
    if tickers is not None:
        b = b[b["ticker"].isin(set(map(str, tickers)))]
    return b[["timestamp", "ticker", "futures_dv01", "ctd"]].drop_duplicates(["timestamp", "ticker"]).reset_index(drop=True)
