"""Inputs of the Treasury futures basis (CLAUDE.md 22): per day, every deliverable of every
listed contract with its cash price, accrued, coupons and funding to each candidate
delivery day, plus the futures prices at an aligned snap. Reads disk only, never fetches.

Point in time: everything is what was known at the end of ``day``:
* cash = FedInvest END OF DAY (~15:30 New York, BID side - its "Sell" column); moved toward
  mid by ``cash_mid_frac`` x half the posted buy/sell spread (the posted spread is a
  convention, 0.5 / 1 / 2 32nds by maturity, not the market's - so the true adjustment is
  unknown; 0.5 = a quarter of the posted spread, the v1 default);
* futures = the ``BASIS_SNAP`` (NY1530) mid from the futures-snap store (CLAUDE.md 23 -
  15:30 New York, the instant FedInvest's END OF DAY is taken at, verified by the basis-noise
  minimum; quotes exist for each root's front contract), else the exchange settlement (15:00
  New York, flagged), else nothing. The settlement is always carried along, so outputs
  can also be stated against it (the street's convention, e.g. a DLV screen);
* funding = ``infra.pipeline.financing`` as of ``day``, per bond and delivery day.
Settlement of a cash purchase is the next business day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from infra.analytics import financing as fa
from infra.analytics.futures_basis import delivery_window
from infra.analytics.sofr_curve import business_days
from infra.pipeline import financing as pf
from infra.config import BASIS_SNAP, TREASURY_BASKET_RULES, TREASURY_OTR_TENORS
from infra.pipeline.futures_snaps import read_futures_snaps
from infra.pipeline.daily import read_daily_from_disk
from infra.pipeline.futures_baskets import read_baskets
from infra.pipeline.treasury_prices import read_prices
from infra.pipeline.treasury_ref import read_securities
from infra.processing.futures_baskets import conversion_factor, eligible, expected_issue_coupon, expected_issues
from infra.processing.treasury_prices import accrued_and_yield, coupon_dates, full_price_from_yield

US_ROOTS = ("ZT", "Z3N", "ZF", "ZN", "TN", "ZB", "UB")
_ONE_DAY = pd.Timedelta(days=1)


@dataclass
class BasisDay:
    """One day's inputs. ``bonds``: one row per (contract, cusip, delivery) - the columns
    ``infra.analytics.futures_basis.basis_table`` needs, plus ``root``, ``coupon``,
    ``maturity``, ``yield_eod``, ``dv01`` (full-price DV01 per 1bp at the bid yield),
    ``delivery_kind`` ("first"/"last"). ``futures``: one row per contract with
    ``futures`` (the price used), ``futures_source``, ``settlement``, ``bid``, ``ask``,
    window dates."""
    day: pd.Timestamp
    settle: pd.Timestamp
    bonds: pd.DataFrame
    futures: pd.DataFrame
    meta: dict = field(default_factory=dict)


class BasisInputs:
    """Preloads ``[start, end]`` once; ``day(t)`` serves one point-in-time day."""

    def __init__(self, start, end, *, roots=US_ROOTS, cash_mid_frac: float = 0.5, funding_model: str = "v1",
                 snap: str = BASIS_SNAP, contracts: str = "quoted"):
        self.start, self.end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        self.roots, self.cash_mid_frac, self.funding_model = tuple(roots), cash_mid_frac, funding_model
        self.contracts = contracts  # "quoted" (with a bbo quote at the snap) or "all" listed
        stop = self.end + _ONE_DAY
        self.baskets = read_baskets(self.start, stop)
        self.baskets = self.baskets[self.baskets["root"].isin(self.roots)]
        self.prices = read_prices(self.start, stop)
        self.securities_table = read_securities()
        self.securities = self.securities_table.set_index("cusip")
        tickers = sorted(self.baskets["contract"].astype(str).unique())
        snaps = read_futures_snaps(self.start, stop + _ONE_DAY, snap=snap, tickers=tickers) if tickers else pd.DataFrame()
        self._snap_quotes = {(pd.Timestamp(t).normalize(), k): {"bid": b, "ask": a, "mid": m}
                             for t, k, b, a, m in zip(snaps.get("timestamp", []), snaps.get("ticker", []),
                                                      snaps.get("bid", []), snaps.get("ask", []), snaps.get("mid", []))}
        sett = read_daily_from_disk(tickers, self.start, stop, adjusted=False) if tickers else pd.DataFrame()
        self.settlements = (sett.assign(ticker=sett["ticker"].astype(str)).dropna(subset=["settlement_price"])
                            .set_index(["timestamp", "ticker"])["settlement_price"].astype(float)) if len(sett) else pd.Series(dtype=float)
        self.business_days = business_days(self.start - pd.Timedelta(days=40), self.end + pd.Timedelta(days=400))
        days = sorted(set(self.prices["timestamp"]) & set(self.baskets["timestamp"]))
        self.days = pd.DatetimeIndex(days)
        self.snap = snap

    def _quote(self, ticker: str, day: pd.Timestamp):
        return self._snap_quotes.get((pd.Timestamp(day).normalize(), ticker))

    def _accrued(self, cusip: str, day) -> float:
        s = self.securities.loc[cusip]
        return accrued_and_yield(np.nan, s["coupon"], s["maturity_date"], day, int(s["coupons_per_year"]),
                                 s["dated_date"], s["first_coupon_date"])[0]

    def day(self, t, *, financing: fa.FinancingInputs | None = None) -> BasisDay:
        t = pd.Timestamp(t).normalize()
        bd = self.business_days
        settle = bd[bd > t][0]
        cash = self.prices[self.prices["timestamp"] == t].set_index("cusip")
        bask = self.baskets[self.baskets["timestamp"] == t]
        fin = pf.financing_inputs(t, self.funding_model) if financing is None else financing
        fut_rows, bond_rows = [], []
        for (root, contract), b in bask.groupby(["root", "contract"]):
            contract = str(contract)
            win = delivery_window(root, b["delivery_month"].iloc[0], bd)
            if win["last_delivery"] <= settle:
                continue
            q = self._quote(contract, t)
            sett = self.settlements.get((t, contract), np.nan)
            if q is not None:
                price, source = float(q["mid"]), "bbo_mid"
            elif np.isfinite(sett):
                price, source = float(sett), "settlement"
            else:
                price, source = np.nan, None
            if self.contracts == "quoted" and q is None:
                continue
            fut_rows.append({"root": root, "contract": contract, "futures": price, "futures_source": source,
                             "settlement": sett, "bid": None if q is None else float(q["bid"]),
                             "ask": None if q is None else float(q["ask"]), **win})
            deliveries = {"first": max(win["first_delivery"], settle), "last": win["last_delivery"]}
            for cusip, cf in zip(b["cusip"].astype(str), b["conversion_factor"].astype(float)):
                if cusip not in cash.index or cusip not in self.securities.index:
                    continue
                row = cash.loc[cusip]
                if not np.isfinite(row["price_eod"]):
                    continue
                spread = row["price_buy"] - row["price_sell"]
                adj = self.cash_mid_frac * spread / 2 if np.isfinite(spread) and spread > 0 else 0.0
                price_mid = float(row["price_eod"]) + adj
                s = self.securities.loc[cusip]
                per_year = int(s["coupons_per_year"])
                y = float(row["yield_eod"]) if np.isfinite(row["yield_eod"]) else np.nan
                dv01 = (full_price_from_yield(y - 0.005, s["coupon"], s["maturity_date"], settle, per_year)
                        - full_price_from_yield(y + 0.005, s["coupon"], s["maturity_date"], settle, per_year)) \
                    if np.isfinite(y) else np.nan
                for kind, d in deliveries.items():
                    if kind == "first" and d == deliveries["last"]:
                        continue
                    pays = [c for c in coupon_dates(pd.Timestamp(s["maturity_date"]), d, per_year) if settle < c <= d]
                    coupons = [(c, s["coupon"] / per_year) for c in pays]
                    q_fin = pf.financing_rate(t, settle, d, cusip=cusip, model=self.funding_model, inputs=fin)
                    bond_rows.append({"root": root, "contract": contract, "cusip": cusip, "cf": cf,
                                      "delivery_kind": kind, "delivery": d, "settle": settle,
                                      "price": price_mid, "price_bid": float(row["price_eod"]),
                                      "ai_settle": float(row["accrued"]), "ai_delivery": self._accrued(cusip, d),
                                      "coupons": coupons, "repo": q_fin.rate, "repo_base": q_fin.base,
                                      "coupon": float(s["coupon"]), "maturity": pd.Timestamp(s["maturity_date"]),
                                      "yield_eod": y, "dv01": dv01})
        futures = pd.DataFrame(fut_rows)
        return BasisDay(t, settle, pd.DataFrame(bond_rows), futures,
                        meta={"snap": self.snap, "cash_mid_frac": self.cash_mid_frac, "funding_model": self.funding_model,
                              "future_issues": self.future_issues(t, futures, cash)})

    def future_issues(self, t, futures: pd.DataFrame, cash: pd.DataFrame) -> pd.DataFrame:
        """Per listed contract, the EXPECTED new issues (``infra.processing.futures_baskets.
        expected_issues``: not yet auctioned on ``t``) its basket rule will accept, issued by
        its last delivery day: ``root``, ``contract``, ``cusip`` (placeholder), ``issue_date``,
        ``maturity``, ``coupon`` (the reference yield rounded down to 1/8), ``cf``,
        ``ref_yield`` (the latest same-tenor issue priced on ``t``), ``predecessor``."""
        cols = ["root", "contract", "cusip", "issue_date", "maturity", "coupon", "cf", "ref_yield", "predecessor"]
        if futures.empty:
            return pd.DataFrame(columns=cols)
        sec = self.securities_table
        auctioned = sec[pd.to_datetime(sec["auction_date"]) < pd.Timestamp(t)]
        exp = expected_issues(auctioned, t, futures["last_delivery"].max(), TREASURY_OTR_TENORS)
        if exp.empty:
            return pd.DataFrame(columns=cols)
        # reference yield: the latest issue of the tenor that has a price today
        priced = cash["yield_eod"].dropna()
        ref = {}
        for tenor, (stype, term) in TREASURY_OTR_TENORS.items():
            s = auctioned[(auctioned["security_type"] == stype) & (auctioned["original_term"] == term)]
            s = s.sort_values("issue_date", ascending=False)
            hit = [c for c in s["cusip"].astype(str) if c in priced.index]
            ref[tenor] = (hit[0], float(priced[hit[0]])) if hit else (None, np.nan)
        exp["predecessor"] = [ref[x][0] or p for x, p in zip(exp["tenor"], exp["predecessor"])]
        exp["ref_yield"] = [ref[x][1] for x in exp["tenor"]]
        exp = exp.dropna(subset=["ref_yield"])
        exp["coupon"] = exp["ref_yield"].map(expected_issue_coupon)
        rows = []
        for f in futures.itertuples():
            rule = TREASURY_BASKET_RULES.get(f.root)
            if rule is None:
                continue
            month = pd.Timestamp(f.first_delivery).replace(day=1)
            e = exp[exp["issue_date"] <= f.last_delivery]
            e = e[eligible(e, month, rule)]
            for r in e.itertuples():
                rows.append({"root": f.root, "contract": f.contract, "cusip": r.cusip, "issue_date": r.issue_date,
                             "maturity": pd.Timestamp(r.maturity_date), "coupon": r.coupon,
                             "cf": conversion_factor(r.coupon, r.maturity_date, month, rule.cf_months),
                             "ref_yield": r.ref_yield, "predecessor": r.predecessor})
        return pd.DataFrame(rows, columns=cols)


def deterministic_futures_dv01(start, end, tickers=None, *, roots=US_ROOTS) -> pd.DataFrame:
    """Per day and contract, the futures DV01 of the DETERMINISTIC cheapest-to-deliver (the
    basis models' M0 rule: lowest implied futures price over bonds x {first, last} delivery
    day; DV01 = that bond's forward DV01 / CF, price points per 1bp per 100 face) - for the
    daily cycle's bmk step, which may not import ``infra/models`` (this uses the same pure
    analytics M0 does). No futures price needed. Days whose inputs fail (no basket, no cash
    prices, no funding - funding starts 2018-05) are skipped and listed in ``attrs["errors"]``.
    Columns: ``timestamp``, ``ticker``, ``futures_dv01``, ``ctd``, ``delivery_kind``."""
    from infra.analytics.futures_basis import basis_table, futures_dv01
    cols = ["timestamp", "ticker", "futures_dv01", "ctd", "delivery_kind"]
    src = BasisInputs(start, end, roots=roots, contracts="all")
    want = None if tickers is None else set(map(str, tickers))
    rows, errors = [], {}
    for day in src.days:
        try:
            d = src.day(day)
        except Exception as exc:
            errors[pd.Timestamp(day)] = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            continue
        if d.bonds.empty:
            continue
        for contract, g in d.bonds.groupby("contract"):
            if want is not None and contract not in want:
                continue
            t = basis_table(g.dropna(subset=["dv01"]), None)
            if t.empty:
                continue
            c = t.loc[t["implied_futures"].idxmin()]
            rows.append((pd.Timestamp(day), contract, futures_dv01(c["dv01"], c["repo"], c["settle"], c["delivery"], c["cf"]),
                         c["cusip"], c["delivery_kind"]))
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["errors"] = errors
    return out
