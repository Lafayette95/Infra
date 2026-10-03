"""Treasury futures delivery baskets with per-contract conversion factors (CLAUDE.md 18).
Pure functions, no I/O.

A bond's conversion factor is per CONTRACT (root + delivery month): the same bond has a
different factor in each contract it's deliverable into, since its remaining term at
delivery differs.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from infra.config import CME_TCF_ROOTS, BasketRule

BASKET_COLUMNS = ["timestamp", "root", "contract", "delivery_month", "cusip", "conversion_factor", "source"]
BASKET_KEYS = ["timestamp", "root", "contract", "cusip", "source"]
MONTH_CODES = "FGHJKMNQUVXZ"


def contract_ticker(root: str, delivery_month: pd.Timestamp) -> str:
    """``("ZN", 2026-12)`` -> ``"ZNZ6"`` (CME style: month code + last digit of the year)."""
    m = pd.Timestamp(delivery_month)
    return f"{root}{MONTH_CODES[m.month - 1]}{m.year % 10}"


def parse_tcf(text: str | bytes, day) -> pd.DataFrame:
    """One CME TCF file (as published) -> basket rows (``source = "cme"``), stamped with
    the file's day. Product codes not in CME_TCF_ROOTS are dropped."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    raw = raw[raw["PFCode"].isin(CME_TCF_ROOTS)]
    if raw.empty:
        return pd.DataFrame(columns=BASKET_COLUMNS)
    month = pd.to_datetime(raw["Period"].str.strip(), format="%Y%m")
    root = raw["PFCode"].map(CME_TCF_ROOTS)
    out = pd.DataFrame({
        "timestamp": pd.Timestamp(day).normalize(),
        "root": root.to_numpy(),
        "contract": [contract_ticker(r, m) for r, m in zip(root, month)],
        "delivery_month": month.to_numpy(),
        "cusip": raw["CUSIP"].str.strip().to_numpy(),
        "conversion_factor": pd.to_numeric(raw["Invoice_Conversion_Factor"], errors="coerce").to_numpy(),
        "source": "cme",
    })
    for c in ("timestamp", "delivery_month"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    return out[BASKET_COLUMNS].sort_values(["root", "contract", "cusip"]).reset_index(drop=True)


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Whole months from ``start`` to ``end`` (a day short of a month doesn't count)."""
    m = (end.year - start.year) * 12 + (end.month - start.month)
    return m - 1 if end.day < start.day else m


def conversion_factor(coupon_pct: float, maturity, delivery_month, cf_months: int) -> float:
    """CME's conversion factor: the bond's price per 1 of face at a 6% yield, with its
    term from the first day of ``delivery_month`` to ``maturity`` rounded DOWN to whole
    months (``cf_months`` = 1: 2y/3y/5y contracts) or quarters (3: 10y and longer)."""
    first = pd.Timestamp(delivery_month).replace(day=1)
    total = _months_between(first, pd.Timestamp(maturity))
    total -= total % cf_months
    n, z = divmod(total, 12)
    cpn = coupon_pct / 100.0
    if cf_months == 1:
        v = z if z < 7 else z - 6
    else:
        v = z if z < 7 else 3
    a = 1 / 1.03 ** (v / 6)
    b = cpn / 2 * (6 - v) / 6
    c = 1 / 1.03 ** (2 * n) if z < 7 else 1 / 1.03 ** (2 * n + 1)
    d = cpn / 0.06 * (1 - c)
    return round(a * (cpn / 2 + c + d) - b, 4)


def issued_terms(auctions: pd.DataFrame, day) -> pd.Series:
    """Per CUSIP, the SHORTEST term it has been auctioned at before ``day`` - its original
    issue or any reopening. The Treasury sometimes reopens an older CUSIP at a shorter
    tenor (a 7y note reissued as the 5y when maturity and coupon match: 13 times since
    2016), and CME then judges the original-term cap by that reissue: 91282CGQ8 (7y of
    2023) entered ZF the day after its reopening as the 5y on 2025-02-25. ``auctions``:
    the raw auction rows (``cusip``, ``auction_date``, ``security_term``)."""
    from infra.processing.treasury_ref import term_months
    a = auctions[pd.to_datetime(auctions["auction_date"]) < pd.Timestamp(day)]
    return a.assign(_m=a["security_term"].map(term_months)).groupby("cusip")["_m"].min()


def eligible(securities: pd.DataFrame, delivery_month, rule: BasketRule, terms: pd.Series | None = None) -> pd.Series:
    """Which coupon securities meet ``rule`` for a contract delivering in ``delivery_month``.
    ``terms`` (``issued_terms``) replaces the original term for the original-term CAP."""
    first = pd.Timestamp(delivery_month).replace(day=1)
    last = first + pd.offsets.MonthEnd(0)
    mat = pd.to_datetime(securities["maturity_date"])
    ok = securities["security_type"].isin(["Note", "Bond"]) & (mat >= first + pd.DateOffset(months=rule.min_remaining_months))
    if rule.max_remaining_months is not None:
        cap = (last if rule.max_from_last_day else first) + pd.DateOffset(months=rule.max_remaining_months)
        ok &= (mat <= cap) if rule.max_inclusive else (mat < cap)
    term = securities["term_months"]
    if rule.max_original_months is not None:
        capped = term if terms is None else securities["cusip"].map(terms).fillna(term)
        ok &= capped <= rule.max_original_months
    if rule.original_months is not None:
        ok &= term == rule.original_months
    return ok


def computed_basket(securities: pd.DataFrame, root: str, delivery_month, rule: BasketRule, day,
                    terms: pd.Series | None = None) -> pd.DataFrame:
    """The basket of one contract as of ``day`` from the rules (``source = "computed"``):
    every eligible coupon security auctioned BEFORE ``day`` - CME's file, posted ~06:30
    Chicago, adds a new issue the morning after its auction (verified on every file
    2023-12..2026-10) - with its conversion factor. ``terms``: ``issued_terms`` for the
    day (reopenings at a shorter tenor)."""
    known = securities[pd.to_datetime(securities["auction_date"]) < pd.Timestamp(day)]
    b = known[eligible(known, delivery_month, rule, terms)]
    out = pd.DataFrame({
        "timestamp": pd.Timestamp(day).normalize(), "root": root,
        "contract": contract_ticker(root, delivery_month), "delivery_month": pd.Timestamp(delivery_month),
        "cusip": b["cusip"].to_numpy(),
        "conversion_factor": [conversion_factor(c, m, delivery_month, rule.cf_months)
                              for c, m in zip(b["coupon"], b["maturity_date"])],
        "source": "computed",
    })
    for c in ("timestamp", "delivery_month"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    return out[BASKET_COLUMNS]


# contracts whose last trading day is the delivery month's last business day; the rest
# stop trading 7 business days before it (CME specs; holidays ignored - a day off at most)
_MONTH_END_LAST_TRADE = {"ZT", "ZF", "Z3N"}


def last_trading_day(root: str, delivery_month) -> pd.Timestamp:
    last_bd = pd.Timestamp(delivery_month).replace(day=1) + pd.offsets.BMonthEnd(0)
    return last_bd if root in _MONTH_END_LAST_TRADE else last_bd - pd.offsets.BDay(7)


def listed_contracts(root: str, day, n: int = 3) -> list[pd.Timestamp]:
    """The delivery months CME lists on ``day``: the next ``n`` quarterly contracts not yet
    past their last trading day (each stays listed through it - verified on every CME
    file 2023-12..2026-10)."""
    day = pd.Timestamp(day).normalize()
    m = pd.Timestamp(year=day.year, month=((day.month - 1) // 3) * 3 + 3, day=1)
    out = []
    while len(out) < n:
        if last_trading_day(root, m) >= day:
            out.append(m)
        m = m + pd.DateOffset(months=3)
    return out


def _shift_months(day: pd.Timestamp, months: int, eom: bool) -> pd.Timestamp:
    """``day`` moved by whole months, keeping end-of-month, else the day (clipped)."""
    d = pd.Timestamp(day) + pd.DateOffset(months=months)
    return d + pd.offsets.MonthEnd(0) if eom else d


def expected_issues(securities: pd.DataFrame, as_of, until, tenors: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Original issues EXPECTED after ``as_of`` and up to ``until`` - not yet auctioned, so
    absent from ``securities`` (the reference table as known at ``as_of``). Each tenor's
    pattern is inferred from its own three latest issues: the cycle in months (monthly
    2y/3y/5y/7y, quarterly 10y/20y/30y new CUSIPs; reopenings add none) and whether issue
    and maturity days are month-ends. One row per expected issue: ``cusip`` (a
    placeholder, ``NEW:<tenor>:<issue date>``), ``tenor``, ``security_type``,
    ``term_months``, ``issue_date``, ``maturity_date``, ``auction_date`` (~ issue - 4
    business days), ``predecessor`` (the latest ACTUAL issue of the tenor, whose history
    and yield stand in for it)."""
    as_of, until = pd.Timestamp(as_of), pd.Timestamp(until)
    rows = []
    for tenor, (stype, term) in tenors.items():
        s = securities[(securities["security_type"] == stype) & (securities["original_term"] == term)]
        s = s.dropna(subset=["issue_date"]).sort_values("issue_date").tail(3)
        if len(s) < 2:
            continue
        issues = pd.to_datetime(s["issue_date"])
        cycle = int(round(np.median(np.diff(issues.to_numpy()).astype("timedelta64[D]").astype(int)) / 30.44))
        if cycle < 1:
            continue
        last = s.iloc[-1]
        iss, mat = pd.Timestamp(last["issue_date"]), pd.Timestamp(last["maturity_date"])
        mat_eom = (mat + pd.Timedelta(days=1)).day == 1
        # Issue-day convention: month-end issues (2y/5y/7y; the 20y too, though it matures
        # on the 15th) are nominally the month's last calendar day, rolled FORWARD to a
        # business day (June 30 2019, a Sunday -> issued Monday July 1); mid-month issues
        # fall on the maturity's day of month (the 15th), rolled forward likewise.
        month_end_issue = mat_eom or iss.day >= 25 or (iss.day <= 3 and iss.day != mat.day)
        nominal_month = pd.Timestamp(year=iss.year, month=iss.month, day=1)
        if month_end_issue and iss.day <= 3:
            nominal_month = nominal_month - pd.DateOffset(months=1)
        k = 1
        while True:
            month = nominal_month + pd.DateOffset(months=cycle * k)
            if month_end_issue:
                nominal = month + pd.offsets.MonthEnd(0)
            else:
                nominal = month.replace(day=min(mat.day, (month + pd.offsets.MonthEnd(0)).day))
            nxt = nominal if nominal.weekday() < 5 else nominal + pd.offsets.BDay(1)
            if nxt > until:
                break
            if nxt > as_of:
                rows.append({"cusip": f"NEW:{tenor}:{nxt.date()}", "tenor": tenor, "security_type": stype,
                             "term_months": int(last["term_months"]), "issue_date": nxt,
                             "maturity_date": _shift_months(mat, cycle * k, mat_eom),
                             "auction_date": nxt - pd.offsets.BDay(4), "predecessor": str(last["cusip"])})
            k += 1
    return pd.DataFrame(rows, columns=["cusip", "tenor", "security_type", "term_months", "issue_date",
                                       "maturity_date", "auction_date", "predecessor"])


def expected_issue_coupon(yield_pct: float) -> float:
    """The Treasury sets a new coupon at the highest 1/8 at or below the auction yield."""
    return float(np.floor(yield_pct * 8 + 1e-9) / 8)
