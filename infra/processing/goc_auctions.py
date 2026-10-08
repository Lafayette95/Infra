"""Government of Canada auctions and outstanding securities - pure parsing (no network, no
files) of the Bank of Canada's Valet groups and its bond-auction-schedule page (root CLAUDE.md
17).

Auctions: one row per (auction day, ISIN, kind) - ``BOND`` (nominal, 1998-10 on), ``RRB`` (real
return), ``ULTRA`` (ultra long) and ``TBILL``; amounts in $ millions, yields in percent, the
bidding deadline as published (12:00 for bonds, 10:30 for bills, Ottawa time). Outstanding: one
row per (as-of day, ISIN), $. Plan: the quarterly bond schedule (auction day, term, maturity,
the day the call for tenders - "further details" - is published, delivery).
"""
from __future__ import annotations

import html as _html
import re

import numpy as np
import pandas as pd

AUCTION_COLUMNS = ["timestamp", "isin", "kind", "term_years", "coupon", "issue_date", "maturity_date", "amount_m",
                   "avg_yield", "high_yield", "low_yield", "coverage", "tail", "bid_deadline", "off_the_run",
                   "outstanding_after_m", "boc_purchase_m"]
OUTSTANDING_COLUMNS = ["timestamp", "isin", "security_type", "instrument_type", "coupon", "issue_date",
                       "maturity_date", "outstanding", "outstanding_inf_adj", "series"]
PLAN_COLUMNS = ["auction_date", "kind", "term_years", "maturity_date", "call_date", "delivery_date"]
SECURITY_COLUMNS = ["isin", "kind", "term_years", "coupon", "maturity_date", "first_auction", "issue_date"]
GROUPS = {"BOND": ("AUC_BOND_RESULTS", "AUC_BOND_"), "RRB": ("AUC_BOND_RR_RESULTS", "AUC_BOND_RR_"),
          "ULTRA": ("AUC_BOND_U_RESULTS", "AUC_BOND_U_"), "TBILL": ("AUC_TBILL_RESULTS", "AUC_TBILL_")}
OUTSTANDING_GROUPS = {"GOC_OUTSTANDING": "", "GOC_OUTSTANDING_2025": "_2025"}


def valet_frame(payload: dict, prefix: str, suffix: str = "") -> pd.DataFrame:
    """A Valet group's observations as strings, columns lower-cased without ``prefix`` /
    ``suffix`` (``AUC_BOND_AVG_YIELD`` -> ``avg_yield``)."""
    rows = []
    for o in payload.get("observations", []):
        r = {}
        for k, v in o.items():
            val = v.get("v") if isinstance(v, dict) else v
            if k.startswith(prefix):
                k = k[len(prefix):]
                if suffix and k.endswith(suffix):
                    k = k[: -len(suffix)]
                r[k.lower()] = val
        rows.append(r)
    return pd.DataFrame(rows)


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _col(df: pd.DataFrame, *names) -> pd.Series:
    for n in names:
        if n in df:
            return df[n]
    return pd.Series(np.nan, index=df.index)


def parse_results(payload: dict, kind: str) -> pd.DataFrame:
    """One results group (``GROUPS[kind]``) as auction rows. Real-return and ultra-long
    auctions are single-price: their allotment yield is the ``high_yield``. ``boc_purchase_m``:
    the Bank of Canada's own purchase at the auction."""
    df = valet_frame(payload, GROUPS[kind][1])
    if df.empty or "auction_date" not in df:
        return pd.DataFrame(columns=AUCTION_COLUMNS)
    term = _num(_col(df, "term_years"))
    if kind == "TBILL":
        term = _num(_col(df, "term_days")) / 365.0
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["auction_date"], errors="coerce"),
        "isin": _col(df, "isin"), "kind": kind, "term_years": term, "coupon": _num(_col(df, "coupon_rate")),
        "issue_date": pd.to_datetime(_col(df, "issue_date"), errors="coerce"),
        "maturity_date": pd.to_datetime(_col(df, "maturity_date"), errors="coerce"),
        "amount_m": _num(_col(df, "amount")), "avg_yield": _num(_col(df, "avg_yield", "median_yield")),
        "high_yield": _num(_col(df, "high_yield", "allotment_yield")), "low_yield": _num(_col(df, "low_yield", "low_5_yield")),
        "coverage": _num(_col(df, "coverage")), "tail": _num(_col(df, "tail")),
        "bid_deadline": _col(df, "bid_deadline"), "off_the_run": _col(df, "off_the_run"),
        "outstanding_after_m": _num(_col(df, "outstanding_after")), "boc_purchase_m": _num(_col(df, "boc_purchase"))})
    # each auction also has a HEADER observation (day, deadline, issue day only, no ISIN) next to
    # its line(s) ("<day>_<id>" vs "<day>_<id>-1"): only the lines are auctions
    out = out.dropna(subset=["timestamp", "isin"])
    return out[AUCTION_COLUMNS].reset_index(drop=True)


def parse_outstanding(payload: dict, suffix: str = "") -> pd.DataFrame:
    df = valet_frame(payload, "DOM_DBT_", suffix)
    if df.empty or "isin" not in df:
        return pd.DataFrame(columns=OUTSTANDING_COLUMNS)
    df = df[df["isin"].notna()]
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["as_of_date"], errors="coerce"), "isin": df["isin"],
        "security_type": _col(df, "security_type"), "instrument_type": _col(df, "instrument_type"),
        "coupon": _num(_col(df, "coupon_rate")), "issue_date": pd.to_datetime(_col(df, "issue_date"), errors="coerce"),
        "maturity_date": pd.to_datetime(_col(df, "maturity_date"), errors="coerce"),
        "outstanding": _num(_col(df, "outstanding_amount")), "outstanding_inf_adj": _num(_col(df, "outstanding_amount_inf_adj")),
        "series": None})
    return out.dropna(subset=["timestamp"])[OUTSTANDING_COLUMNS].reset_index(drop=True)


def parse_dmb_csv(text: str, isin_of: dict) -> pd.DataFrame:
    """One of the Bank of Canada's monthly "Unmatured Debt Details: Domestic Marketable Bonds"
    CSVs (2018-01..2022-01; series codes, no ISIN): outstanding bonds at the month end. A bond
    gets its ISIN by (maturity, coupon) from ``isin_of`` (built from the auctions and the daily
    snapshots - unique on both, checked); one never seen there (issued before 1998) keeps the
    id ``SERIES:<code>``. ``issue_date`` = its first issue date; real return bonds (an inflation
    adjustment column) are ``BD-REAL``."""
    m = re.search(r"Period End Date, ([A-Za-z]+ \d+ \d{4})", text)
    if not m:
        return pd.DataFrame(columns=OUTSTANDING_COLUMNS)
    day = pd.to_datetime(m.group(1))
    rows = []
    for line in text.splitlines():
        if not re.match(r"[A-Z][a-z]{2} \d{2} \d{4},", line):
            continue
        p = [x.strip() for x in line.split(",")]
        mat, cpn = pd.to_datetime(p[0]), float(p[1])
        issues = pd.to_datetime([x.strip() for x in p[2].split(";") if x.strip()], errors="coerce")
        infl = float(p[5]) if len(p) > 5 and p[5] else 0.0
        isin = isin_of.get((mat, round(cpn, 4)))
        rows.append({"timestamp": day, "isin": isin or f"SERIES:{p[3]}", "security_type": "BOND",
                     "instrument_type": "BD-REAL" if infl else "BD-FIX", "coupon": cpn,
                     "issue_date": issues.min() if len(issues) else pd.NaT, "maturity_date": mat,
                     "outstanding": float(p[4]), "outstanding_inf_adj": float(p[6]) if len(p) > 6 and p[6] else np.nan,
                     "series": p[3]})
    return pd.DataFrame(rows, columns=OUTSTANDING_COLUMNS)


def _kind(auction_type: str) -> str:
    s = str(auction_type).lower()
    return "RRB" if "real" in s else "ULTRA" if "ultra" in s else "BOND"


def parse_schedule_valet(payload: dict) -> pd.DataFrame:
    df = valet_frame(payload, "AUC_SCHED_")
    if df.empty or "auction_date" not in df:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    return pd.DataFrame({
        "auction_date": pd.to_datetime(df["auction_date"], errors="coerce"),
        "kind": _col(df, "auction_type").map(_kind), "term_years": _num(_col(df, "term_years")),
        "maturity_date": pd.to_datetime(_col(df, "maturity_date"), errors="coerce"),
        "call_date": pd.to_datetime(_col(df, "further_details"), errors="coerce"),
        "delivery_date": pd.to_datetime(_col(df, "delivered"), errors="coerce")}).dropna(subset=["auction_date"])[PLAN_COLUMNS]


def parse_schedule_page(page: str) -> pd.DataFrame:
    """The schedule page's table (Wayback captures: the headers moved over the years, so
    columns are found by name)."""
    i = page.find("<table")
    j = page.find("</table>", i)
    if i < 0:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page[i:j], flags=re.S):
        rows.append([re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S)])
    if not rows:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    head = [h.lower() for h in rows[0]]
    idx = lambda *keys: next((k for k, h in enumerate(head) if all(x in h for x in keys)), None)  # noqa: E731
    cols = {"auction_date": idx("auction date"), "type": idx("type"), "term": idx("term"), "maturity": idx("maturity"),
            "call": idx("further"), "delivery": idx("deliver")}
    get = lambda r, c: r[cols[c]] if cols[c] is not None and cols[c] < len(r) else None  # noqa: E731
    body = [r for r in rows[1:] if len(r) >= 2]
    df = pd.DataFrame({
        "auction_date": pd.to_datetime([get(r, "auction_date") for r in body], errors="coerce", format="mixed"),
        "kind": [_kind(get(r, "type")) for r in body], "term_years": _num(pd.Series([get(r, "term") for r in body])),
        "maturity_date": pd.to_datetime([get(r, "maturity") for r in body], errors="coerce", format="mixed"),
        "call_date": pd.to_datetime([get(r, "call") for r in body], errors="coerce", format="mixed"),
        "delivery_date": pd.to_datetime([get(r, "delivery") for r in body], errors="coerce", format="mixed")})
    return df.dropna(subset=["auction_date"])[PLAN_COLUMNS].reset_index(drop=True)


def securities(auctions: pd.DataFrame, outstanding: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per ISIN: from its first auction, plus ISINs seen only in the outstanding
    snapshots (bonds auctioned before 1998, syndications) with no first auction."""
    a = auctions.dropna(subset=["isin"]).sort_values("timestamp")
    first = a.groupby("isin", as_index=False).first().rename(columns={"timestamp": "first_auction"})
    out = first[SECURITY_COLUMNS]
    if outstanding is not None and len(outstanding):
        o = outstanding.sort_values("timestamp").groupby("isin", as_index=False).first()
        o = o[~o["isin"].isin(out["isin"])]
        kind = o["instrument_type"].map({"BD-FIX": "BOND", "BD-REAL": "RRB", "MM-TBILL": "TBILL"}).fillna(o["security_type"])
        term = ((o["maturity_date"] - o["issue_date"]).dt.days / 365.25).round()
        extra = pd.DataFrame({"isin": o["isin"], "kind": kind, "term_years": term, "coupon": o["coupon"],
                              "maturity_date": o["maturity_date"], "first_auction": pd.NaT, "issue_date": o["issue_date"]})
        out = pd.concat([out, extra[SECURITY_COLUMNS]], ignore_index=True)
    return out.reset_index(drop=True)


CFT_MIN_BUSINESS_DAYS = 2   # the call for tenders comes 2-5 business days before a bond auction


def cft_known_from(auction_days) -> pd.DatetimeIndex:
    """When a bond auction is certainly known, from its CALL FOR TENDERS: 2 business days
    before (the SHORTEST lead in every schedule snapshot, 2022-2026: 153 auctions, 2-5 business
    days, median 5) - never earlier than the real call, often days later."""
    return pd.DatetimeIndex(auction_days) - pd.offsets.BDay(CFT_MIN_BUSINESS_DAYS)


def event_id(kind: str, term_years) -> str:
    """``CA_AUCTION_<t>Y`` for the regular nominal terms, else one event per kind."""
    if kind == "BOND":
        t = None if pd.isna(term_years) else int(round(float(term_years)))
        return f"CA_AUCTION_{t}Y" if t in (2, 3, 5, 7, 10, 30) else "CA_AUCTION_OTHER"
    return {"RRB": "CA_AUCTION_RRB", "ULTRA": "CA_AUCTION_ULTRA", "TBILL": "CA_AUCTION_TBILL"}.get(kind, "CA_AUCTION_OTHER")
