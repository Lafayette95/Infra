"""German Federal securities - pure parsing (no network, no files): the Finanzagentur's
issuance history into auctions and a securities table, and the Bundesbank's BBSSY per-ISIN
prices / yields into a daily price frame.

Auctions: one row per (auction day, ISIN) - a multi-ISIN auction is one row per ISIN.
Securities: one row per ISIN from its FIRST issuance (type, coupon, maturity, maturity
segment, first issue day, flags); the ISSUE (value) day is not in the file, so it is taken
as the first auction / syndication day + 2 weekdays (German Federal auctions settle T+2 -
an approximation next to a TARGET holiday). Prices: ``price_clean`` (KCP), ``price_dirty``
(KDP, 2022-06 on), ``yield`` (REN, ISMA), percent / % of par.
"""
from __future__ import annotations

import datetime
import io

import numpy as np
import pandas as pd

AUCTION_COLUMNS = ["timestamp", "isin", "type", "coupon", "maturity_date", "segment", "volume_m", "issue_kind",
                   "process", "bids_m", "competitive_m", "noncompetitive_m", "allotted_m", "lowest_price",
                   "avg_price", "avg_yield", "retention_m", "bid_to_cover"]
SECURITY_COLUMNS = ["isin", "type", "segment", "tenor_years", "coupon", "maturity_date", "first_auction",
                    "issue_date", "green", "inflation_linked", "bill"]
PRICE_COLUMNS = ["timestamp", "isin", "security_class", "price_clean", "price_dirty", "yield"]
PRICE_VALUES = ["price_clean", "price_dirty", "yield"]
ITEMS = {"KCP": "price_clean", "KDP": "price_dirty", "REN": "yield"}
SETTLEMENT_DAYS = 2


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_issuance_history(xlsx: bytes) -> pd.DataFrame:
    """Every issuance row of the workbook (footnotes and headers dropped). Coupons are in
    PERCENT (the file gives 0.0375 for 3.75%)."""
    raw = pd.read_excel(io.BytesIO(xlsx), header=None)
    is_day = raw[1].map(lambda v: isinstance(v, (pd.Timestamp, datetime.datetime)))
    body = raw[raw[2].astype(str).str.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d") & is_day]
    body = body.iloc[:, :len(AUCTION_COLUMNS) + 1]
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(body[1]).dt.normalize(),
        "isin": body[2].astype(str).str.strip(),
        "type": body[3].astype(str).str.strip(),
        "coupon": (_num(body[4]) * 100.0).round(6),
        "maturity_date": pd.to_datetime(body[5], errors="coerce").dt.normalize(),
        "segment": body[6].astype(str).str.strip(),
        "volume_m": _num(body[7]),
        "issue_kind": body[8].astype(str).str.strip(),
        "process": body[9].astype(str).str.strip(),
        "bids_m": _num(body[10]), "competitive_m": _num(body[11]), "noncompetitive_m": _num(body[12]),
        "allotted_m": _num(body[13]), "lowest_price": _num(body[14]), "avg_price": _num(body[15]),
        "avg_yield": _num(body[16]), "retention_m": _num(body[17]), "bid_to_cover": _num(body[18]),
    })
    return out.sort_values(["timestamp", "isin"]).reset_index(drop=True)


def _tenor_years(segment: str) -> float:
    n, _, unit = segment.partition(" ")
    try:
        return float(n) / (12.0 if unit.upper().startswith("M") else 1.0)
    except ValueError:
        return np.nan


def securities(auctions: pd.DataFrame) -> pd.DataFrame:
    """One row per ISIN, from its first issuance."""
    if auctions.empty:
        return pd.DataFrame(columns=SECURITY_COLUMNS)
    a = auctions.sort_values(["timestamp", "isin"])
    first = a.groupby("isin", as_index=False).first()
    out = pd.DataFrame({
        "isin": first["isin"], "type": first["type"], "segment": first["segment"],
        "tenor_years": first["segment"].map(_tenor_years), "coupon": first["coupon"].fillna(0.0).round(6),
        "maturity_date": first["maturity_date"], "first_auction": first["timestamp"],
        "issue_date": first["timestamp"] + pd.offsets.BDay(SETTLEMENT_DAYS),
        "green": first["type"].str.contains("Green", case=False),
        "inflation_linked": first["type"].str.upper().eq("ILB"),
        "bill": first["type"].str.contains("Bubill", case=False),
    })
    return out[SECURITY_COLUMNS].sort_values("isin").reset_index(drop=True)


def parse_bbssy_csv(text: str) -> pd.DataFrame:
    """``timestamp, isin, security_class, price_clean, price_dirty, yield`` - one row per
    ISIN-day with at least one value (weekends / holidays are listed as ".")."""
    if not text.strip():
        return pd.DataFrame(columns=PRICE_COLUMNS)
    raw = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    raw = raw[raw["BBK_SEIS_ITEM"].isin(ITEMS)].assign(value=lambda d: _num(d["OBS_VALUE"]))
    raw = raw.dropna(subset=["value"])
    if raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    wide = raw.pivot_table(index=["TIME_PERIOD", "BBK_SEIS_ISIN", "BBK_SEIS_SECURITY_CLASS"],
                           columns="BBK_SEIS_ITEM", values="value", aggfunc="last").rename(columns=ITEMS)
    wide = wide.reindex(columns=PRICE_VALUES).reset_index()
    wide.columns.name = None
    out = wide.rename(columns={"TIME_PERIOD": "timestamp", "BBK_SEIS_ISIN": "isin", "BBK_SEIS_SECURITY_CLASS": "security_class"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out[PRICE_COLUMNS].sort_values(["timestamp", "isin"]).reset_index(drop=True)


def encode_prices(df: pd.DataFrame) -> pd.DataFrame:
    """x10000 nullable Int32 (CLAUDE.md 6b)."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    for c in PRICE_VALUES:
        out[c] = (pd.to_numeric(out[c], errors="coerce") * 10000).round().astype("Int32")
    return out


def decode_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in PRICE_VALUES:
        out[c] = out[c].astype("float64") / 10000.0
    for c in ("isin", "security_class"):
        out[c] = out[c].astype(str)
    return out


# ------------------------------------------------------------------ irregular first coupons
# German new issues accrue from their issue date and pay a LONG first coupon: the first
# maturity anniversary after the issue is skipped (verified 2026-10-07 on the Bundesbank's
# published accrued: 40 of 41 issues since 2022-06 with data across that anniversary; the
# exception, a 2023 Schatz, paid a short one - detected from the data, ``short_first``).
REGULAR_STUB_DAYS = 7     # an interest start within this of an anniversary is a regular period


def implied_commencement(accrued: float, coupon: float, settle, maturity) -> pd.Timestamp:
    """Interest start implied by a published accrued, ACT/ACT on the annual period."""
    from infra.processing.treasury_prices import coupon_dates
    prev, nxt = coupon_dates(pd.Timestamp(maturity), pd.Timestamp(settle), 1)[:2]
    days = accrued / coupon * (nxt - prev).days
    if not np.isfinite(days) or abs(days) > 800:      # no sensible start (an index-linked accrued, a typo)
        return pd.NaT
    return (pd.Timestamp(settle) - pd.Timedelta(days=round(days))).normalize()


def first_period_flows(coupon: float, maturity, settle, commencement, short_first: bool = False):
    """``(t, a)`` of a bond in its FIRST coupon period (interest from ``commencement``), or
    None if it isn't in one (or starts on an anniversary): coupon amounts ICMA ACT/ACT -
    a long first coupon = one full coupon + the stub's share of the quasi-period before it."""
    from infra.analytics.treasury_curve import YEAR
    from infra.processing.treasury_prices import coupon_dates
    maturity, settle, c0 = pd.Timestamp(maturity), pd.Timestamp(settle), pd.Timestamp(commencement)
    q = coupon_dates(maturity, c0, 1)          # quasi-coupon dates from the one at or before c0
    a0, a1 = q[0], q[1]
    if (a1 - c0).days <= REGULAR_STUB_DAYS or (c0 - a0).days <= REGULAR_STUB_DAYS:
        return None
    stub = (a1 - c0).days / (a1 - a0).days
    first = a1 if short_first or len(q) < 3 else q[2]
    if settle >= first:
        return None
    first_amt = coupon * (stub if first == a1 else 1.0 + stub)
    dates = [d for d in q if d >= first]
    t = np.array([(d - settle).days / YEAR for d in dates])
    a = np.full(len(dates), float(coupon))
    a[0] = first_amt
    a[-1] += 100.0
    return t, a
