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
