"""FTSE-Tradeweb gilt and EuroGov closing prices - pure parsing (no network, no files) of
InSite's CSV export (root CLAUDE.md 18).

Columns as published: name, close-of-business date, ISIN, type (Conventional / Index-linked /
Strips / Bills; EuroGov lines are all "Conventional", bills included), coupon, maturity, clean
and dirty price, yield, modified duration, accrued interest ("N/A" where it doesn't apply).
Stored x10000 nullable Int32 like the other price stores (root CLAUDE.md 6b); the source gives
up to 6 decimals, so a stored price is exact to 0.0001 and a yield to 0.01bp.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "isin", "name", "security_type", "country", "coupon", "maturity_date", "price_clean",
           "price_dirty", "yield", "mod_duration", "accrued"]
SCALED = ["coupon", "price_clean", "price_dirty", "yield", "mod_duration", "accrued"]
_RENAME = {"Gilt Name": "name", "Close of Business Date": "timestamp", "ISIN": "isin", "Type": "security_type",
           "Coupon": "coupon", "Maturity": "maturity_date", "Clean Price": "price_clean", "Dirty Price": "price_dirty",
           "Yield": "yield", "Mod Duration": "mod_duration", "Accrued Interest": "accrued"}


def parse_export(text: str) -> pd.DataFrame:
    """One export (any number of days and securities) as rows; issuer country = the ISIN's
    first two letters (``EU`` for the European Union's bonds)."""
    if not text or not text.strip():
        return pd.DataFrame(columns=COLUMNS)
    raw = pd.read_csv(io.StringIO(text), dtype=str, na_values=["N/A", ""], keep_default_na=False)
    df = raw.rename(columns=_RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y")
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], format="%m/%d/%Y", errors="coerce")
    for c in SCALED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["name"] = df["name"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["country"] = df["isin"].str[:2]
    return df[COLUMNS].drop_duplicates(["timestamp", "isin"], keep="last").reset_index(drop=True)


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in SCALED:
        out[c] = (out[c] * 10000).round().astype("Int32")
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    out["maturity_date"] = pd.to_datetime(out["maturity_date"]).astype("datetime64[ms]")
    for c in ("isin", "name", "security_type", "country"):
        out[c] = out[c].astype("string")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in SCALED:
        out[c] = out[c].astype("float64") / 10000.0
    for c in ("isin", "name", "security_type", "country"):
        out[c] = out[c].astype(str)
    return out
