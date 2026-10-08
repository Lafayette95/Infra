"""Japan Ministry of Finance JGB interest rates (constant maturity) - NETWORK ONLY, no files.

Free, keyless (verified 2026-10-07, https://www.mof.go.jp/english/policy/jgbs/reference/
interest_rate/index.htm and its Q&A): JGB yields at 1-10, 15, 20, 25, 30 and 40 years, daily
since 1974-09-24, "semiannual compound interest rate[s]" read off a curve through selected
JGBs at the JSDA reference prices of the 15:00 Tokyo close, released 09:30 the next business
day. Two CSVs: the history to the end of last month (~1.2MB) and the current month. "-" =
no value (the long tenors before they were issued).
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

BASE = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
HISTORY = BASE + "historical/jgbcme_all.csv"
CURRENT = BASE + "jgbcme.csv"
TIMEOUT_S = 120
_COLS = ["timestamp", "maturity", "value"]


def fetch_csv(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8-sig", "replace")


def parse_csv(text: str) -> pd.DataFrame:
    """Long ``timestamp, maturity (years), value (%)``; rows with no value dropped."""
    if not text.strip():
        return pd.DataFrame(columns=_COLS)
    raw = pd.read_csv(io.StringIO(text), skiprows=1, dtype=str)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["timestamp"] = pd.to_datetime(raw["Date"], format="%Y/%m/%d", errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    tenors = [c for c in raw.columns if c.endswith("Y") and c[:-1].isdigit()]
    long = raw.melt(id_vars=["timestamp"], value_vars=tenors, var_name="tenor", value_name="value")
    long["maturity"] = long["tenor"].str[:-1].astype(float)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])[_COLS].sort_values(["timestamp", "maturity"]).reset_index(drop=True)


def fetch_curve(start: pd.Timestamp, end: pd.Timestamp, *, fetch=fetch_csv, now: pd.Timestamp | None = None):
    """Yields for days in ``[start, end)`` and the interval genuinely COVERED: up to the last
    published day, never beyond. The history file only when ``start`` is before this month."""
    now = pd.Timestamp.now().normalize() if now is None else pd.Timestamp(now)
    month_start = now.replace(day=1)
    frames = [parse_csv(fetch(CURRENT))]
    if start < month_start:
        frames.insert(0, parse_csv(fetch(HISTORY)))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["timestamp", "maturity"], keep="last")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    covered = [] if df.empty else [(start, min(end, df["timestamp"].max() + pd.Timedelta(days=1)))]
    return df, covered


# ---------------------------------------------------------------- JGB auctions
# Verified 2026-10-08 (https://www.mof.go.jp/english/policy/jgbs/auction/): results workbooks
# per security type (coupon JGBs 1979 on, T-bills FY2008 on, Liquidity Enhancement Auctions
# 2006 on; updated about monthly, so the latest weeks are on the calendar pages), one auction
# CALENDAR page per month from 2023 (``calendar/<yymm>e.htm``, its month announced at the end
# of the month three months before) and dated ALTERATION notices (``<yymm>ae.htm``: the
# calendar before and after the change).
AUCTION_BASE = "https://www.mof.go.jp/english/policy/jgbs/auction/"
RESULT_FILES = {"jgb": "past_auction_results/Auction_Results_for_JGBs.xls",
                "tbill": "past_auction_results/Auction_Results_for_T-bills.xls",
                "liquidity": "past_auction_results/e-ryudousei_historical_data.xls"}
CALENDAR_INDEX = AUCTION_BASE + "calendar/index.htm"
# the JAPANESE calendar publishes each month ahead (December 2026 on 2026-09-29, the end of the
# month three months before); the English pages only appear once the month has started
AUCTION_BASE_JA = "https://www.mof.go.jp/jgbs/auction/"
CALENDAR_INDEX_JA = AUCTION_BASE_JA + "calendar/index.htm"


def fetch_url(url: str) -> tuple[bytes, str | None]:
    """The body and its ``Last-Modified`` header; raises on an HTTP error (a 404 month)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read(), resp.headers.get("Last-Modified")


def last_modified(url: str) -> str | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.headers.get("Last-Modified")


def calendar_links(index_html: str) -> list[str]:
    """Page codes linked from a calendar index: ``<yymm>`` / ``<yymm>a`` (Japanese),
    ``<yymm>e`` / ``<yymm>ae`` (English)."""
    import re
    return sorted(set(re.findall(r"calendar/(\d{4}a?e?)\.htm", index_html)))
