"""US Treasury Daily Par Yield Curve Rates (the CMT curve) - NETWORK ONLY, no files.

Free, public, no key. One CSV per calendar year; each row is one business day with the
par yield (percent, semi-annual bond-equivalent basis) per tenor column ("2 Yr", ...).
Verified 2026-09-30: columns vary by year (e.g. "1.5 Month" / "4 Mo" were added later),
so tenors are read by column NAME, never by position.
"""
from __future__ import annotations

import io
import re
import urllib.request

import pandas as pd

URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
       "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
       "&field_tdr_date_value={year}&page&_format=csv")
TIMEOUT_S = 60
_YEAR_COLUMN = re.compile(r"^(\d+) Yr$")


def fetch_year_csv(year: int) -> str:
    req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8-sig")


def parse_par_csv(text: str) -> pd.DataFrame:
    """Long frame ``timestamp, maturity (years), value (percent)`` for the whole-year
    tenors; blank cells (a tenor not issued at the time) are dropped."""
    cols = ["timestamp", "maturity", "value"]
    if not text.strip():
        return pd.DataFrame(columns=cols)
    wide = pd.read_csv(io.StringIO(text))
    if "Date" not in wide.columns:
        return pd.DataFrame(columns=cols)
    tenors = {c: int(m.group(1)) for c in wide.columns if (m := _YEAR_COLUMN.match(c))}
    long = wide.melt(id_vars="Date", value_vars=list(tenors), var_name="column", value_name="value")
    long["timestamp"] = pd.to_datetime(long["Date"], format="%m/%d/%Y")
    long["maturity"] = long["column"].map(tenors).astype("float64")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])[cols].reset_index(drop=True)


def fetch_par_curve(start: pd.Timestamp, end: pd.Timestamp, *, fetch_year=fetch_year_csv):
    """Par yields for days in ``[start, end)`` plus the ``[start, end)`` intervals the
    response genuinely COVERS: up to the last published day, never beyond - a day not yet
    published is never claimed covered (days before it with no row are holidays)."""
    frames = [parse_par_csv(fetch_year(y)) for y in range(start.year, (end - pd.Timedelta(days=1)).year + 1)]
    df = pd.concat([f for f in frames if not f.empty] or [parse_par_csv("")], ignore_index=True)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    covered = [] if df.empty else [(start, min(end, df["timestamp"].max() + pd.Timedelta(days=1)))]
    return df, covered
