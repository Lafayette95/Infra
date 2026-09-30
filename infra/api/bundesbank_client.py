"""Deutsche Bundesbank German federal-securities PAR curve - NETWORK ONLY, no files.

Free, public, no key: the Bundesbank SDMX-REST API, dataset BBSIS, item ``ZAR`` - "yields
derived from the term structure of interest rates" for Federal securities with ANNUAL
coupon payments (German: "Aus der Zinsstruktur abgeleitete Renditen für
Bundeswertpapiere mit jährl. Kuponzahlungen"), residual maturities 1..30y, percent, 2
decimals, published the same day. Verified 2026-09-30: rebuilding it from the same
model's zero curve (item ``ZST``, Svensson, annually compounded) with annual coupons
matches within +-0.8bp (2-decimal rounding of both series), continuous compounding is
5-8bp off - i.e. it is the par curve of the Bundesbank's fitted Svensson model.

The API answers a plain script normally from a residential connection, but redirects
EVERY request to a JavaScript bot challenge (303 -> ``/.enodia/challenge``) from a VPN
address (verified 2026-09-30, NordVPN) - raised here as ``BundesbankChallenge``, never
worked around.
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

URL = ("https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
       "D.I.ZAR.ZI.EUR.S1311.B.A604.{maturities}.R.A.A._Z._Z.A?startPeriod={start}&endPeriod={end}")
ACCEPT = "application/vnd.sdmx.data+csv;version=1.0.0"
TENORS = (2, 3, 5, 7, 10, 20, 30)
TIMEOUT_S = 120
_COLS = ["timestamp", "maturity", "value"]


class BundesbankChallenge(RuntimeError):
    """The API answered with its bot challenge instead of data (typically: a VPN is on)."""


def fetch_csv(start: pd.Timestamp, end_inclusive: pd.Timestamp, tenors=TENORS) -> str:
    """One request for every tenor over ``[start, end_inclusive]``."""
    maturities = "+".join(f"R{t:02d}XX" for t in tenors)
    url = URL.format(maturities=maturities, start=start.date(), end=end_inclusive.date())
    req = urllib.request.Request(url, headers={"Accept": ACCEPT, "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # follows the 303, if any
        if "enodia" in resp.geturl() or "csv" not in resp.headers.get("Content-Type", ""):
            raise BundesbankChallenge(f"bot challenge instead of data ({resp.geturl()}) - VPN connected?")
        return resp.read().decode("utf-8-sig")


def parse_par_csv(text: str) -> pd.DataFrame:
    """Long frame ``timestamp, maturity (years), value (percent)``; missing values dropped."""
    if not text.strip():
        return pd.DataFrame(columns=_COLS)
    raw = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    raw = raw[raw["BBK_SEIS_MATURITY"].str.fullmatch(r"R\d\dXX", na=False)]
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw["TIME_PERIOD"]),
        "maturity": raw["BBK_SEIS_MATURITY"].str[1:3].astype(int).astype("float64"),
        "value": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
    })
    return out.dropna(subset=["value"]).reset_index(drop=True)


def fetch_par_curve(start: pd.Timestamp, end: pd.Timestamp, *, fetch=fetch_csv):
    """Par yields for days in ``[start, end)`` plus the interval genuinely COVERED - up to
    the last published day, never beyond (days before it with no row are holidays)."""
    df = parse_par_csv(fetch(start, end - pd.Timedelta(days=1)))
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    covered = [] if df.empty else [(start, min(end, df["timestamp"].max() + pd.Timedelta(days=1)))]
    return df, covered
