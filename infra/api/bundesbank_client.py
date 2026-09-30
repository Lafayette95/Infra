"""Deutsche Bundesbank SVENSSON PARAMETERS of its German Federal-securities term
structure - NETWORK ONLY, no files.

Free, public, no key: the Bundesbank SDMX-REST API, dataset BBSIS, item ``ZST`` (term
structure), valuations ``B0 B1 B2 B3 T1 T2`` (Beta0..3, Tau1..2), issuer ``S1311`` /
security class ``A604`` (listed Federal securities), daily, 5 decimals - e.g.
``BBSIS.D.I.ZST.B0.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A``. (The same valuations under
issuer ``S122`` are the Pfandbrief curve - not ours.)

Verified 2026-09-30 against the Bundesbank's own published curves: the Svensson formula
on these parameters reproduces its zero curve (item ``ZST``, valuation ``ZI``) within
+-0.5bp, and - reading that zero rate as ANNUALLY compounded, ``DF = (1+r)^-T`` - its
annual-coupon par curve (item ``ZAR``) within +-0.45bp (both published to 2 decimals).
Reading it as continuously compounded is 5-8bp off.

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
       "D.I.ZST.B0+B1+B2+B3+T1+T2.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A?startPeriod={start}&endPeriod={end}")
ACCEPT = "application/vnd.sdmx.data+csv;version=1.0.0"
TIMEOUT_S = 120
PARAMS = ["B0", "B1", "B2", "B3", "T1", "T2"]


class BundesbankChallenge(RuntimeError):
    """The API answered with its bot challenge instead of data (typically: a VPN is on)."""


def fetch_csv(start: pd.Timestamp, end_inclusive: pd.Timestamp) -> str:
    """One request for all six parameters over ``[start, end_inclusive]``."""
    url = URL.format(start=start.date(), end=end_inclusive.date())
    req = urllib.request.Request(url, headers={"Accept": ACCEPT, "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # follows the 303, if any
        if "enodia" in resp.geturl() or "csv" not in resp.headers.get("Content-Type", ""):
            raise BundesbankChallenge(f"bot challenge instead of data ({resp.geturl()}) - VPN connected?")
        return resp.read().decode("utf-8-sig")


def parse_params_csv(text: str) -> pd.DataFrame:
    """One row per day: ``timestamp, B0, B1, B2, B3, T1, T2``; a day missing any parameter
    ("." - weekends/holidays are listed blank) is dropped."""
    cols = ["timestamp", *PARAMS]
    if not text.strip():
        return pd.DataFrame(columns=cols)
    raw = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    raw = raw[raw["BBK_SEIS_VALUATION"].isin(PARAMS)]
    raw = raw.assign(value=pd.to_numeric(raw["OBS_VALUE"], errors="coerce"))
    wide = raw.pivot_table(index="TIME_PERIOD", columns="BBK_SEIS_VALUATION", values="value", aggfunc="last")
    wide = wide.reindex(columns=PARAMS).dropna()
    out = wide.reset_index().rename(columns={"TIME_PERIOD": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out.columns.name = None
    return out[cols].reset_index(drop=True)


def fetch_svensson_params(start: pd.Timestamp, end: pd.Timestamp, *, fetch=fetch_csv):
    """Parameters for days in ``[start, end)`` plus the interval genuinely COVERED - up to
    the last published day, never beyond (days before it with no row are holidays)."""
    df = parse_params_csv(fetch(start, end - pd.Timedelta(days=1)))
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    covered = [] if df.empty else [(start, min(end, df["timestamp"].max() + pd.Timedelta(days=1)))]
    return df, covered
