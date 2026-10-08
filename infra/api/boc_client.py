"""Bank of Canada benchmark Government of Canada bond yields - NETWORK ONLY, no files.

Valet API, free and keyless (https://www.bankofcanada.ca/valet/, terms
https://www.bankofcanada.ca/terms/; verified 2026-10-07): group ``bond_yields_benchmark``,
daily from 2001-01-02, 2 decimals - "mid-market closing yields of selected Government of
Canada bond issues that mature approximately in the indicated terms" (2, 3, 5, 7, 10 years
and "long", currently the 2057 bond). A benchmark is "generally changed when a building
benchmark bond is adopted by financial markets as a benchmark, typically after the last
auction for that bond" - so a switch day's change mixes two bonds (no bond id is published).
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

URL = "https://www.bankofcanada.ca/valet/observations/group/bond_yields_benchmark/csv?start_date={start}&end_date={end}"
TIMEOUT_S = 120
SERIES = {"BD.CDN.2YR.DQ.YLD": 2.0, "BD.CDN.3YR.DQ.YLD": 3.0, "BD.CDN.5YR.DQ.YLD": 5.0, "BD.CDN.7YR.DQ.YLD": 7.0,
          "BD.CDN.10YR.DQ.YLD": 10.0, "BD.CDN.LONG.DQ.YLD": 30.0}   # "long" stored as the 30y
_COLS = ["timestamp", "maturity", "value"]


def fetch_csv(start: pd.Timestamp, end_inclusive: pd.Timestamp) -> str:
    url = URL.format(start=pd.Timestamp(start).date(), end=pd.Timestamp(end_inclusive).date())
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8-sig", "replace")


def parse_csv(text: str) -> pd.DataFrame:
    """The OBSERVATIONS block (after its ``"date",...`` header line) as long rows."""
    lines = text.splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith('"date"')), None)
    if head is None:
        return pd.DataFrame(columns=_COLS)
    raw = pd.read_csv(io.StringIO("\n".join(lines[head:])), dtype=str)
    raw["timestamp"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    long = raw.melt(id_vars=["timestamp"], value_vars=[c for c in SERIES if c in raw], var_name="series", value_name="value")
    long["maturity"] = long["series"].map(SERIES)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])[_COLS].sort_values(["timestamp", "maturity"]).reset_index(drop=True)


def fetch_benchmark_yields(start: pd.Timestamp, end: pd.Timestamp, *, fetch=fetch_csv):
    """Yields for days in ``[start, end)`` and the interval COVERED: up to the last
    published day, never beyond."""
    df = parse_csv(fetch(start, end - pd.Timedelta(days=1)))
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)
    covered = [] if df.empty else [(start, min(end, df["timestamp"].max() + pd.Timedelta(days=1)))]
    return df, covered


# ------------------------------------------------------------------ benchmark page, zero curve
PAGE = "https://www.bankofcanada.ca/rates/interest-rates/canadian-bonds/"
ZERO_URL = "https://www.bankofcanada.ca/stats/results/csv"


def fetch_benchmark_page() -> bytes:
    """The "Selected bond yields" page: it names each current benchmark bond and the date it
    became the benchmark (parsed by ``infra.processing.boc_benchmarks``)."""
    req = urllib.request.Request(PAGE, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read()


def fetch_zero_curve(start: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.DataFrame:
    """The BoC fitted zero-coupon curve (Bolder-Johnson-Metzler; 0.25-30y every quarter year,
    decimals, continuously compounded), published weekly on Thursdays with a two-week lag;
    from 1986. Long ``timestamp, maturity, zero`` (decimal); "na" rows dropped."""
    import urllib.parse
    body = urllib.parse.urlencode({"lookupPage": "lookup_yield_curve.php", "startRange": "1986-01-01",
                                   "searchRange": "", "dFrom": str(pd.Timestamp(start).date()),
                                   "dTo": str(pd.Timestamp(end_inclusive).date())}).encode()
    req = urllib.request.Request(ZERO_URL, data=body, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = resp.read().decode("utf-8-sig", "replace")
    raw = pd.read_csv(io.StringIO(text), skipinitialspace=True, na_values=["na"], dtype=str)
    raw = raw.rename(columns=lambda c: c.strip())
    if "Date" not in raw:
        return pd.DataFrame(columns=["timestamp", "maturity", "zero"])
    raw["timestamp"] = pd.to_datetime(raw["Date"], errors="coerce")
    cols = [c for c in raw.columns if c.startswith("ZC")]
    long = raw.dropna(subset=["timestamp"]).melt(id_vars=["timestamp"], value_vars=cols, var_name="col", value_name="zero")
    long["maturity"] = long["col"].str[2:-2].astype(int) / 100.0
    long["zero"] = pd.to_numeric(long["zero"], errors="coerce")
    return long.dropna(subset=["zero"])[["timestamp", "maturity", "zero"]].reset_index(drop=True)


# ---------------------------------------------------------------- Government of Canada auctions
# Valet groups (verified 2026-10-08): AUC_BOND_RESULTS (nominal bonds, 1998-10 on, with ISIN and
# each auction's bidding deadline), AUC_BOND_RR_RESULTS (real return), AUC_BOND_U_RESULTS (ultra
# long), AUC_TBILL_RESULTS (T-bills), AUC_SCHED (the CURRENT quarterly bond auction schedule),
# GOC_OUTSTANDING / GOC_OUTSTANDING_2025 (every outstanding bill and bond by ISIN, daily, 2025 on;
# the 2025 group's fields carry a "_2025" suffix).
VALET_GROUP = "https://www.bankofcanada.ca/valet/observations/group/{group}/json"
SCHEDULE_PAGE = ("https://www.bankofcanada.ca/markets/government-securities-auctions/calls-for-tenders-and-results/"
                 "bond-auction-schedule/")


def fetch_valet_group(group: str) -> dict:
    import json
    req = urllib.request.Request(VALET_GROUP.format(group=group), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))

# monthly outstanding-bonds CSVs (series codes, no ISIN): only 2018-01..2022-01 exist (probed 2026-10-08)
DMB_CSV = "https://www.bankofcanada.ca/stats/assets/csv/en-GOC_DMB_{ym}.csv"
DMB_MONTHS = pd.period_range("2018-01", "2022-01", freq="M")


def fetch_dmb_csv(ym: str) -> str:
    req = urllib.request.Request(DMB_CSV.format(ym=ym), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "replace")
