"""Bank of England nominal gilt SPOT curve (zero-coupon, VRP spline) - NETWORK ONLY.

Free, public, no key. Published as Excel workbooks inside two zips (verified 2026-09-30,
https://www.bankofengland.co.uk/statistics/yield-curves):

* ``glcnominalddata.zip`` (~39 MB) - the archive, one workbook per period
  ("GLC Nominal daily data_2016 to 2024.xlsx", "..._2025 to present.xlsx", ...), updated
  once a month is complete;
* ``latest-yield-curve-data.zip`` - the CURRENT month ("GLC Nominal daily data current
  month.xlsx").

Every workbook's sheet "4. spot curve" has the maturities (years, 0.5 .. 40) on row 3 and
one row per business day from row 5, column 0 = date. The archive is only downloaded when
a request reaches before the current month.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile

import pandas as pd

BASE = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
ARCHIVE_ZIP = "glcnominalddata.zip"
LATEST_ZIP = "latest-yield-curve-data.zip"
SPOT_SHEET = "4. spot curve"
TIMEOUT_S = 300
_ARCHIVE_FILE = re.compile(r"GLC Nominal daily data_(\d{4}) to (\d{4}|present)\.xlsx$")
_LATEST_FILE = "GLC Nominal daily data current month.xlsx"
_COLS = ["timestamp", "maturity", "value"]


def fetch_zip(name: str) -> bytes:
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read()


def parse_spot_sheet(xlsx: bytes) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """``(long frame timestamp/maturity/value, first date row, last date row)``. Date rows
    with no values at all (a holiday row the BoE still lists) count for the span."""
    raw = pd.read_excel(io.BytesIO(xlsx), sheet_name=SPOT_SHEET, header=None)
    maturities = pd.to_numeric(raw.iloc[3, 1:], errors="coerce")
    body = raw.iloc[5:]
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    body = body[dates.notna()]
    dates = dates[dates.notna()]
    if body.empty:
        return pd.DataFrame(columns=_COLS), None, None
    wide = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    wide.columns = maturities.to_numpy()
    wide = wide.loc[:, wide.columns.notna()]
    wide.index = dates.to_numpy()
    long = wide.stack().rename("value").reset_index()
    long.columns = _COLS
    long["maturity"] = long["maturity"].astype("float64")
    long = long.dropna(subset=["value"]).reset_index(drop=True)  # holiday rows are listed blank
    return long, dates.min().normalize(), dates.max().normalize()


def _workbooks(start: pd.Timestamp, end: pd.Timestamp, fetch, now: pd.Timestamp | None = None) -> list[tuple[str, bytes]]:
    """``[(kind, xlsx bytes)]`` needed for ``[start, end)``, archive first."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    month_start = now.normalize().replace(day=1)
    out = []
    if start < month_start:
        with zipfile.ZipFile(io.BytesIO(fetch(ARCHIVE_ZIP))) as z:
            for name in sorted(z.namelist()):
                m = _ARCHIVE_FILE.search(name)
                if not m:
                    continue
                first = pd.Timestamp(f"{m.group(1)}-01-01")
                last = pd.Timestamp.max if m.group(2) == "present" else pd.Timestamp(f"{int(m.group(2)) + 1}-01-01")
                if first < end and start < last:
                    out.append(("archive", z.read(name)))
    if end > month_start - pd.Timedelta(days=7):  # the archive can lag the month's close
        with zipfile.ZipFile(io.BytesIO(fetch(LATEST_ZIP))) as z:
            out.append(("latest", z.read(next(n for n in z.namelist() if n.endswith(_LATEST_FILE)))))
    return out


def fetch_spot_curve(start: pd.Timestamp, end: pd.Timestamp, *, fetch=fetch_zip, now: pd.Timestamp | None = None):
    """Spot yields (percent) for days in ``[start, end)`` plus the intervals genuinely
    COVERED. Each workbook covers from its first date row to its last; an archive
    workbook ending in the month right before the current-month workbook's month is
    complete up to that month's start (the archive is refreshed once a month closes), so
    no weekend/holiday gap is left at the seam - but a STALE archive (ending earlier)
    leaves a real gap, never claimed covered, retried on the next run."""
    books = [(kind, *parse_spot_sheet(x)) for kind, x in _workbooks(start, end, fetch, now)]
    latest_first = next((f for kind, _, f, _ in books if kind == "latest" and f is not None), None)
    frames, covered = [], []
    for kind, df, first, last in books:
        if first is None:
            continue
        frames.append(df)
        stop = last + pd.Timedelta(days=1)
        if kind == "archive" and latest_first is not None:
            seam = latest_first.replace(day=1)
            if last >= seam - pd.offsets.MonthBegin(1):
                stop = max(stop, seam)
        covered.append((max(start, first), min(end, stop)))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLS)
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].drop_duplicates(
        ["timestamp", "maturity"], keep="last").reset_index(drop=True)
    return df, [(s, e) for s, e in covered if s < e]
