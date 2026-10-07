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
_OLD_SPOT_SHEET = "4. nominal spot curve"
TIMEOUT_S = 300
_ARCHIVE_FILE = re.compile(r"GLC Nominal daily data_(\d{4}) to (\d{4}|present)\.xlsx$")
_LATEST_FILE = "GLC Nominal daily data current month.xlsx"
_COLS = ["timestamp", "maturity", "value"]


def fetch_zip(name: str) -> bytes:
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read()


def parse_spot_sheet(xlsx: bytes, sheet: str = SPOT_SHEET) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    """``(long frame timestamp/maturity/value, first date row, last date row)``. Date rows
    with no values at all (a holiday row the BoE still lists) count for the span. The
    maturity row (4th) is in YEARS on every BoE spot sheet (the short-end sheets also carry
    a months row above it)."""
    book = pd.ExcelFile(io.BytesIO(xlsx))
    if sheet not in book.sheet_names and sheet == SPOT_SHEET and _OLD_SPOT_SHEET in book.sheet_names:
        sheet = _OLD_SPOT_SHEET     # the 1979-2004 workbooks: same layout, "nominal" in the name
    raw = pd.read_excel(book, sheet_name=sheet, header=None)
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


def _workbooks(start: pd.Timestamp, end: pd.Timestamp, fetch, now: pd.Timestamp | None = None, *,
               archive_zip: str = ARCHIVE_ZIP, archive_file=None, latest_file: str | None = None) -> list[tuple[str, bytes]]:
    """``[(kind, xlsx bytes)]`` needed for ``[start, end)``, archive first."""
    archive_file = _ARCHIVE_FILE if archive_file is None else archive_file
    latest_file = _LATEST_FILE if latest_file is None else latest_file
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    month_start = now.normalize().replace(day=1)
    out = []
    if start < month_start:
        with zipfile.ZipFile(io.BytesIO(fetch(archive_zip))) as z:
            for name in sorted(z.namelist()):
                m = archive_file.search(name)
                if not m:
                    continue
                first = pd.Timestamp(f"{m.group(1)}-01-01")
                last = pd.Timestamp.max if m.group(2) == "present" else pd.Timestamp(f"{int(m.group(2)) + 1}-01-01")
                if first < end and start < last:
                    out.append(("archive", z.read(name)))
    if end > month_start - pd.Timedelta(days=7):  # the archive can lag the month's close
        with zipfile.ZipFile(io.BytesIO(fetch(LATEST_ZIP))) as z:
            out.append(("latest", z.read(next(n for n in z.namelist() if n.endswith(latest_file)))))
    return out


# The BoE's SONIA OIS curve (2009 on): its own archive zip, the same "latest" zip; spot rates
# SEMI-ANNUALLY compounded (2026-10-07: par SONIA swaps priced off it match DTCC closes to
# -0.4bp mean, sd 1.5bp; read as continuous +4.4bp, annual -5.1bp). The short-end sheet gives
# monthly maturities to 5y, the curve sheet 0.5y-25y.
OIS_ARCHIVE_ZIP = "oisddata.zip"
_OIS_ARCHIVE_FILE = re.compile(r"OIS daily data_(\d{4}) to (\d{4}|present)\.xlsx$")
_OIS_LATEST_FILE = "OIS daily data current month.xlsx"
OIS_SHEETS = ("3. spot, short end", "4. spot curve")
_OIS_SINGLE_SHEET = "2. spot curve"


def fetch_ois_curve(start: pd.Timestamp, end: pd.Timestamp, *, fetch=fetch_zip, now: pd.Timestamp | None = None):
    """The BoE SONIA OIS spot curve, both sheets merged (the short end's monthly points to
    5y, the curve sheet beyond), for days in ``[start, end)`` plus the intervals COVERED -
    as ``fetch_spot_curve``."""
    return fetch_spot_curve(start, end, fetch=fetch, now=now, _ois=True)


def fetch_spot_curve(start: pd.Timestamp, end: pd.Timestamp, *, fetch=fetch_zip, now: pd.Timestamp | None = None,
                     _ois: bool = False):
    """Spot yields (percent) for days in ``[start, end)`` plus the intervals genuinely
    COVERED. Each workbook covers from its first date row to its last; an archive
    workbook ending in the month right before the current-month workbook's month is
    complete up to that month's start (the archive is refreshed once a month closes), so
    no weekend/holiday gap is left at the seam - but a STALE archive (ending earlier)
    leaves a real gap, never claimed covered, retried on the next run."""
    if _ois:
        books = []
        for kind, x in _workbooks(start, end, fetch, now, archive_zip=OIS_ARCHIVE_ZIP, archive_file=_OIS_ARCHIVE_FILE,
                                  latest_file=_OIS_LATEST_FILE):
            names = pd.ExcelFile(io.BytesIO(x)).sheet_names
            # 2016 on: a short-end sheet (monthly to 5y) + the curve sheet (0.5y-25y); the
            # 2009-2015 workbook: ONE spot sheet ("2. spot curve", monthly maturities)
            sheets = OIS_SHEETS if all(sh in names for sh in OIS_SHEETS) else (_OIS_SINGLE_SHEET,)
            parts = [parse_spot_sheet(x, sh) for sh in sheets]
            short = parts[0][0]
            curve = parts[-1][0] if len(parts) > 1 else short.iloc[0:0]
            curve = curve[curve["maturity"] > (short["maturity"].max() if len(short) else 0)]
            firsts = [p[1] for p in parts if p[1] is not None]; lasts = [p[2] for p in parts if p[2] is not None]
            books.append((kind, pd.concat([short, curve], ignore_index=True), min(firsts) if firsts else None,
                          max(lasts) if lasts else None))
    else:
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
