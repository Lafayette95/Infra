"""DTCC Public Price Dissemination (pddata.dtcc.com) - NETWORK ONLY, no files.

Free, no key. Every swap a US person trades is reported to a swap data repository and
publicly disseminated (CFTC Part 43); DTCC publishes one CUMULATIVE zip per report kind
(``RATES``, ``CREDITS``, ``FOREX``, ...) per day. Facts this module relies on (verified
2026-10-01):
* the file for day D covers dissemination over the UTC day D (event timestamps up to
  23:59 UTC) and appears once, complete, shortly after midnight UTC; before that the URL
  answers 404 - and so does every day that has aged out of DTCC's rolling ~2-year archive;
* the CDN answers 503 to bursts of requests (and to every HEAD request), so requests are
  sequential and a 503 is retried with backoff;
* one zip holds one CSV (~15 MB unzipped for a weekday's RATES), every asset class's
  products in it identified by ``UPI FISN`` / ``UPI Underlier Name`` (``Product name`` and
  the underlier ID columns are empty).
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request

import pandas as pd

BASE_URL = "https://pddata.dtcc.com/ppd/api/report/cumulative/cftc"
TIMEOUT_S = 120
RETRIES = 5
BACKOFF_S = 2.0


class DtccError(RuntimeError):
    pass


def file_name(kind: str, day) -> str:
    return f"CFTC_CUMULATIVE_{kind}_{pd.Timestamp(day):%Y_%m_%d}.zip"


def url(kind: str, day) -> str:
    return f"{BASE_URL}/{file_name(kind, day)}"


def fetch_cumulative(kind: str, day, *, sleep=time.sleep) -> bytes | None:
    """The day's cumulative zip as published, or None if DTCC has no file for it (not
    published yet, or aged out). Raises DtccError once retries are exhausted."""
    request = urllib.request.Request(url(kind, day), headers={"User-Agent": "infra-data-pipeline"})
    last = None
    for attempt in range(RETRIES + 1):
        if attempt:
            sleep(BACKOFF_S * 2 ** (attempt - 1))
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                content = resp.read()
            if content[:2] == b"PK":
                return content
            last = f"not a zip ({len(content)} bytes)"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise DtccError(f"{file_name(kind, day)}: {last} after {RETRIES + 1} attempts")



# The DTCC GCF Repo Index history (a different DTCC host, files.dtcc.com; verified
# 2026-10-02): one workbook, sheet "Data 2005-2024", daily weighted average rates of GCF
# repo by collateral (MBS, Treasury, Agency) from 2005-01-03 to 2024-12-31. Frozen: the
# live CSV it was updated from now answers 404. DTCC's terms: personal, informational use
# only - never republished.
GCF_INDEX_URL = "https://files.dtcc.com/download/assets/GCF-Index-Graph.xlsx/c7e6be84ca1f11f090703666cfebbd61"


def fetch_gcf_index_workbook(*, sleep=time.sleep) -> bytes:
    """The GCF Repo Index history workbook (.xlsx bytes)."""
    request = urllib.request.Request(GCF_INDEX_URL, headers={"User-Agent": "Mozilla/5.0"})
    last = None
    for attempt in range(RETRIES + 1):
        if attempt:
            sleep(BACKOFF_S * 2 ** (attempt - 1))
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                content = resp.read()
            if content[:2] == b"PK":
                return content
            last = f"not an xlsx ({len(content)} bytes)"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise DtccError(f"GCF index workbook: {last} after {RETRIES + 1} attempts")
