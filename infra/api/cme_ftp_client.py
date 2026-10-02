"""CME Group public FTP (ftp.cmegroup.com) - NETWORK ONLY, no files.

Free, anonymous. Reachable even when the cmegroup.com WEBSITE blocks an address (it
blocked ours on 2026-10-01 as "suspected web scraping"; the FTP server didn't).
``settle/TCF/``: one ``TCF_YYYYMMDD.csv`` per business day (~06:30 Chicago, plus some
Saturdays; an ``.xlsx`` twin ~19:00) from 2023-12-09 - every deliverable CUSIP and its
invoice conversion factor for the next three contract months of each Treasury future.
Columns ``Exch, Period (YYYYMM), PFCode, CUSIP, Invoice_Conversion_Factor, Create_Time``.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.request

TCF_URL = "ftp://ftp.cmegroup.com/settle/TCF/"
TIMEOUT_S = 60
RETRIES = 4
_NAME = re.compile(r"(TCF_\d{8}\.csv)")


class CmeFtpError(RuntimeError):
    pass


def _get(url: str, sleep=time.sleep) -> bytes:
    last = None
    for attempt in range(RETRIES):
        if attempt:
            sleep(2.0 * 2 ** (attempt - 1))
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
    raise CmeFtpError(f"{url}: {last}")


def list_tcf_files() -> list[str]:
    """Every daily CSV currently on the server (names, sorted)."""
    return sorted(set(_NAME.findall(_get(TCF_URL).decode("utf-8", "replace"))))


def fetch_tcf_file(name: str) -> bytes:
    return _get(TCF_URL + name)
