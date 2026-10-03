"""CFTC public reporting (Socrata, publicreporting.cftc.gov) - NETWORK ONLY, no files.

Free, no key. Used for the Commitments of Traders TRADERS IN FINANCIAL FUTURES report
(``CFTC_TFF_REPORTS``: one dataset for futures only, one for futures + options combined),
verified 2026-10-02: 147 financial markets from 2006-06-13, ~47k rows per dataset, every
value a string. ``Last-Modified`` on any request moves when CFTC publishes (observed
Friday 15:30 New York), so a HEAD-like 1-row request says whether anything is new.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from email.utils import parsedate_to_datetime

import pandas as pd

BASE = "https://publicreporting.cftc.gov/resource"
PAGE = 50_000  # Socrata's max $limit
TIMEOUT_S = 120
RETRIES = 4
BACKOFF_S = 2.0


class CftcError(RuntimeError):
    pass


def _get(dataset: str, params: dict, *, sleep=time.sleep) -> tuple[list[dict], pd.Timestamp | None]:
    url = f"{BASE}/{dataset}.json?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "infra-data-pipeline"})
    last = None
    for attempt in range(RETRIES + 1):
        if attempt:
            sleep(BACKOFF_S * 2 ** (attempt - 1))
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                modified = resp.headers.get("Last-Modified")
                stamp = None if modified is None else \
                    pd.Timestamp(parsedate_to_datetime(modified)).tz_convert("UTC").tz_localize(None)
                return json.loads(resp.read()), stamp
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise CftcError(f"{dataset}: {last}")


def last_modified(dataset: str) -> pd.Timestamp | None:
    """When the dataset last changed (UTC, tz-naive), from a 1-row request."""
    return _get(dataset, {"$limit": 1})[1]


def fetch_tff(dataset: str, since=None) -> list[dict]:
    """Every row of a TFF dataset with a report date on/after ``since`` (all if None), as
    the API returns them, paged."""
    rows, offset = [], 0
    where = {} if since is None else {"$where": f"report_date_as_yyyy_mm_dd >= '{pd.Timestamp(since):%Y-%m-%d}'"}
    while True:
        page, _ = _get(dataset, {**where, "$order": "id", "$limit": PAGE, "$offset": offset})
        rows += page
        if len(page) < PAGE:
            return rows
        offset += PAGE
