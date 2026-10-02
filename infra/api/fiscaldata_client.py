"""US Treasury Fiscal Data API (api.fiscaldata.treasury.gov) - NETWORK ONLY, no files.

Free, keyless. ``fetch_auctions``: the ``auctions_query`` dataset - every marketable
Treasury auction since 1979, one record per auction, 114 fields (terms, CUSIP, offering
amount, high yield, bid-to-cover, allotments by bidder class, competitive close time).
Verified 2026-10-01: an ANNOUNCED auction is listed before it is held, with every result
field the string ``"null"`` - so the same query serves history and the week ahead.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
TIMEOUT_S = 120
PAGE_SIZE = 10000
MAX_RETRIES = 4


def _get(url: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                        timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Fiscal Data {url}: {exc}") from None
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_auctions(start: pd.Timestamp, end: pd.Timestamp | None = None, *, get=_get) -> pd.DataFrame:
    """Every auction with ``auction_date`` in ``[start, end)`` (``end`` None = through the
    announced ones), all fields as the API's strings (``"null"`` = no value)."""
    flt = f"auction_date:gte:{pd.Timestamp(start):%Y-%m-%d}"
    if end is not None:
        flt += f",auction_date:lt:{pd.Timestamp(end):%Y-%m-%d}"
    rows, page = [], 1
    while True:
        query = urllib.parse.urlencode({"filter": flt, "sort": "auction_date", "page[size]": PAGE_SIZE,
                                        "page[number]": page})
        payload = get(f"{AUCTIONS_URL}?{query}")
        rows += payload.get("data", [])
        if page >= int(payload.get("meta", {}).get("total-pages", 1)):
            break
        page += 1
    return pd.DataFrame(rows)
