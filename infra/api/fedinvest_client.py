"""Treasury FedInvest price pages (treasurydirect.gov/GA-FI/FedInvest) - NETWORK ONLY, no files.

Free, no key. Facts this module relies on (verified 2026-10-01):
* one page per day lists EVERY marketable Treasury (~460 on 2026-10-01: notes, bonds,
  bills, TIPS, FRNs) with BUY / SELL / END OF DAY clean prices; history from 2008-09-02;
* END OF DAY matches the 3:30pm CMT curve (on-the-run yields within ~0.5bp, 19 days
  2016-2026); BUY/SELL are the Treasury's ~1pm prices. The END OF DAY column is posted
  LATER than the page itself: at 22:27 New York on the day it was still all zeros;
* a weekend or holiday returns a valid page with the date heading and no rows;
* the page is a form: GET it for a session cookie and a ``_csrf`` token, then POST
  ``priceDate`` (YYYY-MM-DD) and follow the redirect to ``securityPriceDetail``.
"""
from __future__ import annotations

import html
import http.cookiejar
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

FORM_URL = "https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate"
TIMEOUT_S = 60
RETRIES = 4
BACKOFF_S = 3.0
COLUMNS = ["cusip", "security_type", "rate", "maturity", "call_date", "buy", "sell", "end_of_day"]
_CSRF = re.compile(r'name="_csrf" value="([^"]+)"')
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")


class FedInvestError(RuntimeError):
    pass


def parse_page(text: str) -> pd.DataFrame:
    """The price table of one page, as text (COLUMNS). Empty for a day without prices."""
    rows = []
    for r in _ROW.findall(text):
        cells = [html.unescape(_TAG.sub("", c)).strip() for c in _CELL.findall(r)]
        if len(cells) == len(COLUMNS):
            rows.append(cells)
    return pd.DataFrame(rows, columns=COLUMNS)


def _page(day) -> str:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    with opener.open(FORM_URL, timeout=TIMEOUT_S) as resp:
        token = _CSRF.search(resp.read().decode("utf-8", "replace"))
    if token is None:
        raise FedInvestError("no _csrf token on the FedInvest form")
    body = urllib.parse.urlencode({"priceDate": f"{pd.Timestamp(day):%Y-%m-%d}", "submit": "Show Prices",
                                   "_csrf": token.group(1)}).encode()
    with opener.open(urllib.request.Request(FORM_URL, data=body), timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "replace")


def fetch_prices(day, *, sleep=time.sleep) -> pd.DataFrame:
    """One day's price table (COLUMNS, text). Retries transient failures with backoff."""
    last = None
    for attempt in range(RETRIES):
        if attempt:
            sleep(BACKOFF_S * 2 ** (attempt - 1))
        try:
            return parse_page(_page(day))
        except (urllib.error.URLError, TimeoutError, ConnectionError, FedInvestError) as exc:
            last = exc
    raise FedInvestError(f"FedInvest {pd.Timestamp(day).date()}: {last}")
