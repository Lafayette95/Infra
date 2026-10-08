"""Montréal Exchange (TMX MX) historical derivatives data - NETWORK ONLY, no files.

``m-x.ca/en/trading/data/historical?symbol=<root>&from=YYYY-MM-DD&to=YYYY-MM-DD&dnld=1``
returns a CSV, one row per contract per day (verified 2026-10-08): settlement price, open / high /
low / last, closing bid / ask and sizes, volume, open interest, expiry; from 2009-01-02, up to the
day before. Free, no login; the site's terms license downloads for personal, non-commercial use,
and its robots.txt asks for ``Crawl-delay: 15`` - enforced here across every call of the process.
Canada's bond futures (CGB 10y, CGF 5y, CGZ 2y, LGB 30y) aren't on Databento.
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request

URL = "https://www.m-x.ca/en/trading/data/historical"
TIMEOUT_S = 120
CRAWL_DELAY_S = 15.0
_last = [0.0]


def fetch_historical(symbol: str, start, end) -> str:
    """The CSV for ``symbol`` (a root, e.g. ``CGB``) over ``[start, end]`` (dates)."""
    wait = CRAWL_DELAY_S - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    q = urllib.parse.urlencode({"symbol": symbol, "from": str(start)[:10], "to": str(end)[:10], "dnld": 1})
    req = urllib.request.Request(f"{URL}?{q}", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8-sig", "replace")
    finally:
        _last[0] = time.monotonic()
    if not body.lstrip().startswith('"Date"'):
        raise RuntimeError(f"MX historical {symbol}: not a CSV ({len(body)} bytes)")
    return body
