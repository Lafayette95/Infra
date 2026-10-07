"""Eurex's deliverable bonds and conversion factors - NETWORK ONLY, no files.

One free CSV for every Eurex fixed-income future (German, Italian, French, Swiss, EU ...):
``#Contract;ISIN;Coupon;Maturity;ConvFac`` for the currently listed contracts, linked from
the notified-bonds page (its URL carries a hash that changes, so the page is read first).
Verified 2026-10-07.
"""
from __future__ import annotations

import re
import urllib.request

PAGE = "https://www.eurex.com/ex-en/data/clearing-files/notified-deliverable-bonds-conversion-factors"
BASE = "https://www.eurex.com"
TIMEOUT_S = 60
_LINK = re.compile(r'href="(/resource/blob/[^"]+/deliverable-bonds-and-conversion-factors\.csv)"')


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read()


def fetch_deliverables_csv() -> bytes:
    page = _get(PAGE).decode("utf-8", "replace")
    m = _LINK.search(page)
    if not m:
        raise RuntimeError("no deliverable-bonds CSV link on Eurex's notified-bonds page")
    body = _get(BASE + m.group(1))
    if not body.lstrip(b"\xef\xbb\xbf").startswith(b"#Contract"):
        raise RuntimeError(f"Eurex deliverables file has an unexpected layout: {body[:60]!r}")
    return body
