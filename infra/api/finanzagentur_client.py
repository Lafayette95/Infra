"""German Finance Agency (Finanzagentur) issuance history - NETWORK ONLY, no files.

One free workbook, ``emissionshistorie_en.xlsx``: every Federal auction / syndication /
tap since 1999 (Bubills, Schatz, Bobls, Bunds, Green, inflation-linked) with ISIN, coupon,
maturity, maturity segment, volume, new issue vs reopening, bids, prices and yields
(verified 2026-10-07; robots.txt allows it). The German counterpart of Fiscal Data's
auctions (``fiscaldata_client``).
"""
from __future__ import annotations

import urllib.request

URL = ("https://www.deutsche-finanzagentur.de/fileadmin/user_upload/Institutionelle-investoren/"
       "auktionen/emissionshistorie_en.xlsx")
TIMEOUT_S = 120


def fetch_issuance_history() -> bytes:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = resp.read()
    if body[:2] != b"PK":
        raise RuntimeError(f"issuance history is not an xlsx ({len(body)} bytes)")
    return body
