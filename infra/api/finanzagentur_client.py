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


# ---------------------------------------------------------------- issuance plans
# The issuance-calendar page links the current year's ANNUAL outlook and its QUARTERLY updates
# as xlsx, and carries the live "Upcoming Issues" table (verified 2026-10-07; every 2024-2026
# file is still on the server, under names whose case varies by year - see OUTLOOK_CANDIDATES).
SITE = "https://www.deutsche-finanzagentur.de"
CALENDAR_URL = f"{SITE}/en/federal-securities/issuances/issuance-calendar"
OUTLOOK_DIR = f"{SITE}/fileadmin/user_upload/Institutionelle-investoren/auktionen/"


def _request(url: str, method: str = "GET"):
    return urllib.request.Request(url, method=method, headers={"User-Agent": "Mozilla/5.0"})


def fetch_calendar_page() -> str:
    with urllib.request.urlopen(_request(CALENDAR_URL), timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "replace")


def outlook_links(page: str) -> list[str]:
    """File names of the English issuance-outlook workbooks linked from the calendar page."""
    import re
    return sorted({m.split("/")[-1] for m in re.findall(r'href="([^"]+\.xlsx)"', page)
                   if "auktionen/" in m and "outlook" in m.lower() and m.lower().endswith("_en.xlsx")})


def outlook_candidates(year: int) -> list[str]:
    """Every name a year's annual / quarterly outlook has used (2024: lower case; 2026: the
    Q3 / Q4 updates capitalised) - for a history probe, not the daily run."""
    out = []
    for stem in (f"issuance_outlook_{year}", f"Issuance_outlook_{year}"):
        out.append(f"{stem}_en.xlsx")
        for q in ("Q2", "Q3", "Q4"):
            out += [f"{stem}_update_{q}_en.xlsx", f"{stem}_Update_{q}_en.xlsx"]
    return out


def outlook_last_modified(name: str) -> str | None:
    """The file's ``Last-Modified`` header (None if absent); raises on 404."""
    with urllib.request.urlopen(_request(OUTLOOK_DIR + name, "HEAD"), timeout=TIMEOUT_S) as resp:
        return resp.headers.get("Last-Modified")


def fetch_outlook(name: str) -> tuple[bytes, str | None]:
    with urllib.request.urlopen(_request(OUTLOOK_DIR + name), timeout=TIMEOUT_S) as resp:
        body, lm = resp.read(), resp.headers.get("Last-Modified")
    if body[:2] != b"PK":
        raise RuntimeError(f"{name} is not an xlsx ({len(body)} bytes)")
    return body, lm
