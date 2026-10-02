"""National Association of Realtors - NETWORK ONLY, no files.

``fetch_next_existing_home_sales``: NAR states the NEXT Existing-Home Sales release on
its statistics page, one sentence (verified 2026-10-01): "Existing-Home Sales for
September 2026 will be released on Tuesday, October 13, 2026 at 10:00 a.m. Eastern."
NAR follows no fixed weekday rule (best rule 32% on 17 years of release days), so this is
the only forward date there is - one release ahead, re-read daily.
"""
from __future__ import annotations

import html
import re
import urllib.request

import pandas as pd

URL = "https://www.nar.realtor/research-and-statistics/housing-statistics/existing-home-sales"
TIMEOUT_S = 60
_NEXT = re.compile(r"Existing-Home Sales for (\w+ \d{4}) will be released on \w+, (\w+ \d{1,2}, \d{4}) at "
                   r"(\d{1,2}):(\d{2}) ([ap])\.?m\.? Eastern", re.I)


def fetch_page() -> str:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "ignore")


def parse_next_release(page: str) -> dict | None:
    """``{"period": month start, "day": release day, "time_et": "HH:MM"}`` or None."""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page)))
    m = _NEXT.search(text)
    if not m:
        return None
    hour = int(m.group(3)) % 12 + (12 if m.group(5).lower() == "p" else 0)
    return {"period": pd.Timestamp(m.group(1)), "day": pd.Timestamp(m.group(2)), "time_et": f"{hour:02d}:{m.group(4)}"}


def fetch_next_existing_home_sales(*, fetch=fetch_page) -> dict | None:
    return parse_next_release(fetch())
