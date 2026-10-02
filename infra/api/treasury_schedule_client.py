"""US Treasury's TENTATIVE AUCTION SCHEDULE - NETWORK ONLY, no files.

Published at each Quarterly Refunding (early Feb/May/Aug/Nov) as XML (verified 2026-10-01:
"Aug2026 Refunding Auction Calendar Official Ver 2", 2026-08-05 .. 2027-01-30, 213
auctions): per auction the term, type (BILL/NOTE/BOND), reopening, TIPS and FRN flags,
and the announcement, auction and settlement dates - ~4 months ahead, where Fiscal Data
(infra.api.fiscaldata_client) only lists the ~1 announced week. Dates only, no times;
revised between refundings ("Ver 2"), so it is re-read rather than cached.
"""
from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

URL = "https://home.treasury.gov/system/files/221/Tentative-Auction-Schedule.xml"
TIMEOUT_S = 60
COLUMNS = ["calendar", "published", "term", "security_type", "reopening", "tips", "frn",
           "announcement_date", "auction_date", "settlement_date"]


def fetch_xml() -> str:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", "ignore")


def parse(xml: str) -> pd.DataFrame:
    """One row per scheduled auction; ``published`` = the calendar's StartDate (the
    refunding announcement it was released with)."""
    root = ET.fromstring(xml.strip())
    name = (root.findtext("AuctionCalendarName") or "").strip()
    published = pd.Timestamp(root.findtext("StartDate"))
    rows = []
    for a in root.iter("AuctionCalendarDate"):
        t = lambda tag: (a.findtext(tag) or "").strip()  # noqa: E731
        rows.append((name, published, t("SecurityTermWeekYear"), t("SecurityType"), t("ReOpeningIndicator") == "Y",
                     t("TIPS") == "Y", t("FloatingRate") == "Y", pd.Timestamp(t("AnnouncementDate")),
                     pd.Timestamp(t("AuctionDate")), pd.Timestamp(t("SettlementDate"))))
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_schedule(*, fetch=fetch_xml) -> pd.DataFrame:
    return parse(fetch())
