"""Tradeweb Market InSite: FTSE-Tradeweb gilt and EuroGov closing prices - NETWORK ONLY, no
files (root CLAUDE.md 18).

``reports.tradeweb.com/closing-prices/gilts/``, behind the user's own free InSite login
(``TRADEWEB_USER`` / ``TRADEWEB_PASSWORD`` in ``.env``; registered 2026-10-08). Terms (the site's
Terms of Use and the export dialog): personal, non-professional, non-commercial use, no
redistribution. A day's prices are free from 12:00 London the next day.

The page is ASP.NET WebForms with Telerik controls (verified 2026-10-08): dates are read from each
date picker's ``dateInput_ClientState`` JSON, the ISIN from ``SymbolTextBox``; a search posts
``SubmitButton``, an export then posts ``ExportButton`` from the search's own page and returns a
CSV. An export of ALL securities always covers the last 5 working days (whatever dates are asked);
a single ISIN exports its whole range (2017-2026 in one request failed - chunk it). The results
grid pages 30 rows at a time through numbered postback links.

Polite by construction: one session, at least ``MIN_INTERVAL_S`` between requests.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import time

import pandas as pd

BASE = "https://reports.tradeweb.com"
PRICES = BASE + "/closing-prices/gilts/"
LOGIN = BASE + "/account/login/?ReturnUrl=%2fclosing-prices%2fgilts%2f"
P = "ctl00$ctl00$MainContent$MainContent$"
I = "ctl00_ctl00_MainContent_MainContent_"
TIMEOUT_S = 180
MIN_INTERVAL_S = 2.0
SECURITY_TYPES = ("all", "Conventional", "Index-linked", "Strips", "Bills")
CP_TYPES = ("all", "gilts", "iosco")       # all / gilts only / EuroGov only


def has_credentials() -> bool:
    from dotenv import dotenv_values
    env = {**dotenv_values(".env"), **os.environ}
    return bool(env.get("TRADEWEB_USER")) and bool(env.get("TRADEWEB_PASSWORD"))


def _form(page: str) -> dict:
    f = {}
    for m in re.finditer(r"<input([^>]+)>", page):
        a = m.group(1)
        n, v = re.search(r'name="([^"]+)"', a), re.search(r'value="([^"]*)"', a)
        if n and re.search(r'type="(text|hidden)"', a):
            f[n.group(1)] = _html.unescape(v.group(1)) if v else ""
    for m in re.finditer(r'<select[^>]+name="([^"]+)"[^>]*>(.*?)</select>', page, flags=re.S):
        sel = re.search(r'<option[^>]*selected[^>]*value="([^"]*)"', m.group(2)) or re.search(r'<option[^>]*value="([^"]*)"', m.group(2))
        f[m.group(1)] = sel.group(1) if sel else ""
    return f


def _text(page: str) -> str:
    t = re.sub(r"<(script|style).*?</\1>", " ", page, flags=re.S)
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", t)))


class InSite:
    """One logged-in session."""

    def __init__(self, user: str | None = None, password: str | None = None):
        import requests
        from dotenv import dotenv_values
        env = {**dotenv_values(".env"), **os.environ}
        self.user = user or env.get("TRADEWEB_USER") or ""
        self.password = password or env.get("TRADEWEB_PASSWORD") or ""
        if not (self.user and self.password):
            raise RuntimeError("TRADEWEB_USER / TRADEWEB_PASSWORD not set in .env")
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0"
        self._last = 0.0
        self._logged_in = False

    def _pace(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _request(self, method: str, url: str, **kw):
        for attempt in range(3):
            self._pace()
            try:
                r = self.s.request(method, url, timeout=TIMEOUT_S, **kw)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))
                continue
            if "/account/login/" in r.url and url != LOGIN:      # session expired
                self._logged_in = False
                self.login()
                continue
            return r
        raise RuntimeError(f"InSite: {url} kept redirecting to the login page")

    def login(self) -> None:
        self._pace()
        r = self.s.get(LOGIN, timeout=TIMEOUT_S)
        f = _form(r.text)
        f["ctl00$MainContent$LoginUser$UserName"] = self.user
        f["ctl00$MainContent$LoginUser$Password"] = self.password
        f["ctl00$MainContent$LoginUser$LoginButton"] = "Log In"
        self._pace()
        r = self.s.post(r.url, data=f, timeout=TIMEOUT_S)
        if "/account/login/" in r.url:
            raise RuntimeError("InSite login failed (check TRADEWEB_USER / TRADEWEB_PASSWORD)")
        self._logged_in = True

    def _filters(self, f: dict, start, end, isin: str, cp_type: str, security_type: str) -> dict:
        for side, d in (("From", pd.Timestamp(start)), ("To", pd.Timestamp(end))):
            tag, txt = d.strftime("%Y-%m-%d-00-00-00"), f"{d.month}/{d.day}/{d.year}"
            f[P + f"{side}DatePicker"] = d.strftime("%Y-%m-%d")
            f[P + f"{side}DatePicker$dateInput"] = txt
            f[I + f"{side}DatePicker_dateInput_ClientState"] = json.dumps(
                {"enabled": True, "emptyMessage": "", "validationText": tag, "valueAsString": tag,
                 "minDateStr": "1980-01-01-00-00-00", "maxDateStr": "2099-12-31-00-00-00", "lastSetTextBoxValue": txt})
        f[P + "SymbolTextBox"] = isin
        f[P + "CPTypeList"] = cp_type
        f[P + "SecurityTypesList"] = security_type
        return f

    def search(self, start, end, *, isin: str = "", cp_type: str = "all", security_type: str = "all") -> str:
        """The results page of a search (its grid holds the first 30 rows)."""
        if not self._logged_in:
            self.login()
        page = self._request("GET", PRICES).text
        f = self._filters(_form(page), start, end, isin, cp_type, security_type)
        f["__EVENTTARGET"], f["__EVENTARGUMENT"] = P + "SubmitButton", ""
        return self._request("POST", PRICES, data=f).text

    def export(self, start, end, *, isin: str = "", cp_type: str = "all", security_type: str = "all") -> str:
        """The CSV export of a search ("" when the site returned no file)."""
        page = self.search(start, end, isin=isin, cp_type=cp_type, security_type=security_type)
        f = self._filters(_form(page), start, end, isin, cp_type, security_type)
        f["__EVENTTARGET"], f["__EVENTARGUMENT"] = P + "ExportButton", ""
        r = self._request("POST", PRICES, data=f)
        if "csv" not in (r.headers.get("Content-Type") or ""):
            msg = _text(r.text)
            if "has experienced an error" in msg:
                raise RuntimeError("InSite export: the site returned an application error")
            return ""
        return r.content.decode("utf-8-sig")

    def grid_rows(self, start, end, *, cp_type: str = "all", security_type: str = "all",
                  max_pages: int = 10) -> list[list[str]]:
        """Every row of a search's results grid, page by page (up to ``max_pages``)."""
        page = self.search(start, end, cp_type=cp_type, security_type=security_type)
        m = re.search(r"(\d[\d,]*) items in (\d+) pages", _text(page))
        pages = int(m.group(2)) if m else 1
        rows = _grid(page)
        for k in range(2, min(pages, max_pages) + 1):
            f = _form(page)
            f["__EVENTTARGET"] = P + f"ClosePricesGrid$ctl00$ctl03$ctl01$ctl{5 + 2 * (k - 1):02d}"
            f["__EVENTARGUMENT"] = ""
            page = self._request("POST", PRICES, data=f).text
            rows += _grid(page)
        return rows


def _grid(page: str) -> list[list[str]]:
    rows = re.findall(r'<tr[^>]*class="rg(?:Alt)?Row"[^>]*>(.*?)</tr>', page, flags=re.S)
    return [[re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
             for c in re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S)][1:] for r in rows]
