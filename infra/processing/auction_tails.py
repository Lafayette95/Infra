"""Treasury auction TAILS from archived auction recaps. Pure (no I/O).

tail (bp) = official high yield - when-issued (WI) yield at the close, x100. Positive =
a tail (the auction cleared cheaper than the market), negative = a stop-through.

Sources (``SOURCES``), verified 2026-10-01 on archived articles:

* ZeroHedge recaps: the numbers are in prose, e.g. "Stopping at a high yield of 4.917%
  ... the When Issued 4.907%", "the high yield which at 1.553 ... stopped through the When
  Issued 1.559% by 0.6bps", "while the When Issued traded at 1.888% the auction tailed by
  5.2bps". Many also STATE the tail - a second, independent check.
* ForexLive one-liners: both numbers in the TITLE, "US sells 5-year notes at 1.988% vs
  1.980 WI bid".

Every extraction is checked twice (``check``): its reported high yield must equal the
official one (Fiscal Data) to the published 3 decimals - which also IDENTIFIES the
auction (tenor + high yield is unique within a year: ``match``) - and, where the article
states a tail, it must equal high yield - WI. Only checked rows count (``best``).
"""
from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

SOURCES = ("zerohedge", "forexlive")  # best() priority order
COLUMNS = ["timestamp", "cusip", "source", "url", "capture", "tenor_years", "high_yield_reported", "wi_yield",
           "stated_tail_bp", "high_yield_official", "tail_bp", "check_high_yield", "check_stated_tail", "passed"]
KEYS = ["timestamp", "cusip", "source"]
PARSER_VERSION = 3  # bump when extraction changes: the harvest then retries every non-passed article
_NUM = r"(-?\d+\.\d+)"
_FILL = r"(?:\s+[a-z][a-z'-]*){0,4}?"  # a few words, no digits ("which was trading", "a disappointing")
# Phrasings seen 2011-2026 (ZeroHedge): "high yield of 4.917%", "the high yield which at 1.553", "Stopping at a
# high yield of", "The 30 year priced at 3.75%", "the closing yield of 0.90%", "auction pricing a disappointing 2.27%"
_HIGH = [re.compile(r"high yield(?:\s+which)?(?:\s+(?:of|at|was|is))*\s*" + _NUM, re.I),
         re.compile(r"stopp(?:ed|ing) (?:out )?at (?:a high yield of )?" + _NUM, re.I),
         re.compile(r"(?:closing|stop[- ]?out) yield(?:\s+(?:of|was|at))*\s*" + _NUM, re.I),
         re.compile(r"(?:priced|pricing|prices|printed)(?:\s+at)?(?:\s+[a-z][a-z'-]*){0,6}?\s+" + _NUM + r"\s*%", re.I)]
# "When Issued 4.907%", "the When Issued traded at 1.888%", "When Issued which was trading at 3.64%",
# "When Issued of 0.905%", "WI trading at 1.99", "1.980 WI bid"
_WI = [re.compile(r"when[- ]issued" + _FILL + r"\s+" + _NUM, re.I),
       re.compile(r"\bWI\b" + _FILL + r"\s+" + _NUM, re.I),
       re.compile(_NUM + r"%?\s*(?:WI|when[- ]issued)\b", re.I)]
_FXL_TITLE = re.compile(r"at\s+" + _NUM + r"%?\s+vs\.?\s+" + _NUM + r"%?\s+WI", re.I)
# within one sentence: any character but a full stop - a decimal point ("1.559%") is fine
_SENT = r"(?:[^.]|\.\d){0,120}?"
_STATED = [(re.compile(r"(tail(?:ed|ing|s)?|stop(?:ped|ping|s)?[- ]through)" + _SENT + r"\bby\s+(\d+(?:\.\d+)?)\s*bps?\b",
                       re.I), None),
           (re.compile(r"(\d+(?:\.\d+)?)\s*bps?\s+(tail|stop[- ]through)", re.I), None),
           (re.compile(r"(tail|stop[- ]through)\s+(?:was|of)" + _FILL + r"\s+(\d+(?:\.\d+)?)\s*bps?", re.I), None)]
_SCREWS = re.compile(r"on the screws", re.I)
NON_US = re.compile(r"ital(y|ian)|german|bund|spain|spanish|japan|jgb|\bgilts?\b|\buk\b|britain|france|french|"
                    r"\boat\b|greek|greece|portug|irish|ireland|belgi|dutch|austria|china|chinese|euro", re.I)
MAX_ABS_TAIL_BP = 15.0  # the record tail is ~11bp (30Y, Aug 2011): beyond this a number was mis-read
_TENOR = re.compile(r"(?<![\d.])(2|3|5|7|10|20|30)(?:-|\s)?(?:y|yr|year)s?(?![a-z])"
                    r"|\b(two|three|five|seven|ten|twenty|thirty)[- ]year", re.I)
_WORDS = {"two": 2, "three": 3, "five": 5, "seven": 7, "ten": 10, "twenty": 20, "thirty": 30}


def page_text(page: str) -> tuple[str, str]:
    """(title, body text) of an article page."""
    t = re.search(r"<title>(.*?)</title>", page, re.S | re.I)
    title = html.unescape(t.group(1)).strip() if t else ""
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S | re.I)
    body = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))
    return title, body


def tenor(text: str) -> int | None:
    """Tenor in years from a title or URL slug; None for TIPS/FRN/bills or no tenor."""
    t = text.lower().replace("_", "-")
    if re.search(r"\btips\b|\bfrn\b|floating|\bbills?\b|week", t):
        return None
    m = _TENOR.search(t)
    return None if not m else (int(m.group(1)) if m.group(1) else _WORDS[m.group(2).lower()])


def _first(patterns, text):
    for p in patterns:
        m = p.search(text)
        if m:
            return float(next(g for g in m.groups() if g is not None))
    return np.nan


def _decimals(x: str) -> int:
    return len(x.split(".")[1]) if "." in x else 0


def extract(source: str, title: str, body: str) -> dict:
    """The numbers an article reports: ``high_yield_reported``, ``wi_yield`` (percent),
    ``stated_tail_bp`` (+ tail / - stop-through; 0 "on the screws") - NaN where absent -
    and ``hy_decimals`` (the reported high yield's precision, for the check)."""
    hy = wi = stated = np.nan
    hy_dec = 3
    if source == "forexlive":
        m = _FXL_TITLE.search(title)
        if m:
            hy, wi, hy_dec = float(m.group(1)), float(m.group(2)), _decimals(m.group(1))
    if not np.isfinite(hy):
        for p in _HIGH:  # the body first; old recaps put it in the title ("Prices At Record Low Yield Of 0.222%")
            m = p.search(body) or p.search(title)
            if m:
                hy, hy_dec = float(m.group(1)), _decimals(m.group(1))
                break
    if not np.isfinite(wi):
        wi = _first(_WI, body if source == "zerohedge" else title + " " + body)
    for p, _ in _STATED:
        m = p.search(body)
        if m:
            word, num = (m.group(1), m.group(2)) if not m.group(1)[0].isdigit() else (m.group(2), m.group(1))
            stated = float(num) * (1.0 if word.lower().startswith("tail") else -1.0)
            break
    if not np.isfinite(stated) and _SCREWS.search(body):
        stated = 0.0
    return {"high_yield_reported": hy, "wi_yield": wi, "stated_tail_bp": stated, "hy_decimals": hy_dec}


def slug_date(url: str) -> pd.Timestamp | None:
    """A publication date in an article URL: ForexLive's slug suffix ``...-20170125``, or
    a date in the path, ZeroHedge's old ``/news/2012-10-23/...`` (first archived captures
    can be YEARS after publication, so the capture date is no substitute)."""
    path = url.split("?")[0]
    for rx in (r"-(20\d{2})(\d{2})(\d{2})/?$", r"/(20\d{2})-(\d{2})-(\d{2})/", r"/(20\d{2})/(\d{2})/(\d{2})/"):
        m = re.search(rx, path)
        if m:
            try:
                return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
    return None


def tolerance(decimals: int) -> float:
    """How far an official 3-decimal high yield can be from one REPORTED with ``decimals``."""
    return max(5e-4, 0.5 * 10.0 ** -decimals) + 1e-9


def match(tenor_years: int | None, high_yield: float, near: pd.Timestamp, auctions: pd.DataFrame,
          *, window_days: int = 400, decimals: int = 3) -> pd.Series | None:
    """The auction of that tenor whose official high yield equals the reported one (to the
    published 3 decimals) closest to ``near``, within ``window_days``; None if none or ambiguous."""
    if tenor_years is None or not np.isfinite(high_yield):
        return None
    c = auctions[(auctions["tenor_years"] == tenor_years)
                 & ((auctions["high_yield"] - high_yield).abs() <= tolerance(decimals))
                 & ((auctions["timestamp"] - near).abs() <= pd.Timedelta(days=window_days))]
    if c.empty:
        return None
    c = c.assign(_d=(c["timestamp"] - near).abs()).sort_values("_d")
    return c.iloc[0]


def check(row: dict) -> dict:
    """Fill the official comparison, the tail, and the two checks."""
    dec = int(row.get("hy_decimals", 3))
    hy_ok = (np.isfinite(row["high_yield_official"])
             and abs(row["high_yield_reported"] - row["high_yield_official"]) <= tolerance(dec))
    tail = (row["high_yield_official"] - row["wi_yield"]) * 100.0 if np.isfinite(row["wi_yield"]) else np.nan
    # a stated tail is given to 0.1bp but computed from a WI that may be rounded too
    stated_ok = (not np.isfinite(row["stated_tail_bp"])) or (np.isfinite(tail) and abs(tail - row["stated_tail_bp"]) <= 0.55)
    plausible = np.isfinite(tail) and abs(tail) <= MAX_ABS_TAIL_BP
    out = {k: v for k, v in row.items() if k != "hy_decimals"}
    return {**out, "tail_bp": round(tail, 4) if np.isfinite(tail) else np.nan, "check_high_yield": bool(hy_ok),
            "check_stated_tail": bool(stated_ok), "passed": bool(hy_ok and stated_ok and plausible)}


def best(tails: pd.DataFrame, priority=SOURCES) -> pd.DataFrame:
    """One tail per auction: the highest-priority source among rows that PASSED, plus how
    many sources passed and their spread (a disagreement worth a look when > 0.1bp)."""
    ok = tails[tails["passed"]].copy()
    if ok.empty:
        return ok.assign(n_sources=pd.Series(dtype=int), spread_bp=pd.Series(dtype=float))
    ok["_rank"] = ok["source"].map({s: i for i, s in enumerate(priority)}).fillna(len(priority))
    stats = ok.groupby(["timestamp", "cusip"])["tail_bp"].agg(n_sources="size", spread_bp=lambda s: s.max() - s.min())
    top = ok.sort_values("_rank").groupby(["timestamp", "cusip"], as_index=False).head(1).drop(columns="_rank")
    return top.merge(stats.reset_index(), on=["timestamp", "cusip"]).sort_values("timestamp").reset_index(drop=True)
