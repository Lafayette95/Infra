"""German Federal issuance PLANS - pure parsing (no network, no files): the Finanzagentur's
issuance outlook workbooks and its live "Upcoming Issues" table, into one row per planned
auction.

Three layers, each revising the one before (root CLAUDE.md 17):

* the ANNUAL outlook (published mid-December for the next year), "projected";
* the QUARTERLY updates (end of each quarter, for the rest of the year: ``Update_Q2`` ..
  ``Update_Q4``), which replace the annual plan from that quarter on;
* the live table on the issuance-calendar page: the current schedule of auctions still to
  come, including intra-quarter adjustments (a placeholder like "Bund 30 Y" gets its ISIN,
  a volume changes). It only lists auctions not yet held, so a daily snapshot records each
  change as it is published.

Plan rows: ``auction_date, security, term, issue_kind (new_issue / reopening / None),
volume_m, maturity_date, isin, coupon, interest_start, first_coupon``. ``isin`` is None for a
placeholder (a multi-line "15/20/30 years" reopening, a "Green issue") - decided later.
"""
from __future__ import annotations

import datetime
import html as _html
import io
import re

import numpy as np
import pandas as pd

PLAN_COLUMNS = ["auction_date", "security", "term", "issue_kind", "volume_m", "maturity_date", "isin", "coupon",
                "interest_start", "first_coupon"]
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}\d")


def _kind(v) -> str | None:
    s = str(v).strip().lower()
    if s.startswith("new") or s == "n":
        return "new_issue"
    if s.startswith("reop") or s == "r":
        return "reopening"
    return None


def _col(cols: list[str], *keys: str, exclude: tuple[str, ...] = ()) -> int | None:
    for i, c in enumerate(cols):
        if all(k in c for k in keys) and not any(x in c for x in exclude):
            return i
    return None


def parse_outlook(xlsx: bytes) -> tuple[pd.DataFrame, str | None]:
    """One plan row per line of an issuance-outlook workbook, plus its "As of <month year>"
    label if the sheet states one (2025 on). Columns are found by NAME: the layout moved
    between years (2024 has separate term / remaining-term columns, an upper-case header and
    a "VOLUME IN € BN" header over values in € millions). Coupons in PERCENT."""
    raw = pd.read_excel(io.BytesIO(xlsx), header=None)
    as_of = next((str(v).strip() for v in raw[0].dropna() if str(v).strip().lower().startswith("as of")), None)
    hdr = next(i for i in raw.index if {"date", "isin"} <= {str(v).strip().lower() for v in raw.loc[i]})
    cols = [str(v).strip().lower() for v in raw.loc[hdr]]
    body = raw.loc[hdr + 1:]
    is_day = body.iloc[:, _col(cols, "date")].map(lambda v: isinstance(v, (pd.Timestamp, datetime.datetime)))
    body = body[is_day]
    get = lambda i: body.iloc[:, i] if i is not None else pd.Series(np.nan, index=body.index)
    ti = _col(cols, "term")                    # "term to maturity" (2024) / "term to maturity/ remaining term*"
    ri = _col(cols, "remaining")               # 2024 only: a separate "remaining term" column (Bubills)
    term = get(ti)
    if ri is not None and ri != ti:
        term = term.where(term.notna(), get(ri))
    vol = pd.to_numeric(get(_col(cols, "volume")), errors="coerce")
    if vol.median() < 100:                     # stated in € bn
        vol = vol * 1000.0
    cpn = pd.to_numeric(get(_col(cols, "coupon", exclude=("first",))), errors="coerce")
    cpn = cpn.where(cpn > 0.5, cpn * 100.0)    # the file gives 0.029 for 2.90%
    isin = get(_col(cols, "isin")).map(lambda v: v.strip() if isinstance(v, str) and _ISIN.fullmatch(v.strip()) else None)
    out = pd.DataFrame({
        "auction_date": pd.to_datetime(get(_col(cols, "date"))).dt.normalize(),
        "security": get(_col(cols, "security")).astype(str).str.strip(),
        "term": term.map(lambda v: None if pd.isna(v) else str(v).strip()),
        "issue_kind": get(_col(cols, "type")).map(_kind),
        "volume_m": vol,
        "maturity_date": pd.to_datetime(get(_col(cols, "maturity")), errors="coerce"),
        "isin": isin,
        "coupon": cpn,
        "interest_start": pd.to_datetime(get(_col(cols, "start of interest")), errors="coerce"),
        "first_coupon": pd.to_datetime(get(_col(cols, "first coupon")), errors="coerce"),
    })
    return out[PLAN_COLUMNS].reset_index(drop=True), as_of


def _cell(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_upcoming(page: str) -> pd.DataFrame:
    """The issuance-calendar page's "Upcoming Issues" tables (capital and money market) as
    plan rows (``term`` = the label's tenor, e.g. "30 Y"; volume from "5.0 € bn")."""
    i = page.find("Upcoming Issues")
    if i < 0:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", page[i:], flags=re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)
        if len(tds) != 5:
            continue
        date, issuance, vol, cpn, mat = (_cell(t) for t in tds)
        d = pd.to_datetime(date, format="%d.%m.%Y", errors="coerce")
        if pd.isna(d):
            continue
        m = re.match(r"(.*?)\s*\(\s*([NR])\s*\)\s*(\S+)?", issuance)
        label, kind, rest = (m.group(1), m.group(2), m.group(3)) if m else (issuance, None, None)
        sec = label.split()[0] if label else None
        term = " ".join(label.split()[1:]) or None
        v = re.match(r"([\d.,]+)\s*€\s*(bn|mn)", vol)
        vol_m = (float(v.group(1).replace(",", "")) * (1000.0 if v.group(2) == "bn" else 1.0)) if v else np.nan
        c = re.match(r"([\d.,]+)\s*%", cpn)
        rows.append({"auction_date": d, "security": sec, "term": term, "issue_kind": _kind(kind or ""),
                     "volume_m": vol_m, "maturity_date": pd.to_datetime(mat, format="%d.%m.%Y", errors="coerce"),
                     "isin": rest if rest and _ISIN.fullmatch(rest) else None,
                     "coupon": float(c.group(1)) if c else np.nan, "interest_start": pd.NaT, "first_coupon": pd.NaT})
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def line_key(df: pd.DataFrame) -> pd.Series:
    """A plan line's identity across vintages: auction day + security + ISIN (or, for a
    placeholder, its term label). Two lines the same day on the same security are distinct
    ISINs (a 10y and a 30y Bund) or distinct placeholders."""
    ident = df["isin"].where(df["isin"].notna(), "term:" + df["term"].fillna("?").astype(str))
    return df["auction_date"].dt.strftime("%Y-%m-%d") + "|" + df["security"].astype(str) + "|" + ident


# ---------------------------------------------------------------- events
CONVENTIONAL_TENORS = (2, 5, 7, 10, 15, 20, 30)


def _years(label) -> list[int]:
    """Whole-year tenors in a label: "10 years" -> [10], "15/20/30 years" -> [15, 20, 30],
    "30 Y" -> [30]; months -> []."""
    s = str(label or "")
    if "month" in s.lower():
        return []
    return [int(x) for x in re.findall(r"\d+", s)]


def security_class(security) -> str:
    """``bubill`` / ``schatz`` / ``bobl`` / ``bund`` / ``green`` / ``ilb``: the plan's labels
    vary ("Green issue", "Bund/g" and "Bobl/g" for Green lines on the 2025 calendar page)."""
    s = str(security).lower()
    if "green" in s or s.endswith("/g"):
        return "green"
    if "ilb" in s or "inflation" in s or s.endswith("/i"):
        return "ilb"
    for c in ("bubill", "schatz", "bobl", "bund"):
        if s.startswith(c):
            return c
    return s


def event_id(security: str, term, isin, segment_of: dict) -> str:
    """The registry event of a plan line or a held auction (``infra.reference.events``,
    ``DE_AUCTION_*``): Bubills, Green and inflation-linked securities have one event each;
    conventional Schatz / Bobls / Bunds one per tenor - the ISIN's maturity segment in the
    issuance history when known (``segment_of``: ISIN -> "10 Y"), else the plan's term
    label; a Bund line with several tenors and no ISIN yet ("15/20/30 years", decided
    later) is ``DE_AUCTION_LONG``."""
    cls = security_class(security)
    if cls in ("bubill", "green", "ilb"):
        return f"DE_AUCTION_{cls.upper()}"
    if cls == "schatz":
        return "DE_AUCTION_2Y"
    if cls == "bobl":
        return "DE_AUCTION_5Y"
    seg = segment_of.get(isin) if isin else None
    years = _years(seg) if seg else _years(term)
    if len(years) == 1 and years[0] in CONVENTIONAL_TENORS:
        return f"DE_AUCTION_{years[0]}Y"
    return "DE_AUCTION_LONG"
