"""Japanese Government Bond auctions - pure parsing (no network, no files): the Ministry of
Finance's results workbooks into auctions and a securities table, and its monthly auction
calendars and alteration notices into plan rows (root CLAUDE.md 17).

Auctions: one row per (auction day, security). A security's id is its type + original tenor
+ issue number - ``JGB10_384`` (10-year #384), ``ILB10_29`` (inflation-indexed), ``GX5_2``
(Climate Transition), ``FRN15_..``, ``DISC3_..`` - the MoF publishes no ISIN in these
files. T-bills are ``TB<months>_<issue>``; a Liquidity Enhancement Auction (reopening several
off-the-run issues at once) is ``LIQ`` (its issues aren't in the file). Amounts in BILLION yen
(the coupon workbook states 100 million yen).
"""
from __future__ import annotations

import html as _html
import io
import re

import numpy as np
import pandas as pd

AUCTION_COLUMNS = ["timestamp", "security_id", "kind", "tenor_months", "issue_number", "issue_date",
                   "maturity_date", "coupon", "offering_bn", "bids_bn", "accepted_bn", "avg_price", "avg_yield",
                   "low_price", "high_yield"]
SECURITY_COLUMNS = ["security_id", "kind", "tenor_months", "issue_number", "coupon", "maturity_date",
                    "first_auction", "issue_date"]
PLAN_COLUMNS = ["auction_date", "label", "kind", "tenor_months", "issue_number"]
# results workbook sheets -> (kind, original tenor in years)
SHEETS = {"40年債": ("JGB", 40), "30年債": ("JGB", 30), "20年債": ("JGB", 20), "15変動": ("FRN", 15),
          "10年債": ("JGB", 10), "GX10年債": ("GX", 10), "10年物価連動": ("ILB", 10), "5年債": ("JGB", 5),
          "GX5年債": ("GX", 5), "2年債": ("JGB", 2), "6年債": ("JGB", 6), "4年債": ("JGB", 4),
          "割3年": ("DISC", 3)}
_FIELDS = (("issue_number", ("issue number",)), ("timestamp", ("auction date",)), ("issue_date", ("issue date",)),
           ("maturity_date", ("maturity date",)), ("coupon", ("nominal coupon",)),
           ("offering_bn", ("offering amount",)), ("bids_bn", ("amounts of competitive bids",)),
           ("accepted_bn", ("amounts of bids accepted",)), ("avg_price", ("average price",)),
           ("avg_yield", ("yield at the average",)), ("low_price", ("lowest accepted price", "price at the highest")),
           ("high_yield", ("yield at the lowest", "highest accepted yield")))


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed").dt.normalize()


def _frame(raw: pd.DataFrame, hdr: int, unit_bn: float) -> pd.DataFrame:
    cols = [re.sub(r"\s+", " ", str(v)).strip().lower() for v in raw.loc[hdr]]
    body = raw.loc[hdr + 1:]
    out = {}
    for name, keys in _FIELDS:
        idx = next((i for i, c in enumerate(cols) if any(k in c for k in keys)), None)
        out[name] = body.iloc[:, idx] if idx is not None else pd.Series(np.nan, index=body.index)
    df = pd.DataFrame(out)
    df["timestamp"] = _dates(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    for c in ("issue_date", "maturity_date"):
        df[c] = _dates(df[c])
    for c in ("coupon", "offering_bn", "bids_bn", "accepted_bn", "avg_price", "avg_yield", "low_price", "high_yield"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("offering_bn", "bids_bn", "accepted_bn"):
        df[c] = df[c] * unit_bn
    return df


def parse_results(xls: bytes) -> pd.DataFrame:
    """Every coupon-bond auction in the JGB results workbook."""
    book = pd.read_excel(io.BytesIO(xls), sheet_name=None, header=None)
    frames = []
    for sheet, raw in book.items():
        if sheet.strip() not in SHEETS:
            continue
        kind, years = SHEETS[sheet.strip()]
        hdr = next(i for i in raw.index if str(raw.loc[i, 0]).strip().lower() == "issue number")
        df = _frame(raw, hdr, 0.1)                   # 100 million yen -> billion
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["issue_number"])
        frames.append(df.assign(kind=kind, tenor_months=12 * years,
                                security_id=f"{kind}{years}_" + df["issue_number"].astype(str)))
    return pd.concat(frames, ignore_index=True)[AUCTION_COLUMNS] if frames else pd.DataFrame(columns=AUCTION_COLUMNS)


def _months(label) -> int | None:
    m = re.search(r"(\d+)\s*-?\s*(month|year)", str(label), flags=re.I)
    if not m:
        return None
    return int(m.group(1)) * (12 if m.group(2).lower() == "year" else 1)


def parse_tbills(xls: bytes) -> pd.DataFrame:
    """Every T-bill auction (one sheet per fiscal year; amounts already in billion yen)."""
    book = pd.read_excel(io.BytesIO(xls), sheet_name=None, header=None)
    frames = []
    for raw in book.values():
        hdr = next((i for i in raw.index if str(raw.loc[i, 0]).strip().lower() == "issue number"), None)
        if hdr is None:
            continue
        months = raw.loc[hdr + 1:, 1].map(_months)
        df = _frame(raw, hdr, 1.0)
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["issue_number"])
        df["tenor_months"] = months.reindex(df.index)
        frames.append(df.assign(kind="TB", security_id="TB" + df["tenor_months"].astype("Int64").astype(str) + "_"
                                + df["issue_number"].astype(str)))
    return pd.concat(frames, ignore_index=True)[AUCTION_COLUMNS] if frames else pd.DataFrame(columns=AUCTION_COLUMNS)


def parse_liquidity(xls: bytes) -> pd.DataFrame:
    """Liquidity Enhancement Auctions: day, amounts (billion yen), yield SPREADS in the
    ``avg_yield`` / ``high_yield`` columns (bids are spreads to the reopened issues' yields)."""
    raw = pd.read_excel(io.BytesIO(xls), header=None)
    hdr = next(i for i in raw.index if str(raw.loc[i, 0]).strip().lower() == "auction date")
    cols = [re.sub(r"\s+", " ", str(v)).strip().lower() for v in raw.loc[hdr]]
    body = raw.loc[hdr + 1:]
    pick = lambda *keys: body.iloc[:, next(i for i, c in enumerate(cols) if all(k in c for k in keys))]  # noqa: E731
    df = pd.DataFrame({"timestamp": _dates(pick("auction date")), "issue_date": _dates(pick("issue date")),
                       "bids_bn": pd.to_numeric(pick("competitive"), errors="coerce"),
                       "accepted_bn": pd.to_numeric(pick("accepted"), errors="coerce"),
                       "high_yield": pd.to_numeric(pick("highest"), errors="coerce"),
                       "avg_yield": pd.to_numeric(pick("average"), errors="coerce")}).dropna(subset=["timestamp"])
    df = df.assign(security_id="LIQ", kind="LIQ", tenor_months=np.nan, issue_number=pd.NA, maturity_date=pd.NaT,
                   coupon=np.nan, offering_bn=np.nan, avg_price=np.nan, low_price=np.nan)
    return df[AUCTION_COLUMNS].reset_index(drop=True)


def securities(auctions: pd.DataFrame) -> pd.DataFrame:
    """One row per bond / bill from its FIRST auction (Liquidity Enhancement rows have no
    security of their own and are left out)."""
    a = auctions[auctions["kind"] != "LIQ"].sort_values("timestamp")
    if a.empty:
        return pd.DataFrame(columns=SECURITY_COLUMNS)
    first = a.groupby("security_id", as_index=False).first()
    return first.rename(columns={"timestamp": "first_auction"})[SECURITY_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------- calendars
_JA_NUM = str.maketrans("０１２３４５６７８９", "0123456789")


def _months_ja(label: str) -> int | None:
    m = re.search(r"(\d+)\s*(ヶ月|か月|カ月|ケ月|年)", label.translate(_JA_NUM))
    return None if not m else int(m.group(1)) * (12 if m.group(2) == "年" else 1)


def classify(label: str) -> tuple[str, int | None]:
    """(kind, original tenor in months) of a calendar line, English or Japanese."""
    s = str(label)
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", s):
        if "流動性供給" in s:
            return "LIQ", None
        if "国庫短期証券" in s:
            return "TB", _months_ja(s)
        if "物価連動" in s:
            return "ILB", 120
        if "クライメート" in s or "GX" in s:
            return "GX", _months_ja(s)
        if "変動" in s:
            return "FRN", _months_ja(s)
        return "JGB", _months_ja(s)
    if "liquidity enhancement" in s.lower():
        return "LIQ", None
    if "discount bill" in s.lower():
        return "TB", _months(s)
    if "inflation" in s.lower():
        return "ILB", 120
    if "climate" in s.lower() or "gx" in s.lower():
        return "GX", _months(s)
    if "floating" in s.lower():
        return "FRN", _months(s)
    return "JGB", _months(s)


def _cells(tr: str) -> list[str]:
    return [re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", c))).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S)]


def _rows(table_html: str, year: int) -> list[tuple[pd.Timestamp, str]]:
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S):
        c = _cells(tr)
        if len(c) < 2:
            continue
        text = c[0].replace(".", ". ").replace("  ", " ")
        if not re.search(r"\d{4}", text):              # "Apr.2" on alteration notices: add the
            text = f"{text}, {year}"                    # year BEFORE parsing (Feb. 29 has no 1900)
        d = pd.to_datetime(text, errors="coerce", format="mixed")
        if pd.isna(d):
            continue
        out.append((d.normalize(), c[1]))
    return out


def _plan(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["auction_date", "label"])
    if df.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    kinds = df["label"].map(classify)
    num = df["label"].str.extract(r"[\(（]第?(\d+)回?[\)）]\s*$")[0]
    return df.assign(kind=[k for k, _ in kinds], tenor_months=[t for _, t in kinds],
                     issue_number=pd.to_numeric(num, errors="coerce").astype("Int64"))[PLAN_COLUMNS]


def parse_month(page: str, code: str) -> pd.DataFrame:
    """The JGB auctions (table 1) of a month page: English ``<yymm>e`` or Japanese
    ``<yymm>`` ("12月3日（木）" dates)."""
    year = 2000 + int(code[:2])
    i = page.find("Auction Date")
    if i < 0:
        i = page.find("入札予定日")
    if i < 0:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    j = page.find("</table>", i)
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page[i:j], flags=re.S):
        c = _cells(tr)
        if len(c) < 2:
            continue
        m = re.match(r"(\d+)月(\d+)日", c[0].translate(_JA_NUM))
        if m:
            rows.append((pd.Timestamp(year=year, month=int(m.group(1)), day=int(m.group(2))), c[1]))
    return _plan(rows) if rows else _plan(_rows(page[i:j], year))


def parse_alteration(page: str, code: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.DataFrame, pd.DataFrame]:
    """An alteration notice ``<yymm>ae``: (notice day, the original calendar's announcement
    day, the calendar before, the calendar after)."""
    year = 2000 + int(code[:2])
    text = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", page)))
    m = re.search(r"([A-Z][a-z]+\.? \d{1,2}, 20\d\d) Ministry of Finance", text)
    notice = pd.to_datetime(m.group(1).replace(".", ""), errors="coerce") if m else None
    m = re.search(r"Announced on ([A-Z][a-z]+\.? ?\d{1,2}, 20\d\d)", text)
    announced = pd.to_datetime(m.group(1).replace(".", ""), errors="coerce") if m else None
    tables = [t for t in re.split(r"<th[^>]*>\s*Auction Date\s*</th>", page)[1:]]
    before = _plan(_rows(tables[0], year)) if len(tables) > 0 else pd.DataFrame(columns=PLAN_COLUMNS)
    after = _plan(_rows(tables[1], year)) if len(tables) > 1 else pd.DataFrame(columns=PLAN_COLUMNS)
    return notice, announced, before, after


def announced_by(code: str) -> pd.Timestamp:
    """When a month's calendar is known, conservatively: the END of the month three months
    before (observed: announced on the 24th-28th of that month - Feb 2024 on 2023-11-28,
    May 2024 on 2024-02-27, Apr / May / Jun 2026 on 2026-01-27 / 02-24 / 03-26)."""
    first = pd.Timestamp(year=2000 + int(code[:2]), month=int(code[2:4]), day=1)
    return first - pd.DateOffset(months=2) - pd.Timedelta(days=1)


# ---------------------------------------------------------------- events
def event_id(kind: str, tenor_months) -> str:
    """``JP_AUCTION_<t>Y`` for the regular coupon tenors, else one event per kind."""
    if kind == "JGB" and tenor_months and int(tenor_months) % 12 == 0 and int(tenor_months) // 12 in (2, 5, 10, 20, 30, 40):
        return f"JP_AUCTION_{int(tenor_months) // 12}Y"
    return {"TB": "JP_AUCTION_TBILL", "LIQ": "JP_AUCTION_LIQ", "ILB": "JP_AUCTION_ILB",
            "GX": "JP_AUCTION_GX"}.get(kind, "JP_AUCTION_OTHER")
