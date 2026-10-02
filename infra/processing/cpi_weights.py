"""CPI relative importance (the weights of the CPI item tree) - parsing, pure, no I/O.

BLS publishes, once a year, each item's relative importance in DECEMBER of year Y (percent
of all items, CPI-U and CPI-W, U.S. city average = its "Table 1"), released with the
January Y+1 CPI. It is the weight set the index is aggregated from in Y+1 (updated month
to month by relative price changes - BLS doesn't publish those intermediate weights).

Three layouts, verified 2026-09-30 on every year 1987-2025:
* ``coded`` (1987-1996 ``.txt``): ``CODE  NAME  cpi_u  cpi_w``, one flat list SORTED BY
  ITEM CODE (the codes of the time) - no indentation and no sections: the special
  aggregates (``SA0L1E`` ...) sit among the items, and the tree is in the codes themselves
  (``SA1`` > ``SA11`` > ``SA111``). ``indent`` and ``section`` are null for these years;
* ``dotted`` (1997-2019 ``.txt``): the tree by indentation (one leading space = level 0),
  names padded with dots; a long name wraps onto preceding number-less lines;
* ``xlsx`` (2020 on): sheet "Table 1", an explicit ``Indent Level`` column.
``indent`` is AS PUBLISHED, and BLS's indentation is not a strict tree in a few places
(e.g. "Alcoholic beverages" sits one level under "Food" in both the text and the xlsx
layouts, so "Food"'s direct children over-sum): never derive weights from it by summing
children - every published aggregate carries its own weight.
The dotted and xlsx layouts list the expenditure-category tree, then the special
aggregates ("All items" again, then "Commodities", "Energy", ...) - kept, as ``section``.

Output rows: ``weight_year, section, line, indent, item_name, item_code, cpi_u, cpi_w`` -
``line`` the row's order within the year (stable, a key), ``item_code`` as published
(coded years only; ``match_item_codes`` fills it from the CPI catalog for the others).
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

WEIGHT_COLUMNS = ["weight_year", "section", "line", "indent", "item_name", "item_code", "cpi_u", "cpi_w"]
EXPENDITURE, SPECIAL = "expenditure", "special"

_NUMS = re.compile(r"^(?P<text>.*?)\s+(?P<u>-?\d*\.\d+|\d+)\s+(?P<w>-?\d*\.\d+|\d+)\s*$")
# an 8-character code fills its column, leaving ONE space before the name (``SAS2LSRS HOUSE...``)
_CODED = re.compile(r"^(?P<code>[A-Z][A-Z0-9]{1,7})\s+(?P<text>\S.*)$")


def _frame(rows: list[dict], weight_year: int) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["section", "indent", "item_name", "item_code", "cpi_u", "cpi_w"])
    df.insert(0, "weight_year", int(weight_year))
    df.insert(2, "line", np.arange(len(df), dtype="int32"))
    return df.astype({"weight_year": "int32", "indent": "Int32", "cpi_u": "float64", "cpi_w": "float64"})[WEIGHT_COLUMNS]


def _clean_name(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*\.{2,}\s*$", "", text)).strip().rstrip(".").strip()


def parse_txt(text: str, weight_year: int) -> pd.DataFrame:
    """Table 1 of a ``.txt`` archive year (``coded`` or ``dotted`` layout, auto-detected)."""
    rows, pending, section, seen_all, started = [], None, EXPENDITURE, False, False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _NUMS.match(line)
        if m is None:
            # a wrapped name's first part (dotted layout): remember its indentation. Before
            # the first number row, a number-less line is the title/header, never a name.
            body = line.strip()
            if not started or body in ("Expenditure category", "Special aggregate indexes"):
                continue
            pending = (len(line) - len(line.lstrip(" ")), body) if pending is None else \
                (pending[0], pending[1] + " " + body)
            continue
        started = True
        head = m.group("text")
        code = None
        coded = _CODED.match(head.strip())
        if coded and not head.startswith(" "):
            code, name, indent = coded.group("code"), coded.group("text"), None
        elif pending is not None:
            indent, name = pending[0] - 1, pending[1] + " " + head.strip()
        else:
            indent, name = len(head) - len(head.lstrip(" ")) - 1, head
        pending = None
        name = _clean_name(name)
        if name.lower() == "all items":
            section = SPECIAL if seen_all else EXPENDITURE
            seen_all = True
        rows.append({"section": None if code else section, "indent": indent, "item_name": name, "item_code": code,
                     "cpi_u": float(m.group("u")), "cpi_w": float(m.group("w"))})
    return _frame(rows, weight_year)


def parse_xlsx(content: bytes, weight_year: int) -> pd.DataFrame:
    """Sheet "Table 1" of a 2020+ ``.xlsx`` year."""
    raw = pd.read_excel(io.BytesIO(content), sheet_name="Table 1", header=None)
    head = next(i for i in range(len(raw)) if str(raw.iat[i, 0]).strip() == "Indent Level")
    rows, section, seen_all = [], EXPENDITURE, False
    for _, r in raw.iloc[head + 1:].iterrows():
        name, u, w = r.iloc[1], pd.to_numeric(r.iloc[2], errors="coerce"), pd.to_numeric(r.iloc[3], errors="coerce")
        if pd.isna(name) or pd.isna(u):
            continue  # blank spacer rows, section captions ("Expenditure category")
        name = _clean_name(str(name))
        if name.lower() == "all items":
            section = SPECIAL if seen_all else EXPENDITURE
            seen_all = True
        indent = pd.to_numeric(r.iloc[0], errors="coerce")
        rows.append({"section": section, "indent": None if pd.isna(indent) else int(indent), "item_name": name,
                     "item_code": None, "cpi_u": float(u), "cpi_w": float(w) if pd.notna(w) else np.nan})
    return _frame(rows, weight_year)


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def match_item_codes(weights: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Fill ``item_code`` where it's missing, from the CPI item catalog (``cu.item``:
    ``item_code, item_name``) by normalized name; a name matching no item, or several,
    stays null (codes already published - the coded years - are kept as they are)."""
    pairs = items[["item_code", "item_name"]].drop_duplicates()  # the catalog repeats each item per series
    lookup = pairs.assign(_k=pairs["item_name"].map(_norm)).drop_duplicates("_k", keep=False)
    by_name = dict(zip(lookup["_k"], lookup["item_code"]))
    out = weights.copy()
    missing = out["item_code"].isna()
    out.loc[missing, "item_code"] = out.loc[missing, "item_name"].map(lambda n: by_name.get(_norm(n)))
    return out
