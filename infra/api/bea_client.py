"""BEA NIPA bulk flat files (apps.bea.gov/national/Release/TXT) - NETWORK ONLY, no files.

Free, no key (the BEA API needs one and pages by table; these files hold everything).
Verified 2026-09-30:
* ``NipaDataM.txt`` (monthly; ``...A``/``...Q`` annual/quarterly) is CSV ``%SeriesCode,
  Period, Value`` for EVERY published NIPA series, incl. the "underlying detail" tables
  (U-prefixed, e.g. U20404 = PCE price indexes by type of product, ~390 lines deep);
  ``Period`` is ``2026M08`` / ``2026Q2`` / ``2026``; values carry thousands separators;
* ``SeriesRegister.txt`` maps each series to its tables and lines (``TableId:LineNo``,
  ``|``-separated), with label, metric, calculation type and scale (``DefaultScale``: -6
  = millions); ``TablesRegister.txt`` names the tables;
* the files are regenerated at each release (HTTP ``Last-Modified`` 12:30 UTC = 08:30 New
  York on the Personal Income and Outlays day), the publication time the store stamps;
* BEA keeps no vintages here: only the latest revised history.
"""
from __future__ import annotations

import io
import urllib.request

import pandas as pd

from infra.api.bls_client import http_last_modified

BASE_URL = "https://apps.bea.gov/national/Release/TXT"
TIMEOUT_S = 300
VALUE_COLUMNS = ["series_id", "date", "value"]


class BeaError(RuntimeError):
    pass


def _request(u: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(u, method=method, headers={"User-Agent": "infra-data-pipeline"})


def url(name: str) -> str:
    return f"{BASE_URL}/{name}"


def get_text(name: str) -> str:
    with urllib.request.urlopen(_request(url(name)), timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8-sig", errors="replace")


def last_modified(name: str) -> pd.Timestamp:
    return http_last_modified(url(name), request=_request)


def parse_period(period: pd.Series) -> pd.Series:
    """``2026M08`` -> 2026-08-01, ``2026Q2`` -> 2026-04-01 (quarter's first month), ``2026``
    -> 2026-01-01 - the same period convention as the macro-release store."""
    p = period.str.strip()
    year = p.str[:4]
    month = pd.Series("01", index=p.index)
    is_m, is_q = p.str[4] == "M", p.str[4] == "Q"
    month[is_m] = p[is_m].str[5:].str.zfill(2)
    month[is_q] = ((p[is_q].str[5:].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2)
    return pd.to_datetime(year + "-" + month + "-01").astype("datetime64[ms]")


def parse_data(text: str, series: set[str] | None = None) -> pd.DataFrame:
    """``series_id, date, value`` (float, the series' own scale), optionally only ``series``."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.columns = ["series_id", "period", "value"]
    if series is not None:
        df = df[df["series_id"].isin(series)]
    out = pd.DataFrame({
        "series_id": df["series_id"],
        "date": parse_period(df["period"]),
        "value": pd.to_numeric(df["value"].str.replace(",", "", regex=False), errors="coerce").astype("float64"),
    })
    return out.dropna(subset=["value"]).reset_index(drop=True)


def parse_register(text: str) -> pd.DataFrame:
    """``SeriesRegister.txt`` -> ``ticker, label, metric, calculation, scale, table_lines,
    parents`` (``table_lines`` as published, e.g. ``T20804:1|U20404:1``)."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df.columns = ["ticker", "label", "metric", "calculation", "scale", "table_lines", "parents"]
    df["scale"] = pd.to_numeric(df["scale"], errors="coerce").astype("Int64")
    return df


def in_tables(register: pd.DataFrame, tables: tuple[str, ...]) -> pd.DataFrame:
    """Register rows for the series that appear in any of ``tables``."""
    wanted = set(tables)
    hit = register["table_lines"].map(lambda s: any(t.split(":")[0] in wanted for t in s.split("|") if t))
    return register[hit].reset_index(drop=True)


def fetch_nipa(name: str, tables: tuple[str, ...], *, get=get_text, modified=last_modified):
    """``(values, catalog, published)`` for the series of ``tables`` in data file ``name``
    (all of it when ``tables`` is empty). Publication time read before and after the
    download, like ``infra.api.bls_client.fetch_survey``."""
    published = modified(name)
    register = parse_register(get("SeriesRegister.txt"))
    catalog = in_tables(register, tables) if tables else register
    values = parse_data(get(name), set(catalog["ticker"]) if tables else None)
    if modified(name) != published:
        raise BeaError(f"{name} was republished during the download - retry")
    return values, catalog, published
