"""BLS bulk time-series files (download.bls.gov/pub/time.series) - NETWORK ONLY, no files.

Free, no key, no request limits - unlike the BLS API (50 series / 500 queries a day), one
file holds EVERY series of a survey: ``cu`` (CPI-U, ~8,100 series: every item x area,
SA/NSA), ``wp`` (PPI by commodity, incl. final demand), ``pc`` (PPI by industry).

Facts this module relies on (verified 2026-09-30):
* the server answers 403 unless the User-Agent carries a contact email - set
  ``BLS_CONTACT_EMAIL`` in ``.env`` (``has_contact``); no email is ever sent otherwise;
* ``<s>.data.0.Current`` is tab-separated ``series_id, year, period, value,
  footnote_codes``, every field space-padded; ``period`` is ``M01``..``M12`` (months),
  ``M13`` (annual average), ``S01``-``S03`` (half-years / annual, areas published
  semiannually); a missing value is ``-`` (footnote ``X``: the 2025 appropriations lapse);
* the files are regenerated at the release (HTTP ``Last-Modified`` 08:30 New York on
  release day), which is the publication time the vintage store stamps;
* BLS keeps no vintages: the file only ever holds the latest revised history.
"""
from __future__ import annotations

import io
import os
import urllib.error
import urllib.request
import zipfile
from email.utils import parsedate_to_datetime

import pandas as pd
from dotenv import load_dotenv

from infra.config import BLS_CONTACT_ENV, PROJECT_ROOT

BASE_URL = "https://download.bls.gov/pub/time.series"
TIMEOUT_S = 300
VALUE_COLUMNS = ["series_id", "date", "value"]


class BlsError(RuntimeError):
    pass


def contact(explicit: str | None = None) -> str | None:
    load_dotenv(PROJECT_ROOT / ".env")
    return explicit or os.environ.get(BLS_CONTACT_ENV) or None


def has_contact(explicit: str | None = None) -> bool:
    return contact(explicit) is not None


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    email = contact()
    if not email:
        raise BlsError(f"Set {BLS_CONTACT_ENV} in .env: download.bls.gov refuses requests without a contact email")
    return urllib.request.Request(url, method=method, headers={"User-Agent": f"infra-data-pipeline ({email})"})


def data_file(survey: str) -> str:
    return f"{survey}.data.0.Current"


def url(survey: str, name: str) -> str:
    return f"{BASE_URL}/{survey}/{name}"


def get_text(survey: str, name: str) -> str:
    with urllib.request.urlopen(_request(url(survey, name)), timeout=TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_last_modified(u: str, request=_request) -> pd.Timestamp:
    """The ``Last-Modified`` header of ``u`` (HEAD), tz-naive UTC."""
    with urllib.request.urlopen(request(u, "HEAD"), timeout=60) as resp:
        header = resp.headers.get("Last-Modified")
    if not header:
        raise BlsError(f"{u}: no Last-Modified header")
    return pd.Timestamp(parsedate_to_datetime(header)).tz_convert("UTC").tz_localize(None)


def last_modified(survey: str) -> pd.Timestamp:
    """When the survey's data file was last regenerated (= published), UTC."""
    return http_last_modified(url(survey, data_file(survey)))


def read_tsv(text: str) -> pd.DataFrame:
    """A BLS tab-separated file with every cell and header stripped; all columns str."""
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda col: col.str.strip())


def parse_data(text: str) -> pd.DataFrame:
    """``series_id, date (month start), value`` for the MONTHLY periods; annual averages,
    half-years and missing (``-``) values dropped."""
    df = read_tsv(text)
    monthly = df[df["period"].str.fullmatch(r"M(0[1-9]|1[0-2])")]
    value = pd.to_numeric(monthly["value"], errors="coerce")
    out = pd.DataFrame({
        "series_id": monthly["series_id"],
        "date": pd.to_datetime(monthly["year"] + "-" + monthly["period"].str[1:] + "-01").astype("datetime64[ms]"),
        "value": value.astype("float64"),
    })
    return out.dropna(subset=["value"]).reset_index(drop=True)


def parse_catalog(series_text: str, mappings: dict[str, str]) -> pd.DataFrame:
    """The survey's ``<s>.series`` table (one row per series: codes, seasonal flag, title,
    base period, ...) with each code column's NAME joined on from its mapping file
    (``mappings``: ``{"item": <s>.item text, "area": ...}`` -> ``item_name``, ``area_name``).
    Used to navigate the component tree; not a vintage (latest only)."""
    cat = read_tsv(series_text)
    for name, text in mappings.items():
        m = read_tsv(text)
        code, label = f"{name}_code", f"{name}_name"
        if code in cat.columns and code in m.columns and label in m.columns:
            # a code can be scoped by another column (wp.item: group_code + item_code)
            on = [c for c in ("group_code",) if c in m.columns and c in cat.columns and c != code] + [code]
            m = m.drop_duplicates(on)
            cat = cat.merge(m[[*on, label]], on=on, how="left")
    return cat.rename(columns={"series_id": "ticker"})


# mapping files per survey worth joining (each exists on the server, verified 2026-09-30)
CATALOG_MAPPINGS = {"cu": ("item", "area"), "wp": ("group", "item"), "pc": ("industry", "product")}


def fetch_survey(survey: str, *, get=get_text, modified=last_modified):
    """``(values, catalog, published)``: every monthly value of every series in the
    survey's current file, its series catalog, and the file's publication time (UTC).
    The publication time is read before AND after the download: a file regenerated
    mid-download can't be tied to one version, so that raises (the next run retries)."""
    published = modified(survey)
    values = parse_data(get(survey, data_file(survey)))
    mappings = {m: get(survey, f"{survey}.{m}") for m in CATALOG_MAPPINGS.get(survey, ())}
    catalog = parse_catalog(get(survey, f"{survey}.series"), mappings)
    if modified(survey) != published:
        raise BlsError(f"{data_file(survey)} was republished during the download - retry")
    return values, catalog, published


# ------------------------------------------------------------------ CPI relative importance
# The CPI weights (infra.processing.cpi_weights), one file per "weight year" Y = December Y,
# released with the January Y+1 CPI. Verified 2026-09-30: www.bls.gov (same contact-email
# rule) serves 2020+ as ``<Y>.xlsx``; earlier years only inside per-decade zip archives, as
# ``<Y>.txt`` (1987-2019; also PDFs, ignored). A decade's loose files may move into a new
# archive once it closes, so a year not found loose is looked up in its decade's archive too.
RI_URL = "https://www.bls.gov/cpi/tables/relative-importance"


def ri_archive(year: int) -> tuple[str, str]:
    """``(zip name, member prefix)`` of the archive holding ``year``."""
    lo, hi = (1987, 1989) if year < 1990 else (year // 10 * 10, year // 10 * 10 + 9)
    return f"ri-archive-{lo}-{hi}.zip", f"{lo}-{hi}_RI_archive/{year}"


def get_www(name: str) -> bytes | None:
    """One file under RI_URL; None when it doesn't exist (404)."""
    try:
        with urllib.request.urlopen(_request(f"{RI_URL}/{name}"), timeout=TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_relative_importance(years, *, get=get_www) -> dict[int, tuple[str, bytes]]:
    """``{year: (format "xlsx" | "txt", content)}`` for every requested year BLS has
    published; a year it hasn't is simply absent. Each archive is downloaded once per call."""
    out, archives = {}, {}
    for year in sorted(set(years)):
        if year >= 2020 and (content := get(f"{year}.xlsx")) is not None:
            out[year] = ("xlsx", content)
            continue
        name, prefix = ri_archive(year)
        if name not in archives:
            archives[name] = get(name)
        if archives[name] is None:
            continue
        with zipfile.ZipFile(io.BytesIO(archives[name])) as zf:
            for fmt in ("xlsx", "txt"):
                if f"{prefix}.{fmt}" in zf.namelist():
                    out[year] = (fmt, zf.read(f"{prefix}.{fmt}"))
                    break
    return out
