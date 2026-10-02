"""FRED / ALFRED (St. Louis Fed) - NETWORK ONLY, no files.

Free, needs an API key (``FRED_API_KEY`` in the environment or ``.env``). One endpoint
serves every series, so every FRED-sourced macro release (``MACRO_RELEASES``) goes
through ``fetch_vintages`` - no per-series code.

ALFRED is FRED's archive of every published VINTAGE: asked for a real-time window, it
returns one row per (observation ``date``, value) with the ``realtime_start``/
``realtime_end`` days that value was the published one. So a series' Advance /
Preliminary / Final estimates and every later revision come back as separate rows, each
dated by the day it was published - exactly the release calendar a real-time nowcast
needs, with no separate calendar feed.

Two facts about the API this module relies on (docs: https://fred.stlouisfed.org/docs/api/fred/):
* ``realtime_start``/``realtime_end`` bound the REAL-TIME (publication) axis;
  ``9999-12-31`` means "still current". A value published before the requested
  ``realtime_start`` and still current is returned too - possibly with its
  ``realtime_start`` clipped to the request's - so the pipeline de-duplicates against
  what it already holds (``infra.processing.releases.drop_unchanged``) rather than trust
  a returned ``realtime_start`` at the window's edge as a new publication.
* missing values are the string ``"."``; results are paged (``limit`` <= 100000).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from dotenv import load_dotenv

from infra.config import FRED_API_KEY_ENV, PROJECT_ROOT

BASE_URL = "https://api.stlouisfed.org/fred"
TIMEOUT_S = 60
PAGE_LIMIT = 100_000
# The API allows 120 requests/minute; on HTTP 429 back off and retry.
MAX_RETRIES = 5
BACKOFF_S = 2.0
EPOCH = pd.Timestamp("1776-07-04")  # ALFRED's own "beginning of time" for the real-time axis
STILL_CURRENT = "9999-12-31"
# A US publication day D is complete by 06:00 UTC on D+1 (02:00 New York in summer,
# 01:00 in winter); before that, D may still get releases and must not be claimed covered.
DAY_COMPLETE_LAG = pd.Timedelta(hours=6)

VINTAGE_COLUMNS = ["realtime_start", "date", "value"]


class FredError(RuntimeError):
    pass


def api_key(explicit: str | None = None) -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = explicit or os.environ.get(FRED_API_KEY_ENV)
    if not key:
        raise FredError(f"Set {FRED_API_KEY_ENV} in the environment or in .env "
                        "(free: fred.stlouisfed.org -> My Account -> API Keys)")
    return key


def has_key(explicit: str | None = None) -> bool:
    load_dotenv(PROJECT_ROOT / ".env")
    return bool(explicit or os.environ.get(FRED_API_KEY_ENV))


def get_json(endpoint: str, params: dict, *, key: str | None = None) -> dict:
    """One GET, retried on rate limiting / transient server errors."""
    query = urllib.parse.urlencode({**params, "api_key": api_key(key), "file_type": "json"})
    url = f"{BASE_URL}/{endpoint}?{query}"
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_S * 2 ** attempt)
                continue
            try:
                message = json.loads(exc.read().decode("utf-8")).get("error_message", "")
            except Exception:
                message = ""
            raise FredError(f"FRED {endpoint} {params.get('series_id', '')}: HTTP {exc.code} {message}") from None
    raise FredError(f"FRED {endpoint}: gave up after {MAX_RETRIES} attempts")


def parse_observations(payload: dict) -> pd.DataFrame:
    """``realtime_start`` (publication day), ``date`` (observation period start),
    ``value`` (float) - one row per published value; ``"."`` (no value) dropped."""
    obs = payload.get("observations", [])
    if not obs:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ms]" if c != "value" else "float64")
                             for c in VINTAGE_COLUMNS})
    df = pd.DataFrame(obs)
    df = df[df["value"] != "."]
    return pd.DataFrame({
        "realtime_start": pd.to_datetime(df["realtime_start"]).astype("datetime64[ms]"),
        "date": pd.to_datetime(df["date"]).astype("datetime64[ms]"),
        "value": pd.to_numeric(df["value"], errors="coerce").astype("float64"),
    }).dropna(subset=["value"]).reset_index(drop=True)


def complete_day_end(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Exclusive end of the publication days that are over (see DAY_COMPLETE_LAG)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    return (now - DAY_COMPLETE_LAG).normalize()


def fetch_vintages(series_id: str, start: pd.Timestamp, end: pd.Timestamp, *, key: str | None = None,
                   get=get_json, now: pd.Timestamp | None = None):
    """Every value of ``series_id`` published in ``[start, end)`` (publication days),
    plus values still current at ``start`` (see module doc), and the interval genuinely
    COVERED: up to the last COMPLETE publication day, never beyond.

    Returns ``(frame, covered)`` like the other raw fetchers (infra.api.treasury_client)."""
    start = max(pd.Timestamp(start), EPOCH)
    frames, offset = [], 0
    while True:
        payload = get("series/observations", {
            "series_id": series_id, "realtime_start": start.strftime("%Y-%m-%d"),
            "realtime_end": STILL_CURRENT, "limit": PAGE_LIMIT, "offset": offset, "sort_order": "asc",
        }, key=key)
        frames.append(parse_observations(payload))
        offset += PAGE_LIMIT
        if offset >= int(payload.get("count", 0)):
            break
    df = pd.concat(frames, ignore_index=True)
    df = df[df["realtime_start"] < pd.Timestamp(end)].reset_index(drop=True)
    covered_end = min(pd.Timestamp(end), complete_day_end(now))
    return df, ([(start, covered_end)] if covered_end > start else [])


def fetch_series_info(series_id: str, *, key: str | None = None, get=get_json) -> dict:
    """Series metadata (title, frequency, units, seasonal adjustment, observation range)."""
    seriess = get("series", {"series_id": series_id}, key=key).get("seriess", [])
    if not seriess:
        raise FredError(f"FRED series {series_id}: not found")
    return seriess[0]


def search_series(text: str, *, limit: int = 20, key: str | None = None, get=get_json) -> pd.DataFrame:
    """Full-text series search - for finding/verifying ids, never used by the pipeline."""
    rows = get("series/search", {"search_text": text, "limit": limit, "order_by": "popularity"},
               key=key).get("seriess", [])
    cols = ["id", "title", "frequency_short", "units_short", "seasonal_adjustment_short",
            "observation_start", "observation_end", "popularity"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def fetch_release_dates(release_id: int, *, key: str | None = None, get=get_json) -> pd.DatetimeIndex:
    """Every publication date of a FRED release, PAST and SCHEDULED (verified 2026-10-01:
    one call returns e.g. the Employment Situation's 867 dates from 1955-05-06 through the
    scheduled 2026-12-04). Dates only - FRED carries no release times."""
    payload = get("release/dates", {"release_id": release_id, "realtime_start": EPOCH.strftime("%Y-%m-%d"),
                                    "realtime_end": STILL_CURRENT, "include_release_dates_with_no_data": "true",
                                    "limit": 10000, "sort_order": "asc"}, key=key)
    return pd.DatetimeIndex(sorted({pd.Timestamp(x["date"]) for x in payload.get("release_dates", [])}))
