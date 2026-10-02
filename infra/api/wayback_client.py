"""Internet Archive Wayback Machine - NETWORK ONLY, no files.

Two endpoints, both free and keyless:

* CDX (``/cdx/search/cdx``): the list of captures of a URL (prefix) in a time window -
  ``list_captures``;
* playback with the ``id_`` flag (``/web/<timestamp>id_/<original url>``): the page
  EXACTLY as captured, without the archive's toolbar or link rewriting -
  ``fetch_capture``. Bytes are returned as-is: some captures come back still
  gzip-compressed (infra.processing.econ_calendar.decode_html handles it).

Be polite: callers space requests out (``infra.pipeline.econ_calendar`` waits between
page fetches); HTTP 429 / 5xx are retried here with backoff.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

CDX_URL = "https://web.archive.org/cdx/search/cdx"
PLAYBACK_URL = "https://web.archive.org/web/{timestamp}id_/{original}"
TIMEOUT_S = 120
MAX_RETRIES = 5
BACKOFF_S = 10.0
USER_AGENT = "Mozilla/5.0 (compatible; personal research; low-rate archive reader)"
CAPTURE_COLUMNS = ["timestamp", "original", "digest"]


class WaybackError(RuntimeError):
    pass


def make_get(timeout_s: float = TIMEOUT_S, max_retries: int = MAX_RETRIES, backoff_s: float = BACKOFF_S):
    """A GET with its own patience. The default is patient (a multi-hour history harvest
    rides out the archive's slow spells); the daily cycle uses a FAST one - the archive
    was down for hours on 2026-10-01 (CDX timing out, playback HTTP 503), and a patient
    client would have held a scheduled run for hours."""
    def get(url: str) -> bytes:
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(backoff_s * 2 ** attempt)
                    continue
                raise WaybackError(f"{url}: HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                if attempt < max_retries - 1:
                    time.sleep(backoff_s * 2 ** attempt)
                    continue
                raise WaybackError(f"{url}: {exc}") from None
        raise WaybackError(f"{url}: gave up after {max_retries} attempts")
    return get


_get = make_get()
FAST_GET = make_get(timeout_s=30, max_retries=2, backoff_s=5)


def list_captures(url_prefix: str, start: pd.Timestamp, end: pd.Timestamp, *, url_regex: str | None = None,
                  first_per_url: bool = False, get=_get) -> pd.DataFrame:
    """Every successful HTML capture of URLs starting with ``url_prefix`` (query-string
    variants included) in ``[start, end)``: ``timestamp`` (capture time, UTC),
    ``original`` (the URL as captured), ``digest`` (content hash). ``url_regex`` filters
    on the archive's side (a site-wide prefix can list 100k+ URLs); ``first_per_url``
    keeps only each URL's first capture."""
    params = {
        "url": url_prefix, "matchType": "prefix", "output": "json",
        "fl": "timestamp,original,digest",
        "filter": ["statuscode:200", "mimetype:text/html"] + ([f"original:{url_regex}"] if url_regex else []),
        "from": pd.Timestamp(start).strftime("%Y%m%d%H%M%S"),
        "to": (pd.Timestamp(end) - pd.Timedelta(seconds=1)).strftime("%Y%m%d%H%M%S"),
    }
    if first_per_url:
        params["collapse"] = "urlkey"
    body = get(f"{CDX_URL}?{urllib.parse.urlencode(params, doseq=True)}").decode("utf-8", "ignore").strip()
    rows = json.loads(body)[1:] if body else []
    df = pd.DataFrame(rows, columns=CAPTURE_COLUMNS)
    if df.empty:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ms]"), "original": pd.Series(dtype=str),
                             "digest": pd.Series(dtype=str)})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S").astype("datetime64[ms]")
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_capture(timestamp: pd.Timestamp, original: str, *, get=_get) -> bytes:
    """The captured page's raw bytes (possibly gzip-compressed)."""
    ts = pd.Timestamp(timestamp).strftime("%Y%m%d%H%M%S")
    return get(PLAYBACK_URL.format(timestamp=ts, original=original))
