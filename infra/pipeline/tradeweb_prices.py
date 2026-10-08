"""FTSE-Tradeweb gilt and EuroGov closing prices - archive / store / read (root CLAUDE.md 18;
client ``infra.api.tradeweb_client``, parsing ``infra.processing.tradeweb_prices``).

* Raw archive ``TRADEWEB_RAW_DIR``: every export kept as delivered - ``daily/<export day>.csv``
  (all securities, the last ``DAILY_LOOKBACK`` working days: the site's limit for an all-security
  export, and enough to heal missed runs) and ``history/<isin>__<from>_<to>.csv`` (one security,
  one chunk). A file on disk is its own coverage (Rule 2.1).
* Prices ``TRADEWEB_PRICES_DIR`` (keys ``timestamp`` = close-of-business day, ``isin``),
  upserted from every export.
* Universe ``TRADEWEB_RAW_DIR/universe.parquet``: every ISIN seen (exports, and the site's
  search grid on sampled days - the only way to find a security that matured before today),
  with its first / last day seen.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from infra.config import TRADEWEB_PRICES_DIR, TRADEWEB_RAW_DIR
from infra.processing import tradeweb_prices as tp
from infra.storage import parquet_store

log = logging.getLogger(__name__)
KEYS = ["timestamp", "isin"]
HISTORY_START = pd.Timestamp("2017-07-24")    # the site's first day
DAILY_LOOKBACK = 5                            # working days an all-security export may cover
CHUNK_YEARS = 3                               # a single-ISIN export of 2017-2026 in one request failed
ERROR_RETRIES = 2                             # tries again after ERROR_PAUSE_S x attempt before splitting
ERROR_PAUSE_S = 30
SESSION = None                                # one InSite session per process (network hook; tests stub it)
_ONE_DAY = pd.Timedelta(days=1)


def _session():
    global SESSION
    if SESSION is None:
        from infra.api.tradeweb_client import InSite
        SESSION = InSite()
    return SESSION


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, "utf-8")
    tmp.replace(path)


def _store(df: pd.DataFrame, root: Path) -> int:
    if df.empty:
        return 0
    parquet_store.write_partitioned(tp.encode(df), root, KEYS)
    return len(df)


# ------------------------------------------------------------------ universe
def _universe_path(raw_root: Path) -> Path:
    return Path(raw_root) / "universe.parquet"


def read_universe(*, raw_root: Path = TRADEWEB_RAW_DIR) -> pd.DataFrame:
    p = _universe_path(raw_root)
    cols = ["isin", "name", "security_type", "country", "coupon", "maturity_date", "first_seen", "last_seen"]
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=cols)


def _merge_universe(rows: pd.DataFrame, raw_root: Path) -> int:
    """Fold seen (day, ISIN, terms) rows into the universe; returns the ISINs it now holds."""
    if rows.empty:
        return len(read_universe(raw_root=raw_root))
    seen = rows.groupby("isin").agg(name=("name", "last"), security_type=("security_type", "last"),
                                    country=("country", "last"), coupon=("coupon", "last"),
                                    maturity_date=("maturity_date", "last"), first_seen=("timestamp", "min"),
                                    last_seen=("timestamp", "max")).reset_index()
    u = read_universe(raw_root=raw_root)
    both = pd.concat([u, seen], ignore_index=True)
    out = both.groupby("isin").agg(name=("name", "last"), security_type=("security_type", "last"),
                                   country=("country", "last"), coupon=("coupon", "last"),
                                   maturity_date=("maturity_date", "last"), first_seen=("first_seen", "min"),
                                   last_seen=("last_seen", "max")).reset_index()
    p = _universe_path(raw_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    return len(out)


def discover(days, *, cp_type: str = "gilts", security_types=("Conventional", "Index-linked"),
             raw_root: Path = TRADEWEB_RAW_DIR) -> int:
    """The ISINs in the site's search grid on ``days`` (one search + a page per 30 rows each),
    folded into the universe - for securities that matured before today's export."""
    from infra.api.tradeweb_client import _grid
    s = _session()
    frames = []
    for day in pd.DatetimeIndex(days):
        for st in security_types:
            rows = s.grid_rows(day, day, cp_type=cp_type, security_type=st)
            g = pd.DataFrame([r[:6] for r in rows if len(r) >= 6],
                             columns=["name", "timestamp", "isin", "security_type", "coupon", "maturity_date"])
            if g.empty:
                continue
            g["timestamp"] = pd.to_datetime(g["timestamp"], format="%m/%d/%Y")
            g["maturity_date"] = pd.to_datetime(g["maturity_date"], format="%m/%d/%Y", errors="coerce")
            g["coupon"] = pd.to_numeric(g["coupon"], errors="coerce")
            g["country"] = g["isin"].str[:2]
            frames.append(g)
        log.info("tradeweb discover %s: %d rows so far", day.date(), sum(len(f) for f in frames))
    return _merge_universe(pd.concat(frames, ignore_index=True), raw_root) if frames else 0


# ------------------------------------------------------------------ prices
def update_daily(*, now=None, raw_root: Path = TRADEWEB_RAW_DIR, root: Path = TRADEWEB_PRICES_DIR) -> dict:
    """The daily refresh: one export of every security over the last ``DAILY_LOOKBACK`` working
    days (a day is free from 12:00 London the next day), archived and upserted."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    end = (now.normalize() - pd.offsets.BDay(1)).normalize()
    start = (end - pd.offsets.BDay(DAILY_LOOKBACK - 1)).normalize()
    text = _session().export(start, end)
    df = tp.parse_export(text)
    if df.empty:
        return {"rows": 0, "days": []}
    _write(Path(raw_root) / "daily" / f"{now.strftime('%Y-%m-%d')}.csv", text)
    _merge_universe(df, raw_root)
    n = _store(df, root)
    return {"rows": n, "days": sorted(df["timestamp"].dt.date.unique())}


def _export_range(isin: str, lo: pd.Timestamp, hi: pd.Timestamp, raw_root: Path, failed: list) -> list[str]:
    """The exports covering ``[lo, hi]`` for one security: the archived file if there is one,
    else one request. The site sometimes answers with an application error that goes away on a
    later try (one gilt's 2021-2023 failed, then worked whole), so a failure is retried
    ``ERROR_RETRIES`` times after a pause before the range is split in two (down to a single day;
    a day that still fails is listed in ``failed``). An empty range is not an error (a header
    only)."""
    path = Path(raw_root) / "history" / f"{isin}__{lo:%Y%m%d}_{hi:%Y%m%d}.csv"
    split_marker = path.with_suffix(".split")       # this range was split: its halves are the archive
    if path.exists():
        return [path.read_text("utf-8")]
    if split_marker.exists():
        mid = (lo + (hi - lo) / 2).normalize()
        return (_export_range(isin, lo, mid, raw_root, failed)
                + _export_range(isin, mid + _ONE_DAY, hi, raw_root, failed))
    text, err = None, None
    for attempt in range(1 + ERROR_RETRIES):        # the site's errors are mostly transient
        try:
            text = _session().export(lo, hi, isin=isin)
            break
        except RuntimeError as exc:
            if "application error" not in str(exc):
                raise
            err = exc
            time.sleep(ERROR_PAUSE_S * (attempt + 1))
    if text is None:
        if hi <= lo:
            failed.append(lo)
            log.warning("tradeweb %s: %s fails even alone - skipped", isin, lo.date())
            return []
        mid = (lo + (hi - lo) / 2).normalize()
        split_marker.parent.mkdir(parents=True, exist_ok=True)
        split_marker.write_text(f"{err}\n", "utf-8")
        return (_export_range(isin, lo, mid, raw_root, failed)
                + _export_range(isin, mid + _ONE_DAY, hi, raw_root, failed))
    _write(path, text)
    return [text]


def backfill_isin(isin: str, start=None, end=None, *, raw_root: Path = TRADEWEB_RAW_DIR,
                  root: Path = TRADEWEB_PRICES_DIR) -> dict:
    """One security's history in ``CHUNK_YEARS`` chunks (split further where the site fails); an
    archived chunk is read from disk, never asked again (Rule 2.1)."""
    start = max(pd.Timestamp(start) if start is not None else HISTORY_START, HISTORY_START)
    end = pd.Timestamp(end) if end is not None else (pd.Timestamp.now().normalize() - _ONE_DAY)
    n, failed, lo = 0, [], start
    while lo <= end:
        hi = min(lo + pd.DateOffset(years=CHUNK_YEARS) - _ONE_DAY, end)
        for text in _export_range(isin, lo, hi, raw_root, failed):
            df = tp.parse_export(text)
            df = df[df["isin"] == isin]
            _merge_universe(df, raw_root)
            n += _store(df, root)
        lo = hi + _ONE_DAY
    return {"rows": n, "failed_days": failed}


def read_prices(start=None, end=None, *, isins=None, root: Path = TRADEWEB_PRICES_DIR) -> pd.DataFrame:
    if not parquet_store.has_data(root):
        return pd.DataFrame(columns=tp.COLUMNS)
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end) + _ONE_DAY,
                                        equals_in={"isin": list(isins)} if isins is not None else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=tp.COLUMNS)
    return tp.decode(df)[tp.COLUMNS].sort_values(KEYS).reset_index(drop=True)
