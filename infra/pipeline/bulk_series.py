"""Full-granularity inflation detail (``BULK_DATASETS``: CPI, PPI, PCE component trees) from
the agencies' bulk files: read / plan / fetch / store + a ``load`` parent - the same shape
as infra/pipeline/releases.py, on the same raw vintage layout (infra.processing.releases:
``timestamp`` = publication day, ``ticker`` = the source's own series id, ``period``,
``value``), one store per source under ``Database/RawData`` (``BLS``, ``BEA``).

What differs from the FRED releases, all because these sources keep NO vintages - a file
only ever holds the latest revised history:

* a request can't ask for a past publication window. What's fetched is the file's CURRENT
  version, stamped with its HTTP ``Last-Modified`` day (the publication); ``drop_unchanged``
  keeps only the values that version changed. So vintages exist from the first snapshot on
  (that first snapshot stamps the whole history with its own day - honest: nothing earlier
  was seen), and a version is only ever seen if some run happens while it's current (a
  monthly release, the cycle runs daily; a same-day correction replaces that day's rows);
* coverage (keyed by dataset) is on the publication-TIME axis: ``[first snapshot, latest
  ingested Last-Modified]``. A file whose Last-Modified is already covered is never
  downloaded again (Rule 2.1 - the sources are free, but a CPI+PPI file set is ~190 MB);
  a cheap HEAD request is all an unchanged day costs;
* the series CATALOG (titles, item/area codes, BEA table lines and scale) is kept alongside,
  latest only, one flat file per dataset under ``_catalog`` - it's how you navigate the tree.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from infra.api import bea_client, bls_client
from infra.config import BULK_CATALOG_DIR, BULK_COVERAGE_FILE, BULK_DATASETS, RAW_DATA_ROOT, BulkDataset
from infra.coverage.intervals import to_utc_day
from infra.processing import releases as pr
from infra.storage import coverage_store, parquet_store

log = logging.getLogger(__name__)

# source name -> fn(dataset) -> (values series_id/date/value, catalog, published UTC)
SOURCES: dict[str, Callable[[BulkDataset], tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]]] = {
    "bls": lambda d: bls_client.fetch_survey(d.file),
    "bea": lambda d: bea_client.fetch_nipa(d.file, d.tables),
}
# source name -> fn(dataset) -> the current file's publication time (HEAD, no download)
LAST_MODIFIED: dict[str, Callable[[BulkDataset], pd.Timestamp]] = {
    "bls": lambda d: bls_client.last_modified(d.file),
    "bea": lambda d: bea_client.last_modified(d.file),
}
# source name -> fn() -> bool: is the source usable here (its configuration present)?
CONFIGURED: dict[str, Callable[[], bool]] = {"bls": bls_client.has_contact, "bea": lambda: True}
_ONE_MS = pd.Timedelta(milliseconds=1)


def store_dir(dataset: BulkDataset, raw_root: Path = RAW_DATA_ROOT) -> Path:
    return raw_root / dataset.store


def catalog_file(dataset: BulkDataset, catalog_dir: Path = BULK_CATALOG_DIR) -> Path:
    return catalog_dir / f"{dataset.key}.parquet"


def read_bulk_from_disk(
    dataset: str,
    tickers: list[str] | None = None,
    as_of=None,
    *,
    start=None,
    raw_root: Path = RAW_DATA_ROOT,
) -> pd.DataFrame:
    """Decoded vintage rows of ``dataset``'s store PUBLISHED by the end of day ``as_of``
    (inclusive; None = everything), optionally from publication day ``start``. No network.
    ``tickers`` None = every series in the store (which may hold several datasets of one
    source); point-in-time views via ``infra.processing.releases.snapshot``."""
    end = None if as_of is None else to_utc_day(as_of) + pd.Timedelta(days=1)
    raw = parquet_store.read_partitioned(store_dir(BULK_DATASETS[dataset], raw_root), start=start, end=end,
                                         equals_in=None if tickers is None else {"ticker": list(tickers)})
    if raw is None or raw.empty:
        return pr.decode_raw(pr.empty_raw())
    return pr.decode_raw(raw)


def read_catalog(dataset: str, *, catalog_dir: Path = BULK_CATALOG_DIR) -> pd.DataFrame:
    """The dataset's latest series catalog (empty if never fetched)."""
    path = catalog_file(BULK_DATASETS[dataset], catalog_dir)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["ticker"])


def latest_ingested(key: str, *, coverage_file: Path = BULK_COVERAGE_FILE) -> pd.Timestamp | None:
    """Publication time of the latest file version already on disk (None = never)."""
    covered = coverage_store.read_covered(coverage_file, key)
    return covered[-1][1] - _ONE_MS if covered else None


def needs_download(key: str, published: pd.Timestamp | None, *, coverage_file: Path = BULK_COVERAGE_FILE,
                   force_refetch: bool = False) -> bool:
    """Whether the version published at ``published`` still has to be fetched: never when
    nothing is published (None), always with ``force_refetch``, else only if newer than
    the latest ingested version."""
    if published is None:
        return False
    if force_refetch:
        return True
    last = latest_ingested(key, coverage_file=coverage_file)
    return last is None or pd.Timestamp(published) > last


def fetch_bulk_raw(dataset: BulkDataset, *, sources=None):
    """NETWORK ONLY: ``(values, catalog, published)`` of the dataset's current file."""
    sources = SOURCES if sources is None else sources
    return sources[dataset.source](dataset)


def store_bulk_raw(
    dataset: BulkDataset,
    values: pd.DataFrame,
    catalog: pd.DataFrame,
    published: pd.Timestamp,
    *,
    raw_root: Path = RAW_DATA_ROOT,
    coverage_file: Path = BULK_COVERAGE_FILE,
    catalog_dir: Path = BULK_CATALOG_DIR,
) -> int:
    """FILES ONLY: keep what this version changed (``drop_unchanged`` vs what is stored),
    save it and the catalog, and record the version as ingested."""
    root = store_dir(dataset, raw_root)
    incoming = pr.from_snapshot(values, published)
    stored = parquet_store.read_partitioned(root, equals_in={"ticker": sorted(incoming["ticker"].unique())})
    new = pr.drop_unchanged(incoming, pr.empty_raw() if stored is None else stored)
    parquet_store.write_partitioned(pr.encode_raw(new), root, pr.RAW_KEYS)
    path = catalog_file(dataset, catalog_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    catalog.astype({c: str for c in catalog.columns if catalog[c].dtype == object}).to_parquet(
        tmp, engine="pyarrow", index=False, compression="zstd", compression_level=5)
    tmp.replace(path)
    published = pd.Timestamp(published)
    first = coverage_store.read_covered(coverage_file, dataset.key)
    coverage_store.record_covered(coverage_file, dataset.key, [(first[0][0] if first else published, published + _ONE_MS)])
    log.info("bulk %s version %s: %d new of %d values", dataset.key, published, len(new), len(incoming))
    return len(new)


def load_bulk_series(
    dataset: str,
    tickers: list[str] | None = None,
    as_of=None,
    *,
    fetch_missing: bool = True,
    raw_root: Path = RAW_DATA_ROOT,
    coverage_file: Path = BULK_COVERAGE_FILE,
    catalog_dir: Path = BULK_CATALOG_DIR,
    sources=None,
    last_modified=None,
) -> pd.DataFrame:
    """Parent: ingest the current file version if it's newer than what's on disk (a HEAD
    request when it isn't), then read point-in-time. A past ``as_of`` never fetches - a
    current file can't add anything published by a past day that isn't already stored."""
    ds_ = BULK_DATASETS[dataset]
    live = as_of is None or to_utc_day(as_of) >= to_utc_day(pd.Timestamp.now())
    if fetch_missing and live:
        published = (LAST_MODIFIED if last_modified is None else last_modified)[ds_.source](ds_)
        if needs_download(ds_.key, published, coverage_file=coverage_file):
            store_bulk_raw(ds_, *fetch_bulk_raw(ds_, sources=sources), raw_root=raw_root,
                           coverage_file=coverage_file, catalog_dir=catalog_dir)
    return read_bulk_from_disk(dataset, tickers, as_of, raw_root=raw_root)
