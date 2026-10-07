"""The Fed's GSW Treasury curve as a reference for our own curve (infra/models/curves):
fetch / store / read. The whole history is one small file, so the store is that file,
re-fetched only when the stored copy is ``REFRESH_DAYS`` old (Rule 2.1: a fresh copy is
never re-requested). Point in time: a row's ``known_from`` is its date + ``PUBLICATION_LAG``
(the Fed posts with ~a week's lag; history isn't time-stamped, so this is an
approximation for backtests - the curve is used for validation, not as a model input).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from infra.api import fed_gsw_client
from infra.config import RAW_DATA_ROOT

FED_GSW_FILE = RAW_DATA_ROOT / "FedGSW" / "feds200628.csv"
FED_GSW_TIPS_FILE = RAW_DATA_ROOT / "FedGSW" / "feds200805.csv"  # the TIPS (real) curve
REFRESH_DAYS = 7
PUBLICATION_LAG = pd.Timedelta(days=7)
FETCH = fed_gsw_client.fetch_csv  # network hook; tests stub it


def parse(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("Date,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Date": "timestamp"})
    df["known_from"] = df["timestamp"] + PUBLICATION_LAG
    return df.dropna(subset=["BETA0"]).reset_index(drop=True)


def update(*, path: Path = FED_GSW_FILE, now=None, url: str | None = None) -> dict:
    """Refresh the stored file if missing or ``REFRESH_DAYS`` old. Never raises. ``url``:
    the file to fetch (default nominal; ``fed_gsw_client.TIPS_URL`` with
    ``path=FED_GSW_TIPS_FILE`` for the TIPS curve)."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    p = Path(path)
    if p.exists() and now - pd.Timestamp(p.stat().st_mtime, unit="s") < pd.Timedelta(days=REFRESH_DAYS):
        return {"status": "fresh", "error": None}
    try:
        text = FETCH() if url is None else FETCH(url=url)
        parse(text)  # refuse to store something that doesn't parse
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(p)
        return {"status": "fetched", "error": None}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def read_gsw(start=None, end=None, *, as_of=None, path: Path = FED_GSW_FILE) -> pd.DataFrame:
    """Stored GSW rows in ``[start, end]``; with ``as_of``, only rows published by then."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["timestamp"])
    df = parse(Path(path).read_text())
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    if as_of is not None:
        df = df[df["known_from"] <= pd.Timestamp(as_of)]
    return df.reset_index(drop=True)
