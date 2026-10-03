"""Any stored series, by one uniform id, as a wide ``timestamp x id`` frame. Disk only.

The statistical models (``infra/models/stats``) and the dashboard's model page take any
wide frame; this module is the convenience of building one from the database without
knowing which store each series lives in. A series id is ``<source>:<key>``:

=================  ========================================================  ==============
id                 what                                                      index
=================  ========================================================  ==============
``fut:ZN.v.0``     back-adjusted continuous daily settlement of a relative   trading day
                   futures ticker (each day's change on the contract held
                   that day - a roll gap is never a move)
``settle:SR3.c.4`` raw daily settlement of a relative or absolute contract   trading day
                   (roll jumps kept: a constant-rank level)
``stir:SR3.c.4``   ``100 - settlement``: a STIR contract's implied rate, %   trading day
``bond:US_BOND_10y``  CMT / BoE / Bundesbank par yield, %                    day
``otr:US_BOND_10y``   on-the-run END OF DAY yield, % (section 18)            day
``swap:USD:10y``   swap close (default ``NY1500``, ``pure``; also            day (the snap's
                   ``swap:USD:10y:NY1530:adjusted``), %                      date)
``repo:SOFR``      repo rate (``read_repo`` series name, best status), %     day
``release:PAYEMS`` a macro series as published by ``as_of`` (snapshot),      observation
                   in the source's units                                     period
``bar:ZN.v.0``     1-minute trade-bar close (``load_series``, no fetch)      minute (UTC)
=================  ========================================================  ==============

Everything stays tz-naive UTC (root CLAUDE.md 7). ``as_of`` is the point-in-time cutoff
(root CLAUDE.md 3): rows after it are never returned, and ``release:`` reads the vintage
published by then. Missing history must be loaded through the pipeline first; nothing
here fetches.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from infra.pipeline import daily as dl
from infra.pipeline.relative_daily import load_relative_daily
from infra.processing import continuous
from infra.relative.symbology import parse_relative

log = logging.getLogger(__name__)

_ONE_DAY = pd.Timedelta(days=1)
# Absolute settlements are read this far before ``start`` so the first roll has its prior.
_PRIOR_DAYS = pd.Timedelta(days=10)


# --------------------------------------------------------------------------- futures
def continuous_futures(tickers: list[str], start, end) -> pd.DataFrame:
    """Back-adjusted continuous settlement prices, wide ``timestamp x ticker``, over
    ``[start, end]`` (inclusive days) for relative futures tickers. Disk only."""
    specs = []
    for t in tickers:
        spec = parse_relative(t)
        if spec is None:
            raise ValueError(f"{t!r} is not a relative futures ticker (e.g. ZN.v.0)")
        specs.append(spec)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rel = load_relative_daily(specs, start, end + _ONE_DAY, fetch_missing=False)
    if rel.empty:
        return pd.DataFrame(columns=tickers, dtype="float64")
    contracts = sorted(rel["contract"].astype(str).unique())
    absolute = dl.read_daily_from_disk(contracts, start - _PRIOR_DAYS, end + _ONE_DAY)
    changes = continuous.same_contract_changes(rel, absolute)
    unknown = continuous.unknown_changes(changes)
    if len(unknown):
        log.warning("continuous_futures: %d day(s) with no same-contract prior settlement (counted as no "
                    "move): %s", len(unknown),
                    ", ".join(f"{r.ticker} {r.timestamp.date()}" for r in unknown.head(10).itertuples()))
    return continuous.back_adjusted(changes).reindex(columns=tickers)


def raw_settlements(tickers: list[str], start, end) -> pd.DataFrame:
    """Daily settlement as stored, wide, for relative and/or absolute tickers over
    ``[start, end]`` (inclusive days). A relative ticker keeps its roll jumps."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    rel_specs = [parse_relative(t) for t in tickers if parse_relative(t) is not None]
    absolute = [t for t in tickers if parse_relative(t) is None]
    if rel_specs:
        rel = load_relative_daily(rel_specs, start, end + _ONE_DAY, fetch_missing=False)
        if not rel.empty:
            frames.append(rel[["timestamp", "ticker", "settlement_price"]])
    if absolute:
        ab = dl.read_daily_from_disk(absolute, start, end + _ONE_DAY)
        if not ab.empty:
            frames.append(ab[["timestamp", "ticker", "settlement_price"]])
    if not frames:
        return pd.DataFrame(columns=tickers, dtype="float64")
    long = pd.concat(frames, ignore_index=True)
    long["ticker"] = long["ticker"].astype(str)
    wide = long.pivot_table(index="timestamp", columns="ticker", values="settlement_price", aggfunc="last")
    return wide.astype("float64").reindex(columns=tickers)


# --------------------------------------------------------------------------- readers
def _fut(keys, start, end, as_of):
    return continuous_futures(keys, start, end)


def _settle(keys, start, end, as_of):
    return raw_settlements(keys, start, end)


def _stir(keys, start, end, as_of):
    return 100.0 - raw_settlements(keys, start, end)


def _long_to_wide(df: pd.DataFrame, key_col: str, value_col: str, keys) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(keys), dtype="float64")
    wide = df.pivot_table(index="timestamp", columns=key_col, values=value_col, aggfunc="last")
    wide.columns = wide.columns.astype(str)
    return wide.astype("float64").reindex(columns=list(keys))


def _bond(source: str):
    def read(keys, start, end, as_of):
        from infra.pipeline.bond_yields import read_bond_yields
        df = read_bond_yields(keys, start, pd.Timestamp(end) + _ONE_DAY, source=source)
        return _long_to_wide(df, "ticker", "yield", keys)
    return read


def _swap(keys, start, end, as_of):
    from infra.pipeline.swap_closes import read_swap_closes
    out = {}
    for key in keys:
        parts = key.split(":")
        if len(parts) < 2:
            raise ValueError(f"swap id needs currency and tenor, e.g. swap:USD:10y (got {key!r})")
        currency, tenor = parts[0], int(parts[1].rstrip("yY"))
        close = parts[2] if len(parts) > 2 else "NY1500"
        method = parts[3] if len(parts) > 3 else "pure"
        df = read_swap_closes(start, pd.Timestamp(end) + _ONE_DAY, close=close, currency=currency, method=method)
        df = df[df["tenor"] == tenor]
        # one close a day: dated by the snap's date (the instant itself is in the store)
        out[key] = pd.Series(df["rate"].to_numpy(dtype="float64"),
                             index=pd.DatetimeIndex(df["timestamp"]).normalize()).groupby(level=0).last()
    return pd.DataFrame(out).reindex(columns=list(keys))


def _repo(keys, start, end, as_of):
    from infra.pipeline.repo import read_repo
    df = read_repo(start, pd.Timestamp(end) + _ONE_DAY, series=list(keys))
    return _long_to_wide(df, "series", "rate", keys)


def _release(keys, start, end, as_of):
    from infra.pipeline.releases import read_releases_from_disk
    from infra.processing.releases import snapshot
    raw = read_releases_from_disk(list(keys), as_of=as_of)
    snap = snapshot(raw, as_of)
    snap = snap[(snap["period"] >= pd.Timestamp(start)) & (snap["period"] <= pd.Timestamp(end))]
    snap = snap.rename(columns={"timestamp": "published"}).rename(columns={"period": "timestamp"})
    return _long_to_wide(snap, "ticker", "value", keys)


def _bar(keys, start, end, as_of):
    from infra.pipeline.series import load_series
    df = load_series(list(keys), start, pd.Timestamp(end) + _ONE_DAY, fetch_missing=False)
    if df.empty:
        return pd.DataFrame(columns=list(keys), dtype="float64")
    df = df.reset_index() if "timestamp" not in df.columns else df
    df["ticker"] = df["ticker"].astype(str)
    return _long_to_wide(df, "ticker", "close", keys)


@dataclass(frozen=True)
class SeriesSource:
    reader: Callable  # (keys, start, end, as_of) -> wide frame, columns = keys
    description: str
    examples: tuple[str, ...] = ()
    point_in_time_index: bool = True  # False: indexed by observation period, not by when it was known


SERIES_SOURCES: dict[str, SeriesSource] = {
    "fut": SeriesSource(_fut, "back-adjusted continuous futures settlement",
                        ("fut:ZN.v.0", "fut:ZF.v.0", "fut:ZT.v.0", "fut:TN.v.0", "fut:ZB.v.0", "fut:UB.v.0")),
    "settle": SeriesSource(_settle, "raw daily settlement (roll jumps kept)", ("settle:ZQ.c.1",)),
    "stir": SeriesSource(_stir, "STIR implied rate, 100 - settlement (%)",
                         tuple(f"stir:SR3.c.{i}" for i in range(12)) + tuple(f"stir:ESR.c.{i}" for i in range(4))),
    "bond": SeriesSource(_bond("cmt"), "par yield (CMT / BoE / Bundesbank), %",
                         tuple(f"bond:{c}_BOND_{t}y" for c in ("US", "UK", "DE") for t in (2, 3, 5, 7, 10, 20, 30))),
    "otr": SeriesSource(_bond("otr"), "on-the-run Treasury END OF DAY yield, %",
                        tuple(f"otr:US_BOND_{t}y" for t in (2, 3, 5, 7, 10, 20, 30))),
    "swap": SeriesSource(_swap, "swap close (currency:tenor[:close[:method]]), %",
                         tuple(f"swap:USD:{t}y" for t in (1, 2, 3, 5, 7, 10, 15, 20, 30))),
    "repo": SeriesSource(_repo, "repo rate, %", ("repo:SOFR", "repo:TGCR", "repo:BGCR")),
    "release": SeriesSource(_release, "macro series as published by as_of (indexed by period)",
                            ("release:PAYEMS", "release:UNRATE", "release:CPIAUCSL"), point_in_time_index=False),
    "bar": SeriesSource(_bar, "1-minute trade-bar close (UTC)", ("bar:ZN.v.0",)),
}


def parse_id(series_id: str) -> tuple[str, str]:
    source, sep, key = series_id.partition(":")
    if not sep or source not in SERIES_SOURCES or not key:
        raise ValueError(f"bad series id {series_id!r}: expected <source>:<key>, source one of {sorted(SERIES_SOURCES)}")
    return source, key


def read_panel(series_ids: list[str], start, end, *, as_of=None, how: str = "outer") -> pd.DataFrame:
    """Wide ``timestamp x series_id`` (float) over ``[start, end]`` (inclusive days),
    nothing after ``as_of``. ``how="outer"`` keeps every timestamp any series has (NaN
    where a market was closed - for a model to handle explicitly); ``"inner"`` keeps only
    timestamps where every series has a value."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if as_of is not None:
        end = min(end, pd.Timestamp(as_of))
    by_source: dict[str, list[str]] = {}
    for sid in dict.fromkeys(series_ids):
        source, key = parse_id(sid)
        by_source.setdefault(source, []).append(key)
    frames = []
    for source, keys in by_source.items():
        wide = SERIES_SOURCES[source].reader(keys, start, end, as_of)
        wide = wide.rename(columns={k: f"{source}:{k}" for k in keys})
        frames.append(wide)
    out = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    out.index = pd.DatetimeIndex(out.index, name="timestamp")
    out = out[(out.index >= start) & (out.index < end.normalize() + _ONE_DAY)]
    out = out.reindex(columns=list(dict.fromkeys(series_ids))).astype("float64")
    if how == "inner":
        out = out.dropna(how="any")
    elif how != "outer":
        raise ValueError(f"how {how!r} (outer | inner)")
    return out.dropna(how="all")


def example_ids() -> list[str]:
    """Every source's example ids (for pickers)."""
    return [e for s in SERIES_SOURCES.values() for e in s.examples]
