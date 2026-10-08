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

**Availability** (``SeriesSource.availability``, ``available_at``, ``read_available``; user
decision 2026-10-05: one global set of rules, here next to the readers): WHEN a value became
public - to the MARKET, not to our pipeline (a settlement is public that evening; our licence
and the morning cycle deliver it later - a live-scheduling concern). A row's ``timestamp``
is a LABEL (the trading day, the period), not when it was known: a condition, a feature or
any point-in-time join must use ``available_at``. Rules sit at the LATE end of the usual
publication window; a store's own publication instant (release vintages) beats a rule;
``verified=False`` marks a deliberately conservative rule whose real time is not verified.
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
class Availability:
    """When a value labelled day D (or instant t) became public.

    ``kind="day"``: D moved ``days`` business days (``calendar``), at ``local_time`` in
    ``timezone``. ``kind="instant"``: the label instant + ``offset``. ``kind="publication"``:
    the store's own publication instant per row (release vintages)."""
    kind: str = "day"
    days: int = 0
    local_time: str = "23:59"
    timezone: str = "America/New_York"
    offset: pd.Timedelta = pd.Timedelta(0)
    calendar: str = "federal"
    verified: bool = True
    note: str = ""


def _settle_avail(key: str) -> Availability:
    """A settlement for trading day D: public by the session's close on D (CME settles at
    14:00 CT, the session closes 16:00 CT - the close is the safe side)."""
    from infra.config import FUTURES_ROOTS, TRADING_HOURS
    spec = parse_relative(key)
    root = spec.root if spec is not None else next((r for r in sorted(FUTURES_ROOTS, key=len, reverse=True)
                                                    if key.startswith(r)), None)
    session = TRADING_HOURS[FUTURES_ROOTS[root].dataset]
    return Availability("day", 0, session.close_time, session.timezone,
                        note=f"settlement by the session close ({session.close_time} {session.timezone})")


def _bond_avail(key: str) -> Availability:
    if key.startswith("UK_"):
        return Availability("day", 1, "12:00", "Europe/London", calendar="market", verified=False,
                            note="BoE curve: next business day noon (conservative; often later - month ends)")
    if key.startswith("DE_"):
        return Availability("day", 1, "09:00", "Europe/Berlin", calendar="market", verified=False,
                            note="Bundesbank Svensson parameters: next morning (conservative; same-day in practice)")
    if key.startswith("JP_"):
        return Availability("day", 1, "09:30", "Asia/Tokyo", calendar="weekday", verified=False,
                            note="MoF: 09:30 Tokyo the next business day (no Japanese holiday calendar)")
    if key.startswith("CA_"):
        return Availability("day", 1, "10:00", "America/New_York", verified=False,
                            note="BoC Valet benchmark yields: next morning (conservative; publication time unverified)")
    return Availability("day", 1, "09:00", "America/New_York", verified=False,
                        note="Treasury par yield curve (CMT): next morning (conservative; published the same evening)")


def _bmk_avail(key: str) -> Availability:
    """bmk P&L of day D is public when its inputs are: per issuer where it differs from the
    generic rule (D+1 10:00 New York - FedInvest's posting, the late one)."""
    src, _, ticker = key.partition(":")
    if src in ("mof", "yield_mof") or ticker.startswith("JP_"):
        return Availability("day", 1, "09:30", "Asia/Tokyo", calendar="weekday", verified=False,
                            note="MoF JGB yields: 09:30 Tokyo the next business day (MoF Q&A); no Japanese holiday "
                                 "calendar - on the day after one the rule is a day early")
    if src.endswith("@LDN1615"):
        if src.startswith("fut"):
            return Availability("day", 0, "16:16", "Europe/London", calendar="weekday",
                                note="futures mids at the 16:15 London snap: known then")
        if src.startswith("ois"):
            return Availability("day", 0, "20:15", "Europe/London", calendar="weekday",
                                note="OIS closes at 16:15 London may use prints up to 4h after (EUR / GBP fallback "
                                     "window): known by 20:15")
        # cash moved by futures: as late as its source curve (bond: rules)
        return _bond_avail(ticker.split("__")[0] if "__" not in ticker else ticker.split("__")[1])
    return Availability("day", 1, "10:00", "America/New_York", calendar="market", verified=False,
                        note="yield P&L: FedInvest-based (otr, curve) posts D+1 ~10:00 New York; CMT is out the same "
                             "evening - one rule, the late one")


def _swap_avail(key: str) -> Availability:
    from infra.config import SWAP_CLOSES
    parts = key.split(":")
    close = parts[2] if len(parts) > 2 else "NY1500"
    spec = SWAP_CLOSES[close]
    return Availability("day", 0, spec.local_time, spec.timezone, offset=pd.Timedelta(minutes=90),
                        note=f"the {close} snap + 90 min (the latest trade a close can use; DTCC disseminates in "
                             f"real time)")


def _repo_avail(key: str) -> Availability:
    if key in ("SOFR", "TGCR", "BGCR"):
        return Availability("day", 1, "08:00", "America/New_York",
                            note="NY Fed: ~08:00 New York on D+1 (root CLAUDE.md 19; may be revised that afternoon)")
    return Availability("day", 2, "12:00", "America/New_York", verified=False,
                        note="OFR / DTCC series: two business days (conservative)")


@dataclass(frozen=True)
class SeriesSource:
    reader: Callable  # (keys, start, end, as_of) -> wide frame, columns = keys
    description: str
    examples: tuple[str, ...] = ()
    point_in_time_index: bool = True  # False: indexed by observation period, not by when it was known
    availability: Availability | Callable[[str], Availability] = Availability()
    kind: str = "level"   # level (yields, settlements, indices) | moves (daily P&L / returns): the feature
                          # maker differences a level and sums moves (infra.processing.features)


def _model_out(keys, start, end, as_of):
    """A stored model run's prediction column: key ``<run>:<column>`` (the column may itself contain
    ``:``, e.g. ``c_rpca_curve:residual:bmk:otr:US_BOND_10y``), indexed by the row's label."""
    from infra.config import MODEL_RUNS_DIR
    from infra.storage import model_runs as store
    out = {}
    for k in keys:
        run, _, col = k.partition(":")
        p = store.read_predictions(run, root=MODEL_RUNS_DIR)
        if p.empty or col not in p:
            out[k] = pd.Series(dtype="float64")
            continue
        sl = p[col].astype("float64")
        sl.index = pd.DatetimeIndex(sl.index).normalize()
        sl = sl[~sl.index.duplicated(keep="last")]
        out[k] = sl[(sl.index >= pd.Timestamp(start)) & (sl.index <= pd.Timestamp(end))]
    return pd.DataFrame(out).reindex(columns=list(keys))


def _feat(keys, start, end, as_of):
    """Any feature-maker expression as a series (``feat:<expr>``, e.g.
    ``feat:vol(otr:US_BOND_10y,20) | lvl``), labelled by the DAY it became available
    (conservative: a value known on D+1 is row D+1, never row D); warm-up read from 1500 days back."""
    from infra.pipeline.features import feature            # features reads this module: import late
    out = {}
    for k in keys:
        f = feature(k, pd.Timestamp(start) - pd.Timedelta(days=1500), pd.Timestamp(end) + _ONE_DAY)
        f = f.dropna()
        f.index = pd.DatetimeIndex(f.index).normalize()
        f = f.groupby(level=0).last()
        out[k] = f[(f.index >= pd.Timestamp(start)) & (f.index <= pd.Timestamp(end))]
    return pd.DataFrame(out).reindex(columns=list(keys))


def kind_of(series_id: str) -> str:
    """``level`` or ``moves`` (the source's declared input kind)."""
    return SERIES_SOURCES[parse_id(series_id)[0]].kind


def _bmk(keys, start, end, as_of):
    """Daily benchmark P&L in bp (+ = a long position made money), from ``Bmk/Pnl``: key
    ``<source>:<ticker or structure>`` - ``curve:US_BOND_10y``,
    ``otr:FLY__US_BOND_5y__US_BOND_7y__US_BOND_10y`` (bmk ``yield_<source>`` for the yield sources
    cmt / otr / curve), ``swsp_cmt:US_SWSP_10y`` (any other bmk name as is: the swap-spread P&L);
    a structure is its legs' P&L x ``infra.reference.structures.yield_structure_weights``."""
    from infra.config import BMK_ROOT
    from infra.reference.structures import yield_structure_weights
    from infra.storage import parquet_store
    out = {}
    by_src: dict[str, list[str]] = {}
    for k in keys:
        src, _, name = k.partition(":")
        by_src.setdefault(src, []).append(name)
    for src, names in by_src.items():
        weights = {n: (yield_structure_weights(n) if "__" in n else {n: 1.0}) for n in names}
        tickers = sorted({t for w in weights.values() for t in w})
        from infra.config import BMK_YIELD_OFFICIAL, BMK_YIELD_SOURCES
        short = set(BMK_YIELD_SOURCES) | set(BMK_YIELD_OFFICIAL.values())
        bmk = f"yield_{src}" if src in short else src     # curve -> yield_curve, boe -> yield_boe; swsp_cmt as is
        raw = parquet_store.read_partitioned(BMK_ROOT / "Pnl", start=pd.Timestamp(start), end=pd.Timestamp(end) + pd.Timedelta(days=1),
                                             equals_in={"bmk": [bmk], "ticker": tickers})
        if raw is None or raw.empty:
            for n in names:
                out[f"{src}:{n}"] = pd.Series(dtype="float64")
            continue
        raw = raw.assign(ticker=raw["ticker"].astype(str))
        legs = raw.pivot_table(index="timestamp", columns="ticker", values="pnl_per_dv01").reindex(columns=tickers)
        for n, w in weights.items():
            out[f"{src}:{n}"] = legs[list(w)].mul(pd.Series(w)).sum(axis=1, min_count=len(w))
    df = pd.DataFrame(out)
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df.reindex(columns=list(keys))


def _pos(keys, start, end, as_of):
    """Stored positioning measures (``infra.pipeline.positioning``): key
    ``<spec>:<measure>:<instrument>``, e.g. ``ust_daily:rel_asym:otr:US_BOND_2y``."""
    from infra.pipeline.positioning import read_asymmetry
    out = {}
    for k in keys:
        spec, measure, instrument = k.split(":", 2)
        rows = read_asymmetry(spec, measure, instrument, start, end)
        out[k] = pd.Series(rows["value"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(rows["timestamp"]))
    return pd.DataFrame(out).reindex(columns=list(keys))


SERIES_SOURCES: dict[str, SeriesSource] = {
    "fut": SeriesSource(_fut, "back-adjusted continuous futures settlement",
                        ("fut:ZN.v.0", "fut:ZF.v.0", "fut:ZT.v.0", "fut:TN.v.0", "fut:ZB.v.0", "fut:UB.v.0"),
                        availability=_settle_avail),
    "settle": SeriesSource(_settle, "raw daily settlement (roll jumps kept)", ("settle:ZQ.c.1",),
                           availability=_settle_avail),
    "stir": SeriesSource(_stir, "STIR implied rate, 100 - settlement (%)",
                         tuple(f"stir:SR3.c.{i}" for i in range(12)) + tuple(f"stir:ESR.c.{i}" for i in range(4)),
                         availability=_settle_avail),
    "bond": SeriesSource(_bond("cmt"), "par yield (CMT / BoE / Bundesbank), %",
                         tuple(f"bond:{c}_BOND_{t}y" for c in ("US", "UK", "DE") for t in (2, 3, 5, 7, 10, 20, 30)),
                         availability=_bond_avail),
    "otr": SeriesSource(_bond("otr"), "on-the-run Treasury END OF DAY yield, %",
                        tuple(f"otr:US_BOND_{t}y" for t in (2, 3, 5, 7, 10, 20, 30)),
                        availability=Availability("day", 1, "10:00", "America/New_York",
                                                  note="FedInvest END OF DAY: posted D+1 06:00-~10:00 New York "
                                                       "(root CLAUDE.md 18)")),
    "bmk": SeriesSource(_bmk, "daily benchmark P&L in bp of a long position (bmk yield_<source>; yield structures too)",
                        ("bmk:curve:US_BOND_10y", "bmk:otr:FLY__US_BOND_5y__US_BOND_7y__US_BOND_10y"),
                        availability=_bmk_avail, kind="moves"),
    "swap": SeriesSource(_swap, "swap close (currency:tenor[:close[:method]]), %",
                         tuple(f"swap:USD:{t}y" for t in (1, 2, 3, 5, 7, 10, 15, 20, 30)), availability=_swap_avail),
    "repo": SeriesSource(_repo, "repo rate, %", ("repo:SOFR", "repo:TGCR", "repo:BGCR"), availability=_repo_avail),
    "release": SeriesSource(_release, "macro series as published by as_of (indexed by period)",
                            ("release:PAYEMS", "release:UNRATE", "release:CPIAUCSL"), point_in_time_index=False,
                            availability=Availability("publication", note="each vintage at its publication day, "
                                                      "at the release's registry time")),
    "feat": SeriesSource(_feat, "any feature-maker expression (infra.pipeline.features), labelled by its availability day",
                         ("feat:vol(otr:US_BOND_10y,20) | lvl",),
                         availability=Availability("day", 0, "00:00", "America/New_York", calendar="market",
                                                   verified=False, note="the label IS the availability day"),
                         kind="level"),
    "model": SeriesSource(_model_out, "a stored model run's prediction column (model:<run>:<column>), e.g. a PCA "
                          "residual = that residual portfolio's daily P&L with the fit's frozen weights",
                          ("model:c_rpca_curve:residual:bmk:otr:US_BOND_10y",),
                          availability=Availability("day", 1, "10:00", "America/New_York", calendar="market",
                                                    verified=False,
                                                    note="row D is computed from inputs known by D+1 10:00 New York "
                                                         "(the bmk yield P&L); the late one of the usual inputs"),
                          kind="moves"),
    "pos": SeriesSource(_pos, "positioning measure (spec:measure:instrument), e.g. asymmetric reaction",
                        ("pos:ust_daily:rel_asym:otr:US_BOND_2y", "pos:ust_daily:semivar_asym:factor"),
                        availability=Availability("day", 1, "10:00", "America/New_York", calendar="market",
                                                  verified=False,
                                                  note="day D's value uses moves through D: the late one of its "
                                                       "inputs (FedInvest END OF DAY, D+1 ~10:00 New York)"),
                        kind="level"),
    "bar": SeriesSource(_bar, "1-minute trade-bar close (UTC)", ("bar:ZN.v.0",),
                        availability=Availability("instant", offset=pd.Timedelta(minutes=1),
                                                  note="a bar is stamped at its START: known one minute later")),
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


# --------------------------------------------------------------------------- availability
def availability_of(series_id: str) -> Availability:
    source, key = parse_id(series_id)
    rule = SERIES_SOURCES[source].availability
    return rule(key) if callable(rule) else rule


def available_at(series_id: str, labels) -> pd.DatetimeIndex:
    """UTC instant each labelled value became public (``Availability``); not for
    ``kind="publication"`` sources (use ``read_available``)."""
    from infra.processing.schedule_rules import business_days
    from infra.trading_calendar import snap_instants
    rule = availability_of(series_id)
    labels = pd.DatetimeIndex(pd.to_datetime(labels))
    if rule.kind == "instant":
        return labels + rule.offset
    if rule.kind != "day":
        raise ValueError(f"{series_id}: availability {rule.kind!r} needs read_available")
    days = labels.normalize()
    if rule.days:
        bd = business_days(days.min() - pd.Timedelta(days=10), days.max() + pd.Timedelta(days=10 + 2 * rule.days),
                           rule.calendar)
        pos = bd.searchsorted(days, side="right") - 1 + rule.days  # from the day (or the business day before it)
        days = bd[pos]
    return snap_instants(days, rule.local_time, rule.timezone) + rule.offset


def _release_available(key: str, start, end) -> pd.DataFrame:
    """Every vintage of a macro series with its publication instant: the publication day at
    the release's registry time (the event of the registry series stored under this id), else
    the end of the publication day. ``label`` = the period."""
    from infra.pipeline.releases import read_releases_from_disk
    from infra.reference.events import EVENTS, SERIES
    from infra.trading_calendar import snap_instants
    raw = read_releases_from_disk([key])
    raw = raw[(raw["timestamp"] >= pd.Timestamp(start) - pd.DateOffset(years=2)) & (raw["timestamp"] <= pd.Timestamp(end))]
    ev = next((EVENTS[s_.event] for s_ in SERIES.values() if s_.store_id == key), None)
    t = ev.time_local if ev is not None and ev.time_local else "23:59"
    tz = ev.timezone if ev is not None else "America/New_York"
    pub = snap_instants(pd.DatetimeIndex(raw["timestamp"]).normalize(), t, tz)
    return pd.DataFrame({"label": pd.DatetimeIndex(raw["period"]), "available_at": pub,
                         "value": raw["value"].to_numpy(dtype="float64")})


def read_available(series_id: str, start, end) -> pd.DataFrame:
    """One series as a point-in-time timeline: ``label`` (the row's day / period / instant),
    ``available_at`` (UTC, when it became public) and ``value``, sorted by availability. For a
    vintage series, every vintage is a row (a revision is a later row for the same label)."""
    source, key = parse_id(series_id)
    if availability_of(series_id).kind == "publication":
        out = _release_available(key, start, end)
    else:
        wide = read_panel([series_id], start, end)
        s = wide[series_id].dropna() if series_id in wide else pd.Series(dtype="float64")
        out = pd.DataFrame({"label": s.index, "available_at": available_at(series_id, s.index) if len(s) else
                            pd.DatetimeIndex([]), "value": s.to_numpy(dtype="float64")})
    return out.sort_values(["available_at", "label"], kind="stable").reset_index(drop=True)
