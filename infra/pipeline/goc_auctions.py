"""Government of Canada auctions, outstanding securities and the bond auction schedule -
fetch / store / point-in-time plan (root CLAUDE.md 17; pure parsing
``infra.processing.goc_auctions``, client ``infra.api.boc_client``).

* Auctions ``CA_AUCTIONS_DIR`` (keys ``timestamp``, ``isin``, ``kind``): the Bank of Canada's
  Valet results groups, re-read whole each run (free, small) and upserted.
* Outstanding ``CA_OUTSTANDING_DIR`` (keys ``timestamp`` = as-of day, ``isin``): every bill
  and bond outstanding, daily from 2025, plus the monthly bond CSVs of 2018-01..2022-01
  (``backfill_dmb``; ISINs by maturity + coupon, ``SERIES:<code>`` for 5 bonds issued before
  1998 with no ISIN anywhere) - with the auctions, ``read_securities(as_of)`` is the universe
  point in time.
* Plan ``CA_AUCTION_PLAN_DIR``: the quarterly bond auction schedule, archived whenever it
  changes (Valet, daily) plus the Wayback Machine's captures of its page (2022 on). Replayed
  like the German plan: each snapshot replaces the schedule from its own day on. T-bills
  aren't scheduled there (a fixed bi-weekly cycle) - their dates come from the held auctions.
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import pandas as pd

from infra.api import boc_client as bc
from infra.config import CA_AUCTION_PLAN_DIR, CA_AUCTIONS_DIR, CA_DMB_FILES_DIR, CA_OUTSTANDING_DIR
from infra.processing import goc_auctions as ga
from infra.storage import parquet_store

log = logging.getLogger(__name__)
PLAN_SOURCE, HELD_SOURCE, CFT_SOURCE = "ca_plan", "ca_auctions", "ca_cft"
KEYS = ["timestamp", "isin", "kind"]
OUT_KEYS = ["timestamp", "isin"]
STAMP = "%Y%m%dT%H%M%SZ"
VINTAGE_COLUMNS = ga.PLAN_COLUMNS + ["vintage", "known_from"]
FETCH_GROUP = bc.fetch_valet_group   # network hooks; tests stub them
FETCH_DMB = bc.fetch_dmb_csv
_ONE_DAY = pd.Timedelta(days=1)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


# ------------------------------------------------------------------ fetch / store
def update(*, auctions_root: Path = CA_AUCTIONS_DIR, outstanding_root: Path = CA_OUTSTANDING_DIR,
           plan_root: Path = CA_AUCTION_PLAN_DIR, with_2025: bool = False, now=None,
           errors: dict | None = None) -> dict:
    """The daily refresh: every results group, the current outstanding group (``with_2025``
    also the 2025 one - a one-off), and the schedule (archived if it changed). Each group fails
    alone (``errors``)."""
    errors = {} if errors is None else errors
    now = _now() if now is None else pd.Timestamp(now)
    frames = []
    for kind, (group, _) in ga.GROUPS.items():
        try:
            frames.append(ga.parse_results(FETCH_GROUP(group), kind))
        except Exception as exc:
            errors[f"ca_{kind.lower()}"] = f"{type(exc).__name__}: {exc}"
    n_auc = store_auctions(pd.concat(frames, ignore_index=True), root=auctions_root) if frames else 0
    n_out = 0
    for group, suffix in ga.OUTSTANDING_GROUPS.items():
        if suffix and not with_2025:
            continue
        try:
            n_out += store_outstanding(ga.parse_outstanding(FETCH_GROUP(group), suffix), root=outstanding_root)
        except Exception as exc:
            errors[f"ca_{group.lower()}"] = f"{type(exc).__name__}: {exc}"
    changed = False
    try:
        payload = FETCH_GROUP("AUC_SCHED")
        rows = ga.parse_schedule_valet(payload)
        prev = _latest_snapshot(plan_root)
        changed = bool(len(rows)) and (prev is None or not _same(rows, prev))
        if changed:
            _write(Path(plan_root) / f"{now.strftime(STAMP)}__valet.json", json.dumps(payload).encode("utf-8"))
    except Exception as exc:
        errors["ca_schedule"] = f"{type(exc).__name__}: {exc}"
    return {"auction_rows": n_auc, "outstanding_rows": n_out, "schedule_changed": changed}


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    cols = ["auction_date", "kind", "term_years", "maturity_date"]
    return a[cols].reset_index(drop=True).equals(b[cols].reset_index(drop=True))


def store_auctions(df: pd.DataFrame, *, root: Path = CA_AUCTIONS_DIR) -> int:
    if df.empty:
        return 0
    df = df.drop_duplicates(KEYS, keep="last")
    out = df.assign(**{c: pd.to_datetime(df[c]).astype("datetime64[ms]") for c in ("timestamp", "issue_date", "maturity_date")},
                    **{c: df[c].astype("string") for c in ("isin", "kind", "bid_deadline", "off_the_run")})
    parquet_store.write_partitioned(out, root, KEYS)
    return len(out)


def store_outstanding(df: pd.DataFrame, *, root: Path = CA_OUTSTANDING_DIR) -> int:
    if df.empty:
        return 0
    out = df.drop_duplicates(OUT_KEYS, keep="last")
    out = out.assign(**{c: pd.to_datetime(out[c]).astype("datetime64[ms]") for c in ("timestamp", "issue_date", "maturity_date")},
                     **{c: out[c].astype("string") for c in ("isin", "security_type", "instrument_type", "series")})
    parquet_store.write_partitioned(out, root, OUT_KEYS)
    return len(out)


def read_auctions(start=None, end=None, *, root: Path = CA_AUCTIONS_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end)) \
        if parquet_store.has_data(root) else None
    if df is None or df.empty:
        return pd.DataFrame(columns=ga.AUCTION_COLUMNS)
    for c in ("isin", "kind", "bid_deadline", "off_the_run"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df[ga.AUCTION_COLUMNS].sort_values(KEYS).reset_index(drop=True)


def read_outstanding(start=None, end=None, *, root: Path = CA_OUTSTANDING_DIR) -> pd.DataFrame:
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end)) \
        if parquet_store.has_data(root) else None
    if df is None or df.empty:
        return pd.DataFrame(columns=ga.OUTSTANDING_COLUMNS)
    if "series" not in df:                      # files written before the column existed
        df["series"] = None
    for c in ("isin", "security_type", "instrument_type"):
        df[c] = df[c].astype(str)
    return df[ga.OUTSTANDING_COLUMNS].sort_values(OUT_KEYS).reset_index(drop=True)


def isin_lookup(*, auctions_root: Path = CA_AUCTIONS_DIR, outstanding_root: Path = CA_OUTSTANDING_DIR) -> dict:
    """(maturity, coupon) -> ISIN over the auctions and the daily snapshots; raises if a key is
    ambiguous (two ISINs), which would mislabel a monthly-CSV bond."""
    a = read_auctions(root=auctions_root)
    o = read_outstanding(root=outstanding_root)
    o = o[~o["isin"].str.startswith("SERIES:")]
    both = pd.concat([a[["maturity_date", "coupon", "isin"]], o[["maturity_date", "coupon", "isin"]]]).dropna()
    both = both.assign(coupon=both["coupon"].round(4)).drop_duplicates()
    dup = both.duplicated(["maturity_date", "coupon"], keep=False)
    if dup.any():
        raise ValueError(f"ambiguous (maturity, coupon) -> ISIN: {both[dup].head(4).to_dict('records')}")
    return {(pd.Timestamp(m), c): i for m, c, i in zip(both["maturity_date"], both["coupon"], both["isin"])}


def backfill_dmb(*, files_root: Path = CA_DMB_FILES_DIR, auctions_root: Path = CA_AUCTIONS_DIR,
                 outstanding_root: Path = CA_OUTSTANDING_DIR) -> int:
    """One-off: the monthly outstanding CSVs (2018-01..2022-01), each fetched once (a file on
    disk is its coverage), loaded into the outstanding store with ISINs where known."""
    files_root = Path(files_root)
    files_root.mkdir(parents=True, exist_ok=True)
    for ym in bc.DMB_MONTHS.strftime("%Y-%m"):
        path = files_root / f"{ym}.csv"
        if not path.exists():
            text = FETCH_DMB(ym)
            if "Period End Date" in text:
                path.write_text(text, "utf-8")
    lookup = isin_lookup(auctions_root=auctions_root, outstanding_root=outstanding_root)
    frames = [ga.parse_dmb_csv(p.read_text("utf-8"), lookup) for p in sorted(files_root.glob("*.csv"))]
    frames = [f for f in frames if len(f)]
    return store_outstanding(pd.concat(frames, ignore_index=True), root=outstanding_root) if frames else 0


def read_securities(as_of=None, *, auctions_root: Path = CA_AUCTIONS_DIR,
                    outstanding_root: Path = CA_OUTSTANDING_DIR) -> pd.DataFrame:
    """The universe: one row per ISIN auctioned (or seen outstanding) by ``as_of``."""
    hi = None if as_of is None else pd.Timestamp(as_of).normalize() + _ONE_DAY
    return ga.securities(read_auctions(None, hi, root=auctions_root), read_outstanding(None, hi, root=outstanding_root))


# ------------------------------------------------------------------ plan
def backfill_wayback(*, root: Path = CA_AUCTION_PLAN_DIR) -> int:
    """One-off history: the Wayback Machine's captures of the schedule page, each archived at
    its capture time."""
    from infra.api import wayback_client as wb
    have = {p.name.split("__")[0] for p in Path(root).glob("*")} if Path(root).exists() else set()
    caps = wb.list_captures(bc.SCHEDULE_PAGE.split("://")[1], pd.Timestamp("2015-01-01"), _now())
    n = 0
    for cap in caps.drop_duplicates("digest").itertuples():
        stamp = pd.Timestamp(cap.timestamp).strftime(STAMP)
        if stamp in have:
            continue
        body = wb.fetch_capture(cap.timestamp, cap.original)
        if body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        if ga.parse_schedule_page(body.decode("utf-8", "replace")).empty:
            log.info("wayback capture %s: no schedule table, skipped", stamp)
            continue
        _write(Path(root) / f"{stamp}__wayback.html", body)
        n += 1
    return n


def _parse(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return ga.parse_schedule_valet(json.loads(path.read_text("utf-8")))
    return ga.parse_schedule_page(path.read_text("utf-8", "replace"))


def _latest_snapshot(root: Path) -> pd.DataFrame | None:
    files = sorted(p for p in Path(root).glob("*__*") if p.suffix in (".json", ".html")) if Path(root).exists() else []
    return _parse(files[-1]) if files else None


def vintages(*, root: Path = CA_AUCTION_PLAN_DIR) -> pd.DataFrame:
    files = sorted(p for p in Path(root).glob("*__*") if p.suffix in (".json", ".html")) if Path(root).exists() else []
    frames = []
    for p in files:
        df = _parse(p)
        if len(df):
            frames.append(df.assign(vintage=p.name, known_from=pd.Timestamp(p.name.split("__")[0].rstrip("Z"))))
    return pd.concat(frames, ignore_index=True)[VINTAGE_COLUMNS] if frames else pd.DataFrame(columns=VINTAGE_COLUMNS)


def replay(v: pd.DataFrame):
    """``(known_from, plan)`` after each snapshot: it replaces the schedule from its own day."""
    plan = pd.DataFrame(columns=v.columns)
    for _, g in v.groupby("vintage", sort=False):
        t = g["known_from"].iloc[0]
        plan = pd.concat([plan[plan["auction_date"] < t.normalize()], g[g["auction_date"] >= t.normalize()]],
                         ignore_index=True)
        yield t, plan


def plan_state(as_of=None, *, root: Path = CA_AUCTION_PLAN_DIR) -> pd.DataFrame:
    as_of = _now() if as_of is None else pd.Timestamp(as_of)
    v = vintages(root=root)
    state = pd.DataFrame(columns=VINTAGE_COLUMNS)
    for _, state in replay(v[v["known_from"] <= as_of]):
        pass
    return state[state["auction_date"] >= as_of.normalize()].sort_values("auction_date").reset_index(drop=True)


def release_rows(*, observed=None, root: Path = CA_AUCTION_PLAN_DIR, auctions_root: Path = CA_AUCTIONS_DIR) -> pd.DataFrame:
    """Release-calendar rows: the schedule replayed (``ca_plan``, at the registry's 12:00
    Ottawa), the held auctions (``ca_auctions``, at each auction's own bidding deadline; stage
    new_issue / reopening by the ISIN's first auction, known from their own day) and the same
    bond auctions known from their call for tenders (``ca_cft``: 2 business days before, the
    shortest lead observed - so every auction since 1998 is known ahead, if only just)."""
    from infra.processing import release_calendar as rc
    from infra.reference.events import EVENTS
    observed = _now().normalize() if observed is None else pd.Timestamp(observed)
    frames = []
    for t, plan in replay(vintages(root=root)):
        ev = [ga.event_id(k, y) for k, y in zip(plan["kind"], plan["term_years"])]
        for e, g in plan.assign(event=ev).groupby("event"):
            frames.append(rc.event_rows(EVENTS[e], g["auction_date"], source=PLAN_SOURCE, known_from=t.normalize(),
                                        observed=t.normalize()))
    a = read_auctions(root=auctions_root)
    if len(a):
        first = a.groupby("isin")["timestamp"].transform("min")
        a = a.assign(event=[ga.event_id(k, y) for k, y in zip(a["kind"], a["term_years"])],
                     stage=(a["timestamp"] == first).map({True: "new_issue", False: "reopening"}),
                     day=pd.to_datetime(a["timestamp"]).dt.normalize())
        for (e, st), g in a.groupby(["event", "stage"]):
            g = g.drop_duplicates("day")
            frames.append(rc.event_rows(EVENTS[e], g["day"], source=HELD_SOURCE, known_from=g["day"], observed=observed,
                                        stage=st, times=list(g["bid_deadline"])))
            if e != "CA_AUCTION_TBILL":           # bonds: known by their call for tenders (ga.cft_known_from)
                frames.append(rc.event_rows(EVENTS[e], g["day"], source=CFT_SOURCE, known_from=ga.cft_known_from(g["day"]),
                                            observed=observed, stage=st, times=list(g["bid_deadline"])))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return rc.empty()
    return rc.merge_observation(frames[0], pd.concat(frames[1:], ignore_index=True)) if len(frames) > 1 else frames[0]
