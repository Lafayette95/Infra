"""SYNCHRONIZED benchmark P&L, part of the ``bmk_pnl`` step (root CLAUDE.md 12; config
``BMK_SYNC_*``): every series marked at the one instant ``BMK_SYNC_SNAP`` (16:15 London), so a
cross-issuer strategy compares the same 24 hours. Stored in ``Bmk/Pnl`` next to the other
bmks, names ending ``@LDN1615``; pure maths ``infra.processing.sync_pnl``. Local only:

* ``fut@LDN1615`` - ``FUT_<root>``: the bbo-1m mid at the snap of the contract held over the
  interval (``infra.pipeline.swap_hedge.hedge_contracts``: US ``.v.0``, Eurex / Long Gilt the
  most-quoted contract); ``pnl`` per 1 contract, ``pnl_per_dv01`` per the previous day's DV01
  (US / Eurex: the CTD futures DV01 in ``Bmk/Risk``; Long Gilt: 1 / |the swap-hedge regression
  ratio| - an empirical DV01, until gilt prices give a CTD).
* ``ois@LDN1615`` - ``<CCY>_OIS_<t>y``: -(change of the 16:15 London par OIS close, adjusted
  where present) x 100.
* ``yield_<src>@LDN1615`` - US CMT (15:30 New York) and our German curve (11:15 Frankfurt,
  ``yield_curve@LDN1615`` on ``DE_BOND_<t>y``) yields moved to the snap by the hedge future's move x the swap-hedge ratio; UK's BoE curve
  is already at 16:15 London (copied as is, for a complete family).
* Canada (no intraday Canadian futures): ``yield_boc@LDN1615`` on ``CA_BOND_<t>y`` (the BoC's
  16:00 Toronto yields) and ``fut@LDN1615`` on ``FUT_CGB`` (the MX settlement, ~15:00 Toronto),
  each moved by its US hedge future's quote move x a point-in-time ratio (``_canada``); the
  CGB's ``pnl_per_dv01`` uses an empirical DV01 (its settlement change on the BoC 10y change).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from infra.config import (BMK_SYNC_CA_CASH, BMK_SYNC_CA_FUTURE, BMK_SYNC_CA_WINDOW, BMK_SYNC_CASH, BMK_SYNC_FUTURES,
                          BMK_SYNC_SNAP, BMK_SYNC_SWAP_TENORS, BMK_YIELD_MAX_GAP_DAYS, BMK_YIELD_OFFICIAL_TENORS,
                          FUTURES_ROOTS, MX_POINT_VALUE, SWAP_CLOSES, SWAP_HEDGES)
from infra.cycle.core import StepContext
from infra.cycle.paths import CyclePaths
from infra.processing import sync_pnl as sp
from infra.processing import yield_pnl as yp
from infra.storage import parquet_store

_LOOKBACK = pd.Timedelta(days=14)
_ONE_DAY = pd.Timedelta(days=1)
PNL_KEYS = ["timestamp", "ticker", "bmk"]


def _snap():
    s = SWAP_CLOSES[BMK_SYNC_SNAP]
    return s.local_time, s.timezone


def _bbo_root(paths: CyclePaths):
    from infra.cycle.intraday import _ipaths
    return _ipaths(paths).bbo_dir


def _futures(lo, hi, paths: CyclePaths, book) -> list[pd.DataFrame]:
    from infra.pipeline.swap_hedge import hedge_contracts, snap_mids
    held = hedge_contracts(lo, hi, roots=list(BMK_SYNC_FUTURES), daily_root=paths.daily_futures_dir,
                           contracts_file=paths.contracts_file, bbo_root=_bbo_root(paths))
    risk = parquet_store.read_partitioned(paths.bmk_root / "Risk", start=lo, end=hi + _ONE_DAY,
                                          equals_in={"root": list(BMK_SYNC_FUTURES)}) \
        if parquet_store.has_data(paths.bmk_root / "Risk") else None
    out = []
    for root in BMK_SYNC_FUTURES:
        h = held.get(root)
        if h is None or h.empty:
            continue
        marks = snap_mids(sorted(set(h.dropna())), lo, hi + _ONE_DAY, *_snap(), bbo_root=_bbo_root(paths))
        if marks.empty:
            continue
        cfg = FUTURES_ROOTS[root]
        if root == "R":    # no CTD for the Long Gilt yet: the regression DV01 (points per bp)
            ratio = book.ratio.get(root, pd.Series(dtype=float))
            dv = pd.DataFrame({c: (1.0 / ratio.abs()).reindex(marks.index, method="ffill") for c in marks.columns})
        elif risk is not None and len(risk):
            r = risk[risk["root"].astype(str) == root]
            dv = r.assign(ticker=r["ticker"].astype(str)).pivot_table(index="timestamp", columns="ticker", values="value") \
                / cfg.point_value
        else:
            dv = pd.DataFrame()
        out.append(sp.held_futures_pnl(marks, h, dv, ticker=f"FUT_{root}", bmk=f"fut@{BMK_SYNC_SNAP}",
                                       currency=cfg.currency, point_value=cfg.point_value or 1.0,
                                       max_gap_days=BMK_YIELD_MAX_GAP_DAYS))
    return out


def _swaps(lo, hi, paths: CyclePaths) -> list[pd.DataFrame]:
    from infra.pipeline.swap_closes import read_swap_closes
    out = []
    for ccy in SWAP_CLOSES[BMK_SYNC_SNAP].currencies:
        c = read_swap_closes(lo, hi + _ONE_DAY, close=BMK_SYNC_SNAP, currency=ccy, method="best", root=paths.swap_closes_dir)
        c = c[c["tenor"].isin(BMK_SYNC_SWAP_TENORS)]
        if c.empty:
            continue
        y = pd.DataFrame({"timestamp": pd.to_datetime(c["timestamp"]).dt.normalize(),
                          "ticker": ccy + "_OIS_" + c["tenor"].astype(int).astype(str) + "y", "yield": c["rate"].astype(float)})
        out.append(yp.level_change_pnl(y, f"ois@{BMK_SYNC_SNAP}", max_gap_days=BMK_YIELD_MAX_GAP_DAYS, currency=ccy))
    return out


def _cash_yields(country: str, src: str, lo, hi, paths: CyclePaths) -> pd.DataFrame:
    """``timestamp, ticker, yield`` (%) at the source's own price time: an official curve
    (Daily/Bonds), or our German curve's Svensson par (``src="curve"`` for DE)."""
    from infra.pipeline.bonds import read_bonds_from_disk
    if country == "DE" and src == "curve":
        from infra.config import BUND_CURVE_BMK_METHOD
        from infra.pipeline.bund_curves import read_bund_curves
        c = read_bund_curves(lo, hi, method=BUND_CURVE_BMK_METHOD, root=paths.bund_curves_dir)
        if c.empty:
            return pd.DataFrame(columns=["timestamp", "ticker", "yield"])
        cols = [f"par_{t}y" for t in BMK_YIELD_OFFICIAL_TENORS if f"par_{t}y" in c]
        long = c.melt(id_vars=["timestamp"], value_vars=cols, var_name="col", value_name="yield")
        return pd.DataFrame({"timestamp": pd.to_datetime(long["timestamp"]), "ticker": "DE_BOND_" + long["col"].str[4:],
                             "yield": long["yield"]})
    b = read_bonds_from_disk([f"{country}_BOND_{t}y" for t in BMK_YIELD_OFFICIAL_TENORS], lo, hi + _ONE_DAY,
                             root=paths.daily_bonds_dir)
    return b.rename(columns={"par_yield": "yield"})[["timestamp", "ticker", "yield"]].assign(ticker=lambda d: d["ticker"].astype(str))


def _cash(lo, hi, paths: CyclePaths, book) -> list[pd.DataFrame]:
    from infra.pipeline.swap_hedge import snap_mids
    out = []
    for country, (src, src_time, hedge_ccy) in BMK_SYNC_CASH.items():
        b = _cash_yields(country, src, lo, hi, paths)
        if b.empty:
            continue
        currency = {"US": "USD", "DE": "EUR", "UK": "GBP"}[country]
        if src_time is not None:
            hedges = SWAP_HEDGES[hedge_ccy]
            moved = []
            for root in sorted({hedges[t] for t in BMK_YIELD_OFFICIAL_TENORS if t in hedges}):
                h = book.contract.get(root)
                if h is None:
                    continue
                tick = sorted(set(h.dropna()))
                at_snap = snap_mids(tick, lo, hi + _ONE_DAY, *_snap(), bbo_root=_bbo_root(paths))
                at_src = snap_mids(tick, lo, hi + _ONE_DAY, *src_time, bbo_root=_bbo_root(paths))
                for t in [t for t in BMK_YIELD_OFFICIAL_TENORS if hedges.get(t) == root]:
                    rows = b[b["ticker"] == f"{country}_BOND_{t}y"]
                    for r in rows.itertuples(index=False):
                        day = pd.Timestamp(r.timestamp)
                        c = h.get(day)
                        ms = at_snap.at[day, c] if (c in at_snap.columns and day in at_snap.index) else np.nan
                        m0 = at_src.at[day, c] if (c in at_src.columns and day in at_src.index) else np.nan
                        moved.append((day, r.ticker, sp.moved_yield(r._2, book.ratio_on(root, day), ms, m0)))
            b = pd.DataFrame(moved, columns=["timestamp", "ticker", "yield"])
        out.append(yp.level_change_pnl(b.dropna(subset=["yield"]), f"yield_{src}@{BMK_SYNC_SNAP}",
                                       max_gap_days=BMK_YIELD_MAX_GAP_DAYS, currency=currency))
    return out


def _held_mids(root: str, lo, hi, local_time: str, tz: str, paths: CyclePaths, held: pd.Series) -> pd.Series:
    """Day -> the held contract's mid at ``local_time`` (``tz``) that day."""
    from infra.pipeline.swap_hedge import snap_mids
    m = snap_mids(sorted(set(held.dropna())), lo, hi + _ONE_DAY, local_time, tz, bbo_root=_bbo_root(paths))
    return pd.Series({d: m.at[d, c] for d, c in held.items() if d in m.index and c in m.columns}, dtype=float)


def _same_contract_change(mids: pd.Series, held: pd.Series) -> pd.Series:
    h = held.reindex(mids.index)
    return mids.diff().where(h.eq(h.shift()))


def _canada(lo, hi, paths: CyclePaths) -> list[pd.DataFrame]:
    """Canada at the snap (``BMK_SYNC_CA_*``): the BoC benchmark yields (16:00 Toronto) and the
    CGB settlement (~15:00 Toronto), each moved by its US hedge future's quote move between its
    own time and the snap x a point-in-time ratio (``sync_pnl.rolling_beta`` of the Canadian
    daily change on the future's mid change at the Canadian series' own time)."""
    from infra.config import MX_SETTLEMENT_LOCAL_TIME
    from infra.pipeline import mx_futures as mx
    from infra.pipeline.bonds import read_bonds_from_disk
    from infra.pipeline.swap_hedge import hedge_contracts
    from infra.processing import mx_futures as mf
    warm = lo - pd.offsets.BDay(BMK_SYNC_CA_WINDOW + 20)
    src, (src_time, src_tz) = BMK_SYNC_CA_CASH
    fut_root, fut_hedge = BMK_SYNC_CA_FUTURE
    usd = SWAP_HEDGES["USD"]
    tenors = [t for t in BMK_YIELD_OFFICIAL_TENORS if t in usd]
    roots = sorted({usd[t] for t in tenors} | {fut_hedge})
    held = hedge_contracts(warm, hi, roots=roots, daily_root=paths.daily_futures_dir, contracts_file=paths.contracts_file,
                           bbo_root=_bbo_root(paths))
    out = []
    # --- the BoC benchmark yields
    b = read_bonds_from_disk([f"CA_BOND_{t}y" for t in tenors], warm, hi + _ONE_DAY, root=paths.daily_bonds_dir)
    if len(b):
        b = b.assign(ticker=b["ticker"].astype(str))
        moved = []
        for t in tenors:
            h = held.get(usd[t])
            y = b[b["ticker"] == f"CA_BOND_{t}y"].set_index("timestamp")["par_yield"].astype(float).sort_index()
            if h is None or h.empty or y.empty:
                continue
            at_src = _held_mids(usd[t], warm, hi, src_time, src_tz, paths, h)
            at_snap = _held_mids(usd[t], warm, hi, *_snap(), paths, h)
            beta = sp.rolling_beta(y.diff() * 100, _same_contract_change(at_src, h), BMK_SYNC_CA_WINDOW)  # bp per point
            for day, v in y.items():
                moved.append((day, f"CA_BOND_{t}y", sp.moved_yield(v, beta.get(day), at_snap.get(day), at_src.get(day))))
        m = pd.DataFrame(moved, columns=["timestamp", "ticker", "yield"]).dropna(subset=["yield"])
        if len(m):
            out.append(yp.level_change_pnl(m, f"yield_{src}@{BMK_SYNC_SNAP}", max_gap_days=BMK_YIELD_MAX_GAP_DAYS,
                                           currency="CAD"))
    # --- the CGB future
    f = mx.read_mx_futures(warm, hi, roots=[fut_root], root=paths.mx_futures_dir)
    h = held.get(fut_hedge)
    if len(f) and h is not None and not h.empty:
        front = mf.front(f)
        settle = f.pivot_table(index="timestamp", columns="ticker", values="settlement").sort_index()
        fs = mx.front_series(fut_root, warm, hi, root=paths.mx_futures_dir).set_index("timestamp")
        at_src = _held_mids(fut_hedge, warm, hi, *MX_SETTLEMENT_LOCAL_TIME, paths, h)
        at_snap = _held_mids(fut_hedge, warm, hi, *_snap(), paths, h)
        beta = sp.rolling_beta(fs["change"], _same_contract_change(at_src, h), BMK_SYNC_CA_WINDOW)   # points per point
        gap = (at_snap - at_src).reindex(settle.index)
        marks = settle.add(beta.reindex(settle.index) * gap, axis=0)
        # empirical DV01 (points per bp): the CGB settlement change on the BoC 10y change, prior days
        y10 = b[b["ticker"] == "CA_BOND_10y"].set_index("timestamp")["par_yield"].astype(float).sort_index() if len(b) else pd.Series(dtype=float)
        dv = sp.rolling_beta(fs["change"], y10.diff() * 100, BMK_SYNC_CA_WINDOW).abs()
        dv01 = pd.DataFrame({c: dv.reindex(marks.index) for c in marks.columns})
        out.append(sp.held_futures_pnl(marks, front, dv01, ticker=f"FUT_{fut_root}", bmk=f"fut@{BMK_SYNC_SNAP}",
                                       currency="CAD", point_value=MX_POINT_VALUE[fut_root],
                                       max_gap_days=BMK_YIELD_MAX_GAP_DAYS))
    return out


def compute_sync_pnl(start, end, *, paths: CyclePaths | None = None) -> pd.DataFrame:
    from infra.pipeline.swap_hedge import build_hedge_book
    paths = paths or CyclePaths.default()
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    lo, hi = start - _LOOKBACK, end
    book = build_hedge_book(lo, hi, currencies=["USD", "EUR", "GBP"], daily_root=paths.daily_futures_dir,
                            bonds_root=paths.daily_bonds_dir, bbo_root=_bbo_root(paths), contracts_file=paths.contracts_file)
    parts = [p for p in (_futures(lo, hi, paths, book) + _swaps(lo, hi, paths) + _cash(lo, hi, paths, book)
                         + _canada(lo, hi, paths)) if len(p)]
    if not parts:
        return pd.DataFrame(columns=sp.PNL_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)


def backfill_daily_sync_pnl(start, end, *, paths: CyclePaths | None = None) -> dict:
    paths = paths or CyclePaths.default()
    df = compute_sync_pnl(start, end, paths=paths)
    if len(df):
        out = df.copy()
        for col in ("ticker", "bmk", "currency", "cusip", "prev_cusip"):
            out[col] = out[col].astype("string")
        for col in ("timestamp", "prev_timestamp"):
            out[col] = pd.to_datetime(out[col]).astype("datetime64[ms]")
        for col in ("yield", "prev_yield", "pnl", "pnl_per_dv01"):
            out[col] = out[col].astype("float64")
        parquet_store.write_partitioned(out, paths.bmk_root / "Pnl", PNL_KEYS)
    by = df.groupby("bmk").size().to_dict() if len(df) else {}
    last = df.groupby("bmk")["timestamp"].max().dt.date.astype(str).to_dict() if len(df) else {}
    return {"rows": len(df), "by_bmk": by, "last_day": last}


# ------------------------------------------------------------------------ checks
def _read(ctx: StepContext) -> pd.DataFrame:
    df = parquet_store.read_partitioned(ctx.paths.bmk_root / "Pnl", start=ctx.start, end=ctx.end + _ONE_DAY)
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "ticker", "bmk", "pnl_per_dv01"])
    df["bmk"] = df["bmk"].astype(str)
    return df[df["bmk"].str.endswith(f"@{BMK_SYNC_SNAP}")]


def check_present(ctx: StepContext):
    """Every synchronized family has rows in the window (warn: a family missing its quotes,
    closes or DV01s shows here)."""
    want = {f"fut@{BMK_SYNC_SNAP}", f"ois@{BMK_SYNC_SNAP}", *(f"yield_{s}@{BMK_SYNC_SNAP}" for s, _, _ in BMK_SYNC_CASH.values())}
    df = _read(ctx)
    missing = sorted(want - set(df["bmk"]))
    return (not missing, "every synchronized family has P&L in the window" if not missing else f"no rows: {missing}", None)


def check_sane(ctx: StepContext):
    df = _read(ctx)
    bad = df[df["pnl_per_dv01"].abs() > 75]
    return (bad.empty, "every synchronized daily move within 75bp" if bad.empty else f"{len(bad)} move(s) beyond 75bp",
            None if bad.empty else bad[["timestamp", "ticker", "bmk", "pnl_per_dv01"]])
