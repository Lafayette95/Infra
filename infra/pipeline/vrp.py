"""Volatility risk premium per day (root CLAUDE.md 16; spec ``infra.config.VRP``), disk
only: swaption ATM vols (``infra.pipeline.swaptions``) against the forward swap rate on the
OIS curves, and futures options' ATM vol (``infra.pipeline.futures_iv``) against the front
contract's settlements.

Store ``Derived/VolRiskPremium``, keys ``timestamp`` (day), ``instrument``
(``SWPT_<expiry>_<tenor>y``, ``<root>_<n>d``): ``units`` (bp / points per year), ``iv``,
``rv_<w>`` / ``vrp_<w>`` per trailing window, ``rv_ewma`` / ``vrp_ewma`` / ``ratio_ewma``,
and the ex-post ``rv_life`` / ``vrp_life`` with ``life_end`` (the day they became known -
``read_vrp(as_of=)`` hides them before it).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from infra.analytics import swap_curve as sc
from infra.analytics import swaptions as sw
from infra.analytics import vrp as va
from infra.config import (
    DAILY_FUTURES_DIR,
    DAILY_OPTIONS_DIR,
    OIS_CURVES_DIR,
    SWAPTION_VOLS_DIR,
    SWAPTIONS,
    VRP,
    VRP_DIR,
    VrpSpec,
)
from infra.pipeline.ois_curves import read_ois_curves
from infra.pipeline.swaptions import read_vols
from infra.storage import parquet_store

KEYS = ["timestamp", "instrument"]
_ONE_DAY = pd.Timedelta(days=1)


def _curves(start, end, ois_root: Path) -> dict:
    n = read_ois_curves(start, pd.Timestamp(end) + _ONE_DAY, curve=SWAPTIONS["USD_SOFR"].curve, root=ois_root)
    if n.empty:
        return {}
    n["day"] = pd.to_datetime(n["timestamp"]).dt.normalize()
    return {d: sc.curve_from_nodes(g) for d, g in sorted(n.groupby("day"))}


def _fwd(curve, day, start, tenor: int) -> float:
    f, _ = sw.forward_annuity(curve, day, start, start + pd.DateOffset(years=tenor))
    return f * 1e4  # bp


def _assemble(iv: pd.Series, changes: pd.Series, life: pd.DataFrame, spec: VrpSpec, instrument: str,
              units: str) -> pd.DataFrame:
    """One instrument's rows on the days it has an implied vol."""
    rv = {w: va.trailing_vol(changes, w, trading_days=spec.trading_days, min_obs=spec.min_obs) for w in spec.windows}
    ew = va.ewma_vol(changes, spec.ewma_lambda, trading_days=spec.trading_days, min_obs=spec.min_obs)
    out = pd.DataFrame({"iv": iv})
    for w, s in rv.items():
        out[f"rv_{w}"] = s.reindex(out.index)
        out[f"vrp_{w}"] = out["iv"] - out[f"rv_{w}"]
    out["rv_ewma"] = ew.reindex(out.index)
    out["vrp_ewma"] = out["iv"] - out["rv_ewma"]
    out["ratio_ewma"] = out["iv"] / out["rv_ewma"]
    out = out.join(life, how="left")
    out["vrp_life"] = out["iv"] - out["rv_life"]
    out = out.rename_axis("timestamp").reset_index()
    out["instrument"], out["units"] = instrument, units
    return out


def swaption_vrp(start, end, *, spec: VrpSpec = VRP, vols_root: Path = SWAPTION_VOLS_DIR,
                 ois_root: Path = OIS_CURVES_DIR) -> pd.DataFrame:
    """Every ``swaption_points`` point on its days in ``[start, end]``. Realised side from
    the OIS curves (history before ``start`` for the trailing windows; days after ``end``,
    as far as stored, for the ex post)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    hist = start - pd.Timedelta(days=int(max(spec.windows) * 1.6) + 30)
    curves = _curves(hist, end + pd.Timedelta(days=120), ois_root)
    days = sorted(curves)
    if not days:
        return pd.DataFrame()
    vols = read_vols(start, end + _ONE_DAY, root=vols_root)
    frames = []
    for name, months, tenor in spec.swaption_points:
        iv = vols[(vols["expiry"] == name) & (vols["tenor"] == tenor)].set_index("timestamp")["vol_bp"]
        iv = iv[(iv.index >= start) & (iv.index <= end)]
        if iv.empty:
            continue
        cm = pd.Series({d: _fwd(curves[d], d, sc.add_business_days(d + pd.DateOffset(months=months), 2), tenor)
                        for d in days if d <= end})
        changes = cm.diff().dropna()
        life = {}
        for t in iv.index:
            expiry = t + pd.DateOffset(months=months)
            u_start = sc.add_business_days(expiry, 2)
            path_days = [d for d in days if t <= d <= expiry]
            if not path_days or path_days[-1] < expiry - pd.Timedelta(days=4):
                continue  # the option's life isn't over on the stored curves yet
            path = pd.Series({d: _fwd(curves[d], d, u_start, tenor) for d in path_days})
            life[t] = {"rv_life": va.realised_over(path, trading_days=spec.trading_days,
                                                   min_obs=max(5, spec.min_obs * months // 3)),
                       "life_end": pd.Timestamp(path_days[-1])}
        frames.append(_assemble(iv, changes, pd.DataFrame.from_dict(life, orient="index"), spec,
                                f"SWPT_{name}_{tenor}y", "bp"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def futures_vrp(start, end, *, spec: VrpSpec = VRP, options_root: Path = DAILY_OPTIONS_DIR,
                futures_root: Path = DAILY_FUTURES_DIR) -> pd.DataFrame:
    """Each ``futures`` root: ATM vol at ``futures_horizon_days`` (points/yr = lognormal x
    future) against the front contract's settlement changes (points)."""
    from infra.pipeline import futures_iv as fi
    from infra.pipeline import daily as dl
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    frames = []
    for root in spec.futures:
        atm = fi.atm_iv(root, start, end, store=options_root)
        if atm.empty:
            continue
        days = sorted(atm["timestamp"].unique())
        hz = pd.Series([pd.Timestamp(d) + pd.Timedelta(days=spec.futures_horizon_days) for d in days],
                       index=pd.DatetimeIndex(days))
        h = fi.vol_at_horizon(atm, hz).set_index("timestamp")
        iv = h["iv"] * h["future"]  # points per year
        hist = start - pd.Timedelta(days=int(max(spec.windows) * 1.6) + 30)
        stop = end + pd.Timedelta(days=spec.futures_horizon_days + 10)
        names = parquet_store.read_partitioned(futures_root, start=hist, end=stop, columns=["ticker"])
        tick = pd.Series(names["ticker"].astype(str).unique()) if names is not None else pd.Series([], dtype=str)
        tick = sorted(tick[tick.str.fullmatch(rf"{root}[FGHJKMNQUVXZ]\d")])
        fut = dl.read_daily_from_disk(tick, hist, stop, root=futures_root)
        fut["ticker"] = fut["ticker"].astype(str)
        w = fut.pivot(index="timestamp", columns="ticker", values="settlement_price").sort_index()
        oi = fut.pivot(index="timestamp", columns="ticker", values="open_interest").sort_index()
        ch = fi.front_changes(w, oi)
        life = {}
        for t in iv.index:
            endt = t + pd.Timedelta(days=spec.futures_horizon_days)
            seg = ch[(ch.index > t) & (ch.index <= endt)]
            if len(seg) and seg.index[-1] >= endt - pd.Timedelta(days=4):
                life[t] = {"rv_life": float(np.sqrt((seg ** 2).mean() * spec.trading_days)),
                           "life_end": pd.Timestamp(seg.index[-1])}
        frames.append(_assemble(iv, ch[ch.index <= end], pd.DataFrame.from_dict(life, orient="index"), spec,
                                f"{root}_{spec.futures_horizon_days}d", "points"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_vrp(start, end, **kw) -> pd.DataFrame:
    sw_kw = {k: v for k, v in kw.items() if k in ("spec", "vols_root", "ois_root")}
    fu_kw = {k: v for k, v in kw.items() if k in ("spec", "options_root", "futures_root")}
    parts = [p for p in (swaption_vrp(start, end, **sw_kw), futures_vrp(start, end, **fu_kw)) if len(p)]
    if not parts:
        return pd.DataFrame(columns=KEYS)
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    out["life_end"] = pd.to_datetime(out["life_end"]).astype("datetime64[ms]")
    return out.sort_values(KEYS).reset_index(drop=True)


def store_vrp(df: pd.DataFrame, start, end, *, root: Path = VRP_DIR) -> int:
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize() + _ONE_DAY
    parquet_store.delete_where(root, lambda p: pd.to_datetime(p["timestamp"]).ge(lo) & pd.to_datetime(p["timestamp"]).lt(hi))
    if df.empty:
        return 0
    parquet_store.write_partitioned(df, root, KEYS)
    return len(df)


def read_vrp(start=None, end=None, *, as_of=None, instruments=None, root: Path = VRP_DIR) -> pd.DataFrame:
    """Stored rows in ``[start, end)``. ``as_of``: point in time - the ex-post fields of
    an option whose life ended after ``as_of`` are blanked."""
    df = parquet_store.read_partitioned(root, start=None if start is None else pd.Timestamp(start),
                                        end=None if end is None else pd.Timestamp(end),
                                        equals_in={"instrument": list(instruments)} if instruments else None)
    if df is None or df.empty:
        return pd.DataFrame(columns=KEYS)
    df["instrument"] = df["instrument"].astype(str)
    if as_of is not None:
        late = df["life_end"].isna() | (pd.to_datetime(df["life_end"]) > pd.Timestamp(as_of))
        df.loc[late, ["rv_life", "vrp_life"]] = np.nan
    return df.sort_values(KEYS).reset_index(drop=True)
