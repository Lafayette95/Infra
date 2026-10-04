"""Treasury event study - pure, no I/O (infra/models/basis/CLAUDE.md, add-on MS).

How a bond's RICHNESS moves around known events: auctions, reopenings, a successor issue
taking over the on-the-run, a new issue's first weeks. Richness = the bond's yield
RESIDUAL to a smooth curve fitted that day across all notes and bonds (bp; negative =
rich, i.e. yields less than its neighbours). Built as the FIRST PROVIDER of the
event-profile interface (``EVENT_PROFILE_COLUMNS``): expected residual path and its
dispersion by event type x tenor x business days from the event. A later project-wide
event study plugs in as another provider with the same columns (user decision 2026-10-02).

* ``curve_residuals``: per day, a least-squares cubic regression spline of yield on
  maturity (truncated-power basis, knots ``KNOTS`` years) over bonds of ``MIN_MATURITY`` ..
  30 years, one trimming pass (|residual| > 4 x MAD dropped, refit). Coupon effects and
  the curve's own fit error are not removed; an event study measures CHANGES around the
  event, which they barely touch.
* ``event_paths``: each event's residual change from its anchor day (the business day
  before the event, or the event day itself for ``anchor_offset=0``).
* ``event_profiles``: mean / median / sd / n by (event_type, tenor, rel_day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KNOTS = (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0)
MIN_MATURITY = 1.0
EVENT_PROFILE_COLUMNS = ["event_type", "tenor", "rel_day", "mean_bp", "median_bp", "sd_bp", "n"]


def _basis(m: np.ndarray, knots=KNOTS) -> np.ndarray:
    cols = [np.ones_like(m), m, m ** 2, m ** 3] + [np.clip(m - k, 0, None) ** 3 for k in knots]
    return np.column_stack(cols)


def fit_curve(maturity_years: np.ndarray, yields: np.ndarray, *, knots=KNOTS, trim: float = 4.0) -> np.ndarray:
    """Spline coefficients for one day (one trimming pass)."""
    X = _basis(maturity_years, knots)
    coef, *_ = np.linalg.lstsq(X, yields, rcond=None)
    r = yields - X @ coef
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    keep = np.abs(r) <= trim * max(mad, 1e-6)
    if keep.sum() >= X.shape[1] + 3 and not keep.all():
        coef, *_ = np.linalg.lstsq(X[keep], yields[keep], rcond=None)
    return coef


def curve_residuals(prices: pd.DataFrame, maturity: pd.Series, *, min_maturity: float = MIN_MATURITY,
                    knots=KNOTS) -> pd.DataFrame:
    """``prices``: ``timestamp, cusip, yield_eod`` (%); ``maturity``: cusip -> date.
    Returns ``timestamp, cusip, maturity_years, residual_bp`` for every bond with a yield
    (bonds outside the fitted range get NaN)."""
    df = prices.dropna(subset=["yield_eod"]).copy()
    df["cusip"] = df["cusip"].astype(str)
    df["maturity_years"] = (pd.to_datetime(df["cusip"].map(maturity)) - pd.to_datetime(df["timestamp"])).dt.days / 365.25
    df = df.dropna(subset=["maturity_years"])
    out = []
    for day, g in df.groupby("timestamp"):
        fit = g[(g["maturity_years"] >= min_maturity) & (g["maturity_years"] <= 30.5)]
        if len(fit) < len(knots) + 8:
            continue
        coef = fit_curve(fit["maturity_years"].to_numpy(), fit["yield_eod"].to_numpy(), knots=knots)
        r = (g["yield_eod"].to_numpy() - _basis(g["maturity_years"].to_numpy(), knots) @ coef) * 100
        r[(g["maturity_years"] < min_maturity).to_numpy() | (g["maturity_years"] > 30.5).to_numpy()] = np.nan
        out.append(pd.DataFrame({"timestamp": day, "cusip": g["cusip"].to_numpy(),
                                 "maturity_years": g["maturity_years"].to_numpy(), "residual_bp": r}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["timestamp", "cusip", "maturity_years", "residual_bp"])


def event_paths(residuals: pd.DataFrame, events: pd.DataFrame, *, window: tuple[int, int],
                anchor_offset: int = -1) -> pd.DataFrame:
    """``events``: ``event_type, tenor, cusip, event_day``. For each event, the bond's
    residual change from its ANCHOR (``anchor_offset`` business days from the event: -1 =
    the day before, 0 = the event day) over ``window`` = (first, last) business-day
    offsets, on the residuals' own day index (business days the prices exist). Rows:
    ``event_type, tenor, cusip, event_day, rel_day, change_bp``."""
    wide = residuals.pivot_table(index="timestamp", columns="cusip", values="residual_bp")
    days = wide.index
    rows = []
    for e in events.itertuples(index=False):
        if e.cusip not in wide.columns:
            continue
        i0 = days.searchsorted(pd.Timestamp(e.event_day))  # first business day on/after the event
        if i0 >= len(days):
            continue
        ia = i0 + anchor_offset
        if ia < 0 or ia >= len(days):
            continue
        base = wide[e.cusip].iloc[ia]
        if not np.isfinite(base):
            continue
        lo, hi = max(i0 + window[0], 0), min(i0 + window[1], len(days) - 1)
        seg = wide[e.cusip].iloc[lo:hi + 1]
        rel = np.arange(lo - i0, hi - i0 + 1)
        ok = np.isfinite(seg.to_numpy())
        rows.append(pd.DataFrame({"event_type": e.event_type, "tenor": e.tenor, "cusip": e.cusip,
                                  "event_day": pd.Timestamp(e.event_day), "rel_day": rel[ok],
                                  "date": days[lo:hi + 1][ok], "change_bp": seg.to_numpy()[ok] - base}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["event_type", "tenor", "cusip", "event_day", "rel_day", "date", "change_bp"])


def event_profiles(paths: pd.DataFrame) -> pd.DataFrame:
    """The event-profile interface: by (event_type, tenor, rel_day)."""
    g = paths.groupby(["event_type", "tenor", "rel_day"])["change_bp"]
    out = pd.DataFrame({"mean_bp": g.mean(), "median_bp": g.median(), "sd_bp": g.std(), "n": g.size()}).reset_index()
    return out[EVENT_PROFILE_COLUMNS]
