"""Treasury event study: build the inputs and run ``infra.analytics.event_study`` -
reads disk only (FedInvest END OF DAY yields, the auctions store, the on/off-the-run map).

Events (``treasury_events``), per nominal coupon tenor (2/3/5/7/10/20/30y, ORIGINAL term):
* ``new_issue``   - a new original issue, from its ISSUE day (anchor = issue day: it has
                    no price before): its richness over its first weeks.
* ``otr_roll``    - the outgoing on-the-run bond (``issue`` convention rank 0 the day
                    before the successor's issue) around the successor's ISSUE day.
* ``reopening``   - a reopened bond around its reopening AUCTION day.
* ``auction``     - every auction of the tenor (new or reopening), for the current
                    on-the-run bond of that tenor: the concession into the auction and the
                    recovery after.
"""
from __future__ import annotations

import pandas as pd

from infra.analytics.event_study import curve_residuals, event_paths, event_profiles
from infra.pipeline.treasury_otr import read_otr
from infra.pipeline.treasury_prices import read_prices
from infra.pipeline.treasury_ref import read_securities
from infra.pipeline.tsy_auctions import read_auctions
from infra.processing.tsy_auctions import nominal_coupons

WINDOWS = {"new_issue": ((0, 60), 0), "otr_roll": ((-20, 40), -1), "reopening": ((-10, 20), -1),
           "auction": ((-5, 5), -6)}  # (window, anchor offset); auction anchored a week before


def residuals(start, end) -> pd.DataFrame:
    """Daily curve residuals (bp) of every note and bond."""
    p = read_prices(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))
    p = p[p["security_type"].astype(str).isin(["Note", "Bond"])]
    sec = read_securities().drop_duplicates("cusip").set_index("cusip")
    return curve_residuals(p[["timestamp", "cusip", "yield_eod"]], sec["maturity_date"])


def treasury_events(start, end) -> pd.DataFrame:
    """``event_type, tenor, cusip, event_day`` (module doc)."""
    a = nominal_coupons(read_auctions())
    a = a[(pd.to_datetime(a["auction_date"]) >= pd.Timestamp(start)) & (pd.to_datetime(a["auction_date"]) <= pd.Timestamp(end))]
    a = a.assign(tenor=a["original_security_term"].astype(str).str.extract(r"(\d+)-Year")[0] + "y")
    reopen = a["reopening"].astype(str).eq("Yes")
    ev = [pd.DataFrame({"event_type": "new_issue", "tenor": a.loc[~reopen, "tenor"], "cusip": a.loc[~reopen, "cusip"],
                        "event_day": pd.to_datetime(a.loc[~reopen, "issue_date"])}),
          pd.DataFrame({"event_type": "reopening", "tenor": a.loc[reopen, "tenor"], "cusip": a.loc[reopen, "cusip"],
                        "event_day": pd.to_datetime(a.loc[reopen, "auction_date"])})]
    otr = read_otr(pd.Timestamp(start) - pd.Timedelta(days=10), pd.Timestamp(end))
    otr = otr[(otr["convention"] == "issue") & (otr["rank"] == 0)][["timestamp", "tenor", "cusip"]]
    otr = otr.sort_values("timestamp")
    for kind, day_col, rows in (("otr_roll", "issue_date", a[~reopen]), ("auction", "auction_date", a)):
        for r in rows.itertuples(index=False):
            d = pd.Timestamp(getattr(r, day_col))
            prev = otr[(otr["tenor"] == r.tenor) & (otr["timestamp"] < d)]
            if prev.empty:
                continue
            cusip = prev["cusip"].iloc[-1]
            if kind == "otr_roll" and cusip == r.cusip:
                continue
            ev.append(pd.DataFrame({"event_type": [kind], "tenor": [r.tenor], "cusip": [cusip], "event_day": [d]}))
    out = pd.concat(ev, ignore_index=True).dropna()
    out["cusip"] = out["cusip"].astype(str)
    return out.sort_values(["event_day", "event_type"], ignore_index=True)


def in_study_profiles(start, end, *, res: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The event-profile interface's first provider: (profiles, paths)."""
    res = residuals(start, end) if res is None else res
    ev = treasury_events(start, end)
    paths = pd.concat([event_paths(res, ev[ev["event_type"] == k], window=w, anchor_offset=a)
                       for k, (w, a) in WINDOWS.items()], ignore_index=True)
    return event_profiles(paths), paths


# ---------------------------------------------------------------- add-on MS provider
_MEMO: dict = {}
HISTORY_START = "2008-09-02"  # FedInvest's first day


def net_paths_all(end=None) -> pd.DataFrame:
    """Every event path since ``HISTORY_START``, netted of specialness carry
    (``infra.analytics.event_netting``); memoised per process."""
    from infra.analytics.event_netting import modified_duration, net_paths, specialness_carry_bp
    from infra.pipeline.specialness import observed_specialness
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    key = ("paths", end)
    if key not in _MEMO:
        res = residuals(HISTORY_START, end)
        prof, paths = in_study_profiles(HISTORY_START, end, res=res)
        paths["anchor_offset"] = paths["event_type"].map({k: a for k, (w, a) in WINDOWS.items()})
        days = pd.DatetimeIndex(sorted(res["timestamp"].unique()))
        y = read_prices(pd.Timestamp(HISTORY_START), end + pd.Timedelta(days=1))[["timestamp", "cusip", "yield_eod"]]
        y["cusip"] = y["cusip"].astype(str)
        d = res.merge(y, on=["timestamp", "cusip"])
        keep = set(paths["cusip"])
        d = d[d["cusip"].isin(keep)]
        dur = d.assign(dur=modified_duration(d["maturity_years"], d["yield_eod"])).set_index(["timestamp", "cusip"])["dur"]
        sp = observed_specialness(HISTORY_START, end + pd.Timedelta(days=1))
        sp = sp[sp.index.get_level_values("cusip").isin(keep)]
        _MEMO[key] = (net_paths(paths, specialness_carry_bp(paths, sp, dur, days)), days)
    return _MEMO[key]


def net_profiles_as_of(as_of, *, end=None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """The event-profile interface, NET of specialness carry, from path observations
    dated BEFORE ``as_of`` only (point in time) - ``mean_bp`` = mean net change; and the
    business-day calendar the paths use."""
    paths, days = net_paths_all(end)
    p = paths[pd.to_datetime(paths["date"]) < pd.Timestamp(as_of)]
    g = p.groupby(["event_type", "tenor", "rel_day"])["net_change_bp"]
    prof = pd.DataFrame({"mean_bp": g.mean(), "median_bp": g.median(), "sd_bp": g.std(), "n": g.size()}).reset_index()
    return prof, days


def aging_profiles_as_of(as_of, *, end=None) -> pd.DataFrame:
    """Add-on MS (v1 default): the aging profile per tenor (``infra.analytics.event_drift.
    aging_profile``) from bond-days BEFORE ``as_of`` only, each daily residual change net
    of that day's specialness carry; memoised inputs."""
    from infra.analytics.event_drift import aging_profile
    from infra.analytics.event_netting import modified_duration
    from infra.pipeline.specialness import observed_specialness, successors
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    key = ("aging_daily", end)
    if key not in _MEMO:
        res = residuals(HISTORY_START, end)
        succ = successors(read_securities()).set_index("cusip")
        r = res[res["cusip"].isin(succ.index)].sort_values(["cusip", "timestamp"]).copy()
        r["tenor"] = r["cusip"].map(succ["tenor"])
        days = pd.DatetimeIndex(sorted(res["timestamp"].unique()))
        pos = pd.Series(range(len(days)), index=days)
        r["age_bd"] = r["timestamp"].map(pos) - days.searchsorted(pd.to_datetime(r["cusip"].map(succ["issue_date"])))
        y = read_prices(pd.Timestamp(HISTORY_START), end + pd.Timedelta(days=1))[["timestamp", "cusip", "yield_eod"]]
        y["cusip"] = y["cusip"].astype(str)
        r = r.merge(y, on=["timestamp", "cusip"], how="left")
        sp = observed_specialness(HISTORY_START, end + pd.Timedelta(days=1)).rename("sp").reset_index()
        sp["cusip"] = sp["cusip"].astype(str)
        r = r.merge(sp, on=["timestamp", "cusip"], how="left")
        carry = r["sp"].fillna(0.0) / 360.0 / modified_duration(r["maturity_years"], r["yield_eod"].fillna(3.0))
        r["d1_bp"] = r.groupby("cusip")["residual_bp"].diff() - carry
        _MEMO[key] = r.dropna(subset=["d1_bp"])[["timestamp", "tenor", "age_bd", "d1_bp"]]
    d = _MEMO[key]
    return aging_profile(d[d["timestamp"] < pd.Timestamp(as_of)])
