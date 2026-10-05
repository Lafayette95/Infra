"""Strategy accounting: target positions in absolute contracts -> executed positions -> P&L and
costs -> checks (root CLAUDE.md 27). Pure: marks come in (``infra.pipeline.futures_marks``),
nothing is read or written here. Independent of how the positions were made (any strategy).

1. **Targets**: ``positions_abs`` pivoted to label x contract (a contract not held = 0), moved
   ``exec_lag`` labels later (decided at T, traded at T + lag).
2. **Executable?** per label and contract (``reason``): ``halt`` (the venue's daily halt),
   ``no_fresh_quote`` (no quote within ``max_quote_age_min``: closed, holiday or dead),
   ``one_sided``, ``wide_spread`` (> ``wide_spread_k`` x the contract's trailing median spread,
   from earlier labels only), ``no_settlement`` (daily: no settlement that day).
3. **Executed positions** (``policy``): ``defer`` - a trade waits for the contract's next
   executable label (the target at that label then); ``ignore`` - executed = target.
4. **P&L**, labels = decision / execution times, P&L stamped at the END of each step:
   gross at L = executed at the previous label x (mark at L - mark there) x point value;
   cost at L = |trade at L| x half-spread x ``spread_paid`` x point value; net = gross - cost.
   A roll is two trades (old contract out, new in). A held contract without a mark gives NaN
   gross (never a silent 0) and a check.
5. **Checks** (flags, never fixes): ``deferred`` (episodes; warn beyond ``defer_warn_labels``),
   ``exceeds_top_of_book``, ``held_in_delivery`` (on/after first notice), ``held_past_last_trade``,
   ``no_mark``, ``cost_fallback`` / ``cost_missing``, ``mark_jump`` (a big move on a held contract
   that REVERSES at the next label - a bad quote; a big move that stays is news), ``target_not_reached``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from infra.strategies.config.accounting import AccountingSpec

REASONS = ("halt", "no_fresh_quote", "one_sided", "wide_spread", "no_settlement")
CHECK_COLUMNS = ["check", "severity", "label", "contract", "n", "detail"]


@dataclass
class AccountingResult:
    contracts: pd.DataFrame      # long: label, contract, instrument, target, executed, trade, mark, half_spread,
                                 # gross, cost, net, reason
    totals: pd.DataFrame         # per label: gross, cost, net, and gross:/cost:/net: per instrument
    checks: pd.DataFrame         # CHECK_COLUMNS
    summary: dict


def targets(positions_abs: pd.DataFrame, labels: pd.DatetimeIndex, lag: int = 0) -> pd.DataFrame:
    """label x contract target positions (0 where not held), shifted ``lag`` labels later."""
    p = positions_abs.dropna(subset=["contract"])
    wide = p.pivot_table(index="label", columns="contract", values="position", aggfunc="sum")
    wide = wide.reindex(pd.DatetimeIndex(labels)).fillna(0.0)
    return wide.shift(lag).fillna(0.0) if lag else wide


def _wide(marks: pd.DataFrame, col: str, labels, contracts) -> pd.DataFrame:
    w = marks.pivot_table(index="timestamp", columns="contract", values=col, aggfunc="last", dropna=False)
    return w.reindex(index=pd.DatetimeIndex(labels), columns=contracts)


def _trailing_median(x: pd.DataFrame, lookback: int, min_obs: int) -> pd.DataFrame:
    """Per column, the median of the previous ``lookback`` non-missing values (this row excluded)."""
    out = {}
    for c in x.columns:
        s = x[c].dropna()
        med = s.rolling(lookback, min_periods=min_obs).median().shift(1)
        out[c] = med.reindex(x.index).ffill()
    return pd.DataFrame(out, index=x.index)


def market_state(marks: pd.DataFrame, labels, contracts, spec: AccountingSpec) -> dict[str, pd.DataFrame]:
    """Wide ``mark``, ``half`` (half-spread), ``reason`` ("" = executable), and for bbo the
    top-of-book sizes; ``cost_note`` per cell ("", "cost_fallback", "cost_missing")."""
    contracts = list(contracts)
    if spec.marks == "bbo":
        bid, ask = _wide(marks, "bid", labels, contracts), _wide(marks, "ask", labels, contracts)
        mark = _wide(marks, "mark", labels, contracts)
        qt = marks.assign(quote_time=pd.to_datetime(marks["quote_time"]))
        age = (pd.DatetimeIndex(labels).to_numpy()[:, None]
               - _wide(qt, "quote_time", labels, contracts).to_numpy(dtype="datetime64[ns]"))
        stale = ~(age <= np.timedelta64(int(spec.max_quote_age_min * 60e9), "ns"))
        halt = _wide(marks.assign(halt=marks["halt"].astype(float)), "halt", labels, contracts).fillna(0).astype(bool)
        spread = ask - bid
        fresh_two = spread.where(~stale & spread.notna())
        med = _trailing_median(fresh_two, spec.spread_lookback, spec.spread_min_obs)
        reason = np.full(mark.shape, "", dtype=object)
        wide_m = (spread > spec.wide_spread_k * med).to_numpy()
        for flag, msk in (("wide_spread", wide_m), ("one_sided", (bid.isna() | ask.isna()).to_numpy()),
                          ("no_fresh_quote", stale), ("halt", halt.to_numpy())):
            reason[msk] = flag                      # later flags win: halt > stale > one-sided > wide
        cost_note = np.where(spread.isna().to_numpy(), "cost_missing", "")
        return {"mark": mark, "half": spread / 2, "reason": pd.DataFrame(reason, index=mark.index, columns=contracts),
                "bid_size": _wide(marks, "bid_size", labels, contracts),
                "ask_size": _wide(marks, "ask_size", labels, contracts),
                "cost_note": pd.DataFrame(cost_note, index=mark.index, columns=contracts)}
    if spec.marks == "settlement":
        mark = _wide(marks, "settlement", labels, contracts)
        half = (_wide(marks, "ask", labels, contracts) - _wide(marks, "bid", labels, contracts)) / 2
        # fallback: the contract's trailing median half-spread over earlier execution days (cost_fallback)
        per_day = marks.dropna(subset=["exec_day"]).assign(half=(marks["ask"] - marks["bid"]) / 2)
        fb = {}
        for c in contracts:
            d = per_day[per_day["contract"] == c].drop_duplicates("exec_day").set_index("exec_day")["half"].sort_index()
            med = d.dropna().rolling(spec.cost_fallback_days, min_periods=1).median().shift(1)
            ex = marks[marks["contract"] == c].set_index("timestamp")["exec_day"].reindex(pd.DatetimeIndex(labels))
            fb[c] = med.reindex(d.index).ffill().reindex(pd.DatetimeIndex(ex.to_numpy())).to_numpy()
        fb = pd.DataFrame(fb, index=mark.index)
        # a contract with no snap history of its own (e.g. the next contract, rolled into before
        # the snaps cover it) takes its ROOT's trailing median half-spread (same tick, same market)
        from infra.pipeline.event_pnl import root_of
        roots = {c: root_of(c) for c in contracts}
        for r in set(roots.values()):
            cols = [c for c in contracts if roots[c] == r]
            d = per_day[per_day["contract"].isin(cols)].groupby("exec_day")["half"].median().sort_index()
            med = d.dropna().rolling(spec.cost_fallback_days, min_periods=1).median().shift(1)
            for c in cols:
                ex = marks[marks["contract"] == c].set_index("timestamp")["exec_day"].reindex(pd.DatetimeIndex(labels))
                root_fb = med.reindex(d.index).ffill().reindex(pd.DatetimeIndex(ex.to_numpy())).to_numpy()
                fb[c] = fb[c].fillna(pd.Series(root_fb, index=mark.index))
        note = np.where(half.isna() & fb.notna(), "cost_fallback", np.where(half.isna() & fb.isna(), "cost_missing", ""))
        reason = np.where(mark.isna(), "no_settlement", "")
        return {"mark": mark, "half": half.fillna(fb),
                "reason": pd.DataFrame(reason.astype(object), index=mark.index, columns=contracts),
                "bid_size": None, "ask_size": None,
                "cost_note": pd.DataFrame(note.astype(object), index=mark.index, columns=contracts)}
    raise ValueError(f"marks {spec.marks!r} (bbo | settlement)")


def execute(target: pd.DataFrame, reason: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Executed positions: ``defer`` keeps the previous executed position wherever the market
    can't be traded; ``ignore`` = target."""
    if policy == "ignore":
        return target.copy()
    if policy != "defer":
        raise ValueError(f"policy {policy!r} (defer | ignore)")
    ok = reason.reindex_like(target).fillna("").eq("")
    return target.where(ok).ffill().fillna(0.0)


def _episodes(mask: pd.Series) -> list[tuple[pd.Timestamp, int]]:
    """Runs of True: (first label, length)."""
    m = mask.to_numpy(dtype=bool)
    if not m.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return [(mask.index[a], int(b - a)) for a, b in zip(edges[::2], edges[1::2])]


def account(positions_abs: pd.DataFrame, labels, marks: pd.DataFrame, calendar: pd.DataFrame,
            spec: AccountingSpec, *, exec_lag: int = 0) -> AccountingResult:
    """See the module docstring. ``calendar``: ``futures_marks.contract_calendar`` (index =
    contract: point_value, last_trade, first_notice); ``marks`` from ``quote_marks`` (bbo) or
    ``settlement_marks`` (settlement), covering every contract in ``positions_abs``."""
    labels = pd.DatetimeIndex(labels).sort_values().unique()
    tgt = targets(positions_abs, labels, exec_lag)
    contracts = list(tgt.columns)
    if not contracts:
        empty = pd.DataFrame(columns=["label", "contract", "instrument", "target", "executed", "trade", "mark",
                                      "half_spread", "gross", "cost", "net", "reason"])
        return AccountingResult(empty, pd.DataFrame({"gross": 0.0, "cost": 0.0, "net": 0.0}, index=labels),
                                pd.DataFrame(columns=CHECK_COLUMNS), {"contracts": 0})
    st = market_state(marks, labels, contracts, spec)
    ex = execute(tgt, st["reason"], spec.policy)
    trade = ex.diff()
    trade.iloc[0] = ex.iloc[0]
    pv = calendar["point_value"].reindex(contracts).astype(float)
    mark = st["mark"]
    held_prev = ex.shift(1).fillna(0.0)
    dmark = mark - mark.shift(1)
    gross = (held_prev * dmark * pv).where(held_prev != 0, 0.0)
    cost = (trade.abs() * st["half"] * spec.spread_paid * pv).where(trade != 0, 0.0)
    cost_known = cost.fillna(0.0)
    net = gross - cost_known

    inst_of = (positions_abs.dropna(subset=["contract"]).drop_duplicates("contract")
               .set_index("contract")["instrument"].to_dict())
    keep = (tgt != 0) | (ex != 0) | (held_prev != 0) | (trade != 0)
    stacked = {name: frame.where(keep).stack(future_stack=True) for name, frame in
               (("target", tgt), ("executed", ex), ("trade", trade), ("mark", mark), ("half_spread", st["half"]),
                ("gross", gross), ("cost", cost), ("net", net))}
    long = pd.DataFrame(stacked)
    long["reason"] = st["reason"].where(keep & (tgt != ex)).stack(future_stack=True).reindex(long.index)
    long = long[keep.stack(future_stack=True).reindex(long.index).to_numpy(dtype=bool)]
    long = long.rename_axis(["label", "contract"]).reset_index()
    long["instrument"] = long["contract"].map(inst_of)
    long["reason"] = long["reason"].fillna("")
    long = long[["label", "contract", "instrument", "target", "executed", "trade", "mark", "half_spread", "gross",
                 "cost", "net", "reason"]]

    unknown = gross.isna().any(axis=1)
    totals = pd.DataFrame({"gross": gross.sum(axis=1).mask(unknown), "cost": cost_known.sum(axis=1)}, index=labels)
    totals["net"] = totals["gross"] - totals["cost"]
    for inst in sorted(set(inst_of.values())):
        cols = [c for c in contracts if inst_of.get(c) == inst]
        g = gross[cols]
        totals[f"gross:{inst}"] = g.sum(axis=1).mask(g.isna().any(axis=1))
        totals[f"cost:{inst}"] = cost_known[cols].sum(axis=1)
        totals[f"net:{inst}"] = totals[f"gross:{inst}"] - totals[f"cost:{inst}"]
    totals.index.name = "label"

    checks = run_checks(tgt, ex, trade, held_prev, gross, mark, st, calendar, spec)
    summary = {"labels": len(labels), "contracts": len(contracts), "trades": int((trade != 0).sum().sum()),
               "gross": float(np.nansum(gross.to_numpy())), "cost": float(cost_known.to_numpy().sum()),
               "net": float(np.nansum(net.to_numpy())),
               "checks": checks.groupby(["check", "severity"]).size().to_dict() if len(checks) else {}}
    return AccountingResult(long.reset_index(drop=True), totals, checks, summary)


def run_checks(tgt, ex, trade, held_prev, gross, mark, st, calendar, spec) -> pd.DataFrame:
    rows = []

    def add(check, severity, label, contract, n, detail):
        rows.append({"check": check, "severity": severity, "label": label, "contract": contract, "n": n,
                     "detail": detail})

    labels = tgt.index
    for c in tgt.columns:
        # deferred trades: target not reached, as episodes
        gap = (tgt[c] - ex[c]).abs() > 1e-12
        for start, n in _episodes(gap):
            why = st["reason"].loc[start, c] if start in st["reason"].index else ""
            add("deferred", "warn" if n > spec.defer_warn_labels else "info", start, c, n,
                f"target {tgt.at[start, c]:.2f} vs executed {ex.at[start, c]:.2f} for {n} label(s); {why}")
        # top of book
        if st["bid_size"] is not None:
            t = trade[c]
            size = np.where(t > 0, st["ask_size"][c], st["bid_size"][c])
            over = (t != 0) & (t.abs() > pd.Series(size, index=labels))
            for lab in labels[over.to_numpy()]:
                add("exceeds_top_of_book", "warn", lab, c, 1,
                    f"trade {trade.at[lab, c]:.1f} vs top-of-book size {size[labels.get_loc(lab)]:.0f}")
        # delivery / expiry
        held = ex[c] != 0
        day = labels.normalize()
        fn, lt = calendar.at[c, "first_notice"] if c in calendar.index else pd.NaT, \
            calendar.at[c, "last_trade"] if c in calendar.index else pd.NaT
        if pd.notna(fn):
            for start, n in _episodes(held & (day >= fn)):
                add("held_in_delivery", "warn", start, c, n, f"position held on/after first notice {fn.date()}")
        if pd.notna(lt):
            for start, n in _episodes(held & (day > lt)):
                add("held_past_last_trade", "fail", start, c, n, f"position held after last trade {lt.date()}")
        # marks
        for start, n in _episodes((held_prev[c] != 0) & gross[c].isna()):
            add("no_mark", "warn", start, c, n, "held contract without a mark: gross P&L NaN")
        d = (mark[c] - mark[c].shift(1)).abs()
        typical = d[d > 0].rolling(spec.mark_jump_lookback, min_periods=50).median().shift(1).reindex(labels).ffill()
        move = mark[c] - mark[c].shift(1)
        reverted = -(mark[c].shift(-1) - mark[c]) / move >= spec.mark_jump_reversal
        jump = (held_prev[c] != 0) & (d > spec.mark_jump_k * typical) & reverted
        for lab in labels[jump.to_numpy()]:
            add("mark_jump", "warn", lab, c, 1, f"mark move {d.at[lab]:.4f} vs typical {typical.at[lab]:.4f}")
        # costs
        note = st["cost_note"][c].where(trade[c] != 0, "")
        for kind, sev in (("cost_fallback", "info"), ("cost_missing", "warn")):
            n = int((note == kind).sum())
            if n:
                add(kind, sev, labels[(note == kind).to_numpy()][0], c, n,
                    "trailing-median half-spread used" if kind == "cost_fallback" else "no spread: cost counted 0")
    if len(labels):
        last = labels[-1]
        off = (tgt.loc[last] - ex.loc[last]).abs() > 1e-12
        for c in tgt.columns[off.to_numpy()]:
            add("target_not_reached", "info", last, c, 1, f"target {tgt.at[last, c]:.2f}, executed {ex.at[last, c]:.2f}")
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)
