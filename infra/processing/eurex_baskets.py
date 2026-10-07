"""Eurex German government-bond futures: delivery baskets and conversion factors - pure, no
I/O. Contract specs ``infra.config.EUREX_BOND_FUTURES``.

Conversion factor (Eurex's notional-coupon price of a bond at delivery, per 1 nominal,
6 decimals): with K the notional coupon (6%; Buxl 4%), c the bond's coupon, A the NEXT coupon
actually paid (a long first coupon pays c plus its stub, a short one the stub only), tau the
time to it in coupon periods (ACT/ACT ICMA, counted from the delivery day), n the regular
coupons after it:

    CF = [(1+K)^-tau * (A + c * (1 - (1+K)^-n) / K + 100 * (1+K)^-n) - (A - c * tau)] / 100

``A - c * tau`` is the accrued interest at delivery. Verified 2026-10-07 on every German
contract in Eurex's own file (FGBS / FGBM / FGBL / FGBX, Dec 2026 - Jun 2027): every regular
bond exact to 6 decimals, and the four bonds in a long first coupon period exact too.
"""
from __future__ import annotations

import pandas as pd

from infra.processing.treasury_prices import coupon_dates


def delivery_day(year: int, month: int) -> pd.Timestamp:
    """The 10th of the contract month, or the next weekday (Eurex: next exchange day)."""
    d = pd.Timestamp(year, month, 10)
    while d.dayofweek >= 5:
        d += pd.Timedelta(days=1)
    return d


def last_trading_day(delivery: pd.Timestamp) -> pd.Timestamp:
    """Two exchange days before delivery (weekdays)."""
    return pd.Timestamp(delivery) - pd.offsets.BDay(2)


def next_payment(coupon: float, maturity, day, *, commencement=None, short_first: bool = False):
    """``(amount A, tau, date)`` of a bond's next coupon after ``day``: tau = coupon periods
    to it (ACT/ACT ICMA). A bond in a long first period pays c plus its stub, a short one the
    stub only, at its first coupon date. Accrued interest at ``day`` = A - c * tau."""
    c = float(coupon)
    maturity, day = pd.Timestamp(maturity), pd.Timestamp(day)
    q = coupon_dates(maturity, day, 1)
    prev, nxt = q[0], q[1]
    tau = (nxt - day).days / (nxt - prev).days
    pay, amount = nxt, c
    if commencement is not None and pd.notna(commencement):
        c0 = pd.Timestamp(commencement)
        qc = coupon_dates(maturity, c0, 1)
        a0, a1 = qc[0], qc[1]
        stub = (a1 - c0).days / (a1 - a0).days
        regular_start = (a1 - c0).days <= 7 or (c0 - a0).days <= 7
        first_pay = a1 if (short_first or len(qc) < 3) else qc[2]
        if not regular_start and day < first_pay:
            amount = c * (stub if first_pay == a1 else 1.0 + stub)
            if first_pay > nxt:                        # before the skipped anniversary
                tau += 1.0
                pay = first_pay
    return amount, tau, pay


def accrued_at(coupon: float, maturity, day, **first) -> float:
    a, tau, _ = next_payment(coupon, maturity, day, **first)
    return a - float(coupon) * tau


def conversion_factor(coupon: float, maturity, delivery, notional_pct: float, *, commencement=None,
                      short_first: bool = False) -> float:
    """Eurex conversion factor (module docstring). ``commencement``: the interest start of a
    bond possibly still in its first coupon period at delivery (None = regular schedule)."""
    k = notional_pct / 100.0
    c = float(coupon)
    maturity, delivery = pd.Timestamp(maturity), pd.Timestamp(delivery)
    amount, tau, pay = next_payment(c, maturity, delivery, commencement=commencement, short_first=short_first)
    n = sum(1 for d in coupon_dates(maturity, delivery, 1) if d > pay)   # regular coupons after it
    v = (1.0 + k) ** -n
    dirty = (1.0 + k) ** -tau * (amount + c * (1.0 - v) / k + 100.0 * v)
    return round((dirty - (amount - c * tau)) / 100.0, 6)


def listed_contracts(root: str, day, n: int = 3) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """``(contract, delivery, last trading day)`` of the ``n`` quarterly contracts listed on
    ``day`` (Mar / Jun / Sep / Dec, each through its last trading day). The contract is named
    as Databento's raw symbol: ``FGBL SI <last trading day> PS``."""
    day = pd.Timestamp(day).normalize()
    out, y, m = [], day.year, ((day.month - 1) // 3 + 1) * 3
    while len(out) < n:
        d = delivery_day(y, m)
        ltd = last_trading_day(d)
        if ltd >= day:
            out.append((f"{root} SI {ltd:%Y%m%d} PS", d, ltd))
        m += 3
        if m > 12:
            y, m = y + 1, m - 12
    return out


def implied_repo(dirty_settle: float, settle, delivery, futures: float, cf: float, accrued_delivery: float,
                 coupons: float) -> float:
    """Implied repo (%, ACT/360) of buying the bond at ``settle`` and delivering it: invoice
    (futures x CF + accrued at delivery) plus coupons received, over the dirty price paid."""
    days = (pd.Timestamp(delivery) - pd.Timestamp(settle)).days
    if days <= 0 or dirty_settle <= 0:
        return float("nan")
    return ((futures * cf + accrued_delivery + coupons) / dirty_settle - 1.0) * 360.0 / days * 100.0


def basket(securities: pd.DataFrame, volumes: pd.Series, delivery, spec, as_of=None) -> pd.DataFrame:
    """Deliverable securities for a contract delivering on ``delivery``: German Federal
    coupon securities (Green included, inflation-linked not) with remaining term in
    ``[spec.min_years, spec.max_years]`` at delivery, original term at most
    ``spec.max_original_years`` (None = no cap) and cumulative issued volume (``volumes``, EUR
    m by ISIN) at least ``spec.min_volume_m``. Terms are measured in calendar months and days
    (``_years``), as Eurex states them ("8 1/2 to 10 1/2 years"). ``as_of``: only securities
    ISSUED by then (a new issue enters Eurex's list once issued, not at its auction - found
    2026-10-07: a Schatz auctioned the day before, settling two days later, wasn't listed yet)."""
    delivery = pd.Timestamp(delivery)
    s = securities[~securities["bill"] & ~securities["inflation_linked"]
                   & securities["type"].isin(["Schatz", "Bobl", "Bund", "Green"])]
    if as_of is not None:
        s = s[s["issue_date"] <= pd.Timestamp(as_of)]
    rem = s["maturity_date"].map(lambda m: _years(delivery, m))
    orig = [(_years(i, m)) for i, m in zip(s["issue_date"], s["maturity_date"])]
    ok = rem.between(spec.min_years, spec.max_years)
    if spec.max_original_years is not None:
        ok &= pd.Series(orig, index=s.index) <= spec.max_original_years
    ok &= s["isin"].map(volumes).fillna(0) >= spec.min_volume_m
    return s[ok].assign(remaining_years=rem[ok])


def _years(start, end) -> float:
    """Whole months + day fraction, in years (Eurex measures terms in years and months)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    months = (end.year - start.year) * 12 + (end.month - start.month)
    days = end.day - start.day
    return (months + days / 31.0) / 12.0
