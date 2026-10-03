"""Treasury futures basis: forwards, implied futures prices, net basis, implied repo, the
cheapest-to-deliver over bonds AND delivery dates, futures DV01. Pure functions, no I/O.

Conventions (CLAUDE.md 22):
* Prices are CLEAN per 100 face; ``ai`` = accrued interest. Invoice for bond i delivered on
  day d = F x CF_i + AI_i(d), so accrued cancels and the cost of delivering i is
  P_i - CF_i x F. Each bond IMPLIES a futures price, its forward over its conversion
  factor: F_i(d) = fwd_i(d) / CF_i. The short delivers the (bond, day) with the LOWEST
  implied futures price - the CTD - and in a deterministic world the futures' fair value
  is that minimum. Ranking by net basis / CF is the same ranking (net basis / CF =
  F_i - F); ranking by raw net basis is not.
* Forwards from the trade's settlement day ``s`` to delivery ``d``: full price financed at
  ``repo`` (%, simple ACT/360 over [s, d), rolling overnight per the funding model);
  coupons paid in (s, d] are reinvested at the same rate to d.
* Delivery windows (CME, verified against the contract specs; CLAUDE.md 22): first
  delivery = first business day of the delivery month; ZT/Z3N/ZF trade to the month's
  last business day and deliver up to the 3rd business day after it; the others stop
  trading 7 business days before the month's last business day and deliver up to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_END_LAST_TRADE = frozenset({"ZT", "Z3N", "ZF"})


def delivery_window(root: str, delivery_month, business_days: pd.DatetimeIndex) -> dict[str, pd.Timestamp]:
    """``first_delivery``, ``last_trading``, ``last_delivery`` for a contract."""
    bd = pd.DatetimeIndex(business_days)
    m0 = pd.Timestamp(delivery_month).replace(day=1)
    m1 = m0 + pd.offsets.MonthBegin(1)
    in_month = bd[(bd >= m0) & (bd < m1)]
    last_bd = in_month[-1]
    if root in MONTH_END_LAST_TRADE:
        after = bd[bd > last_bd]
        return {"first_delivery": in_month[0], "last_trading": last_bd, "last_delivery": after[2]}
    before = bd[bd < last_bd]
    return {"first_delivery": in_month[0], "last_trading": before[-7], "last_delivery": last_bd}


def forward_clean(price: float, ai_settle: float, ai_delivery: float, coupons: list[tuple[pd.Timestamp, float]],
                  settle, delivery, repo_pct: float) -> float:
    """Clean forward price for delivery on ``delivery``, bought on ``settle``."""
    settle, delivery = pd.Timestamp(settle), pd.Timestamp(delivery)
    n = (delivery - settle).days
    full_fwd = (price + ai_settle) * (1 + repo_pct / 100 * n / 360)
    for pay_day, amount in coupons:
        full_fwd -= amount * (1 + repo_pct / 100 * (delivery - pd.Timestamp(pay_day)).days / 360)
    return full_fwd - ai_delivery


def implied_repo_pct(price: float, ai_settle: float, ai_delivery: float, coupons, settle, delivery,
                     futures: float, cf: float) -> float:
    """The rate (%, ACT/360) at which buying the bond and delivering it into ``futures``
    breaks even: solves forward_clean(rate) = CF x F (coupons reinvested at that rate)."""
    settle, delivery = pd.Timestamp(settle), pd.Timestamp(delivery)
    n = (delivery - settle).days
    if n <= 0:
        return float("nan")
    receive = cf * futures + ai_delivery + sum(a for _, a in coupons)
    reinvest_days = sum(a * (delivery - pd.Timestamp(d)).days for d, a in coupons)
    # (P+AI)(1 + r n/360) - sum c_k (1 + r t_k/360) = receive - sum c_k  ->  linear in r
    full = price + ai_settle
    return float((receive - full) / (full * n / 360 - reinvest_days / 360) * 100)


def basis_table(bonds: pd.DataFrame, futures: float | None) -> pd.DataFrame:
    """Per bond x delivery date. ``bonds``: one row per (cusip, delivery) with ``price``,
    ``ai_settle``, ``ai_delivery``, ``coupons`` (list of (date, amount)), ``settle``,
    ``delivery``, ``repo`` (%), ``cf``. Adds ``fwd``, ``implied_futures`` (= fwd / cf),
    ``carry`` (= price - fwd), and - with a market ``futures`` price - ``gross_basis``,
    ``net_basis`` and ``irr``. Prices per 100."""
    out = bonds.copy()
    out["fwd"] = [forward_clean(r.price, r.ai_settle, r.ai_delivery, r.coupons, r.settle, r.delivery, r.repo)
                  for r in out.itertuples()]
    out["implied_futures"] = out["fwd"] / out["cf"]
    out["carry"] = out["price"] - out["fwd"]
    if futures is not None and np.isfinite(futures):
        out["gross_basis"] = out["price"] - out["cf"] * futures
        out["net_basis"] = out["fwd"] - out["cf"] * futures
        out["irr"] = [implied_repo_pct(r.price, r.ai_settle, r.ai_delivery, r.coupons, r.settle, r.delivery,
                                       futures, r.cf) for r in out.itertuples()]
    return out


def cheapest(table: pd.DataFrame) -> pd.Series:
    """The (bond, delivery date) row with the lowest implied futures price."""
    return table.loc[table["implied_futures"].idxmin()]


def futures_dv01(full_price_dv01: float, repo_pct: float, settle, delivery, cf: float) -> float:
    """Futures DV01 (price points per 1bp, per 100 face of futures) implied by one bond:
    its forward DV01 over its conversion factor. Forward DV01 = spot full-price DV01
    carried to delivery at the repo rate."""
    n = (pd.Timestamp(delivery) - pd.Timestamp(settle)).days
    return full_price_dv01 * (1 + repo_pct / 100 * n / 360) / cf


def price_schedule(coupon: float, maturity, at, per_year: int = 2) -> tuple[int, float, float]:
    """``(coupons still to come, fraction of the current period left, accrued)`` on ``at`` -
    computed once per bond and date, then reused for every yield (a root-finder or a
    simulation evaluates the price thousands of times on the same schedule)."""
    from infra.processing.treasury_prices import coupon_dates
    maturity, at = pd.Timestamp(maturity), pd.Timestamp(at)
    dates = coupon_dates(maturity, at, per_year)
    period = (dates[1] - dates[0]).days
    return len(dates) - 1, (dates[1] - at).days / period, coupon / per_year * (at - dates[0]).days / period


def clean_price_from_yield(yield_pct, coupon: float, maturity, at, per_year: int = 2, schedule=None):
    """CLEAN price per 100 at a street-convention yield (%) on settlement day ``at`` -
    vectorised over ``yield_pct`` (closed-form annuity: one evaluation per path, not per
    coupon). Regular periods; same convention as infra.processing.treasury_prices. Pass
    ``schedule`` (``price_schedule``) to skip rebuilding it."""
    n, w, accrued = price_schedule(coupon, maturity, at, per_year) if schedule is None else schedule
    y = np.asarray(yield_pct, dtype="float64") / 100.0 / per_year
    v = 1.0 / (1.0 + y)
    annuity = np.where(np.abs(y) > 1e-12, v ** w * (1 - v ** n) / np.where(np.abs(1 - v) > 1e-15, 1 - v, 1.0), n * 1.0)
    full = coupon / per_year * annuity + 100.0 * v ** (n - 1 + w)
    return full - accrued


def forward_yield(fwd_clean: float, coupon: float, maturity, delivery, per_year: int = 2) -> float:
    """The yield (%) at which the bond's clean price on ``delivery`` equals its forward."""
    from scipy.optimize import brentq
    sched = price_schedule(coupon, maturity, delivery, per_year)
    f = lambda y: float(clean_price_from_yield(y, coupon, maturity, delivery, per_year, sched)) - fwd_clean  # noqa: E731
    try:
        return brentq(f, -5.0, 40.0)
    except ValueError:
        return float("nan")


def simulate_delivery(fwd: np.ndarray, cf: np.ndarray, fwd_yield: np.ndarray, coupon: np.ndarray, maturity,
                      delivery, shocks_bp: np.ndarray, *, bump_bp: float = 0.0):
    """Delivery at ``delivery``: every bond's forward yield moves by its shock (bp) -
    ``shocks_bp`` either one per path, shared by all bonds (one factor), or one per path
    and bond (``paths x bonds``, a multi-factor model); ``bump_bp`` is added to all, for DV01. Each bond's simulated
    price is centred so its mean over paths equals its forward (martingale: the forward
    is the expected delivery-day price). Returns ``(implied futures per path [paths x
    bonds], fair futures = mean of the per-path minimum, CTD share per bond)``."""
    shocks_bp = np.asarray(shocks_bp, dtype="float64")
    paths = shocks_bp.shape[0]
    prices = np.empty((paths, fwd.size))
    for j in range(fwd.size):
        sh = shocks_bp[:, j] if shocks_bp.ndim == 2 else shocks_bp
        sched = price_schedule(coupon[j], maturity[j], delivery)
        p = clean_price_from_yield(fwd_yield[j] + (sh + bump_bp) / 100.0, coupon[j], maturity[j], delivery,
                                   schedule=sched)
        base = p if bump_bp == 0.0 else clean_price_from_yield(fwd_yield[j] + sh / 100.0, coupon[j],
                                                               maturity[j], delivery, schedule=sched)
        prices[:, j] = p - base.mean() + fwd[j]  # same centring with and without the bump
    implied = prices / cf
    m = implied.min(axis=1)
    share = np.bincount(implied.argmin(axis=1), minlength=fwd.size) / paths
    return implied, float(m.mean()), share


def bachelier_exchange(m: float, s: float) -> float:
    """E[max(S, 0)] for S ~ N(m, s^2): the two-bond (Margrabe-type) switch value when the
    spread between the runner-up's and the CTD's implied futures is normal."""
    from scipy.stats import norm
    if s <= 0:
        return max(m, 0.0)
    return float(s * norm.pdf(m / s) + m * norm.cdf(m / s))
