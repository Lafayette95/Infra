"""Asymmetric reaction measures: positioning inferred from how the market MOVES (pure).

One question behind every measure: after normalising for how the market usually responds,
does it respond more in one direction than the other? A crowded position is visible as
amplified moves AGAINST it (stops, forced unwinds) and muted moves in its favour.

**Sign convention (all measures):** moves are in YIELD direction (+ = yields up = a loss
for duration longs). A POSITIVE asymmetry = yield-up moves are amplified = consistent with
a crowded LONG duration position (for a relative measure: long that instrument against
the factor). Inputs must already be in that direction; ``infra.pipeline.positioning``
flips futures P&L.

Three families, by what defines the shock:

1. **Surprise** (needs events and a consensus): the market's response to its own
   expected move, split by the sign of the news (``impact_betas``,
   ``surprise_asymmetry``); and whether the first reaction continues or reverses, split by
   its direction (``continuation_asymmetry``).
2. **Big moves, one instrument** (no events): realised semivariance asymmetry
   (Barndorff-Nielsen, Kinnebrock & Shephard 2010) and big-move tail asymmetry. Mostly
   MACRO skew (a zero bound, a hiking cycle), not positioning: read against its own
   history, or net it out with family 3.
3. **Relative to the usual covariance** (no events, no timestamps): each instrument's
   residual against a factor, with a beta estimated on a window that ENDS BEFORE the
   measurement window (estimated on the same days it would absorb the asymmetry being
   measured); then its excess beta on big factor-up moves minus on big factor-down moves
   (``relative_asymmetry``). A macro skew moves the factor itself and cancels here.

Every measure is a ratio of ROLLING SUMS over the last ``window`` observations (days,
grid steps or events), trailing only, so the value at t uses nothing after t; each comes
with the counts behind it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsum(x, window: int, min_obs: int):
    return x.rolling(window, min_periods=min_obs).sum()


def _ratio(num, den):
    out = num / den
    return out.where(np.isfinite(out))


def trailing_vol(moves, span: int, min_obs: int):
    """EWMA root mean square of moves (around 0), KNOWN BEFORE each move (shifted one
    observation): the scale a move is judged against."""
    return moves.pow(2).ewm(span=span, min_periods=min_obs).mean().pow(0.5).shift(1)


# ------------------------------------------------------------------ family 2

def semivariance_asymmetry(moves, window: int, min_obs: int):
    """(upside - downside) realised semivariance over the window, / total: in [-1, 1].
    Returns ``(asym, n)``."""
    up = moves.clip(lower=0).pow(2)
    down = moves.clip(upper=0).pow(2)
    su, sd = _rsum(up, window, min_obs), _rsum(down, window, min_obs)
    return _ratio(su - sd, su + sd), _rsum(moves.notna().astype(float), window, min_obs)


def rolling_skew(moves, window: int, min_obs: int):
    """Sample skewness of the moves over the window, around the WINDOW'S OWN MEAN: the
    shape of the distribution without its direction. Semivariance and tail asymmetry are
    measured around 0, so over a trending window they mostly report the trend (found on
    real data 2026-10-07: -0.56..-0.71 correlation with the CTA model's positions); the
    skew is the de-trended version."""
    return moves.rolling(window, min_periods=min_obs).skew()


def tail_asymmetry(moves, vol, k: float, window: int, min_obs: int, min_big: int):
    """Big moves only (|move / vol| > k): (sum z^2 of big up - of big down) / their total, in
    [-1, 1]; NaN with fewer than ``min_big`` big moves in the window. Returns
    ``(asym, n_up, n_down)``."""
    z = moves / vol
    big_up, big_down = (z > k), (z < -k)
    z2 = z.pow(2)
    su = _rsum(z2.where(big_up, 0.0).where(z.notna()), window, min_obs)
    sd = _rsum(z2.where(big_down, 0.0).where(z.notna()), window, min_obs)
    nu = _rsum(big_up.astype(float).where(z.notna()), window, min_obs)
    nd = _rsum(big_down.astype(float).where(z.notna()), window, min_obs)
    asym = _ratio(su - sd, su + sd).where(nu + nd >= min_big)
    return asym, nu, nd


# ------------------------------------------------------------------ family 3

def trailing_pc1(moves: pd.DataFrame, window: int, refit_every: int, min_obs: int) -> tuple[pd.Series, pd.DataFrame]:
    """A level factor: the first principal component of the panel's second moments (around
    0) over the trailing ``window`` complete rows, refitted every ``refit_every``
    observations and applied only to LATER rows (point in time). Loadings are signed so
    they sum to +1: the factor is a weighted average move, in the panel's units (bp).

    Returns ``(factor, loadings)``; loadings indexed by the row each fit applies from."""
    idx = moves.index
    complete = moves.notna().all(axis=1).to_numpy()
    vals = moves.to_numpy(dtype=float)
    factor = np.full(len(idx), np.nan)
    rows, loads = [], []
    current = None
    for i in range(len(idx)):
        if i % refit_every == 0:
            past = vals[max(0, i - window):i][complete[max(0, i - window):i]]
            if len(past) >= min_obs:
                w, v = np.linalg.eigh(past.T @ past / len(past))
                vec = v[:, np.argmax(w)]
                current = vec / vec.sum() if vec.sum() != 0 else None
                if current is not None:
                    rows.append(idx[i])
                    loads.append(current)
        if current is not None and complete[i]:
            factor[i] = vals[i] @ current
    loadings = pd.DataFrame(loads, index=pd.DatetimeIndex(rows), columns=moves.columns)
    return pd.Series(factor, index=idx, name="pc1"), loadings


def trailing_beta(y, f: pd.Series, window: int, gap: int, min_obs: int):
    """Beta of ``y`` on ``f`` through the origin over a ``window`` that ends ``gap``
    observations before each row (so a measurement window of ``gap`` rows never sees its
    own data in the beta)."""
    if isinstance(y, pd.DataFrame):
        num = y.mul(f, axis=0)
        den = (f * f).where(f.notna())
        both = y.notna().mul(f.notna(), axis=0)
        num = _rsum(num.where(both), window, min_obs)
        den = _rsum(pd.DataFrame({c: den for c in y.columns}).where(both), window, min_obs)
    else:
        both = y.notna() & f.notna()
        num = _rsum((y * f).where(both), window, min_obs)
        den = _rsum((f * f).where(both), window, min_obs)
    return _ratio(num, den).shift(gap)


def relative_asymmetry(moves: pd.DataFrame, factor: pd.Series, *, beta_window: int, window: int, k: float,
                       vol_span: int, min_obs: int, min_big: int) -> dict[str, pd.DataFrame]:
    """Per instrument: residual ``e = move - beta x factor`` (beta from BEFORE the window),
    then over the window, on BIG factor moves (|factor / its vol| > k):

        excess beta up   = sum(e f | f big up)   / sum(f^2 | f big up)
        excess beta down = sum(e f | f big down) / sum(f^2 | f big down)
        asym = up - down

    > 0: in big sell-offs the instrument sells off MORE than usual relative to how it
    rallies in big rallies - long that instrument against the factor is the pained side.
    Also the residuals' own semivariance asymmetry over the window (all moves)."""
    beta = trailing_beta(moves, factor, beta_window, window, min_obs)
    resid = moves - beta.mul(factor, axis=0)
    zf = factor / trailing_vol(factor, vol_span, min_obs)
    up, down = zf > k, zf < -k
    ef = resid.mul(factor, axis=0)
    f2 = factor * factor
    out = {}
    up_num = _rsum(ef.where(up, 0.0, axis=0).where(resid.notna()), window, min_obs)
    dn_num = _rsum(ef.where(down, 0.0, axis=0).where(resid.notna()), window, min_obs)
    valid = resid.notna()
    up_den = _rsum(valid.mul(f2.where(up, 0.0), axis=0).where(valid), window, min_obs)
    dn_den = _rsum(valid.mul(f2.where(down, 0.0), axis=0).where(valid), window, min_obs)
    n_up = _rsum(valid.mul(up.astype(float), axis=0).where(valid), window, min_obs)
    n_dn = _rsum(valid.mul(down.astype(float), axis=0).where(valid), window, min_obs)
    enough = (n_up >= min_big) & (n_dn >= min_big)
    out["excess_beta_up"] = _ratio(up_num, up_den).where(enough)
    out["excess_beta_down"] = _ratio(dn_num, dn_den).where(enough)
    out["rel_asym"] = out["excess_beta_up"] - out["excess_beta_down"]
    out["n_up"], out["n_down"] = n_up, n_dn
    out["beta"] = beta
    out["resid_semivar_asym"], _ = semivariance_asymmetry(resid, window, min_obs)
    out["resid_skew"] = rolling_skew(resid, window, min_obs)
    return out


def trailing_pcs(moves: pd.DataFrame, n: int, window: int, refit_every: int, min_obs: int) -> pd.DataFrame:
    """The first ``n`` principal components of a VOL-SCALED panel (each column divided by
    its own trailing RMS, known before the move - so a cross-asset factor isn't just the
    most volatile asset), refitted every ``refit_every`` rows on the trailing ``window``
    complete rows and applied only to later rows. Each PC is signed so its loadings sum
    to > 0 (with every instrument oriented "+ = a loss for longs", PC1 is the common
    pain direction). Units: vol-scaled moves."""
    scaled = moves / trailing_vol(moves, window, min_obs)
    vals = scaled.to_numpy(dtype=float)
    complete = np.isfinite(vals).all(axis=1)
    out = np.full((len(vals), n), np.nan)
    current = None
    for i in range(len(vals)):
        if i % refit_every == 0:
            lo = max(0, i - window)
            past = vals[lo:i][complete[lo:i]]
            if len(past) >= min_obs:
                w, v = np.linalg.eigh(past.T @ past / len(past))
                vecs = v[:, np.argsort(w)[::-1][:n]]
                current = vecs * np.where(vecs.sum(axis=0) < 0, -1.0, 1.0)
        if current is not None and complete[i]:
            out[i] = vals[i] @ current
    return pd.DataFrame(out, index=moves.index, columns=[f"pc{j + 1}" for j in range(n)])


def multifactor_relative(moves: pd.DataFrame, factors: pd.DataFrame, *, beta_window: int, window: int,
                         refit_every: int, k: float, vol_span: int, min_obs: int, min_big: int,
                         own_factor: dict[str, str] | None = None,
                         beta_min_obs: int | None = None) -> dict[str, pd.DataFrame]:
    """Family 3 with several factors. Per instrument: a trailing OLS of its move on the
    factors, fitted every ``refit_every`` rows on the ``beta_window`` rows that END
    ``window`` rows earlier (never the measurement window), gives the EXPECTED move
    ``m = F b``; the residual ``e = move - m``. Then, on BIG expected moves
    (|m / its trailing vol| > k):

        excess up   = sum(e m | m big > 0) / sum(m^2 | m big > 0)
        excess down = sum(e m | m big < 0) / sum(m^2 | m big < 0)
        asym = up - down

    > 0: when everything else said "this should fall" (a loss for longs) it fell MORE than
    usual, and when it should have rallied it rallied less: crowded long. With one factor
    this is the one-factor measure up to the beta's scale. ``own_factor``: an instrument
    that IS one of the factors (named factors) is regressed on the OTHERS only.
    ``beta_min_obs``: rows a beta fit needs (a 4-factor fit on 60 rows overfits: found
    2026-10-07, ES out-of-sample R^2 -4.5 in a spec starting 2018); default ``min_obs``."""
    own_factor = own_factor or {}
    beta_min_obs = min_obs if beta_min_obs is None else beta_min_obs
    F = factors.to_numpy(dtype=float)
    idx = moves.index
    expected = pd.DataFrame(np.nan, index=idx, columns=moves.columns)
    for col in moves.columns:
        use = [j for j, c in enumerate(factors.columns) if c != own_factor.get(col)]
        y = moves[col].to_numpy(dtype=float)
        Fu = F[:, use]
        ok = np.isfinite(y) & np.isfinite(Fu).all(axis=1)
        b = None
        exp = np.full(len(y), np.nan)
        for i in range(len(y)):
            if i % refit_every == 0:
                hi = i - window
                lo = max(0, hi - beta_window)
                rows = np.arange(lo, max(hi, lo))
                rows = rows[ok[rows]] if len(rows) else rows
                if len(rows) >= beta_min_obs:
                    X, yy = Fu[rows], y[rows]
                    b = np.linalg.lstsq(X, yy, rcond=None)[0]
            if b is not None and np.isfinite(Fu[i]).all():
                exp[i] = Fu[i] @ b
        expected[col] = exp
    resid = moves - expected
    zm = expected / trailing_vol(expected, vol_span, min_obs)
    up, down = zm > k, zm < -k
    valid = resid.notna() & expected.notna()
    em, m2 = resid * expected, expected * expected
    up_num = _rsum(em.where(up & valid, 0.0).where(valid), window, min_obs)
    dn_num = _rsum(em.where(down & valid, 0.0).where(valid), window, min_obs)
    up_den = _rsum(m2.where(up & valid, 0.0).where(valid), window, min_obs)
    dn_den = _rsum(m2.where(down & valid, 0.0).where(valid), window, min_obs)
    n_up = _rsum((up & valid).astype(float).where(valid), window, min_obs)
    n_dn = _rsum((down & valid).astype(float).where(valid), window, min_obs)
    enough = (n_up >= min_big) & (n_dn >= min_big)
    out = {"excess_up": _ratio(up_num, up_den).where(enough), "excess_down": _ratio(dn_num, dn_den).where(enough)}
    out["mf_asym"] = out["excess_up"] - out["excess_down"]
    out["n_up"], out["n_down"] = n_up, n_dn
    # share of the move the factors explain over the window (how much "usual covariance" there is)
    ss_res = _rsum((resid * resid).where(valid), window, min_obs)
    ss_tot = _rsum((moves * moves).where(valid), window, min_obs)
    out["r2"] = 1 - _ratio(ss_res, ss_tot)
    out["resid_skew"] = rolling_skew(resid, window, min_obs)
    return out


# ------------------------------------------------------------------ family 1

def standardise_surprises(surprises: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Each release's surprise / the RMS of its OWN EARLIER surprises (expanding, shifted:
    known before the print), so releases in different units share one scale."""
    out = {}
    for col in surprises.columns:
        s = surprises[col].dropna()
        rms = s.pow(2).expanding(min_periods=min_obs).mean().pow(0.5).shift(1)
        out[col] = s / rms
    return pd.DataFrame(out).reindex(surprises.index)


def impact_betas(z: pd.DataFrame, y: pd.DataFrame, *, refit: pd.DatetimeIndex, lookback: pd.Timedelta,
                 ridge: float, min_events: int) -> pd.DataFrame:
    """The market's usual response, as an EXPECTED MOVE per event window and instrument.

    ``z``: events x releases standardised surprises (0 / NaN where a release didn't print
    in that window - coincident releases, e.g. payrolls and the unemployment rate, share a
    row and are fitted JOINTLY, so their overlap isn't counted twice). ``y``: events x
    instruments window moves. At each ``refit`` date, a ridge regression through the
    origin on the events of the previous ``lookback`` (strictly before the date); applied
    to the events up to the next refit. Point in time. A release needs ``min_events``
    prints in the fit sample, else its coefficient is 0 (no expectation from it)."""
    z = z.fillna(0.0)
    times = z.index
    expected = pd.DataFrame(np.nan, index=times, columns=y.columns)
    bounds = list(refit) + [pd.Timestamp.max]
    for a, b in zip(bounds[:-1], bounds[1:]):
        fit_rows = (times < a) & (times >= a - lookback)
        apply_rows = (times >= a) & (times < b)
        if not apply_rows.any() or not fit_rows.any():
            continue
        Z = z.loc[fit_rows]
        cols = [c for c in Z.columns if (Z[c] != 0).sum() >= min_events]
        if not cols:
            continue
        Zc = Z[cols].to_numpy()
        lhs = Zc.T @ Zc + ridge * np.eye(len(cols))
        for inst in y.columns:
            yy = y.loc[fit_rows, inst].to_numpy()
            ok = np.isfinite(yy)
            if ok.sum() < min_events:
                continue
            beta = np.linalg.solve(Zc[ok].T @ Zc[ok] + ridge * np.eye(len(cols)), Zc[ok].T @ yy[ok]) \
                if not ok.all() else np.linalg.solve(lhs, Zc.T @ yy)
            expected.loc[apply_rows, inst] = z.loc[apply_rows, cols].to_numpy() @ beta
    return expected


def surprise_asymmetry(y: pd.DataFrame, x: pd.DataFrame, window: int, min_obs: int, min_side: int) -> dict:
    """Over the last ``window`` events: the slope of the realised move on the EXPECTED move
    (through the origin), separately where the news was hawkish (x > 0: data implying
    higher yields) and dovish (x < 0):

        b_up = sum(x y | x > 0) / sum(x^2 | x > 0),  b_down likewise,
        asym = (b_up - b_down) / (|b_up| + |b_down|)     in [-1, 1]
        b_all = the same slope on both sides together

    b_up = b_down = 1 is a normal market; asym > 0: hawkish news moves yields more than
    dovish news of the same expected size - pain for longs, crowded long. Bounded on
    purpose: (b_up - b_down) / (b_up + b_down) explodes when the market barely reacts
    (found on daily data 2026-10-07: sd 420). Read it with ``b_all`` - an asymmetry where
    b_all is near 0 is noise. NaN unless each side has ``min_side`` events."""
    up, down = x > 0, x < 0
    xy, x2 = x * y, x * x
    valid = (x.notna() & y.notna())
    num_u = _rsum(xy.where(up & valid, 0.0).where(valid), window, min_obs)
    num_d = _rsum(xy.where(down & valid, 0.0).where(valid), window, min_obs)
    den_u = _rsum(x2.where(up & valid, 0.0).where(valid), window, min_obs)
    den_d = _rsum(x2.where(down & valid, 0.0).where(valid), window, min_obs)
    n_u = _rsum((up & valid).astype(float).where(valid), window, min_obs)
    n_d = _rsum((down & valid).astype(float).where(valid), window, min_obs)
    b_u, b_d = _ratio(num_u, den_u), _ratio(num_d, den_d)
    b_all = _ratio(num_u + num_d, den_u + den_d)
    ok = (n_u >= min_side) & (n_d >= min_side)
    return {"b_up": b_u.where(ok), "b_down": b_d.where(ok), "b_all": b_all.where(ok),
            "surprise_asym": _ratio(b_u - b_d, b_u.abs() + b_d.abs()).where(ok),
            "n_up": n_u, "n_down": n_d}


def continuation_asymmetry(first: pd.DataFrame, after: pd.DataFrame, window: int, min_obs: int,
                           min_side: int) -> dict:
    """Does the first reaction to an event continue or reverse, by its direction? Over the
    last ``window`` events: c_up = sum(first x after | first > 0) / sum(first^2 | first > 0),
    c_down likewise; asym = c_up - c_down. > 0: sell-offs keep going (or reverse less)
    than rallies - shorts covering little, longs being stopped out: crowded long."""
    up, down = first > 0, first < 0
    valid = first.notna() & after.notna()
    fa, f2 = first * after, first * first
    c_u = _ratio(_rsum(fa.where(up & valid, 0.0).where(valid), window, min_obs),
                 _rsum(f2.where(up & valid, 0.0).where(valid), window, min_obs))
    c_d = _ratio(_rsum(fa.where(down & valid, 0.0).where(valid), window, min_obs),
                 _rsum(f2.where(down & valid, 0.0).where(valid), window, min_obs))
    n_u = _rsum((up & valid).astype(float).where(valid), window, min_obs)
    n_d = _rsum((down & valid).astype(float).where(valid), window, min_obs)
    ok = (n_u >= min_side) & (n_d >= min_side)
    return {"cont_up": c_u.where(ok), "cont_down": c_d.where(ok), "cont_asym": (c_u - c_d).where(ok),
            "n_up": n_u, "n_down": n_d}
