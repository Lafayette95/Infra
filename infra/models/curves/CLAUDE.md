# Curves sub-project: our US Treasury zero curve (additive to the root CLAUDE.md)

Code lives in the DATA layer so the daily cycle can run it (root CLAUDE.md 3a, 25):
`infra/analytics/treasury_curve.py` (pure math), `infra/pipeline/treasury_curves.py` (build /
store / read), `scripts/build_treasury_curves.py`; the Fed's GSW reference
`infra/api/fed_gsw_client.py` + `infra/pipeline/fed_gsw.py`. This doc is the methodology.

## 1. Why (user decisions 2026-10-03)
The basis models' MS add-on needed a clean per-bond richness signal: the first one - each
bond's yield-to-maturity minus ONE smooth spline of YTM on maturity - was a poor RV signal
(coupon effect; a bond's own residual drifting as it slides along the fit error; each bond
pulling the curve to itself), and it doubled UB's CTD error. A proper zero curve, fitted on
cash flows, also gives carry, rolldown and z-spreads, and a check on CMT.

## 2. Method
*   **Inputs:** FedInvest END OF DAY per CUSIP (15:30 New York BID, root CLAUDE.md 18), dirty =
    clean + accrued at T+1 settlement; every note and bond's cash flows from the securities
    table (regular schedules, `infra.processing.treasury_prices.coupon_dates`).
*   **Fit universe:** notes and bonds with > 0.5y left (`CURVE_FIT_MIN_YEARS`), minus each
    tenor's on-the-run and first off-the-run (`CURVE_FIT_EXCLUDE_RANKS` = 2, issue
    convention) - their liquidity premium would bend the curve; the Fed's GSW excludes them
    too. **Callable bonds excluded entirely** (11 issued 1979-84, Fiscal Data's `callable`
    flag): priced to their call, they 'yield' 10-13% as bullets and put the 2008-09 curve
    150-260bp off GSW until removed (found 2026-10-04).
*   **Weights:** price errors / (price x modified duration), so the fit is ~in yield bp.
*   **Two methods, both stored (`method` column):**
    *   `spline` (DEFAULT for per-bond work) - the discount function as a cubic regression
        spline, D(t) = 1 + sum b_k f_k(t), knots 1/2/3/5/7/10/15/20/25y; linear in prices, so
        weighted least squares, and a bond's LEAVE-ONE-OUT residual is closed form
        (e / (1 - h_ii)) - its own richness can't pull the curve to itself.
    *   `svensson` - the GSW form (continuously compounded zero rate, 6 parameters),
        nonlinear least squares warm-started from the previous day.
*   **Per bond** (`Derived/TreasuryRV`), every note and bond (on-the-runs too - they just
    don't shape the curve): `ytm`, `zspread_bp` (constant over the curve's zero rates that
    reprices it; negative = rich), `zspread_loo_bp` (spline only), `carry_curve_bp` (today's
    yield minus the yield of the curve-implied FORWARD price at the horizon - funding at the
    curve's own short rate, the screens' convention), `rolldown_bp` (today's yield minus the
    yield at the horizon on today's curve + the bond's z-spread); bp of the bond's own yield,
    + = gain; horizon `CURVE_HORIZON_DAYS` (91). **Repo carry is NOT here** - it needs the
    funding layer (root CLAUDE.md 20; the basis models' forwards already use it): expected
    return = repo carry + rolldown + change in z-spread (user discussion 2026-10-03; both
    carries kept - the market looks at both).
*   **Per day** (`Derived/TreasuryCurves`): parameters (`p0..`), `n_fit`, `rmse_bp`, par and
    zero yields at 1/2/3/5/7/10/20/30y.
*   **Speed:** 0.14s a day (Newton solves for yields / z-spreads; Svensson's objective
    vectorised over flattened cash flows) - 2008-2026 in ~10 minutes.

## 3. Validation (2008-09-02 .. 2026-10-01, 4,523 days)
*   **Fit error** (median, bp): spline 1.0-3.0 from 2010 (2008 13, 2009 4 - the crisis);
    Svensson 1.6-3.0 most years, 6.5 in 2022 (its 6 parameters can't follow 2022's shape).
*   **vs the Fed's GSW** (an INDEPENDENT fit - Fed Board staff, off-the-run notes and bonds,
    Svensson; NOT the CMT curve): our Svensson within ~1bp at every tenor (mean -1.1..+0.9,
    sd 0.9-2.7bp) from a different price source - the method reproduces. The spline differs
    more at 10-30y (sd 5-8bp), by design: it follows local shape Svensson smooths over.
*   **Which curve is which** (asked 2026-10-04): our CMT store comes from the US TREASURY
    (`infra/api/treasury_client.py`, home.treasury.gov "Daily Treasury Par Yield Curve
    Rates", root CLAUDE.md 13). The Fed's H.15 release - FRED's `DGS2`, `DGS10`, ... - REPUBLISHES
    those same Treasury CMT yields (the Fed distributes them, the Treasury fits them). The
    Fed's GSW (`feds200628.csv`) is a separate Fed Board staff fit. CMT is fitted too ("constant
    maturity" = read off a fitted par curve at fixed tenors), mainly on the on-the-runs.
*   **How the Treasury computes CMT** (its methodology page, checked 2026-10-04): inputs are
    ONLY the on-the-run bills (4/6/8/13/17/26/52 weeks), notes (2/3/5/7/10y) and bonds (20/30y);
    prices are indicative BID-side quotes (not trades) collected by the New York Fed at or near
    15:30 New York; since 2021-12-06 the yields bootstrap instantaneous forwards at the input
    maturities and a MONOTONE CONVEX interpolation of those forwards builds the curve (a
    quasi-cubic Hermite spline before). It passes THROUGH the input points ("true par rates") -
    an interpolation, not a smoothing fit. So at a benchmark tenor CMT ~ the on-the-run's own bid
    yield (hence our OTR benchmark within +-0.3bp of it, root CLAUDE.md 18), and our curve minus
    CMT measures the on-the-run premium. Same 15:30 bid timing as FedInvest's END OF DAY.
*   **vs CMT** (the Treasury's par curve, fitted mainly to ON-THE-RUN issues; 2016+): the
    spline sits +2..+3bp above CMT at 2/5/20/30y (sd ~2bp) - the on-the-runs' liquidity
    premium; 10y -1bp (to look at).
