# infra/models/basis - Treasury futures basis & delivery-option models (additive to the root and `infra/models/CLAUDE.md`)

Everything in the root `CLAUDE.md` and `infra/models/CLAUDE.md` applies (prepare -> fit ->
predict, point in time, specs in a registry, inputs in their own module). This file adds
what is specific to the basis models: what they answer, every convention and why, the
research findings behind each design choice (with their numbers), what was fixed along the
way, and how to run it. Open issues live in the root `TOFIX.md` ("Basis: ..." entries).

## 1. What and why
*   **The question:** for each US Treasury futures contract (ZT, Z3N, ZF, ZN, TN, ZB, UB),
    which deliverable will be delivered and when, what the short's delivery options are
    worth (quality, timing), and the futures' DV01 - the input the daily cycle's bmk step
    has been missing for bond futures (`TOFIX.md`), and the basis to compare with a DLV
    screen.
*   **Inputs** (all on disk, never fetched here): the delivery baskets with per-contract
    conversion factors (root `CLAUDE.md` 18), FedInvest cash prices and yields per CUSIP
    (18), the futures snap at 15:30 New York (23), funding model v1 (20), the CMT par curve
    (13) and the auction-derived reference table (18).
*   **The structure (user decisions 2026-10-02; reshaped the same day - see below):** PRICING
    TIERS of increasing realism (M0-M3), SWITCHABLE ADD-ONS (T, IV, MS) any suitable tier can
    carry, and a separate EXPLANATORY layer on top of whatever a tier leaves unexplained. Each tier /
    add-on combination is a spec in `config.BASIS_MODELS` (a class in `model.TIERS`, flags
    on `BasisSpec`), all scored on the same bench (section 6).

    **Pricing tiers** (what moves, and how the short chooses):

    | Tier | Adds | Answers | Status |
    |---|---|---|---|
    | **M0 deterministic** | forward CTD = lowest implied futures price over bonds x {first, last} delivery day; option value 0 | carry-only fair value, baseline futures DV01 | built |
    | **M1 one factor** | one normal level shock to every deliverable's forward yield; quality option by simulation; Margrabe/Bachelier two-bond check | the 6%-crossing switch value, first CTD probabilities | built |
    | **M2 basket factors** | level + a PCA on the basket's SPREADS at the horizon; predecessor backfill for young bonds; EXPECTED NEW ISSUES entering the basket; optionally FAT-TAILED spreads (`spread_df`, spec `M2t`: Student-t, one mixing draw per path) | relative moves that decide the CTD; per-bond probabilities; new-issue CTDs | built (fat tails moved here from M3, user decision 2026-10-02: the spread distribution's SHAPE belongs with M2's spread model) |
    | M3 stochastic funding & timing | the general-collateral funding rate as a front-end FACTOR (SR1-implied path + shocks), so carry moves on each path for the whole basket and the early/late delivery choice is made PER PATH from that path's carry | carry-driven timing - every contract, most in regime shifts (the 2022-24 negative-carry years) | planned - next tier. Macro and common to all bonds; deliberately NOT bundled with the bond-specific microstructure (add-on MS), so each is testable and attributable on its own |

    **Add-ons** (spec flags; T and IV attach to M1, M2, M3; MS to M2, M3; M0 has no simulation to attach to):

    | Add-on | Flag | What | Status |
    |---|---|---|---|
    | **T - timing options** | `timing_options` (spec `M2T`) | the wild card (Bermudan over the intention days, per-window variance with FOMC / quarter-end / month-end multipliers - the DEFAULT) and the end-of-month switch (futures frozen after the last trading day); the quality simulation then stops at the last trading day | built. With M3, T must read M3's PER-PATH delivery decision instead of the deterministic first/last choice (one window, no EOM under negative carry) - a design point inside M3 |
    | IV - implied vol | (planned) | rescales the factor vols in `fit` from Treasury-futures options (CBOT options on ZT/ZF/ZN/ZB/UB, on GLBX): the level from at-the-money implied vol, possibly the curve factors from options across tenors | planned. Needs a PAID Databento fetch (verify the option roots' symbology as for `SR3.OPT`, dry-run the cost for 2019-2026 with near-the-money strikes only - Rule 2.3, user approval). Plumbing exists: the daily options pipeline and Black-76 in `infra/analytics`. Caveat: the options are American - Black-76 approximates them (fine near the money, short-dated) |
    | MS - microstructure | (planned) | the bond-specific side, modelled JOINTLY because it's one phenomenon seen twice - an on-the-run bond is rich partly BECAUSE it's special, and both switch at the same dates: (1) SPECIALNESS linked to idiosyncratic moves (measured on the NY Fed lending fees, not assumed - not the other assistant's "2bp per 1bp"); (2) CALENDAR effects - timed, directional spread moves around known dates: auction concessions (cheapening into an auction or reopening, richening after), reopening supply (a lasting cheapening at settlement), a successor issue settling (the 1-old losing its on-the-run premium), month-end index extension (new issues entering the indices). Each applied as an expected drift of the affected bond's forward yield to delivery (+ extra variance where the event adds dispersion), NET of what funding's specialness lifecycle already puts in carry (or the same richness is counted twice) | planned. Attaches to M2 or M3 (needs M2's per-bond idiosyncratic structure; not M0/M1). Matters most where recent issues sit in the basket: TN, ZN, ZT, UB. Sizes are small (fractions of a bp to a few bp of yield) but comparable to CTD/runner-up gaps (ZN median 0.26/32). **Future-proofing (user decision 2026-10-02): the calendar effects are read through an EVENT-PROFILE interface** - expected spread path and extra variance by event type x tenor x days from the event - whose first provider is MS's own in-study event study (FedInvest yields from 2008, auction / reopening dates from the auctions store, successor dates from the OTR map); a later full-fledged, project-wide event study becomes a second provider that MS reads instead, with no change to MS |

    **The explanatory layer: the basis SPREAD model** (planned) - NOT a pricing tier. What a
    tier leaves unexplained (observed minus model option value) is, in rate terms, the
    CTD's implied repo minus funding v1:

        implied repo - v1 funding ~ TERM premium (left out by the rolling-overnight choice;
          DVP 8-30d measured +3.4bp) + haircut / margin capital cost + balance-sheet charges
          (quarter-ends) + demand-driven RICHNESS (asset managers long futures)

    - i.e. the HEDGE FUNDS' HARVEST spread in the basis trade. A purely STATISTICAL richness
    term (fit a curve to the residual, e.g. decaying into delivery) would add nothing - the
    residual IS the premium - so this layer EXPLAINS it instead: regress the spread on
    funding and positioning variables, report it as its own component (option value ->
    spread -> residual), never fold it into the fair value (that would make fair = market
    by construction). Candidate drivers: SOFR p75 - median (already INSIDE funding v1, so
    its coefficient TESTS v1: ~0 = v1 prices the client spread right, > 0 = hedge funds'
    cost moves more than 1:1 with it), SOFR p99 - p75 (funding-tail stress), the DVP term
    premium (OFR term buckets minus the futures-implied path), quarter-end proximity,
    positioning (CFTC's free weekly Traders in Financial Futures: asset managers' and
    leveraged funds' Treasury-futures positions - a small new free pipeline source), reserves
    (the Fed's H.4.1 on FRED). Its use: a "fair spread" given funding conditions (deviations
    = a signal), the decomposition of the basis, and a validation of the funding model.
    Needs: the M2T (ideally M3) residuals, so the spread is measured AFTER the option values;
    the CFTC source. (This replaces the earlier "M4 market-implied" tier: its timing options
    became add-on T, implied vol add-on IV, and its "futures-richness term" this layer. The
    earlier M3 "funding & microstructure" was split the same day: stochastic funding stayed M3,
    specialness + calendar effects became add-on MS, general fat tails moved to M2.)

    The curve-building sub-project (fitted curve, spread to spline) is NOT a prerequisite
    (user decision 2026-10-02): the basket's own spread PCA defines the relative moves; a
    fitted curve can come later as a variant (e.g. for new-issue proxies).

## 2. Conventions and decisions (2026-10-02)
*   **Rank in futures units.** Invoice for bond i = F x CF_i + AI_i, so the cost of
    delivering i is P_i - CF_i x F (accrued cancels). Each bond IMPLIES a futures price,
    F_i = fwd_i / CF_i (its forward to the delivery day over its CF); the CTD is the
    (bond, day) with the LOWEST implied futures price, and a deterministic fair futures
    price is that minimum. Net basis / CF = F_i - F, so ranking by net basis / CF is the
    same ranking; ranking by RAW net basis is not (CF 0.9 bond with net basis 2/32 = 2.2/32
    of futures; CF 0.6 bond with 1.5/32 = 2.5/32 - the first is cheaper despite the larger
    raw net basis). IRR only approximates the right ranking (it divides by the full price,
    ~ CF x F). Before delivery: F_i(t) = fwd_i / CF_i; at delivery the forward IS the spot,
    so it collapses to argmin P_i / CF_i.
*   **The short's terminal choice ignores history.** On each simulated path the short
    delivers argmin P_i(T) / CF_i: what was bought earlier and the carry earned since are
    SUNK and identical whichever bond is delivered. The option value is fwd_CTD0 / CF_0 -
    E[min_i P_i(T) / CF_i], in futures points - the initial CTD's terminal net basis per CF,
    averaged over paths. A gain of V bond points per contract lowers the fair futures price
    by V / CF_ctd (the invoice is CF x F).
*   **Critique of the user's brainstorm with another assistant** ("Basis Model
    Simulation.pages", read 2026-10-02), kept here so the reasoning isn't re-derived:
    1. Its "time-inconsistency" fix - pick the bond with the highest realised PATH return
       (P&L / day-0 price) - is WRONG: it makes the delivery choice depend on sunk history.
       The real source of IRR vs net-basis disagreement is UNITS (above), not capital
       efficiency. If the simulation disagrees with the day-0 IRR label, the simulation is
       right (given its assumptions).
    2. Its "hidden assumptions" against the terminal-net-basis valuation (sunk carry,
       bid-ask, re-hedging the tail) concern MONETISING the option, not its value; its
       "static vs dynamic futures price" alternative is the same quantity computed twice.
    3. "250-500 days + EWMA lambda 0.94" is self-contradictory (0.94 = ~11-day half-life);
       and a near-singular 21x15 covariance isn't fatal for PCA (the top components are
       well estimated). Adopted instead: SLOW weights for structure, FAST for vol.
    4. Confident but unverified market claims - "forced switches widen co-CTD gaps",
       "reopened bonds jump into the CTD spot", "specialness drops 2bp per 1bp of
       idiosyncratic squeeze" - are not used; each can be tested on our data (M3).
    5. Small errors: its drift E[dP] = (r - coupon) P dt mixes coupon rate and yield; its
       "massive EOM option" was asserted, not shown (see section 4 for what we measured).
    Kept from it: carry drives the timing option; fat-tailed idiosyncratic moves; the
    user's own point that constant repo biases the quality option (a squeezed bond earns
    special carry); splitting general collateral from specialness (= funding v1).
*   **Everything is per 100 face.** Prices, conversion factors, implied futures, net basis and the futures DV01 (points per bp) are per 100 face, so a contract's notional (ZT and Z3N are $200k, the others $100k) never affects the CTD or the probabilities - only money per contract (ZT's 0.019 = ~$38/bp per contract; the bmk step's `FuturesRoot.point_value` already carries it).
*   **Delivery day:** first or last business day of the delivery window, whichever gives
    the lower implied futures price (the sign of carry): positive carry -> last, negative
    -> first. Windows from the CME rules (`infra.analytics.futures_basis.delivery_window`):
    ZT/Z3N/ZF trade to the month's last business day and deliver up to 3 business days
    later; the others stop 7 business days before it and deliver up to it. Business days
    approximate SIFMA's (federal holidays + Good Friday).
*   **Prices at one moment, 15:30 New York** (section 3a): FedInvest END OF DAY and the
    futures' 15:30 mid from the snap store (`infra.config.BASIS_SNAP` = NY1530); the 15:00
    settlement only as a flagged fallback. Outputs also carry the settlement and the
    observed option value against it (`option_value_obs_vs_settle_32`), for comparison with
    a DLV screen (user decision 2026-10-02).
*   **Cash is a BID price** (section 3a). `BasisSpec.cash_mid_frac` (0.5) moves it toward mid
    by that fraction of HALF the posted spread - a quarter of it.
*   **Funding** = `infra.pipeline.financing` v1 (rolling overnight, SR1-implied SOFR + SOFR
    p75 client basis - per-CUSIP specialness), as of the day, per bond and delivery day.
    Forwards finance the full price and reinvest coupons at that rate.

## 3. Research findings (2026-10-02) - data and alignment
### 3a. FedInvest END OF DAY is a 15:30 New York BID price
*   **The time, measured, not assumed:** the CTD's daily gross-basis noise (std of daily
    changes, 32nds, 2024-2026) against futures mids at each New York time has a sharp
    minimum exactly at 15:30:

    | NY time | 14:30 | 15:00 | 15:20 | **15:30** | 15:40 | 16:00 | settlement (15:00) |
    |---|---|---|---|---|---|---|---|
    | ZN | 3.03 | 2.06 | 1.09 | **0.49** | 1.14 | 1.68 | 2.07 |
    | ZB | 5.56 | 3.88 | 2.15 | **1.00** | 2.20 | 3.54 | 3.98 |
    | UB | 6.46 | 4.68 | 2.47 | **0.81** | 2.33 | 4.10 | 4.78 |

    So the street's 15:00 convention (settlement vs 15:00 cash) would mismatch our two legs
    by half an hour; the basis is computed at 15:30. (The 15:30 mid sits a median 0.75/32
    from the 15:00 settlement on the bond fronts - the half hour's market move.)
*   **The side:** END OF DAY equals FedInvest's "Sell" column (median position between Sell
    and Buy: exactly on Sell). Naming trap: FedInvest's "Buy" is the price an investor PAYS
    (the ask), "Sell" what they receive (the bid) - an early analysis read the spread as
    sell - buy and got the adjustment's sign backwards. The posted Buy/Sell spread is a
    CONVENTION (exactly 0.5 / 1 / 2 32nds by maturity), not the market's, so the true mid
    is unknown (`TOFIX.md`).
### 3b. The first net-basis exploration (before any model)
*   Gross basis, carry to the last delivery day at funding v1, net basis and implied repo of
    every deliverable of each front contract, daily 2019-2026 (181,271 bond-days).
*   **TN and UB looked textbook** (positive net basis, implied repo well under funding);
    **ZT/ZF/ZN/ZB showed a CTD net basis median ~0, negative half the time** - not noise
    but a regime: CTD implied repo minus SOFR (ZT/ZF/ZN pooled, median): 2019 +20bp, 2020
    +13, 2021 +3, 2022 -15, 2023 -77, 2024 -37, 2025 +4, 2026 +5. That is the cash-futures
    BASIS TRADE (asset managers' demand keeps futures rich; hedge funds buy cash, sell
    futures, fund in repo; Barth & Kahn's "cash-futures disconnect" for 2019-20).
*   **March 2020, the basis-trade unwind, reproduced:** CTD net basis on 2020-03-18 ZN
    -14/32, ZT -11, ZF -10, ZB -39 (from ~-2 on 2020-03-02), recovering by month-end.
*   **Aligned snaps matter:** CTD daily net-basis noise 0.6-1.2/32 aligned vs 0.9-9/32 pairing
    15:00 settlements with 15:30 cash.
*   **The delivery day must depend on carry:** computing everything to the LAST delivery day
    made 2023-24 implied repo implausibly low (ZT -135bp in 2023) - carry was negative then,
    so the short delivers EARLY. Hence the {first, last} choice in every tier.
*   **Funding layers matter little, specialness barely:** the client basis moves the CTD's
    net basis by ~0.27/32; specialness exceeds 1bp on the CTD on ~5% of days (TN, ZT),
    almost never otherwise; basket bonds special > 5bp, 2016-2026: TN 5.8% (the new 10y is
    deliverable), TWE 2.4%, ZF 1.4%, others < 1%.
### 3c. The basket's structure: why M2 models SPREADS at the HORIZON
*   **CMT gets the level right but relative moves wrong:** over a year, a bond's daily move
    (~4bp) minus CMT's change at its maturity leaves 0.4-1.1bp; but CMT explains only 10-13%
    of the spread moves BETWEEN deliverables (ZN, ZB) and less than nothing for ZT (-3%) and
    UB (-32%) - linear interpolation between sparse CMT points adds noise to spreads. The
    quality option is an option on those spreads, so a CMT-only model misprices it.
*   **Strong local structure:** of what's left after CMT, 73-84% is one factor common to the
    basket (a sector move, or CMT's own fitting error there).
*   **A PCA on yield LEVELS hides the switch drivers:** PC1 = 98-100% of the variance, so
    everything that changes the CTD sits in components holding < 2%, poorly estimated in a
    short window. Hence: remove the level, PCA the spreads.
*   **Spreads mean-revert hard** - variance ratio var(h-day spread change) / (h x var(1-day)),
    2024-2026 (1 = random walk):

    | Basket | 5d | 20d | 60d | 1-day spread sd |
    |---|---|---|---|---|
    | ZT | 0.23 | 0.08 | **0.04** | 0.75bp |
    | ZN | 0.39 | 0.23 | 0.19 | 0.25bp |
    | ZB | 0.72 | 0.49 | 0.32 | 0.33bp |
    | UB | 0.49 | 0.32 | 0.30 | 0.16bp |

    (daily pricing noise in FedInvest's marks plus relative-value reversion). The level is
    closer to a random walk (0.5-1.0 at 60 days). So spread covariance is estimated AT THE
    HORIZON from overlapping h-day changes, never sqrt(h)-scaled from daily (that
    overstated relative variance 3-25x: ZT's option value came out 1.9/32 against 0.44
    observed, its CTD probability 32%).
### 3d. Where M0 misses: new issues and near-ties
*   M0's CTD (deterministic) vs the realised CTD at the last trading day, 2019-2026:

    | Root | Misses | Realised CTD not yet in the basket on the prediction day | Median runner-up gap on miss days |
    |---|---|---|---|
    | ZT | 1,039 / 1,897 | **17%** | 1.9/32 |
    | TN | 63 / 1,897 | **84%** | 5.0/32 |
    | ZN | 405 / 1,897 | 2% | 0.26/32 |
    | ZB, UB, ZF | 2-429 | 0% | 0.8-1.2/32 |

    New issues matter for ZT (e.g. ZTZ2's CTD was the 2y auctioned 2022-09-26) and TN (the
    new 10y) -> expected issues in M2; ZN's misses are near-ties -> probabilistic tiers.
### 3e. Timing options: the user's hypothesis that UB's value is the wild card
*   **The puzzle:** UB's observed option value (fair_M0 - market) has a median of 6.2/32
    while M1 explains ~0 (yields far below the 6% notional, so parallel moves rarely
    switch). The user's hypothesis: the wild card, since UB's CFs are far from 1.
*   **The mechanics, derived** (a DV01-hedged short holds h = 1/CF face of the CTD per
    futures): after the 15:00 settlement fixes the invoice, delivering locks the futures leg
    while the (h - 1) TAIL keeps the window's move: payoff (1/CF - 1) x dP. CF < 1 (a long
    tail) pays on a RALLY after the settlement; CF > 1 on a sell-off. The value scales with
    |1/CF - 1| x the window's price vol - UB's CTD CFs 0.61-0.74 give the largest tail.
    After the last trading day the futures are FROZEN: the tail itself is plain exposure
    (no option - it can be sold any time); the end-of-month option is SWITCHING bonds
    against the frozen price (a ranking level moves change at first order).
*   **The window, 15:00 -> 19:00 New York** (14:00 -> 18:00 Chicago: settlement to the
    18:00 Chicago notice deadline). Share of a day's futures variance in it (bbo-1m mids as
    the proxy for cash moves): 6.1-6.7% overall 2024-2026; on ORDINARY days 2019-2026: ZT
    4.0%, ZF 4.5%, ZN 4.9%, TN 5.3%, ZB 5.6%, UB 6.7%.
*   **Event days carry far more** (window variance vs ordinary days, 2019-2026; the user's
    point):

    | Day | TN | UB | ZB | ZF | ZN | ZT |
    |---|---|---|---|---|---|---|
    | FOMC day (62) | 3.1x | 1.7x | 2.5x | 5.4x | 4.1x | 6.3x |
    | Quarter-end (18) | 2.3x | 2.4x | 2.8x | 2.0x | 2.1x | 2.1x |
    | Month-end (28) | 2.2x | 1.3x | 1.9x | 3.1x | 2.4x | 2.9x |

    FOMC: statement 14:00, press conference ~14:30-15:30, straddling the settlement. The
    tail is one-off events (ZN's largest window move: +29.5/32 on 2025-04-02, the tariff
    announcement after the close).
*   **The last intention day's later deadline is moot for the wild card:** 20:00 Chicago (21:00
    New York; 1.5-1.75x the variance of a 19:00 window) - but that day falls after the last
    trading day, when the futures are already frozen, so it's part of the end-of-month
    period.
*   **The evidence:** the wild card alone (flat window share, Bermudan, from each day's CF,
    DV01 and vol), median 32nds, vs observed:

    | Root | CTD CF | Wild card (model) | Observed, final month | Observed, all days |
    |---|---|---|---|---|
    | UB | 0.61 | **10.5** | **12.3** | 6.2 |
    | TN | 0.80 | 1.6 | 0.8 | 0.7 |
    | ZB | 0.85 | 1.5 | -0.6 | -0.45 |
    | ZN | 0.85 | 0.7 | 0.1 | -0.4 |
    | ZF | 0.88 | 0.4 | -0.2 | -0.6 |
    | ZT | 0.94 | 0.07 | -0.5 | -0.7 |

    UB's final-month value is almost all wild card; the ordering across roots follows the
    tail; the others sit 1-2/32 below - the futures richness plus the cash bid bias.
*   **What the tail does NOT explain:** UB's observed value GROWS into the delivery month
    (2.8/32 at 120+ days, 6-7 at 30-90, 12.3 in the final month), and over time it doesn't
    track tail x vol (within-UB regression slope -0.41, R^2 0.13; observed peaked in 2023 at
    12/32 when CFs rose, a SMALLER tail). Reading: a timing option (UB-dominant) PLUS a
    futures-richness premium that fades into delivery (convergence forces it out) - the
    latter also pushes ZB/ZF/ZN/ZT negative. That premium is what the explanatory spread layer (section 1) is for (`TOFIX.md`).

### 3f. What decides calibration: the spreads' SIZE vs their SHAPE (2026-10-02)
*   **The realised CTD must follow the realised delivery TIMING.** Scoring every contract at
    its last trading day was wrong whenever carry was negative: the shorts deliver at the
    START of the delivery month, and for ZT - whose last trading day is the month's END -
    the bond cheapest a month later was beside the point. On 2024-05-01 for ZTM4, M0 correctly
    ranked the 0.75% note (implied 101.48) over the 4.625% (101.72): CME's conversion factor
    rounds a ZT bond's term DOWN to whole months, which favoured the 0.75% 31-March maturity
    over the 15-March one; carry was negative (funding 5.37%), so delivery was early June.
    Every 2022-2025 ZT "miss" was this. Fixed (`validate.realised_ctd`): M0 on the delivery
    month's FIRST INTENTION day decides; if early (negative carry) its CTD is the realised
    one, else the last trading day's. 78 of 186 contracts 2019-2026 were early deliveries;
    23 realised CTDs changed (ZT 16, ZB 3, UB 2, ZN 2); **ZT's M0 hit rate went 44% -> 82%**.
*   **Corrected scores** (Brier, all horizons pooled; lower = better):

    | Root | M0 | M1 | M2 |
    |---|---|---|---|
    | TN | 0.066 | 0.063 | **0.059** |
    | UB | 0.306 | **0.162** | 0.169 |
    | ZB | 0.404 | **0.296** | 0.349 |
    | ZF | 0.002 | 0.001 | 0.002 |
    | ZN | 0.397 | 0.311 | **0.303** |
    | ZT | 0.358 | **0.317** | 0.368 |

    M2 scored WORSE than M1 on ZB and ZT, which first looked like overstated relative
    variance.
*   **It isn't the size - M2's spread variance is about right, M1's far too small.** For
    sampled days, the CTD / runner-up implied-futures spread's change to the decision day,
    realised vs predicted (32nds):

    | Root | realised sd | M1 predicted (median) | M2 predicted (median) | M2 z-score sd (ideal 1) |
    |---|---|---|---|---|
    | UB | 7.85 | 3.56 | 5.73 | 0.91 |
    | ZB | 6.56 | 2.42 | 6.17 | 0.63 |
    | ZN | 2.75 | 1.05 | 1.71 | 0.92 |
    | ZT | 1.51 | 0.43 | 1.07 | 1.32 |

    (M1's z-scores blow up when two bonds have similar DV01/CF and its predicted sd ~ 0.)
*   **It's the SHAPE:** (1) no drift - the mean z-score is ~0 (-0.24..+0.16), forwards aren't
    biased; (2) FAT TAILS - excess kurtosis 2.6-5.2: the spread usually barely moves (the
    runner-up overtakes in only 16-18% of cases) and occasionally jumps, so a normal with the
    RIGHT variance over-weights moderate moves and predicts too many switches - M1's too-
    small variance accidentally fits the quiet body; (3) DILUTION in big baskets - on ZB, M2
    puts a median 61% on the realised CTD (M1 93%) and 11% on bonds outside the top two: each
    of 50 bonds carries its own normal noise, leaking probability to bonds that never win.
*   **Fix tried: Student-t spreads** (`BasisSpec.spread_df`, spec `M2t`; section 4).
    Results: section 7.

## 4. The models
*   **M0 (`DeterministicBasis`)**: no fitting. Per contract: CTD, delivery day, fair futures,
    the CTD's net basis / implied repo / funding, the OBSERVED option value (fair - market,
    32nds - everything M0 leaves out), the same against the 15:00 settlement, futures DV01 =
    the CTD's forward DV01 / CF, the runner-up's gap in futures 32nds.
*   **M1 (`OneFactorBasis`)**: `fit` = the EWMA vol (lambda 0.94) of daily changes of the CMT
    par yield at each contract's tenor (`level_tenor`: ZT 2y ... UB 30y), data published by
    the day only. `predict`: at the contract's delivery horizon, every deliverable's forward
    yield gets the SAME shock ~ N(0, sigma^2 x business days), 20,000 antithetic paths; each
    bond's simulated price is centred on its forward (no drift); per path the short delivers
    min P/CF. Fair futures = mean of the per-path minimum; model option value = M0 fair -
    model fair (>= 0); probabilities = path shares; futures DV01 by a +-0.5bp bump on the
    same draws; `option_value_margrabe_32` = the two-bond (CTD vs runner-up) value under a
    normal spread, as a check (it badly undershoots when many bonds sit near the switch:
    ZB 1.5 vs 12.5/32 on 2026-09-30, 59 deliverables). Hooks for higher tiers: `_shocks`,
    `_pair_sd`, `_extra_bonds`, `_quality_horizon_end`, `_timing`.
*   **M2 (`FactorBasis`, `factors.py`)**: per contract, `dy_i = L + s_i`: the LEVEL L (the
    basket's mean move; random walk, fast-EWMA daily vol x sqrt(horizon), the SAME draws as
    M1 so the two differ only by the relative structure) and the SPREADS s_i with covariance
    estimated AT the horizon (overlapping h-day changes, slow EWMA lambda 0.99, 500-day
    window, horizon capped at a third of it), top 3 principal components + a diagonal
    idiosyncratic remainder (floor 1% of each bond's variance). Optionally FAT-TAILED
    (`spread_df`, spec `M2t`): the spread part (factors + idio) is multivariate Student-t, ONE
    mixing draw per path shared by the basket (a quiet vs a jump regime), variance kept equal
    to the normal's; the level stays normal. Young bonds take their
    PREDECESSOR's changes (previous original issue of the same type and term; a constant
    spread drops out of changes - the user's chosen backfill), then the nearest-maturity
    basket bond's. Diagnostics per contract: variance shares (level / spread factors /
    idio), the estimated spread variance ratio, how many bonds were backfilled.
*   **Expected new issues (M2, `BasisSpec.future_issues`)**: `infra.processing.futures_baskets.
    expected_issues` infers each tenor's cycle and day conventions from its own latest
    issues, as known on the day (only issues already AUCTIONED count, like CME's file):
    *   month-end issuers (2y/5y/7y, and the 20y - which matures on the 15th) are nominally
        the month's last calendar day, rolled FORWARD to a business day (2019-06-30, a
        Sunday, -> issued Monday 2019-07-01); mid-month issuers (3y/10y/30y) fall on the
        maturity's day (the 15th), rolled forward likewise; the cycle (monthly / quarterly)
        from the median gap between the last issues, anchored on the NOMINAL issue month;
    *   backtest, every quarter-end 2019-2026 with a 6-month horizon: **815 / 845 actual
        issues within 5 days, 747 with the exact issue AND maturity date** (misses: the
        20y's 2020 relaunch; a few issues rolled past holidays - `TOFIX.md`);
    *   each expected issue the contract's basket rule accepts (the same `eligible` as the
        real baskets) gets a coupon = the latest same-tenor issue's yield today rounded DOWN
        to 1/8 (the Treasury's rule), CME's conversion factor, a delivery-day price at that
        yield (no carry: nobody owns it before issue), and its predecessor's history in the
        factor model. Placeholders `NEW:<tenor>:<issue day>`; the bench maps them to the CUSIP
        they became (SCORING only - it looks ahead; the model never sees it).
*   **Timing options (`BasisSpec.timing_options`; spec `M2T`; `infra.analytics.delivery_timing`)**:
    *   the QUALITY simulation then stops at the LAST TRADING day (for a last-day delivery):
        switches after it, with the futures frozen, are the end-of-month option's - no double
        count;
    *   END OF MONTH: the basket's forwards to the last delivery day, futures frozen at their
        lowest implied price, one level shock over the EOM business days (7 for ZB/UB/ZN/TN,
        3 for ZT/ZF/Z3N), savings = cost of the LTD-CTD minus the cheapest cost, cost_i =
        P_i - CF_i x F_frozen (`eom_switch_value`);
    *   WILD CARD: Bermudan by backward induction over the intention days left (from the
        first intention day, 2 business days before the delivery month, to the last trading
        day) with the EOM value as continuation; closed form per window, E[max(a Z, v)];
        each window's sd = the CTD's daily price vol x sqrt(ordinary share x the day's
        multiplier) (`BasisSpec.wildcard_window`, section 3e; `window_kinds`: FOMC >
        quarter-end > month-end > ordinary);
    *   negative carry (first-day delivery): one window, no EOM;
    *   outputs `wildcard_32`, `eom_32`, `timing_32`, `option_value_quality_32`; the timing
        value is added to `option_value_model_32` and taken off `fair_futures`.

## 5. Fixed along the way (2026-10-02)
*   **M2 v1 scaled daily spread covariance by sqrt(h)** -> overstated relative variance 3-25x;
    now estimated at the horizon (3c).
*   **Quality simulation to the last delivery day + EOM switch counted the last days twice**
    -> the quality simulation stops at the last trading day when timing is on.
*   **Expected-issue conventions:** weekend month-end issues were rolled BACK (Friday) - the
    Treasury rolls FORWARD; an issue on the 1st was then read as a mid-month convention and
    later predictions drifted a month; the 20y's month-end issue with a mid-month maturity
    was misread. Fixed by deciding conventions from the maturity and anchoring on the
    nominal issue month: exact predictions 547 -> 747 of 845.
*   **Mid-month issues:** a weekend-shifted last issue (2026-08-17) shifted the next ones
    (Nov 17 instead of Nov 16) -> the nominal day now comes from the maturity's day.
*   **The SOFR path stopped at the SR1 strip's last month** and a 2019 last trading day's
    delivery 2 days past it crashed the realised-CTD pass -> flat extrapolation past the last
    priced day (`infra.analytics.sofr_curve.SofrPath.compounded`).
*   **The realised CTD followed the wrong timing** (scored at the last trading day even when
    carry made the shorts deliver early) -> decided on the first intention day when early
    (3f); ZT's hit rate 44% -> 82%. It is a market fact, the same for every tier -> computed
    once and reused (`--realised`); `run_basis.py --scores-from` scores saved
    predictions (a bench crash no longer costs a 40-minute run).
*   **Smaller:** a day with no deliverable priced crashed M2's fit (guarded); pandas `attrs`
    carrying the level history broke `concat` (cleared on outputs); the 3-month calendar
    in `BasisInputs` didn't reach a third listed contract (widened to 400 days).
*   **Speed** (outputs identical, verified): the holiday calendar was rebuilt on every
    `business_days` call (now built once for 1990-2075); `coupon_dates` stepped with
    `DateOffset` (integer month arithmetic, identical on 12,000 random schedules); the 15:30
    quote lookup scanned years of minute quotes per day (the snap store now); funding's
    base path and client basis were recomputed per bond x date (memoised per day); bond
    schedules were rebuilt inside the yield root-finder and every simulation (reused).
    M0 ~5s -> 0.5s per day, M1 3.2s -> ~1.7s.

## 6. The validation bench (`validate.py`, `scripts/run_basis.py`)
Every tier is scored on the same three things:
1.  **Option value**: model vs observed (fair_M0 - market, 32nds) by root - by time to
    delivery too (the final month is where richness should be smallest).
2.  **CTD calibration**: Brier score and hit rate of the delivery probabilities against the
    REALISED CTD = M0's CTD on the contract's last trading day; by days to it.
3.  **Tracking**: daily changes of market futures vs the model's fair price (same contract
    and source both days), and the futures DV01 against the market's realised sensitivity
    to the CTD's yield (regression slope ~1 = right).
Regression benchmarks (3b): CTD implied repo +13 to +20bp over SOFR in 2019-20; the March
2020 unwind (-14/32 ZN, -39/32 ZB on 2020-03-18).

## 7. Results (2019-01-02 .. 2026-09-30, 1,920 days, each root's front contract)
*   **Observed option value** (fair_M0 - market, 32nds, median): TN 0.69, UB 6.17, ZB -0.45,
    ZF -0.63, ZN -0.41, ZT -0.71 - negative on 58-87% of days for ZB/ZF/ZN/ZT (futures rich
    against funding v1).
*   **M0**: futures DV01 slope 0.97-1.00 (R^2 >= 0.977) on every root; CTD hit rate TN 95-100%,
    ZF ~100%, UB 81-86%, ZB 75-80%, ZN 76-81%, ZT 44-46%.
*   **M1**: Brier improves on UB (0.27-0.38 -> 0.17-0.19), ZB (0.40-0.51 -> 0.29-0.37), ZN
    (0.38-0.48 -> 0.26-0.32); ZT unchanged (a parallel shift never re-ranks ZT); model option
    value ~0 except ZB (median 0.28/32) - yields sat far below the 6% notional. DV01 slope
    1.00-1.06.
*   **Scores above use the FIRST realised-CTD definition**; corrected scores (and the
    size-vs-shape diagnosis) are in 3f - the M0 hit rates for ZT in particular were 44-46%
    before the fix, 82% after.
*   **M2**: corrected Brier best on TN and ZN, worse than M1 on ZB/ZT (3f: right variance,
    wrong shape); DV01 slope 0.99-1.06; the new-issue probability exceeded 10% on 13% of ZT
    days and 5% of ZN days; model option value median ZB 2.6/32 (observed -0.45).
*   **M2t, M2T**: (pending)

## 8. Open (root `TOFIX.md`, "Basis: ...")
Next steps in order: M2t / M2T results; M3 (stochastic funding & timing); add-on MS
(specialness + calendar effects, behind the event-profile interface); the spread layer (needs
the CFTC positioning source); add-on IV (needs the paid options fetch). Combinations are
tested on the bench separately and together (e.g. M3 + T + MS). Known
issues: futures richness and UB's residual; the cash bid/mid guess; only front contracts have a 15:30
quote; the expected-issue generator ignores holidays; the wild-card window's remaining
caveats (constants calibrated on 2019-2026, unscheduled events, the halt, futures as the
cash proxy); wiring the futures DV01 into the cycle's bmk step.

## 9. Running it
    PY=/opt/homebrew/Caskroom/miniconda/base/envs/infra-env/bin/python
    $PY scripts/run_basis.py --model M2T --start 2026-09-01 --end 2026-09-30
    $PY scripts/run_basis.py --model M0 --start 2019-01-02 --end 2026-09-30 --out /tmp/m0.parquet --realised /tmp/realised.parquet
    $PY scripts/run_basis.py --scores-from /tmp/m0.parquet --realised /tmp/realised.parquet
*   ~0.5s per day (M0), ~1-1.7s (M1), ~1.5-3s (M2/M2T). Reads disk only. Models: M0, M1, M2,
    M2T (`config.BASIS_MODELS`).
