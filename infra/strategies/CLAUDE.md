# infra/strategies - Strategies (additive to the root CLAUDE.md)

Everything in the root `CLAUDE.md` applies (layers and time conventions of section 3, the
operating model of 3b, point in time, UTC). The root doc says where things live (section 27);
this one says how the strategies work and what was found. Known gaps: the ONE root `TOFIX.md`,
headings "Strategies: ...". Built 2026-10-05.

## 1. Shape of a strategy
*   **Model predictions -> VIEWS ("signals") -> POSITIONS in contracts -> absolute contracts.**
    A class per strategy type (`base.Strategy` subclasses: `cevt.CEVT`), named configurations
    per class in `config/<class>.py` (`CEVTSpec`, registry `CEVT_STRATEGIES`), registered with
    the jobs in `infra.jobs.strategy_runs.STRATEGY_GROUPS`. Strategies compute and read disk;
    they never write (`infra/jobs` does).
*   **Relative tickers throughout** (`ZN.v.0`); the absolute contract is the LAST step
    (`base.contracts_at`: the contract the relative ticker maps to on the label's CME trading
    day - the same mapping as the event P&L source; past the stored data the last mapping is
    carried, a provisional plan).
*   **Labels are DECISION times** (root CLAUDE.md 3): a view / position labelled T uses only
    what is known at T and is held from T to the next label. Research P&L per instrument:
    `base.pnl(positions, steps, lag)` - the ONE label shift (P&L at L = position decided
    `1 + lag` labels earlier x the step ending at L). Contract-level P&L: section 4.

## 2. Positions (`Strategy.positions`)
*   **Risk unit:** view / ex-ante $ vol per contract = EWMA std of daily back-adjusted
    settlement changes (`vol_span` 60) x point value x sqrt 252, AS AVAILABLE at the label
    (each settlement counts from the session close: the global availability rules,
    root CLAUDE.md 26). Instruments are treated as INDEPENDENT (no covariance yet, `TOFIX.md`).
*   **Scaling to `target_vol_usd` a year:**
    *   `realised` (base default, always-on strategies): the scale that makes the strategy's
        trailing realised $ P&L vol hit the target (`realised_window` days, from days BEFORE
        the label; flat until `realised_min_obs` days of P&L exist).
    *   `full_strength` (sparse strategies, CEVT): fixed - the target is reached only when every
        instrument holds a full-strength (|1|) view (`target / sqrt(n)` per instrument); weaker
        views trade smaller. Vol-targeting a mostly-flat book would lever any weak view up to
        the full target and erase the meaning of its size.

## 3. CEVT - event-study views (`cevt.py`, user design 2026-10-05)
*   **Input:** one or more event-family RUNS (`infra/jobs/family_runs.py`, root CLAUDE.md 26 and
    the event-study doc 4a): each code's walk-forward predictions (per event window, with the
    fit current at its start: `t:`, `expected:`, `passed:`, conditional `cond_passed:`,
    `cond_provisional`) and the family's point-in-time FDR table.
*   **Per code window:** the t-signal `tanh(a * clip(t, -cap, cap)) / tanh(a * cap)` (cap 3,
    a 0.5: concave, |1| at the cap; |t| 1.5 -> 0.70, 2 -> 0.84) and the raw expected move (bp),
    held over the labels `start <= L < end` (decision at the window start, flat at its end).
*   **Gate** (`gate`): `fdr` (default: the family's Benjamini-Hochberg verdict `passed_fdr` AT
    THE FIT THE ROW USED - point in time, BH per fit day across the family's codes), `passed`
    (the code's own tests), `none`. Failing codes are excluded; `include_failing` counts them as
    zeros (dilutes the mean).
*   **Aggregation of the views active at a label:** within a family by `within` (`AGGREGATORS`,
    swappable): `smooth` (default: agreement a = |sum v| / sum |v|; combined = a x the majority
    side's largest |v| + (1 - a) x mean - all agree -> the strongest, evenly split -> ~0),
    `threshold:k` (strongest if a >= k, else mean), `mean`, `absmax`; then the MEAN over the
    families active there (`pool_families`: one stage over every code). `n:` counts the codes,
    `provisional:` marks a view resting on a not-yet-known conditional regime (the latest
    known regime is used, root CLAUDE.md 26).
*   **Positions** from the t-signal (`position_signal = "t"`; `"ev"` = the bp move), scaled
    `full_strength`.
*   **Real data 2026-10-05** (NFP family on ZT/ZN - its 40-code version, 23 of them live - monthly refits 2024-2026): the FDR
    gate passes nothing at recent fits (as the event-study doc reports), so `cevt_nfp` is flat;
    with `gate="passed"` one view switched on (Jul 2024, -102.6 ZN at full strength, `ZNU4`);
    the firm series rebuilds identical.

## 4. Accounting: execution, P&L and costs (`accounting.py`, any strategy)
*   **A separate layer on the ABSOLUTE positions**, knowing nothing about how they were made:
    targets -> marks -> executable? -> executed positions -> P&L and costs -> checks.
    Settings are named (`config/accounting.py`, `ACCOUNTING_MODELS`), picked by
    `StrategySpec.accounting`.
*   **Marks** (`infra/pipeline/futures_marks.py`, disk only):
    *   `bbo` (intraday, `bbo_mid`): the last bbo-1m quote at or before the label (bid, ask,
        sizes, age), the CME halt flag, and the mark = the last two-sided mid within 5 days
        (carried through a halt, weekend, holiday).
    *   `settlement` (DAILY strategies): the execution trading day - the label's own if the
        label is at or before its 14:00 CT settlement, else the next settled day (never
        look-ahead) - its settlement price, and the NY1500 snap's bid/ask (the settlement
        window's book) for the cost.
*   **Executable** per label and contract, else the `reason`: `halt`, `no_fresh_quote` (> 2 min:
    closed, holiday, dead), `one_sided`, `wide_spread` (> 4x the contract's trailing median
    spread, earlier labels only), `no_settlement`. **Policy:** `defer` (default: the trade
    waits for the contract's next executable label, then goes to the target there) or `ignore`
    (trade at the mark regardless; research). Execution lag `StrategySpec.exec_lag` (labels).
*   **P&L** = executed position at the previous label x (mark change) x point value, stamped at
    the step's end. **Cost** = |trade| x half-spread x `spread_paid` x point value;
    `spread_paid` default 0.5 = passively filled on half the trades (user decision 2026-10-05,
    to be researched - `TOFIX.md`). A roll is two trades. A held contract with no mark is NaN
    P&L, never 0. Daily: no NY1500 snap -> the contract's, else its root's, trailing median
    half-spread (`cost_fallback`).
*   **Checks** (flags, never fixes): `deferred` (episodes; warn beyond 8 labels),
    `exceeds_top_of_book` (no impact model yet), `held_in_delivery` (on/after first notice),
    `held_past_last_trade` (fail), `no_mark`, `cost_fallback` / `cost_missing`, `mark_jump` (a
    big move on a held contract that REVERSES at the next label: a bad quote - first version
    flagged on size alone and caught all 45 genuine NFP 08:30-08:45 moves), `target_not_reached`.
*   **Real data 2026-10-05** (NFP family ungated, ZT/ZN 2024-2026): 460 trades over 24
    contracts incl. rolls in 4 s; half-spreads exactly half a tick (ZN 1/128 point, ZT 1/512);
    no NaN; rebuild identical. Found: Friday exits at 19:00 ET (`GRID_END`) can't trade (CME
    shuts 17:00 ET), so they wait for the reopen with weekend risk - the event study values
    them at the 17:00 close (`TOFIX.md`); NFP trades up to 110 ZN contracts against 1-20 at the
    touch. Daily path checked on ZNU6 -> ZNZ6 (Aug-Sep 2026).

## 5. Layered views on curve structures (`layered.py`; definitions `infra/reference/structures.py`)
*   **Why** (user discussion 2026-10-05): the six Treasury futures are 0.8-0.98 correlated, so
    per-future views sized independently double-count one level bet (an NFP rally view on TY and
    on FV is ~sqrt 2 the intended duration risk). Views live where signals live - level (NFP,
    CPI), curve (supply), micro (CTD, specialness) - with a risk budget per layer, netted at the end.
*   **The layers (user decisions 2026-10-05):**

    | Layer | Structures (DV01 weights) | Budget (placeholder) |
    |---|---|---|
    | front | `FRONT__TU__H`: TU hedged on the macro layer | $0.3m/yr - lower: a jump factor (TOFIX: tail-based budget) |
    | macro | `DUR__TY` (default; `DUR__UXY` option), `CURVE__FV__WN` (+FV -WN), `FLY__FV__UXY__WN` (+UXY -0.5 FV -0.5 WN) | $1m/yr |
    | micro | `MICRO__TY__FV__H`, `MICRO__US__WN__H`, hedged on macro | $0.5m/yr |

    Portfolio cap $1.5m/yr on the netted book (full covariance). Conventional DV01 weights for
    duration, curve and fly (user lean, with the alternatives kept as options); hedged structures
    = base legs minus rolling 250-day betas on the set's macro structures, fitted on moves before
    the decision day.
*   **Sizing:** structure position ($/bp) = view x budget / sqrt(structures in layer) /
    structure annual bp vol (EWMA 60); exposures = W' x, netted; the book's vol with the legs'
    EWMA-120 covariance, scaled down above the cap; contracts = exposure / DV01 (as of D-2).
    Diagnostics per label: standalone vol per layer, their sum, the netted total, the scale.
*   **Reconstruction:** with the six structures spanning the six futures, `to_structures` /
    `to_legs` are exact inverses (condition number ~8). Expected moves and per-event moves of any
    linear combination reconstruct exactly; test statistics (t, hit rate, Wilcoxon, FDR) do not -
    rerun them on the reconstructed per-event moves, and declare up front which combinations are
    tested (snooping). Signals (tanh, gate, aggregation) are nonlinear: combine estimates, never
    signals.
*   **Why these structures - the diagnostics** (daily bp moves 2018-10..2026-09, 1,996 days):
    *   PCs of the six futures are stable outside 2020 (yearly cosine to the full sample: level
        >= 0.96, slope >= 0.92, curvature >= 0.93; 2020: 0.83 / 0.76). Curvature is the TU and WN
        wings against a TY / UXY belly.
    *   Conventional content (share of variance by PC level / slope / curvature): CURVE__FV__WN
        7/84/8%, FLY__FV__UXY__WN 13/8/38% (the least level of the 50/50 flies; TU_TY_WN is
        38/0/61%), DUR__TY 99% level. Duration-curve correlation flips sign by regime
        (-0.52 .. +0.58): handled by the portfolio covariance, not by the definitions.
    *   **The front end is its own factor:** TU kurtosis 7.8 (others 1.5-2.8), 4-sigma days 4x
        as often; TU hedged on macro: kurtosis 35, 22% of its variance on 5 days (SVB, Mar 2020,
        CPI 2022-02-10, Feb 2021). Its hedge betas are stable outside the zero bound (duration
        ~0.8, curve ~0.5, fly -1.1..-2.1) but broke in 2021 (duration 0.38).
    *   **Micro dimensions:** with macro on FV / TY / UXY / WN, TY-FV and TY-UXY are the SAME
        residual dimension (only one is in the set); hedged micro is uncorrelated with macro out of
        sample (|corr| <= 0.12), betas stable in the 10y sector (sd 0.02-0.06), less at the long
        end. Micro residuals correlate with each other and with the hedged TU (-0.56): the
        portfolio check, not the definitions, carries that.
    *   **TY vs UXY for duration** (kept TY): UXY is 25% cheaper per bp (half-spread 0.088 vs
        0.118bp), but its book is 3x thinner in DV01 ($50k vs $151k at the touch) and its CTD
        sits among the newest 10s (the most specialness of any contract). Both bp series start
        2018-10 (DV01 history), so UXY's shorter history doesn't matter for bp work. Execution
        could route duration to the cheaper of the two later (TOFIX).
*   **Costs decide which layers can trade intraday** (2025 half-spreads x `spread_paid` 0.5,
    in + out, against each structure's daily vol): DUR__TY 0.12bp (2% of a day's vol),
    CURVE__FV__WN 0.17bp (4%), FLY__FV__UXY__WN 0.17bp (26%), FRONT__TU__H 0.54bp (43%: its
    hedge legs sum to 5.7x its DV01), micro 0.22-0.25bp (44-77%). Vol-parity sizing therefore
    puts huge notional on the low-vol structures (a full fly view ~ $65k/bp = ~800 UXY against
    the wings): on the NFP family (`nfp_intraday_struct`, gate `passed`, 2021-2026 research run)
    costs were 2.8x gross and 739 trades exceeded the top of book. The NFP fly effect (-0.14bp,
    t -2.8) is smaller than its round-trip cost. Low-vol layers need multi-day holding, a
    cost-aware budget, or both (TOFIX).
*   **Event studies on structures:** use `ev_abs_min=None` (a 2bp floor set for single futures
    excludes every fly / micro effect) and judge size against the structure's own vol
    (`ev_vol`). NFP 08:15 -> 10:30 ET (89 events): CURVE__FV__WN steepens +0.96bp (t 2.3, net
    of placebo 2.2), FLY__FV__UXY__WN -0.14bp (t -2.8), duration no consistent sign.

## 5a. Daily strategies (root CLAUDE.md 29; user decisions 2026-10-05)
*   **The same classes**; `frequency` is a REQUIRED field validated against the cycle and the
    accounting marks, presets `INTRADAY` / `DAILY` (`base.py`), two guarded registries per
    config file (`CEVT_STRATEGIES_INTRADAY` / `CEVT_STRATEGIES_DAILY`), stores under
    `Strategies/<frequency>/<name>`.
*   **Decision instant (a):** one decision a day just before the 14:00 CT settlement, traded at
    it (`DAILY_SETTLE` cycle + `settlement` accounting; `exec_lag=1` = decide after the close,
    trade the next settlement).
*   **Sign:** + = long duration, - = short, for views, positions and every benchmark P&L (bp,
    `-dy` for yields): P&L = position x benchmark P&L.
*   **Default daily instrument is the 10y yield** (`US_BOND_10y`), measured against several
    benchmark P&Ls (CMT, on-the-run, our curve; root CLAUDE.md 12), with futures as (a) another
    benchmark - a sanity check, compared at the NY1530 snap - and later (b) an RV view against
    cash. Yields can't be held: views on them need an execution map into futures (TOFIX), so the
    daily strategies so far trade futures or structures.
*   **First run** (`cevt_nfp_days_layers`, gate `passed`, research, 2021-2026): $1.15m gross,
    Sharpe 0.50 (t 1.2); -$0.30m net - day windows on the low-vol layers trade large notional
    (~1,100 TY vs -1,500 UXY on one micro view). Its checks caught a ZB position on ZBU1's
    first-notice day (2021-08-31): `ZB.v.0` rolled late that cycle.

## 5b. Planned: RV signals bias the macro trades' instrument selection (user intent 2026-10-06)
*   **Relative-value signals are not traded on their own:** they choose WHICH instruments carry
    the macro (layer) exposures - which tenor / contract / bond, eventually including cash
    bonds. A tilt changes how a trade that happens anyway is expressed, so its marginal cost is
    ~0 at entry and rebalance; moving an existing position only for the tilt costs a round trip
    and must clear it. Judge a signal by the improvement it gives the macro book's P&L (the fly's
    gross information), not by a standalone Sharpe net of its own trades.
*   **Shape (not built):** macro structure positions -> each active fly view as a per-tenor
    rich / cheap score (belly +1, wings -0.5) summed across signals -> choose the instruments
    holding the macro exposures to maximise expected tilt gain minus switching cost, with
    unintended curve / fly exposure capped by a tilt risk budget (with exposures held exactly,
    six futures leave no freedom) -> accounting unchanged. Generalises the TY <-> UXY routing
    item in `TOFIX.md`.
*   **First candidates** (scratch study 2026-10-06; flies on our curve and on-the-run yields,
    auction windows; walk-forward 2012-2026, gross, yield space): (A) the 7y richens against 5s
    and 10s around the refunding auctions (3y / 10y / 30y) - Sharpe 0.60-0.77, 12-13 of 15 years
    positive; (B) the 5y cheapens into the end-of-month 2y/5y/7y week and recovers after -
    Sharpe 0.52-0.89; A and B uncorrelated, together ~1.0. In futures terms: FV ~ 5y, TY ~ 7y,
    UXY ~ 10y. Open before relying on them: whether they survive in futures (CTD / basis), the
    value of the tilt on an actual macro book, and an out-of-sample selection (the two effects
    were picked from a full-sample scan). CMT is NOT usable for fly / auction studies: it showed
    7x the significant results of the other sources, consistent with its points switching to new
    issues at each auction.

## 6. Running it
```python
from infra.jobs import family_runs, strategy_runs
family_runs.create("nfp", "nfp_intraday", start="2020-01-03", history_start="2016-01-01")
family_runs.rebuild("nfp", "2026-09-30", promote=True)        # the family's history, one pass
strategy_runs.run_daily("cevt_nfp", "2026-09-30")             # firm series, plan, accounting
strategy_runs.account("cevt_nfp")                             # P&L and costs only
strategy_runs.rebuild("cevt_nfp", "2026-09-30")               # recompute + reconcile, incl. P&L
# layered: a CEVTSpec with instruments = structure names and layers="ust_layers", over a family
# run on structures (family "nfp_intraday_struct"); everything else is the same
from infra.pipeline.structures import structure_state             # the point-in-time state, for research
st = structure_state("ust_layers", "2025-01-01", "2026-09-30"); st.W("2026-09-30")
```
CLI: `scripts/strategy_run.py` (same verbs).
