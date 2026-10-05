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
*   **Real data 2026-10-05** (NFP 40-code family on ZT/ZN, monthly refits 2024-2026): the FDR
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

## 5. Running it
```python
from infra.jobs import family_runs, strategy_runs
family_runs.create("nfp", "nfp_intraday", start="2020-01-03", history_start="2016-01-01")
family_runs.rebuild("nfp", "2026-09-30", promote=True)        # the family's history, one pass
strategy_runs.run_daily("cevt_nfp", "2026-09-30")             # firm series, plan, accounting
strategy_runs.account("cevt_nfp")                             # P&L and costs only
strategy_runs.rebuild("cevt_nfp", "2026-09-30")               # recompute + reconcile, incl. P&L
```
CLI: `scripts/strategy_run.py` (same verbs).
