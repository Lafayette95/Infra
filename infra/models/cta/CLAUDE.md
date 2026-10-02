# infra/models/cta - CTA Positioning (additive to the root and `infra/models` CLAUDE.md)

Everything in the root `CLAUDE.md` and `infra/models/CLAUDE.md` still applies (conda env,
one-way dependencies, point-in-time `as_of`, UTC, prepare -> fit -> predict). Known edge
cases go in the ONE root `TOFIX.md`, under headings prefixed "CTA:". This file says what
the model MEANS and HOW to use it.

## 1. What it is
*   **A bottom-up replica of a generic trend-following CTA**, after UBS Q-Series, "CTAs:
    How $375 bln Influences Global Assets?" (2022-09-05; the user's scan,
    `~/Downloads/CTA-bulletproof.pdf`, 48 pages, read in full 2026-10-02; the methodology
    is Part II, pp. 18-26, the results Part III, pp. 27-48). Rule-based: no CTA data goes
    in, only prices; out come the positions a trend follower running this algorithm holds.
*   **Abstract on input.** Any wide `timestamp x asset` price frame: a back-adjusted futures
    price, a yield, an FX rate. Where the series come from is `inputs.py`'s job only.
*   **Outputs, all in [-1, 1] units for now** (scaling to contracts/$ needs the CTAs' size
    footprint, a later step):
    *   `signal`: the trend signal, the risk position before sizing;
    *   `position`: the sized position divided by its own largest absolute value over the
        last 10 years up to the fit date (UBS reports "max abs position last 10y");
    *   `chg_<h>`: the position change over the past h observations (1, 3, 5, 30);
    *   expected flows over the next h (1, 3, 5, 30): `exp_signal`, `exp_vol`,
        `exp_position`, `flow`, split into `flow_from_signal` + `flow_from_vol`;
    *   the spot reaction function (section 4).

## 2. The methodology (UBS Part II), and what the note does NOT say
*   **Signal (Q7 a):** (i) crossovers of EWMAs of the price at 3-5 speeds; (ii) each
    normalised by the underlying's rolling 1-year volatility; (iii) a response function
    making the signal "as close as possible to uniform" on [-1, 1].
*   **Position (Q7 b-d, the rates example):**
    `position = asset-class weight x liquidity factor x signal x portfolio vol scaling / vol`
    with vol = realised 1-3 month (3m in the examples), liquidity factors 1-8 from ADV and
    open interest, and the portfolio scaling a multiplier reaching a 10% vol target over a
    3-year window. **Checked on UBS's own numbers (Fig. 63):** 0.30 x liq x signal x 809.9
    / vol reproduces every published position up to one common factor ~1.075
    (`tests/test_cta.py::test_paper_positions_follow_the_stated_formula`): the formula is as
    stated, with one unexplained constant (a weight nearer 32%, or a unit).
*   **Forecast (Q9):** Monte Carlo price paths over the horizon (2 weeks), centred on the
    forward (zero drift for a futures price); run the algorithm on each path and average the
    terminal signals. Expected vol = the current 2-month realised vol; portfolio scaling held.
    Expected flows split into signal change and vol change.
*   **NOT in the note, so chosen here (each a `CTASpec` field):** the EWMA speeds (default:
    Baz, Granger, Harvey, Le Roux, Rattray 2015, Man AHL - (8,24), (16,48), (32,96), weight
    decay (n-1)/n); how the pairs combine (each pair's crossover divided by its own RMS,
    averaged, then ONE response on the average, so the final signal is the uniform one);
    the response's form (default: the fitted ECDF, `2F - 1`, exactly uniform in sample); the
    path distribution (Gaussian, antithetic, fixed seed).
*   **The note's own charts contradict "uniform".** Fig. 59/60 (ES and TY signals,
    2018-2022) sit at +-1 for months at a time - far more mass at the caps than a uniform
    signal has. Hence `response_gain` (section 5).

## 3. Code (`infra/models/cta/`)
*   `config.py`: `CTASpec` (all parameters) and the `CTA_MODELS` registry; `CTAAsset` /
    `CTAUniverse` (inputs + per-asset metadata: class, liquidity, `returns` = `diff` or
    `log`) and `CTA_UNIVERSES`. Toggle models and universes by name.
*   `prep.py` (stage 1): `prepare(prices, universe=...)` -> `CTAData`. Each series keeps its
    OWN calendar (a holiday elsewhere is not a zero-move day); `anchor` continues changes
    across a fit/predict boundary.
*   `model.py`: `CTAModel(spec)` with `prepare` / `fit(data, as_of)` / `predict(new=None)` /
    `reaction(result, axis)`, and `walk_forward(spec, data, start, end, refit="W-FRI")`.
*   `signal.py`: pure pieces (EWMA, rolling vol, crossovers, responses, portfolio scaling).
*   `state.py`: `AssetParams` (frozen at fit) and `AssetState` (one asset at one observation).
*   `forecast.py`: the Monte Carlo expected signals / positions / flows.
*   `reaction.py`: the reaction-function registry `REACTIONS`.
*   `inputs.py`: stored data -> price frame, disk only (`SOURCES["futures"]`: relative
    tickers, back-adjusted by `infra.processing.continuous`).
*   `paper.py`: UBS's published rates numbers (legible ones only); `compare.py`: ours vs
    theirs.
*   **Fit vs predict.** `fit` estimates and freezes, per asset: each pair's RMS scale, the
    response (ECDF), the position scale (10y max); and the portfolio vol scaling (3y). It
    keeps the EWMA values and the trailing returns at the fit date, so `predict` continues
    every EWMA and rolling vol EXACTLY (`test_predict_continues_the_fit_exactly`). `predict()`
    with no data returns the fit sample (last row = the fit date); `predict(new_prices)`
    runs on rows after the fit date only.
*   **Intraday on a daily fit:** pass the daily rows since the fit with the intraday price as
    the LAST row: it is read as a provisional close (one full EWMA step). Every earlier row
    must be a daily close. The input builder for a live futures price is not written yet
    (`TOFIX.md`).
*   **Point in time:** `fit(as_of)` uses data up to `as_of` only
    (`test_fit_is_point_in_time`). History inside one fit uses that fit's frozen
    parameters (the model's view of its past, as UBS reports "t-2w"); the as-it-happened
    history is `walk_forward`, where each refit date's row comes from that refit.

## 4. The reaction function (`reaction.py`, `CTAModel.reaction`)
*   A registry: a new axis is one function `fn(params, state, spec, grid) -> DataFrame`.
*   `price`: today's price moved by `shock` x one day's vol, all else equal (every EWMA takes
    the move with weight 1/n; vols held). The CTA's spot "gamma".
*   `path`: the same total move (`shock` sigma over the period) spread evenly over
    `reaction_path_days` (5) sessions. `d_position_vs_flat` nets out the unfed trend's own
    decay.
*   `vol`: the sizing vol scaled by (1 + shock) (position ~ 1/vol, signal unchanged); with
    `"norm"` in `vol_shock_targets` the trend normalisation moves too, so the signal does.
*   `price_vol`: both on a grid.
*   **A saturated signal does not react** to an instant shock of any realistic size
    (2026-10-01, US10Y at the 1st percentile of its trend history: a ~20-sigma one-day rally
    is needed to move it), and only slightly to a week-long 4-sigma rally. That is the
    model's statement, not a bug - UBS Fig. 69's "signal capped/floored, not responding to
    additional strength".

## 5. Comparison with the paper (`compare.py`, `scripts/run_cta.py --compare-paper`)
*   **Target:** UBS rates bond futures, data date 2022-09-02 (the note is dated Monday
    2022-09-05, US Labor Day), "2 weeks ago" = 2022-08-19, horizon 10 observations. Our US
    contracts are mapped US2Y=ZT, US5Y=ZF, US10Y=ZN, US20Y=ZB, US30Y=UB (our reading of UBS's
    labels). Positions compared in the same unit: UBS's $DV01 / its 10y max vs our
    `position`. Vols only as ratios (UBS: yield bp; ours: futures price points).
*   **Result 2026-10-02 (US bond futures; Eurex has no history before 2025-07):**

    | | US2Y | US5Y | US10Y | US20Y | US30Y |
    |---|---|---|---|---|---|
    | signal UBS | -1.00 | -1.00 | -0.98 | -0.85 | -1.00 |
    | signal `ubs2022` | -0.64 | -0.67 | -0.69 | -0.66 | -0.81 |
    | signal `ubs2022_cal` | -0.91 | -0.96 | -0.97 | -0.97 | -0.97 |
    | position UBS | -0.25 | -0.25 | -0.29 | -0.31 | -0.39 |
    | position `ubs2022` | -0.10 | -0.15 | -0.20 | -0.28 | -0.34 |
    | position `ubs2022_cal` | -0.13 | -0.22 | -0.29 | -0.39 | -0.38 |

    US10Y in full (UBS / `ubs2022_cal`): signal 2w ago -0.46 / -0.50; expected signal in 2w
    -0.94 / -0.80; position 2w ago -0.12 / -0.15; expected position -0.29 / -0.26; vol 2w
    ago / now 1.026 / 1.005; expected vol / now 0.977 / 0.889; expected flow from signal
    +0.013 / +0.052, from vol -0.007 / -0.029.
*   **`ubs2022_cal`** = fast pairs (4,12), (8,24), (16,48) + ECDF response with
    `response_gain` 2 - the best of a grid of 4 speeds x 4 gains (1, 1.5, 2, 3) x 2
    responses scored on these numbers. Mean abs error over the 5 US contracts: signal 0.06
    (vs 0.27 for `ubs2022`), position 0.05 (vs 0.09). Two parameters chosen on ONE date: a
    calibration to UBS's snapshot, not evidence about real CTAs (`TOFIX.md`).
*   **What still differs:** our expected signals mean-revert more over 2 weeks (-0.80 vs
    -0.94) and our 2-month vol is lower relative to the 3-month one; UBS's vols are in
    yield space (DV01 sizing), ours in price space (a duration difference, slowly varying).
    Values we couldn't read (Fig. 75, 107, 108: the other contracts' t-2w and t+2w columns)
    are None in `paper.py`.

## 6. Running it
*   Load the futures history first (the daily pipeline; US bond futures are on disk from
    2014-12). Then:
    ```
    $PY scripts/run_cta.py --list
    $PY scripts/run_cta.py --model ubs2022_cal --universe ubs_bonds [--as-of D] [--reaction path]
    $PY scripts/run_cta.py --compare-paper --model ubs2022_cal
    ```
*   **Python:**
    ```python
    from infra.models.cta.config import get_universe
    from infra.models.cta.inputs import universe_prices
    from infra.models.cta.model import CTAModel, walk_forward
    uni = get_universe("ubs_us_bonds")
    px = universe_prices(uni, "2014-12-01", "2026-10-01")
    m = CTAModel("ubs2022_cal"); m.fit(m.prepare(px, universe=uni), as_of="2026-09-25")
    res = m.predict(px[px.index > "2026-09-25"])   # light step on new data only
    res.latest(); m.reaction(res, "path")
    ```
*   Speed: a fit of 5 assets x 12 years takes ~0.1s; a weekly walk-forward over a year ~3s.
