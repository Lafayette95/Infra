"""Basis model parameters (infra/models/basis/CLAUDE.md) - specs kept apart from the code so
several tiers / variants sit side by side and are toggled by name (``BASIS_MODELS``).

The structure (user decisions 2026-10-02; infra/models/basis/CLAUDE.md 1): pricing tiers
M0 deterministic -> M1 one factor -> M2 basket factors (optionally fat-tailed spreads) ->
M3 stochastic funding & timing (planned); switchable add-ons - T timing options
(``timing_options``, built), IV implied vol (planned), MS microstructure: specialness +
calendar effects read through an event-profile interface (planned); and a separate
explanatory spread layer (planned). Built specs: M0, M1, M2, M2t, M2T.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BasisSpec:
    name: str
    tier: str  # "M0" .. "M3": which model class runs it (infra.models.basis.model.TIERS)
    description: str = ""
    # Cash is FedInvest's END OF DAY, a BID price; move it toward mid by this fraction of
    # half the posted buy/sell spread (the posted spread is a convention, not the market's:
    # 0.5 = a quarter of it, between "bid" (0) and "posted mid" (1)).
    cash_mid_frac: float = 0.5
    funding_model: str = "v1"  # infra.config.FINANCING_MODELS
    roots: tuple[str, ...] = ("ZT", "Z3N", "ZF", "ZN", "TN", "ZB", "UB")
    # M1+: the one level factor. Its vol = EWMA of daily changes (bp) of the CMT par yield
    # at the contract's tenor, data before the day only; weight lambda per day.
    level_tenor: tuple[tuple[str, int], ...] = (("ZT", 2), ("Z3N", 3), ("ZF", 5), ("ZN", 7), ("TN", 10),
                                                ("ZB", 20), ("UB", 30))
    vol_lambda: float = 0.94
    # Add-on IV (1-factor vol scaling): "ewma" = the level vol as estimated (default);
    # "iv" = scaled by iv_root's implied / realised ratio that day, both in futures points
    # (ATM implied at the contract's delivery horizon / EWMA of the front contract's daily
    # moves), applied to every root - infra/pipeline/futures_iv.py. Validated 2026-10-03:
    # ZN's 1-month implied forecasts the next 21 days' realised vol better than the EWMA
    # (MAE 0.085 vs 0.096 points/day, correlation 0.66 vs 0.55).
    level_vol_source: str = "ewma"
    iv_root: str = "ZN"
    vol_history_days: int = 3 * 365
    n_paths: int = 20_000  # antithetic pairs included
    seed: int = 0
    # M2+: the basket factor model (infra.models.basis.factors). Level vol fast
    # (vol_lambda), relative structure on a slow EWMA over ``factor_window`` days.
    n_spread_factors: int = 3
    factor_lambda: float = 0.99
    factor_window: int = 500
    # M2+: include EXPECTED new issues the basket will accept before delivery
    # (infra.processing.futures_baskets.expected_issues) - ZT's realised CTD was a note not
    # yet auctioned on 17% of M0's ZT misses, TN's on 84% of its misses (2019-2026).
    future_issues: bool = True
    # price expected issues at a FORWARD yield (+ the nearest deliverable's carry shift);
    # False = the first version's spot pricing (too cheap under negative carry, CLAUDE.md 3f)
    future_issue_carry: bool = True
    # The delivery TIMING options (infra.analytics.delivery_timing): the wild card (Bermudan
    # over the intention days up to the last trading day) and the end-of-month switch
    # (futures frozen after the last trading day). Each window's variance = the CTD's daily
    # price variance x the root's ORDINARY-day share of it in 15:00-19:00 New York x the
    # day's multiplier. Measured 2019-2026 on each root's front futures (bbo-1m mids, the
    # day's own 15:00 -> next 15:00 move as the daily variance); day kinds exclusive in the
    # order fomc > quarter_end > month_end > ordinary. ``wildcard_var_share`` is the
    # fallback for a root not listed (the all-days share 2024-2026, 6.3%).
    # M2+: the spread part's distribution - None = normal; a number = Student-t with that
    # many degrees of freedom, one mixing draw per path (factors.simulate_shocks).
    spread_df: float | None = None
    # M2+: multiplier on the idiosyncratic spread VARIANCE (1 = as estimated). Under test
    # 2026-10-02: FedInvest's noisy marks for old off-the-runs may inflate it.
    idio_scale: float = 1.0
    # M2+: mark-noise variances removed from the idiosyncratic variance (factors.fit_factor_model)
    idio_noise_removal: float = 0.0
    # M2+: correlate the idiosyncratic moves by maturity distance (factors.fit_factor_model)
    idio_maturity_corr: bool = False
    # M2+: spreads move with the level (beta per bond; factors.fit_factor_model)
    level_betas: bool = False
    # M2+: the JOINT horizon PCA on total yield changes (factors.fit_joint_model) instead
    # of level + spread PCA; n_joint_factors components
    joint_pca: bool = False
    n_joint_factors: int = 4
    timing_options: bool = False
    # T under NEGATIVE carry: True (default since 2026-10-03, user decision) = Bermudan over
    # every intention day, each passed window costing a day's carry, the end-of-month switch
    # as continuation net of its own days' carry; False = the first version's rule (deliver
    # at the first window, no EOM), which priced UB's T at 2.5/32 in 2023 vs 12.2 observed
    timing_carry_bermudan: bool = True
    wildcard_var_share: float = 0.063
    wildcard_window: tuple[tuple[str, float, float, float, float], ...] = (
        # root, ordinary share, x FOMC day, x quarter-end, x month-end
        ("ZT", 0.040, 6.29, 2.14, 2.88), ("ZF", 0.045, 5.40, 2.00, 3.06), ("ZN", 0.049, 4.12, 2.14, 2.36),
        ("TN", 0.053, 3.14, 2.31, 2.16), ("ZB", 0.056, 2.46, 2.78, 1.85), ("UB", 0.067, 1.67, 2.40, 1.27),
    )


BASIS_MODELS: dict[str, BasisSpec] = {
    "M0": BasisSpec("M0", "M0", "deterministic: forward CTD by lowest implied futures price over bonds x "
                                "{first, last} delivery day; option value 0"),
    "M1": BasisSpec("M1", "M1", "one factor: the same normal shock to every deliverable's forward yield at the "
                                "M0 delivery day; quality option by simulation"),
    "M2": BasisSpec("M2", "M2", "basket factor model: level + PCA on the basket's spreads + idiosyncratic, "
                                "predecessor backfill for young bonds; correlated shocks at the M0 delivery day"),
    "M2T": BasisSpec("M2T", "M2", "M2 + the timing options (wild card and end-of-month switch)", timing_options=True),
    "M2t": BasisSpec("M2t", "M2", "M2 with fat-tailed spreads (Student-t, 6 df, one mixing draw per path)", spread_df=6.0),
    "M2TIV": BasisSpec("M2TIV", "M2", "M2T with the level vol scaled by ZN's implied / realised ratio (add-on IV)",
                       timing_options=True, level_vol_source="iv"),
}
