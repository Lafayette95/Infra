"""Asymmetric reaction specs: named parametrisations, toggled by name (``ASYMMETRY_SPECS``).

A spec says which moves go in (instruments, frequency, how a stored value becomes a
YIELD-direction move in bp), what the "usual covariance" factor is, the windows and
thresholds of each family, and which releases feed the surprise family. Methodology:
``infra.analytics.positioning.asymmetry``. Assembly and storage:
``infra.pipeline.positioning``. Design: TOFIX "Positioning: roadmap" (user decisions
2026-10-07: all three families, daily first then intraday, factor configurable).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# Releases with a usable MarketWatch consensus history (100+ prints from 2009-10, counted
# 2026-10-07; S&P PMIs from 2021). CPI has 5 prints (TOFIX "Forecast": the calendar pattern).
DEFAULT_RELEASES = (
    "NFP TCH Index", "USURTOT Index", "INJCJC Index", "NAPMPMI Index", "NAPMNMI Index",
    "GDP CQOQ Index", "DGNOCHNG Index", "NHSPSTOT Index", "IP CHNG Index", "CONSSENT Index",
    "CONCCONF Index", "ETSLTOTL Index", "OUTFGAF Index", "EMPRGBCI Index", "CHPMINDX Index",
    "ADP CHNG Index", "MPMIUSMA Index", "MPMIUSSA Index", "SBOITOTL Index", "PCE CRCH Index",
)


@dataclass(frozen=True)
class AsymmetrySpec:
    name: str
    description: str = ""
    frequency: str = "daily"                 # daily | intraday
    # daily: series ids (infra.pipeline.series_panel); intraday: futures instruments for the
    # grid P&L source (``pnl_source``).
    instruments: tuple[str, ...] = ()
    # stored value -> YIELD-direction move in bp: a LEVEL is differenced, a MOVES series
    # (bmk / P&L) taken as is; then x ``sign`` x ``scale`` (yields in % -> x100; a long
    # position's P&L -> sign -1).
    sign: float = 1.0
    scale: float = 100.0
    # per-instrument overrides of how a stored value becomes a move: (id, how, multiplier),
    # how = "diff" | "logdiff" | "as_is". Cross-asset convention: + = a LOSS for a long
    # holder of the instrument - a futures price as -100 x its same-contract log return
    # (``fret:``, % price fall),
    # a yield as +100 x its change (bp up).
    instrument_moves: tuple[tuple[str, str, float], ...] = ()
    pnl_source: str = "FUTURE_BPS_BBO"      # intraday moves
    cycle: str = "15MIN_NO_OVERNIGHT"       # intraday grid (families 2 and 3)
    event_cycle: str = "1MIN_NO_OVERNIGHT"  # intraday release windows (family 1)
    # family 3 factor(s): "pc1" (the panel's trailing first principal component),
    # "series:<instrument>" (one instrument; its own relative measures are then left out),
    # or MULTI-factor (user decision 2026-10-07, cross-asset): "pcs:K" (the first K PCs of
    # the vol-scaled panel) or "series:<a>,<b>,..." (named instruments; one that is itself
    # a factor is regressed on the others) - asymmetry.multifactor_relative.
    factor: str = "pc1"
    mf_refit_every: int = 21
    # rows a multi-factor beta fit / PC fit needs (a 4-factor fit on fewer overfits)
    beta_min_obs: int = 252
    pc_window: int = 252
    pc_refit_every: int = 21
    # windows in observations (days, or grid steps intraday)
    window: int = 63                         # measurement window
    beta_window: int = 252                   # usual beta, ending before the measurement window
    vol_span: int = 63                       # trailing vol a move is judged against
    k: float = 1.0                           # "big-ish": |move / vol| > k
    min_obs: int = 40
    min_big: int = 5
    # family 1
    releases: tuple[str, ...] = DEFAULT_RELEASES
    surprise_min_obs: int = 8                # earlier prints before a surprise can be standardised
    impact_lookback_days: int = 5 * 365      # usual impact fitted on the previous ~5 years
    impact_refit: str = "MS"                 # refitted monthly
    impact_ridge: float = 1.0
    impact_min_events: int = 20
    event_window: int = 40                   # events per surprise / continuation window
    event_min_side: int = 8
    event_pre_minutes: int = 1               # intraday window: (release - pre, release + post]
    event_post_minutes: int = 15
    event_after_minutes: int = 120           # continuation: (release + post, release + after]
    start: str = "2008-09-02"

    def __post_init__(self):
        if self.frequency not in ("daily", "intraday"):
            raise ValueError(f"{self.name}: frequency {self.frequency!r} (daily | intraday)")
        if not (self.factor == "pc1" or self.factor.startswith(("series:", "pcs:"))):
            raise ValueError(f"{self.name}: factor {self.factor!r} (pc1 | pcs:K | series:<a>[,<b>...])")
        if any(f not in self.instruments for f in self.factor_instruments):
            raise ValueError(f"{self.name}: factor instrument not among the instruments")
        for inst, how, _ in self.instrument_moves:
            if how not in ("diff", "logdiff", "as_is") or inst not in self.instruments:
                raise ValueError(f"{self.name}: bad instrument_moves entry {inst!r} {how!r}")

    @property
    def factor_instruments(self) -> list[str]:
        return self.factor.split(":", 1)[1].split(",") if self.factor.startswith("series:") else []

    @property
    def factor_instrument(self) -> str | None:
        """The single named factor (one-factor specs only)."""
        names = self.factor_instruments
        return names[0] if len(names) == 1 else None

    @property
    def multifactor(self) -> bool:
        return self.factor.startswith("pcs:") or len(self.factor_instruments) > 1

    @property
    def n_pcs(self) -> int:
        return int(self.factor.split(":", 1)[1]) if self.factor.startswith("pcs:") else 1

    def move_rule(self, instrument: str) -> tuple[str | None, float]:
        """(how, multiplier) for one instrument: its override, else (None = by the source's
        kind, sign x scale)."""
        for inst, how, mult in self.instrument_moves:
            if inst == instrument:
                return how, mult
        return None, self.sign * self.scale


_OTR = tuple(f"otr:US_BOND_{t}y" for t in (2, 5, 10, 30))
_DAILY = AsymmetrySpec(
    name="ust_daily",
    description="US on-the-run yields 2/5/10/30y, daily (FedInvest END OF DAY, from 2008-09); "
                "3-month windows, factor = the curve's level PC.",
    instruments=_OTR,
)
# 15-minute grid: ~53 steps a day. Windows ~ 20 trading days, betas ~ 6 months.
_INTRADAY = AsymmetrySpec(
    name="ust_intraday",
    description="US Treasury futures ZT/ZF/ZN/ZB/UB on the 15-minute grid (bbo-1m mids, bp, from 2018-10); "
                "20-day windows, factor = level PC.",
    frequency="intraday",
    instruments=("ZT.v.0", "ZF.v.0", "ZN.v.0", "ZB.v.0", "UB.v.0"),
    sign=-1.0, scale=1.0,
    pc_window=53 * 63, pc_refit_every=53 * 5,
    window=53 * 20, beta_window=53 * 126, vol_span=53 * 5, k=2.0, min_obs=53 * 10, min_big=10,
    start="2018-10-01",
)

# Cross-asset (user decisions 2026-10-07): US rates + the CME macro roots (equity index,
# G10 FX + MXN, energy, metals), daily settlements from 2010-07; STIR (SR1) from 2018-05.
# fret: = same-contract log returns (NOT logdiff of the back-adjusted level - wrong for
# compounding prices, continuous.log_returns); x -100 = % price fall.
_MACRO_FUT = tuple(f"fret:{r}.v.0" for r in ("ES", "NQ", "RTY", "NKD", "6E", "6J", "6B", "6A", "6C", "6S", "6M",
                                            "CL", "NG", "GC", "SI", "HG"))
_MACRO = AsymmetrySpec(
    name="macro_daily",
    description="US rates (OTR 2/5/10/30y) + CME equity index, FX, energy, metals futures, daily from 2010-07; "
                "multi-factor: the first 4 PCs of the vol-scaled panel. + = a loss for longs everywhere.",
    instruments=_OTR + _MACRO_FUT,
    instrument_moves=tuple((f, "as_is", -100.0) for f in _MACRO_FUT),
    factor="pcs:4", beta_window=504, pc_window=504, min_obs=60, start="2010-07-10",
)
# fut: (back-adjusted, same-contract changes) x -100 = bp of rate rise. NOT stir: - that is
# the raw settlement of whichever contract holds the rank, so every monthly roll is a fake
# jump (found 2026-10-07: it wrecked the whole panel's factors, ES's median R^2 0.62 -> 0.10).
_STIR = ("fut:SR1.c.3",)

# Curve structures (2026-10-08): the bmk OTR yield P&L of a LONG structure (bp; CURVE__a__b
# = long a, short b - a steepener; FLY__a__b__c = long the belly b vs 50/50 wings), x -1 so
# "+ = a loss for a long holder"; the level factor is the 10y itself.
_CURVES = tuple(f"bmk:otr:{s}" for s in (
    "CURVE__US_BOND_2y__US_BOND_10y", "CURVE__US_BOND_5y__US_BOND_30y", "CURVE__US_BOND_2y__US_BOND_5y",
    "CURVE__US_BOND_10y__US_BOND_30y", "FLY__US_BOND_2y__US_BOND_5y__US_BOND_10y",
    "FLY__US_BOND_5y__US_BOND_10y__US_BOND_30y"))
_LEVEL = "bmk:otr:US_BOND_10y"
_CURVE_SPEC = AsymmetrySpec(
    name="ust_curves_daily",
    description="US curve structures (2s10s, 5s30s, 2s5s, 10s30s, 2/5/10 and 5/10/30 flies; OTR, daily from "
                "2008-09) relative to the 10y level; + = a loss for a long structure.",
    instruments=_CURVES + (_LEVEL,),
    instrument_moves=tuple((c, "as_is", -1.0) for c in _CURVES + (_LEVEL,)),
    factor=f"series:{_LEVEL}",
)

ASYMMETRY_SPECS: dict[str, AsymmetrySpec] = {s.name: s for s in (
    _CURVE_SPEC,
    _MACRO,
    replace(_MACRO, name="macro_daily_named",
            factor="series:otr:US_BOND_10y,fret:ES.v.0,fret:6E.v.0,fret:CL.v.0,fret:GC.v.0",
            description="As macro_daily, factors = named: 10y yield, S&P, EUR, crude, gold."),
    replace(_MACRO, name="macro_daily_stir", instruments=_OTR + _STIR + _MACRO_FUT, start="2018-06-01",
            instrument_moves=_MACRO.instrument_moves + (("fut:SR1.c.3", "diff", -100.0),),
            description="As macro_daily plus the SOFR strip (SR1, 3rd contract: rate change in bp), from 2018-06."),
    _DAILY,
    replace(_DAILY, name="ust_daily_vs10y", factor="series:otr:US_BOND_10y",
            description="As ust_daily, factor = the 10y itself (relative measures read 'vs the 10y')."),
    _INTRADAY,
    replace(_INTRADAY, name="ust_intraday_vsZN", factor="series:ZN.v.0",
            description="As ust_intraday, factor = ZN."),
)}


def get_spec(name: str) -> AsymmetrySpec:
    try:
        return ASYMMETRY_SPECS[name]
    except KeyError:
        raise KeyError(f"unknown asymmetry spec {name!r}; known: {sorted(ASYMMETRY_SPECS)}") from None


