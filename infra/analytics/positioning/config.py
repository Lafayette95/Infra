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
    pnl_source: str = "FUTURE_BPS_BBO"      # intraday moves
    cycle: str = "15MIN_NO_OVERNIGHT"       # intraday grid (families 2 and 3)
    event_cycle: str = "1MIN_NO_OVERNIGHT"  # intraday release windows (family 1)
    # family 3 factor: "pc1" (the panel's trailing first principal component) or
    # "series:<instrument>" (one instrument; its own relative measures are then left out)
    factor: str = "pc1"
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
        if not (self.factor == "pc1" or self.factor.startswith("series:")):
            raise ValueError(f"{self.name}: factor {self.factor!r} (pc1 | series:<instrument>)")
        if self.factor.startswith("series:") and self.factor.split(":", 1)[1] not in self.instruments:
            raise ValueError(f"{self.name}: factor instrument not among the instruments")

    @property
    def factor_instrument(self) -> str | None:
        return self.factor.split(":", 1)[1] if self.factor.startswith("series:") else None


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

ASYMMETRY_SPECS: dict[str, AsymmetrySpec] = {s.name: s for s in (
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


