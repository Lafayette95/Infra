"""Conditional autocorrelation (framework B1) parameters: named specs, toggled by name
(``infra/models/CLAUDE.md`` 0; methodology ``infra/models/autocorr/CLAUDE.md``).

A spec says WHAT is tested (``target``: the instrument whose price action may trend or revert;
``x``: the third variable that may decide which), the horizons, the feature and partition of X,
which GATES decide that the conditioning is real (``gates`` + ``thresholds``: swappable - a
looser spec names fewer gates or lower thresholds), what to do when they fail (``fallback``), and
which out-of-sample EVALUATIONS a research run computes (``evaluations``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AutocorrSpec:
    name: str = "custom"
    description: str = ""
    target: str = "bmk:curve:US_BOND_10y"       # series id of DAILY moves (bp, + = a long position made money)
    x: str = "bmk:curve:FLY__US_BOND_5y__US_BOND_7y__US_BOND_10y"   # the third variable (series id)
    x_feature: str = "move:20"                  # move:K (X's K-day move / its vol) | absmove:K (its size, for
                                                # U-shaped effects) | level | z:W (level z-scored)
    mode: str = "chase"                         # chase: E[sign(past) x fwd | X bucket] (symmetric in the past move's
                                                # sign) | cells: E[fwd | X bucket, past-move bucket], a 3x3 with a
                                                # signal per cell (the user's quadrant spec; captures asymmetric /
                                                # co-move vs counter-move effects, including X's direction)
    past_partition: str = "rolling_tercile:504" # cells: the past move's bucket, point in time
    demean_by_x: bool = False                   # cells: subtract each X bucket's mean forward move (isolates the
                                                # autocorrelation part from X's directional part)
    x_is_moves: bool = True                     # X's series is daily moves (bmk P&L); False = a level series
    past_days: int = 5                          # k: the past move whose continuation is tested
    horizon_days: int = 5                       # h: the forward window
    gap_days: int = 1                           # days from the information day D to the forward window's start
                                                # (FedInvest-based yields for D post ~10:00 New York on D+1)
    vol_span: int = 60                          # EWMA of daily moves: the normaliser (known at D)
    x_partition: str = "rolling_tercile:504"    # infra.models.event_study.conditions.PARTITIONERS, point in time
    window: str | None = None                   # fit sample look back ("2520D"); None = all history
    hac_lags: int | None = None                 # None = horizon_days (overlapping forward windows)
    # gates (GATES registry) and their thresholds: the conditioning is used only if ALL pass
    gates: tuple[str, ...] = ("min_obs", "x_t", "bucket_spread_t", "beats_controls")
    thresholds: tuple[tuple[str, float], ...] = (("min_obs", 250.0), ("x_t", 2.0), ("bucket_spread_t", 2.0),
                                                 ("beats_controls", 2.0), ("halves_agree", 1.0),
                                                 ("cell_min_obs", 40.0), ("cell_t", 2.0))
    # cells mode: the model-level gates are only those in `gates` that apply to cells (min_obs); each
    # CELL trades if its own |t| >= cell_t with >= cell_min_obs windows (and, in a family, its q)
    controls: tuple[str, ...] = ("vol", "abs_past")   # what `beats_controls` regresses X's effect beyond
    fallback: str = "flat"                      # gates fail -> flat | benchmark (the unconditional chase/fade rule)
    fixed_map: tuple[tuple[float, float], ...] | None = None
                                                # a NO-FIT rule: (X bucket, +1 chase / -1 fade / 0 flat) - the
                                                # signal map is this, always; fit only reports statistics and no
                                                # gate applies (a pre-stated hypothesis, like the fixed 5s30s rule)
    # out-of-sample evaluation suite (EVALUATIONS registry) for research runs
    evaluations: tuple[str, ...] = ("benchmark", "clark_west", "spanning", "sharpe_diff", "subperiods",
                                    "permutation", "synthetic_ar1", "time_shift")
    n_placebo: int = 100                        # placebo walk-forwards per placebo evaluation
    placebo_block_days: int = 63                # block length of the X permutation (keeps X's persistence)

    def threshold(self, gate: str, default: float | None = None) -> float | None:
        return dict(self.thresholds).get(gate, default)


AUTOCORR_MODELS: dict[str, AutocorrSpec] = {s.name: s for s in (
    AutocorrSpec("default", "strict: every gate at |t| >= 2"),
    AutocorrSpec("loose", "loose: X's own t and the bucket spread at 1.5, no controls gate",
                 gates=("min_obs", "x_t", "bucket_spread_t"),
                 thresholds=(("min_obs", 150.0), ("x_t", 1.5), ("bucket_spread_t", 1.5))),
    AutocorrSpec("none", "no gate: always use the conditional rule (research)", gates=("min_obs",),
                 thresholds=(("min_obs", 100.0),)),
    AutocorrSpec("cells", "the 3x3 cell model (X bucket x past-move bucket), cells at |t| >= 2", mode="cells",
                 gates=("min_obs",)),
    AutocorrSpec("fade_high_chase_low", "no fit: fade the target's move when X is in its + tercile, chase it in "
                 "the - tercile, flat in the middle", fixed_map=((1.0, -1.0), (0.0, 0.0), (-1.0, 1.0)),
                 gates=(), evaluations=("benchmark", "spanning", "subperiods", "permutation", "synthetic_ar1",
                                        "time_shift")),
    AutocorrSpec("chase_high_fade_low", "no fit: the mirror - chase when X is high, fade when low",
                 fixed_map=((1.0, 1.0), (0.0, 0.0), (-1.0, -1.0)), gates=(),
                 evaluations=("benchmark", "spanning", "subperiods", "permutation", "synthetic_ar1", "time_shift")),
    AutocorrSpec("cells_loose", "the cell model, cells at |t| >= 1.5", mode="cells", gates=("min_obs",),
                 thresholds=(("min_obs", 150.0), ("cell_min_obs", 30.0), ("cell_t", 1.5))),
)}


def get_autocorr_spec(spec: AutocorrSpec | str | None = None, **overrides) -> AutocorrSpec:
    base = AUTOCORR_MODELS["default"] if spec is None else (AUTOCORR_MODELS[spec] if isinstance(spec, str) else spec)
    return replace(base, **overrides) if overrides else base
