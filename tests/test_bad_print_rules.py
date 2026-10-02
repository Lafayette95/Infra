"""The bad-print rule's extensions (infra.config.BadPrintRules, CLAUDE.md 12): the market
factor from the other curves of the same currency, and policy-decision exemptions."""
from __future__ import annotations

import pandas as pd

from infra.config import BadPrintRules
from infra.cycle import bad_prints
from infra.cycle.px import peer_outliers
from infra.cycle.universe import UniverseMember

D = pd.Timestamp
DAYS = [d for d in pd.date_range("2025-01-01", "2025-03-14") if d.weekday() < 5]
WIN = (D("2025-03-03"), D("2025-03-14"))
SHOCK = D("2025-03-05")
RULES = BadPrintRules(exempt_events=())


def _curve(tickers, root, dataset="GLBX.MDP3"):
    """Every contract drifts +0.01/weekday, so its typical |move| (the z unit) is 0.01."""
    members = {t: UniverseMember(t, root, dataset, D("2025-01-01"), D("2025-04-30") + pd.DateOffset(months=i), None)
               for i, t in enumerate(tickers)}
    rows = [(t, d, 96.0 + 0.01 * n) for t in tickers for n, d in enumerate(DAYS)]
    return pd.DataFrame(rows, columns=["ticker", "timestamp", "settlement_price"]), members


def _bump(settle, ticker, days, by):
    m = (settle["ticker"] == ticker) & settle["timestamp"].isin(days)
    settle.loc[m, "settlement_price"] += by


def _usd_market(bond_shock: float, n_bonds: int = 2):
    """A ZQ strip with a 10bp kink on SHOCK that reverts (z 11, deviation 10), plus
    Treasury futures that all move ``bond_shock`` that day and keep it."""
    settle, members = _curve(["A", "B", "C", "D", "E"], "ZQ")
    _bump(settle, "C", [SHOCK], 0.10)
    for root in ("ZT", "ZF", "ZN")[:n_bonds]:
        s, m = _curve([f"{root}H5", f"{root}M5"], root)
        for t in m:
            _bump(s, t, [d for d in DAYS if d >= SHOCK], bond_shock)
        settle, members = pd.concat([settle, s], ignore_index=True), {**members, **m}
    return settle, members


def test_a_market_wide_shock_raises_the_tolerance():
    settle, members = _usd_market(bond_shock=0.15)  # Treasuries move 16x their normal
    assert peer_outliers(settle, members, *WIN, rules=None)[0]["ticker"].tolist() == ["C"]  # base rule: flagged
    flagged, _ = peer_outliers(settle, members, *WIN, rules=RULES)
    assert flagged.empty


def test_a_quiet_market_leaves_the_rule_unchanged():
    settle, members = _usd_market(bond_shock=0.0)
    flagged, _ = peer_outliers(settle, members, *WIN, rules=RULES)
    assert flagged["ticker"].tolist() == ["C"] and flagged["market_k"].tolist() == [1.0]


def test_a_bad_print_cannot_excuse_itself_through_its_own_curve():
    """The factor comes from OTHER curves only: two bad prints on a short strip don't
    raise it (2026-09-11, ESRM7 + ESRU7 were 2 of 6 contracts)."""
    settle, members = _curve(["E1", "E2", "E3", "E4", "E5", "E6"], "ESR", dataset="XEUR.EOBI")
    for t in ("E3", "E4"):
        _bump(settle, t, [SHOCK], 0.10)
    bunds, m = _curve(["FGBLH5", "FGBLM5", "FGBMH5", "FGBMM5"], "FGBL", dataset="XEUR.EOBI")
    settle, members = pd.concat([settle, bunds], ignore_index=True), {**members, **m}
    flagged, _ = peer_outliers(settle, members, *WIN, rules=RULES)
    assert set(flagged["ticker"]) >= {"E3"} and (flagged["market_k"] == 1.0).all()


def test_too_few_other_instruments_means_no_scaling():
    settle, members = _usd_market(bond_shock=0.15, n_bonds=1)  # only 2 other instruments
    flagged, _ = peer_outliers(settle, members, *WIN, rules=RULES)
    assert flagged["ticker"].tolist() == ["C"]


def test_the_market_factor_can_be_switched_off():
    settle, members = _usd_market(bond_shock=0.15)
    flagged, _ = peer_outliers(settle, members, *WIN, rules=BadPrintRules(market_scaling=False, exempt_events=()))
    assert flagged["ticker"].tolist() == ["C"]


def test_a_policy_decision_session_is_exempt_for_its_roots(monkeypatch):
    settle, members = _usd_market(bond_shock=0.0)
    monkeypatch.setitem(bad_prints.EXEMPTION_EVENTS, "fomc", lambda: pd.DatetimeIndex([SHOCK]))
    exempt = BadPrintRules(exempt_events=(("fomc", ("ZQ",)),))
    assert peer_outliers(settle, members, *WIN, rules=exempt)[0].empty
    other_root = BadPrintRules(exempt_events=(("fomc", ("SR3",)),))
    assert peer_outliers(settle, members, *WIN, rules=other_root)[0]["ticker"].tolist() == ["C"]


def test_fomc_sessions_move_a_weekend_decision_to_monday():
    from infra.config import FOMCMeeting
    sessions = bad_prints.fomc_decision_sessions((FOMCMeeting("2020-03-03", "2020-03-03", False, scheduled=False),
                                                  FOMCMeeting("2020-03-15", "2020-03-15", False, scheduled=False)))
    assert list(sessions) == [D("2020-03-03"), D("2020-03-16")]
