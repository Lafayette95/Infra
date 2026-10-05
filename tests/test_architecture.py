"""Architecture rule (CLAUDE.md section 3): dependencies point one way - nothing outside
infra/dashboard imports from it, and the headless daily cycle never loads Dash."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

INFRA = Path(__file__).resolve().parents[1] / "infra"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def test_nothing_below_the_dashboard_imports_it():
    offenders = []
    for path in INFRA.rglob("*.py"):
        if "dashboard" in path.relative_to(INFRA).parts:
            continue
        bad = {m for m in _imports(path)
               if m == "infra.dashboard" or m.startswith("infra.dashboard.") or m == "dash" or m.startswith("dash.")}
        if bad:
            offenders.append(f"{path.relative_to(INFRA.parent)}: {sorted(bad)}")
    assert not offenders, "upward/Dash imports:\n" + "\n".join(offenders)


def test_importing_the_daily_cycle_never_loads_dash_or_plotly():
    code = ("import sys, infra.cycle.runner, infra.cycle.flows; "
            "print(sorted(m for m in ('dash', 'plotly', 'infra.dashboard') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=INFRA.parent, check=True).stdout.strip().splitlines()[-1]
    assert out == "[]", out


def test_nothing_below_the_models_imports_them():
    """infra/models (the nowcast) sits above the data pipeline: api / processing /
    pipeline / cycle / storage never import it (only the dashboard may, from above)."""
    offenders = []
    for path in INFRA.rglob("*.py"):
        parts = path.relative_to(INFRA).parts
        if parts[0] in ("models", "dashboard", "strategies", "jobs"):
            continue
        bad = {m for m in _imports(path) if m == "infra.models" or m.startswith("infra.models.")}
        if bad:
            offenders.append(f"{path.relative_to(INFRA.parent)}: {sorted(bad)}")
    assert not offenders, "upward imports into infra/models:\n" + "\n".join(offenders)


def _offenders(package: str, allowed: tuple[str, ...]) -> list[str]:
    out = []
    for path in INFRA.rglob("*.py"):
        if path.relative_to(INFRA).parts[0] in allowed:
            continue
        bad = {m for m in _imports(path) if m == f"infra.{package}" or m.startswith(f"infra.{package}.")}
        if bad:
            out.append(f"{path.relative_to(INFRA.parent)}: {sorted(bad)}")
    return out


def test_layering_above_the_models():
    """Strategies sit above the models (they consume model predictions); jobs above both
    (the only layer writing model / strategy outputs). Nothing below imports them; only the
    dashboard may, from above."""
    assert not _offenders("strategies", ("strategies", "jobs", "dashboard"))
    assert not _offenders("jobs", ("jobs", "dashboard"))


def test_the_suite_cannot_reach_the_outside_world():
    """tests/conftest.py blocks every non-loopback connection: an unstubbed fetch fails
    at once with a clear message instead of hanging the suite."""
    import time
    import urllib.request

    import pytest

    t = time.monotonic()
    with pytest.raises(OSError, match="external network blocked in tests"):
        urllib.request.urlopen("https://api.stlouisfed.org/fred/series?series_id=PAYEMS", timeout=30)
    assert time.monotonic() - t < 5
