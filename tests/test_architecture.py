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
