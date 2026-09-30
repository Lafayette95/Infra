"""Every storage location the daily cycle touches, bundled so a test can point the whole
cycle at a temp directory in one go - the lesson recorded in TOFIX.md (a function with no
path override eventually leaks test data into the real ~/Database)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from infra import config


@dataclass(frozen=True)
class CyclePaths:
    database_root: Path
    daily_futures_dir: Path
    daily_futures_coverage: Path
    daily_bonds_dir: Path
    daily_bonds_coverage: Path
    contracts_file: Path
    defs_coverage: Path
    wirp_dir: Path
    bmk_root: Path
    vintage_root: Path
    adjustments_dir: Path

    @classmethod
    def default(cls) -> CyclePaths:
        return cls(
            database_root=config.DATABASE_ROOT,
            daily_futures_dir=config.DAILY_FUTURES_DIR,
            daily_futures_coverage=config.DAILY_FUTURES_COVERAGE_FILE,
            daily_bonds_dir=config.DAILY_BONDS_DIR,
            daily_bonds_coverage=config.DAILY_BONDS_COVERAGE_FILE,
            contracts_file=config.FUTURES_CONTRACTS_FILE,
            defs_coverage=config.FUTURES_DEFS_COVERAGE_FILE,
            wirp_dir=config.WIRP_DIR,
            bmk_root=config.BMK_ROOT,
            vintage_root=config.VINTAGE_ROOT,
            adjustments_dir=config.ADJUSTMENTS_DIR,
        )

    @classmethod
    def under(cls, root: Path) -> CyclePaths:
        """The default layout rebased under ``root`` - config stays the single source of
        truth for the layout itself, only the root moves."""
        base = cls.default()
        return cls(**{
            f.name: root if f.name == "database_root"
            else root / getattr(base, f.name).relative_to(base.database_root)
            for f in fields(cls)
        })
