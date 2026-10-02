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
    releases_dir: Path
    releases_coverage: Path
    release_calendar_dir: Path
    econ_calendar_dir: Path
    tsy_auctions_dir: Path
    tsy_auctions_coverage: Path
    tsy_tails_dir: Path
    tsy_tails_manifest: Path
    raw_data_root: Path
    bulk_coverage: Path
    bulk_catalog_dir: Path
    cpi_weights_dir: Path
    cpi_weights_coverage: Path

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
            releases_dir=config.RELEASES_DIR,
            releases_coverage=config.RELEASES_COVERAGE_FILE,
            release_calendar_dir=config.RELEASE_CALENDAR_DIR,
            econ_calendar_dir=config.CALENDAR_DIR,
            tsy_auctions_dir=config.TSY_AUCTIONS_DIR,
            tsy_auctions_coverage=config.TSY_AUCTIONS_COVERAGE_FILE,
            tsy_tails_dir=config.TSY_TAILS_DIR,
            tsy_tails_manifest=config.TSY_TAILS_COVERAGE_FILE,
            raw_data_root=config.RAW_DATA_ROOT,
            bulk_coverage=config.BULK_COVERAGE_FILE,
            bulk_catalog_dir=config.BULK_CATALOG_DIR,
            cpi_weights_dir=config.CPI_WEIGHTS_DIR,
            cpi_weights_coverage=config.CPI_WEIGHTS_COVERAGE_FILE,
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
