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
    dtcc_dir: Path
    swap_closes_dir: Path
    ois_curves_dir: Path
    swap_spreads_dir: Path
    inflation_swap_closes_dir: Path
    inflation_curves_dir: Path
    xccy_basis_closes_dir: Path
    swaption_records_dir: Path
    swaption_prints_dir: Path
    swaption_vols_dir: Path
    swaption_oi_dir: Path
    vrp_dir: Path
    daily_options_dir: Path
    cme_tcf_dir: Path
    treasury_securities_dir: Path
    treasury_otr_dir: Path
    treasury_baskets_dir: Path
    treasury_prices_dir: Path
    treasury_prices_coverage: Path
    tips_prices_dir: Path
    tips_prices_coverage: Path
    repo_dir: Path
    repo_coverage: Path
    sec_lending_dir: Path
    sec_lending_coverage: Path
    treasury_curves_dir: Path
    treasury_rv_dir: Path
    tips_curves_dir: Path
    tips_rv_dir: Path
    otr_yields_dir: Path

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
            dtcc_dir=config.DTCC_DIR,
            swap_closes_dir=config.SWAP_CLOSES_DIR,
            ois_curves_dir=config.OIS_CURVES_DIR,
            swap_spreads_dir=config.SWAP_SPREADS_DIR,
            inflation_swap_closes_dir=config.INFLATION_SWAP_CLOSES_DIR,
            inflation_curves_dir=config.INFLATION_CURVES_DIR,
            xccy_basis_closes_dir=config.XCCY_BASIS_CLOSES_DIR,
            swaption_records_dir=config.SWAPTION_RECORDS_DIR,
            swaption_prints_dir=config.SWAPTION_PRINTS_DIR,
            swaption_vols_dir=config.SWAPTION_VOLS_DIR,
            swaption_oi_dir=config.SWAPTION_OI_DIR,
            vrp_dir=config.VRP_DIR,
            daily_options_dir=config.DAILY_OPTIONS_DIR,
            cme_tcf_dir=config.CME_TCF_DIR,
            treasury_securities_dir=config.TREASURY_SECURITIES_DIR,
            treasury_otr_dir=config.TREASURY_OTR_DIR,
            treasury_baskets_dir=config.TREASURY_BASKETS_DIR,
            treasury_prices_dir=config.DAILY_TREASURY_PRICES_DIR,
            treasury_prices_coverage=config.DAILY_TREASURY_PRICES_COVERAGE_FILE,
            tips_prices_dir=config.DAILY_TIPS_PRICES_DIR,
            tips_prices_coverage=config.DAILY_TIPS_PRICES_COVERAGE_FILE,
            repo_dir=config.REPO_DIR,
            repo_coverage=config.REPO_COVERAGE_FILE,
            sec_lending_dir=config.SEC_LENDING_DIR,
            sec_lending_coverage=config.SEC_LENDING_COVERAGE_FILE,
            treasury_curves_dir=config.TREASURY_CURVES_DIR,
            treasury_rv_dir=config.TREASURY_RV_DIR,
            tips_curves_dir=config.TIPS_CURVES_DIR,
            tips_rv_dir=config.TIPS_RV_DIR,
            otr_yields_dir=config.OTR_YIELDS_DIR,
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
