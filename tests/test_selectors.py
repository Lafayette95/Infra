"""Ticker Root / Expiry dropdown options (no Dash needed)."""
from __future__ import annotations

import pandas as pd

from infra.config import DEFAULT_RELATIVE_RANKS, FUTURES_ROOTS
from infra.dashboard.selectors import default_expiry, expiry_options, root_options
from infra.processing import transforms as tf
from infra.storage import contract_store, parquet_store


def _contracts() -> pd.DataFrame:
    return pd.DataFrame({
        "root": ["SR3", "SR3", "ZN"], "ticker": ["SRZ4", "SRH5", "ZNH5"],
        "instrument_id": [1, 2, 3],
        "expiry": pd.to_datetime(["2025-03-18", "2025-06-17", "2025-03-20"]).astype("datetime64[ms]"),
        "activation": pd.NaT,
    })


def test_root_options_cover_every_configured_root():
    opts = root_options()
    assert [o["value"] for o in opts] == list(FUTURES_ROOTS)
    assert "SR3 · 3M SOFR (STIR)" in [o["label"] for o in opts]


def test_expiry_options_relative_first_then_root_contracts_with_disk_marker(tmp_path):
    contracts_file, futures_dir = tmp_path / "contracts.parquet", tmp_path / "Futures"
    contract_store.write_contracts(contracts_file, _contracts())
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-03-10 10:00"]).astype("datetime64[ms]"), "ticker": ["SRZ4"],
        "open": [95.0], "high": [95.0], "low": [95.0], "close": [95.0], "volume": [1], "open_interest": [None],
    }).astype({"volume": "int32", "open_interest": "Int32"})
    parquet_store.write_partitioned(tf.encode_futures(bars), futures_dir, tf.FUTURES_KEYS)

    opts = expiry_options("SR3", contracts_file=contracts_file, futures_dir=futures_dir)
    n_rel = len(DEFAULT_RELATIVE_RANKS)
    assert [o["value"] for o in opts[:n_rel]][:2] == ["SR3.c.0", "SR3.c.1"]
    assert [o["value"] for o in opts[n_rel:]] == ["SRZ4", "SRH5"]  # by expiry, ZN excluded
    assert opts[n_rel]["label"].endswith("expires 2025-03-18 ●")
    assert "●" not in opts[n_rel + 1]["label"]  # SRH5 has no bars on disk


def test_expiry_options_empty_contracts_still_offers_relative(tmp_path):
    opts = expiry_options("SR3", contracts_file=tmp_path / "none.parquet", futures_dir=tmp_path / "F")
    assert [o["value"] for o in opts] == [f"SR3.{k}.{r}" for k, r in DEFAULT_RELATIVE_RANKS]


def test_default_expiry_keeps_relative_choice_across_roots():
    assert default_expiry("ZN", "SR3.c.1") == "ZN.c.1"
    assert default_expiry("ZN", "SR3.v.1") == "ZN.v.1"
    assert default_expiry("ZN", "SRZ4") == "ZN.v.0"  # absolute selection cannot carry over
    assert default_expiry("ZN", None) == "ZN.v.0"
    assert default_expiry("ZN", "ZN.c.9") == "ZN.v.0"  # rank not offered
