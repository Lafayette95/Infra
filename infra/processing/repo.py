"""Repo rates (CLAUDE.md 19): each source's payload -> one long, typed frame. Pure, no I/O.

One row per (``timestamp`` = the trade/effective day, ``series``, ``status``):
* ``SOFR`` / ``TGCR`` / ``BGCR`` (NY Fed): ``rate`` and its 1/25/75/99th volume-weighted
  percentiles, ``volume_bn``; status ``final`` (the NY Fed's value as last published).
* ``OFR_<service>_<bucket>`` (OFR repo release, e.g. ``OFR_DVP_OO``): ``rate`` (volume-
  weighted average) and ``volume_bn``, as ``preliminary`` AND ``final`` rows - both kept,
  so what was known before the final arrived stays visible; ``best`` picks final.
* ``DTCC_GCF_<collateral>`` (DTCC GCF Repo Index, 2005-2024): ``rate`` only, ``final``.
Rates in percent; on disk x10000 nullable Int32 (0.01bp, Rule 6b).
"""
from __future__ import annotations

import io

import pandas as pd

REPO_COLUMNS = ["timestamp", "series", "status", "source", "rate", "volume_bn", "p1", "p25", "p75", "p99"]
REPO_KEYS = ["timestamp", "series", "status"]
RATE_COLUMNS = ("rate", "p1", "p25", "p75", "p99")
FINAL, PRELIMINARY = "final", "preliminary"
_SCALE = 10_000
_OFR_STATUS = {"F": FINAL, "P": PRELIMINARY}
GCF_SHEET = "Data 2005-2024"
GCF_COLUMNS = {"MBS": "DTCC_GCF_MBS", "Treasury": "DTCC_GCF_TSY", "Agency": "DTCC_GCF_AGENCY"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in REPO_COLUMNS}).astype(
        {"timestamp": "datetime64[ms]", "series": object, "status": object, "source": object})


def _frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.reindex(columns=REPO_COLUMNS)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    for c in ("volume_bn",) + RATE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out.dropna(subset=["rate"]).sort_values(REPO_KEYS).reset_index(drop=True)


def nyfed_rates(records: list[dict]) -> pd.DataFrame:
    """NY Fed ``refRates`` records -> rows."""
    if not records:
        return _empty()
    df = pd.DataFrame(records)
    return _frame(pd.DataFrame({
        "timestamp": df["effectiveDate"], "series": df["type"].str.upper(), "status": FINAL, "source": "nyfed",
        "rate": df.get("percentRate"), "volume_bn": df.get("volumeInBillions"),
        "p1": df.get("percentPercentile1"), "p25": df.get("percentPercentile25"),
        "p75": df.get("percentPercentile75"), "p99": df.get("percentPercentile99")}))


def ofr_series(service: str, bucket: str) -> str:
    return f"OFR_{service}_{bucket}"


def ofr_mnemonic(service: str, measure: str, bucket: str, status: str) -> str:
    """e.g. ``("DVP", "AR", "OO", "F")`` -> ``REPO-DVP_AR_OO-F``."""
    return f"REPO-{service}_{measure}_{bucket}-{status}"


def ofr_mnemonics(buckets: dict[str, tuple[str, ...]]) -> list[str]:
    return [ofr_mnemonic(s, m, b, st) for s, bs in buckets.items() for b in bs for m in ("AR", "TV") for st in "PF"]


def ofr_rates(payload: dict[str, list], buckets: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """OFR ``{mnemonic: [[date, value], ...]}`` -> rows (volume in $bn). A day with a volume
    but no rate (a disclosure edit) has no row."""
    parts = []
    for service, bs in buckets.items():
        for bucket in bs:
            for st, status in _OFR_STATUS.items():
                ar = pd.DataFrame(payload.get(ofr_mnemonic(service, "AR", bucket, st)) or [], columns=["timestamp", "rate"])
                tv = pd.DataFrame(payload.get(ofr_mnemonic(service, "TV", bucket, st)) or [], columns=["timestamp", "volume"])
                if ar.empty:
                    continue
                df = ar.merge(tv, on="timestamp", how="left")
                parts.append(pd.DataFrame({
                    "timestamp": df["timestamp"], "series": ofr_series(service, bucket), "status": status,
                    "source": "ofr", "rate": df["rate"],
                    "volume_bn": pd.to_numeric(df["volume"], errors="coerce") / 1e9}))
    return _frame(pd.concat(parts, ignore_index=True)) if parts else _empty()


def gcf_index(workbook: bytes) -> pd.DataFrame:
    """The DTCC GCF Repo Index workbook -> rows (one series per collateral type)."""
    raw = pd.read_excel(io.BytesIO(workbook), sheet_name=GCF_SHEET, header=None)
    head = raw.index[raw[0].astype(str).str.strip() == "Date"][0]
    names = {}
    for col, text in raw.loc[head].items():
        for key, series in GCF_COLUMNS.items():
            if isinstance(text, str) and text.strip().startswith(key):
                names[col] = series
    data = raw.loc[head + 1:].copy()
    data = data[pd.to_datetime(data[0], errors="coerce").notna()]
    parts = [pd.DataFrame({"timestamp": pd.to_datetime(data[0]), "series": series, "status": FINAL,
                           "source": "dtcc_gcf", "rate": data[col]}) for col, series in names.items()]
    return _frame(pd.concat(parts, ignore_index=True)) if parts else _empty()


def best(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (timestamp, series): final where it exists, else preliminary."""
    if df.empty:
        return df
    rank = (df["status"] != FINAL).astype(int)
    return (df.assign(_r=rank).sort_values(["timestamp", "series", "_r"])
            .drop_duplicates(["timestamp", "series"]).drop(columns="_r").reset_index(drop=True))


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Disk form (CLAUDE.md 6b): rates x10000 nullable Int32."""
    out = df[REPO_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ms]")
    for c in RATE_COLUMNS:
        out[c] = (out[c].astype("float64") * _SCALE).round().astype("Int32")
    out["volume_bn"] = out["volume_bn"].astype("float64")
    return out


def decode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in RATE_COLUMNS:
        out[c] = out[c].astype("float64") / _SCALE
    for c in ("series", "status", "source"):
        out[c] = out[c].astype(str)
    return out
