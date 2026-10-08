"""German Federal issuance plans (infra.processing.de_issuance, infra.pipeline.de_issuance):
the workbook and page parsers, the event mapping, and the point-in-time replay."""
from __future__ import annotations

import io

import pandas as pd

from infra.pipeline import de_issuance as pd_
from infra.processing import de_issuance as di

D = pd.Timestamp


def _xlsx(rows, header, preamble=("Issuance outlook", "As of September 2026")) -> bytes:
    top = [[p] + [None] * (len(header) - 1) for p in preamble] + [header]
    buf = io.BytesIO()
    pd.DataFrame(top + rows).to_excel(buf, header=False, index=False)
    return buf.getvalue()


def test_outlook_columns_are_found_by_name_in_both_layouts():
    h2025 = ["Serial number", "Date", "Security", "Term to maturity/ Remaining term*", "Type", "Volume in € mn",
             "Maturity", "ISIN", "Coupon", "Start of interest period", "First coupon date"]
    df, as_of = di.parse_outlook(_xlsx([[1, D("2026-10-13"), "Bobl", "5 years", "Reopening", 5000, D("2031-10-08"),
                                         "DE000BU25075", 0.029, D("2026-07-23"), D("2027-10-08")]], h2025))
    assert as_of == "As of September 2026"
    r = df.iloc[0]
    assert (r["term"], r["issue_kind"], r["volume_m"], r["isin"], round(r["coupon"], 4)) == \
        ("5 years", "reopening", 5000, "DE000BU25075", 2.9)
    # 2024: upper case, separate remaining-term column, a "BN" header over € mn values
    h2024 = ["NUMBER", "DATE", "SECURITY", "TERM TO MATURITY", "REMAINING TERM", "TYPE", "VOLUME IN € BN", "MATURITY",
             "ISIN", "COUPON", "START OF INTEREST PERIOD", "FIRST COUPON DATE"]
    df, as_of = di.parse_outlook(_xlsx([[1, D("2024-01-08"), "Bubill", None, "6 months", "Reopening", 2000,
                                         D("2024-07-17"), "DE000BU0E071", None, None, None],
                                        [2, D("2024-01-17"), "Bund", "30 years", None, "Reopening", 1000, None,
                                         "-", None, None, None]], h2024, preamble=("Issuance outlook",)))
    assert as_of is None
    assert list(df["term"]) == ["6 months", "30 years"] and list(df["volume_m"]) == [2000, 1000]
    assert pd.isna(df["isin"].iloc[1])                      # a placeholder line


PAGE = """<h2>Upcoming Issues</h2><table><tbody>
<tr><td> 13.10.2026 </td><td> Bobl (<abbr title="Reopening">R</abbr>)<br /><a href="x"><span>DE000BU25075</span></a></td>
<td> 5.0 € bn </td><td> 2.90 % </td><td> 08.10.2031 </td></tr>
<tr><td> 14.10.2026 </td><td> Bund 30 Y (<abbr title="Reopening">R</abbr>)<br /></td><td> 1.0 € bn </td><td> - </td><td> - </td></tr>
</tbody></table>"""


def test_live_table():
    df = di.parse_upcoming(PAGE)
    assert list(df["security"]) == ["Bobl", "Bund"] and list(df["term"].fillna("-")) == ["-", "30 Y"]
    assert list(df["isin"].fillna("-")) == ["DE000BU25075", "-"] and list(df["volume_m"]) == [5000.0, 1000.0]
    assert df["coupon"].iloc[0] == 2.9 and df["issue_kind"].iloc[1] == "reopening"


def test_event_mapping():
    seg = {"DE000BU2Z072": "10 Y"}
    assert di.event_id("Bund", "9 years", "DE000BU2Z072", seg) == "DE_AUCTION_10Y"   # the ISIN's segment wins
    assert di.event_id("Bund", "30 Y", None, seg) == "DE_AUCTION_30Y"
    assert di.event_id("Bund", "15/20/30 years", None, seg) == "DE_AUCTION_LONG"
    assert di.event_id("Bund/g", None, None, seg) == "DE_AUCTION_GREEN"
    assert di.event_id("Schatz", None, None, seg) == "DE_AUCTION_2Y"
    assert di.event_id("Bubill", "6 months", None, seg) == "DE_AUCTION_BUBILL"


def _plan(rows, vintage, known, cover):
    df = pd.DataFrame(rows, columns=["auction_date", "security", "isin"])
    for c in di.PLAN_COLUMNS:
        if c not in df:
            df[c] = None
    df["auction_date"] = pd.to_datetime(df["auction_date"])
    return df.assign(vintage=vintage, layer="x", known_from=D(known), cover_from=D(cover))[pd_.VINTAGE_COLUMNS]


def test_replay_replaces_from_coverage_and_only_the_classes_listed():
    annual = _plan([("2026-02-03", "Bobl", "A1"), ("2026-05-05", "Bobl", "A2"), ("2026-05-06", "Green issue", None)],
                   "annual", "2025-12-18", "2026-01-01")
    update = _plan([("2026-05-12", "Bobl", "A3")], "q2", "2026-03-23", "2026-04-01")
    states = list(pd_.replay(pd.concat([annual, update], ignore_index=True)))
    last = states[-1][1]
    # Q1 kept from the annual plan, its Q2 Bobl replaced, the Green line (a class the update
    # doesn't list) kept
    assert sorted(last["isin"].fillna("green")) == ["A1", "A3", "green"]


def test_archive_versions_by_last_modified(tmp_path, monkeypatch):
    lm = {"x_en.xlsx": "Wed, 18 Dec 2024 09:13:28 GMT"}
    monkeypatch.setattr(pd_, "LAST_MODIFIED", lambda name: lm[name])
    monkeypatch.setattr(pd_, "FETCH_OUTLOOK", lambda name: (b"PK..", lm[name]))
    assert [p.name for p in pd_.archive_outlooks(["x_en.xlsx"], root=tmp_path)] == ["x_en__20241218T091328Z.xlsx"]
    assert pd_.archive_outlooks(["x_en.xlsx"], root=tmp_path) == []          # same version: nothing fetched
    lm["x_en.xlsx"] = "Wed, 04 Feb 2026 08:48:28 GMT"                           # replaced in place
    assert [p.name for p in pd_.archive_outlooks(["x_en.xlsx"], root=tmp_path)] == ["x_en__20260204T084828Z.xlsx"]
