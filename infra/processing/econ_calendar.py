"""Economic-calendar pages -> rows -> release vintages. Pure (no I/O).

A calendar page (MarketWatch's U.S. economic calendar, as archived by the Wayback
Machine - infra/api/wayback_client.py) lists one or two weeks of releases: day headers
("TUESDAY, SEPT. 3"), then one row per release with time / report / period / actual /
consensus / previous. Verified 2026-10-01 on captures from 2010, 2018, 2020, 2024 and
2026: the layout changed (2010 "Consensus forecast", 2018 "MEDIAN forecast", 2020+
"Median Forecast", 2026 "Forecast" plus an extra button column and "Add to My Calendar"
rows), so columns are always mapped from each table's OWN header row, never by position.

Three facts the parsing relies on:

* the week shown is the page's, NOT the capture's: an Oct 2020 capture of the old URL
  still shows April 2020 (the page froze when MarketWatch moved it). Day headers carry
  no year, so it is inferred from the weekday: the most recent year in which that month
  and day fall on that weekday, up to two weeks after the capture (the next-week table);
* names drift (``ISM`` / ``ISM manufacturing index`` / ``ISM Report On Business
  Manufacturing PMI``; Markit -> S&P) - ``normalize_report`` + each release's
  ``MacroRelease.calendar_pattern`` absorb that;
* values carry units and typos (``47.2%``, ``-$78.8B``, ``5.02 mln``, ``4.43 miln``,
  ``3.86 million``, ``2.1% (Q4)``) - ``parse_value`` normalizes them to plain numbers.
"""
from __future__ import annotations

import gzip
import re
from html.parser import HTMLParser

import numpy as np
import pandas as pd

ROW_COLUMNS = ["timestamp", "report", "period", "time", "actual", "forecast", "previous",
               "actual_raw", "forecast_raw", "previous_raw", "capture", "page"]
ROW_KEYS = ["timestamp", "report", "period"]

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DAY_HEADER = re.compile(r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\W+([a-z]+)\.?\s+(\d{1,2})\b",
                         re.I)
_MONTH_LABEL = re.compile(r"^([a-z]{3,9})\.?$", re.I)
_MULTIPLIERS = [(re.compile(r"(trillion|tln|trn|t)$"), 1e12), (re.compile(r"(billion|bln|bn|b)$"), 1e9),
                (re.compile(r"(million|miln|mln|mil|m)$"), 1e6), (re.compile(r"(thousand|k)$"), 1e3)]


class _Tables(HTMLParser):
    """Every <tr> as a list of its cells' text (empty cells kept - their POSITION matters)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def decode_html(raw: bytes) -> str:
    """Archived pages are sometimes served still gzip-compressed (seen: 2026 captures)."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "ignore")


def table_rows(html: str) -> list[list[str]]:
    parser = _Tables()
    parser.feed(html)
    return parser.rows


def _column_map(header: list[str]) -> dict[str, int] | None:
    """Column name -> index from a header row, or None if this isn't one."""
    cols = {}
    for i, cell in enumerate(header):
        c = cell.lower()
        if c.startswith("time"):
            cols["time"] = i
        elif c == "report":
            cols["report"] = i
        elif c == "period":
            cols["period"] = i
        elif c == "actual":
            cols["actual"] = i
        elif "forecast" in c or "consensus" in c or c == "median":
            cols["forecast"] = i
        elif c == "previous":
            cols["previous"] = i
    return cols if {"report", "period", "actual", "previous"} <= set(cols) else None


def infer_day(weekday: str, month: int, day: int, capture: pd.Timestamp) -> pd.Timestamp | None:
    """The most recent date with this month/day falling on ``weekday``, at most 14 days
    after ``capture`` (the next-week table) and at most 15 years before it."""
    want = _WEEKDAYS.index(weekday.lower())
    for year in range(capture.year + 1, capture.year - 16, -1):
        try:
            d = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        if d.dayofweek == want and d <= capture.normalize() + pd.Timedelta(days=14):
            return d
    return None


def parse_day_header(text: str, capture: pd.Timestamp) -> pd.Timestamp | None:
    m = _DAY_HEADER.match(text.strip())
    if not m:
        return None
    month = _MONTHS.get(m.group(2).lower()[:3])
    return infer_day(m.group(1), month, int(m.group(3)), capture) if month else None


def resolution(text) -> float:
    """Half a unit of the LAST DISPLAYED digit, in parsed units: ``2.5 million`` -> 5e4,
    ``0.2%`` -> 0.05, ``368,000`` -> 0.5 - the rounding a displayed value carries."""
    value = parse_value(text)
    core = re.search(r"-?\d[\d,]*(?:\.(\d+))?", str(text) or "")
    if not np.isfinite(value) or core is None:
        return np.nan
    number = float(core.group(0).replace(",", ""))
    mult = value / number if number else 1.0
    return 0.5 * 10.0 ** -len(core.group(1) or "") * abs(mult)


_TIME = re.compile(r"^(\d{1,2})(?:[:;.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)$", re.I)


def parse_time(text) -> str | None:
    """A row's release time (``8:30 am``, ``10 am``, ``9:45 AM``, ``8;30 am``) -> ``"HH:MM"``
    (New York wall clock); None for ``Varies``, ``TBA``, blanks."""
    m = _TIME.match(str(text).strip().lower().replace(" ", "") if text else "")
    if not m:
        m = _TIME.match(str(text).strip()) if text else None
    if not m:
        return None
    hour = int(m.group(1)) % 12 + (12 if m.group(3).lower().startswith("p") else 0)
    return f"{hour:02d}:{m.group(2) or '00'}"


# order matters: "second revision" is the THIRD estimate, so "third" is tested first
_STAGES = [("flash", r"\bflash\b"), ("preliminary", r"\bprelim(inary)?\b"), ("advance", r"\badvance\b"),
           ("third", r"\b(3rd|third|second revision)\b"), ("second", r"\b(2nd|second|first revision)\b"),
           ("final", r"\b(final|revised)\b")]


def stage_of(report: str) -> str:
    """The estimate stage a row's NAME states (``flash``, ``preliminary``, ``advance``,
    ``second``, ``third``, ``final``), "" when it says none."""
    t = normalize_report(report)
    return next((name for name, rx in _STAGES if re.search(rx, t)), "")


def parse_value(text) -> float:
    """``47.2%`` -> 47.2, ``-$78.8B`` -> -7.88e10, ``5.02 mln`` -> 5.02e6, ``3.86 million``
    -> 3.86e6, ``2.1% (Q4)`` -> 2.1; blank / ``--`` / ``N/A`` -> NaN."""
    if text is None:
        return np.nan
    t = str(text).strip().lower()
    t = re.sub(r"\(.*?\)", "", t).strip()  # "2.1% (Q4)"
    if not t or t in {"-", "--", "n/a", "na", "tba", "nm"}:
        return np.nan
    t = t.replace("$", "").replace(",", "").replace("%", "").replace("−", "-").strip()
    t = re.sub(r"\s*r$", "", t)  # a trailing revision marker
    mult = 1.0
    for pattern, m in _MULTIPLIERS:
        stripped = pattern.sub("", t).strip()
        if stripped != t:
            t, mult = stripped, m
            break
    try:
        return float(t) * mult
    except ValueError:
        return np.nan


def parse_page(html: str, capture: pd.Timestamp, page: str = "") -> pd.DataFrame:
    """Every release row on a calendar page: ``ROW_COLUMNS``, ``timestamp`` = release day."""
    out, cols, day = [], None, None
    for row in table_rows(html):
        cells = row
        nonempty = [c for c in cells if c]
        header = _column_map(cells)
        if header is not None:
            cols, day = header, None
            continue
        if len(nonempty) == 1 and (d := parse_day_header(nonempty[0], capture)) is not None:
            day = d
            continue
        if cols is None or day is None or len(cells) <= max(cols.values()):
            continue
        report = cells[cols["report"]]
        if not report or report.lower().startswith(("none scheduled", "add to my calendar")):
            continue
        get = lambda k: cells[cols[k]] if k in cols else ""  # noqa: E731
        out.append({
            "timestamp": day, "report": report, "period": get("period"), "time": get("time"),
            "actual": parse_value(get("actual")), "forecast": parse_value(get("forecast")),
            "previous": parse_value(get("previous")), "actual_raw": get("actual"),
            "forecast_raw": get("forecast"), "previous_raw": get("previous"),
            "capture": capture, "page": page,
        })
    if not out:
        return empty_rows()
    df = pd.DataFrame(out, columns=ROW_COLUMNS)
    return df.drop_duplicates(subset=ROW_KEYS, keep="last").reset_index(drop=True)


def empty_rows() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in ROW_COLUMNS})
    for c in ("actual", "forecast", "previous"):
        df[c] = df[c].astype("float64")
    for c in ("timestamp", "capture"):
        df[c] = df[c].astype("datetime64[ms]")
    return df


def encode_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Flat, disk-ready (CLAUDE.md 6a). Values stay float64 (not prices; millions to
    trillions) next to the raw strings as shown, for audit."""
    out = df[ROW_COLUMNS].copy()
    for c in ("timestamp", "capture"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ms]")
    for c in ("report", "period", "time", "actual_raw", "forecast_raw", "previous_raw", "page"):
        out[c] = out[c].fillna("").astype(str)
    for c in ("actual", "forecast", "previous"):
        out[c] = out[c].astype("float64")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- releases
def normalize_report(name: str) -> str:
    """Lower case; ``U.S.``/``US`` and the shutdown markers (``(new date)``, ``*``,
    ``(delayed report)``, ``[delayed due to shutdown]``, ``/delayed``) dropped;
    punctuation (except ``&`` and parentheses) to spaces."""
    t = name.lower()
    t = re.sub(r"\(new date\)|\*", " ", t)  # the 2019 shutdown's rescheduled rows: "Housing starts* (new date)"
    # the 2025 shutdown's: "(delayed report)", "[delayed due to shutdown]", "/delayed*", "(delayed report"
    t = re.sub(r"[\(\[/\-]?\s*delayed( report| due to shutdown)?\s*[\)\]]?", " ", t)
    t = re.sub(r"\bu\.\s?s\.?(?=\s|$)|\bus\b", " ", t)
    t = re.sub(r"[^a-z0-9&()]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def matches(report: str, pattern: str) -> bool:
    return re.search(pattern, normalize_report(report)) is not None


def month_period(label: str, release_day: pd.Timestamp) -> pd.Timestamp | None:
    """A monthly period label (``Aug.``, ``Sept``, ``March``) -> its first day: the latest
    such month not after the release month. Anything else (``Q1``, ``4/25``, ``Aug. 31``)
    -> None: those are not monthly periods."""
    m = _MONTH_LABEL.match(str(label).strip())
    month = _MONTHS.get(m.group(1).lower()[:3]) if m else None
    if month is None:
        return None
    year = release_day.year if month <= release_day.month else release_day.year - 1
    return pd.Timestamp(year=year, month=month, day=1)


_QUARTER_LABEL = re.compile(r"^(?:q([1-4])|([1-4])q)r?$", re.I)
_WEEK_LABEL = re.compile(r"^(?:(\d{1,2})/(\d{1,2})|([a-z]{3,9})\.?\s+(\d{1,2}))(?:\s+week)?$", re.I)


def quarter_period(label: str, release_day: pd.Timestamp) -> pd.Timestamp | None:
    """``Q3`` / ``3Q`` / ``2Q``r -> the quarter's first day: the latest such quarter
    starting before the release day."""
    m = _QUARTER_LABEL.match(str(label).strip())
    if not m:
        return None
    q = int(m.group(1) or m.group(2))
    start = pd.Timestamp(year=release_day.year, month=3 * q - 2, day=1)
    return start if start < release_day else pd.Timestamp(year=release_day.year - 1, month=3 * q - 2, day=1)


def week_period(label: str, release_day: pd.Timestamp) -> pd.Timestamp | None:
    """A week-ending label (``4/7``, ``Aug. 31``, ``June 13 week``) -> that day: the
    latest such date not after the release day."""
    m = _WEEK_LABEL.match(str(label).strip())
    if not m:
        return None
    if m.group(1):
        month, day = int(m.group(1)), int(m.group(2))
    else:
        month, day = _MONTHS.get(m.group(3).lower()[:3]), int(m.group(4))
    if not month:
        return None
    for year in (release_day.year, release_day.year - 1):
        try:
            d = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return None
        if d <= release_day:
            # weeks end on SATURDAY (the agencies' and FRED's dating); the calendar
            # sometimes labels a week by its Friday (verified 2026-10-01 vs ICSA)
            return d + pd.Timedelta(days=(5 - d.dayofweek) % 7)
    return None


_PERIOD = {"M": month_period, "Q": quarter_period, "W": week_period}
_PRIOR = {"M": pd.DateOffset(months=1), "Q": pd.DateOffset(months=3), "W": pd.DateOffset(weeks=1)}


def period_of(label: str, release_day: pd.Timestamp, frequency: str) -> pd.Timestamp | None:
    """The observation period a calendar row refers to, at the release's frequency (the
    source's own period dating: month/quarter start, week-ending day); None when the
    label isn't a period of that frequency."""
    return _PERIOD[frequency](label, release_day)


# Typo filter (``suspect_prints``). A calendar typo is a UNIT/DECIMAL slip - a value off
# by an order of magnitude ("6,280" for 628k housing starts, "50.0" for a 5.0% jobless
# rate, "368.0" for 368k claims, "5.49" for 5.49 million existing home sales; all found
# 2026-10-01 by the FRED cross-check) - not a big print or a revision. "Off" = at least
# SUSPECT_SHIFT_LOG10 orders of magnitude apart; a comparison where BOTH numbers are small
# for the series (< SUSPECT_SMALL x its median |value|: a +20k payroll month, a diffusion
# index near 0) is skipped - a ratio means nothing there. The decision:
#   * the print has a RESTATEMENT (the next release's "previous" for that period): a typo
#     iff it is off the restatement AND off its own row's previous or consensus - a correct
#     small print is restated at about itself and passes; a typo in the restatement alone
#     can't condemn it;
#   * no restatement yet (the newest print): a typo iff off BOTH previous and consensus.
# Calibrated the same day on 1,256 cross-checked prints; two earlier rules (revision-size,
# then ratio-only) flagged 17 and 14 correct prints.
SUSPECT_SHIFT_LOG10 = 0.8  # ~6.3x
SUSPECT_SMALL = 0.2


def _release_rows(rows: pd.DataFrame, pattern: str, frequency: str) -> pd.DataFrame:
    mine = rows[rows["report"].map(lambda r: matches(r, pattern))].copy()
    mine["p"] = [period_of(lbl, day, frequency) for lbl, day in zip(mine["period"], mine["timestamp"])]
    return mine.dropna(subset=["p"]).sort_values("timestamp").reset_index(drop=True)


def suspect_prints(rows: pd.DataFrame, pattern: str, *, frequency: str = "M", staged: bool = False) -> pd.DataFrame:
    """Calendar actuals of one release judged TYPOS (rule above): ``timestamp, period,
    actual, restated, shifted_vs``. ``staged`` (flash/final releases): a period's
    restatement is the next row of the same period (the final's "previous" = the flash)
    or of the next period (the next flash's "previous" = this final)."""
    cols = ["timestamp", "period", "actual", "restated", "shifted_vs"]
    mine = _release_rows(rows, pattern, frequency)
    prints = mine[mine["actual"].notna() & (mine["actual"] != 0)]
    if prints.empty:
        return pd.DataFrame(columns=cols)
    small = SUSPECT_SMALL * prints["actual"].abs().median()
    step = _PRIOR[frequency]

    def off(a: float, ref: float) -> bool | None:
        if not np.isfinite(ref) or ref == 0 or (abs(a) < small and abs(ref) < small):
            return None
        return abs(np.log10(abs(a) / abs(ref))) >= SUSPECT_SHIFT_LOG10

    out = []
    for r in prints.itertuples():
        later = mine[(mine["timestamp"] > r.timestamp) & mine["previous"].notna()]
        later = later[later["p"].isin([r.p, r.p + step] if staged else [r.p + step])]
        restated = later["previous"].iloc[0] if len(later) else np.nan
        vs = {"restated": off(r.actual, restated), "previous": off(r.actual, r.previous),
              "consensus": off(r.actual, r.forecast)}
        if np.isfinite(restated):  # restated (even if too small to compare): it decides
            typo = bool(vs["restated"]) and bool(vs["previous"] or vs["consensus"])
        else:
            typo = bool(vs["previous"]) and bool(vs["consensus"])
        if typo:
            out.append((r.timestamp, r.p, r.actual, restated, ",".join(k for k, v in vs.items() if v)))
    return pd.DataFrame(out, columns=cols)


def release_vintages(rows: pd.DataFrame, pattern: str, *, use_previous: bool = True, frequency: str = "M",
                     scale: float = 1.0, drop_suspect: bool = True) -> pd.DataFrame:
    """One release's vintages from calendar rows (``frequency``: its period labels;
    ``scale``: calendar units -> the release's units, e.g. 1e-3 for payrolls in persons
    -> thousands): ``realtime_start, date, value``
    (the fetchers' shape, infra.api.fred_client.VINTAGE_COLUMNS).

    * an ``actual`` is that period's value published on the release day (a flash and a
      final are each an actual, on their own days - preliminary, then final);
    * a ``previous`` (``use_previous``) is the PRIOR period's value as it stood on the
      release day - a revision when it differs (the Conference Board's, NAR's) and the
      first value we know at all when that period's own week was never archived. Dated
      on the release day, never earlier: conservative (no look-ahead), at worst late.
      OFF for flash/final releases: there the final row's "previous" is the SAME
      period's flash, not the prior period (verified 2026-10-01: "S&P final U.S.
      manufacturing PMI Aug. 47.9, previous 48.0" - 48.0 was August's flash, July's
      final was 49.6), and the 2026 layout no longer says which row is the final.

    ``drop_suspect``: actuals judged typos (``suspect_prints``) are left out - the period
    then gets its value from the next release's "previous", dated that later day.

    Duplicates and unchanged repeats are left to ``infra.processing.releases.
    drop_unchanged`` downstream, like any fetcher's output."""
    cols = ["realtime_start", "date", "value"]
    if rows.empty:
        return pd.DataFrame({c: pd.Series(dtype="datetime64[ms]" if c != "value" else "float64") for c in cols})
    mine = rows[rows["report"].map(lambda r: matches(r, pattern))]
    bad = set()
    if drop_suspect:
        sus = suspect_prints(rows, pattern, frequency=frequency, staged=not use_previous)
        bad = set(zip(sus["timestamp"], sus["period"]))
    out = []
    for r in mine.itertuples():
        period = period_of(r.period, r.timestamp, frequency)
        if period is None:
            continue
        if np.isfinite(r.actual) and (r.timestamp, period) not in bad:
            out.append((r.timestamp, period, r.actual * scale))
        if use_previous and np.isfinite(r.previous):
            out.append((r.timestamp, period - _PRIOR[frequency], r.previous * scale))
    df = pd.DataFrame(out, columns=cols)
    if df.empty:
        return release_vintages(rows.iloc[0:0], pattern, frequency=frequency)
    df["realtime_start"] = df["realtime_start"].astype("datetime64[ms]")
    df["date"] = df["date"].astype("datetime64[ms]")
    # one value per (date, publication day): an actual beats a "previous" for the same
    # period on the same day (a flash and its own period's final never share a day)
    df = df.drop_duplicates(subset=["realtime_start", "date"], keep="first")
    return df.sort_values(["date", "realtime_start"]).reset_index(drop=True)
