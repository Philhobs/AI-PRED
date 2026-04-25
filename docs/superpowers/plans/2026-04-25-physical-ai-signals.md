# Physical AI Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 21 new feature columns (`FEATURE_COLS` 88 → 109) capturing macro demand (4 FRED series), labor demand (BLS JOLTS NAICS 333), and innovation pace (6 USPTO CPC class buckets) for the robotics + autonomy + AI-vision growth thesis.

**Architecture:** New `robotics_signals_ingestion.py` + new `physical_ai_features.py` module. Extend existing `bls_jolts_ingestion.py` (add NAICS 333 series) and `uspto_ingestion.py` (add `--physical-ai` mode for 6 CPC buckets). All features apply uniformly to all 149 tickers (model decides per-ticker weight).

**Tech Stack:** Python 3.11+, Polars, DuckDB, PyArrow, requests, pytest

**Spec:** [docs/superpowers/specs/2026-04-25-physical-ai-signals.md](../specs/2026-04-25-physical-ai-signals.md)

---

### Task 1: New `robotics_signals_ingestion.py` for 4 FRED series

**Files:**
- Create: `ingestion/robotics_signals_ingestion.py`
- Create: `tests/test_robotics_signals_ingestion.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_robotics_signals_ingestion.py`:

```python
"""Tests for FRED-based robotics macro signals ingestion."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl


def _mock_fred_response(observations: list[dict]) -> MagicMock:
    """Build a fake FRED API response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"observations": observations}
    resp.raise_for_status.return_value = None
    return resp


def test_fred_series_constant_has_four_entries():
    from ingestion.robotics_signals_ingestion import _FRED_SERIES
    assert set(_FRED_SERIES) == {"NEWORDER", "NAPM", "IPG3331S", "WPU114"}


def test_fetch_fred_series_schema():
    """fetch_fred_series returns DataFrame with date (Date) and value (Float64)."""
    from ingestion.robotics_signals_ingestion import fetch_fred_series

    obs = [
        {"date": "2025-01-01", "value": "100.0"},
        {"date": "2025-02-01", "value": "101.5"},
    ]
    with patch("ingestion.robotics_signals_ingestion.requests.get",
               return_value=_mock_fred_response(obs)):
        df = fetch_fred_series("NEWORDER")

    assert df.columns == ["date", "value"]
    assert df["date"].dtype == pl.Date
    assert df["value"].dtype == pl.Float64
    assert len(df) == 2
    assert df["date"][0] == date(2025, 1, 1)
    assert df["value"][1] == 101.5


def test_fetch_fred_series_handles_missing_dot():
    """FRED uses '.' for missing observations — must convert to null."""
    from ingestion.robotics_signals_ingestion import fetch_fred_series

    obs = [
        {"date": "2025-01-01", "value": "100.0"},
        {"date": "2025-02-01", "value": "."},
        {"date": "2025-03-01", "value": "102.0"},
    ]
    with patch("ingestion.robotics_signals_ingestion.requests.get",
               return_value=_mock_fred_response(obs)):
        df = fetch_fred_series("NEWORDER")

    assert df["value"][1] is None
    assert df["value"][0] == 100.0
    assert df["value"][2] == 102.0


def test_fetch_fred_series_failure_returns_empty():
    """A 5xx or network exception returns an empty DataFrame, no exception raised."""
    from ingestion.robotics_signals_ingestion import fetch_fred_series

    with patch("ingestion.robotics_signals_ingestion.requests.get",
               side_effect=Exception("boom")):
        df = fetch_fred_series("NEWORDER")

    assert df.is_empty()
    assert df.columns == ["date", "value"]
    assert df["date"].dtype == pl.Date
    assert df["value"].dtype == pl.Float64


def test_save_robotics_signals_writes_parquet(tmp_path: Path):
    """save_robotics_signals writes one parquet per series with snappy compression."""
    from ingestion.robotics_signals_ingestion import save_robotics_signals

    series_dfs = {
        "NEWORDER": pl.DataFrame({
            "date": [date(2025, 1, 1)],
            "value": [100.0],
        }, schema={"date": pl.Date, "value": pl.Float64}),
        "NAPM": pl.DataFrame({
            "date": [date(2025, 1, 1)],
            "value": [50.0],
        }, schema={"date": pl.Date, "value": pl.Float64}),
    }

    save_robotics_signals(tmp_path, series_dfs)

    for series_id in series_dfs:
        path = tmp_path / f"{series_id}.parquet"
        assert path.exists()
        loaded = pl.read_parquet(path)
        assert loaded["date"].dtype == pl.Date
        assert loaded["value"].dtype == pl.Float64


def test_save_robotics_signals_skips_empty(tmp_path: Path):
    """save_robotics_signals does not write a parquet for an empty DataFrame."""
    from ingestion.robotics_signals_ingestion import save_robotics_signals

    save_robotics_signals(tmp_path, {
        "NEWORDER": pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64}),
    })
    assert not (tmp_path / "NEWORDER.parquet").exists()
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `pytest tests/test_robotics_signals_ingestion.py -v`
Expected: 6 fails (`ImportError: cannot import name ...`).

- [ ] **Step 3: Create `ingestion/robotics_signals_ingestion.py`**

```python
"""FRED-based robotics macro signals ingestion.

Fetches 4 FRED series tracking the macro demand backdrop for industrial
automation and physical-AI growth:
  NEWORDER  — Manufacturers' New Orders: Nondefense Capital Goods Ex Aircraft
  NAPM      — ISM Manufacturing PMI (headline)
  IPG3331S  — Industrial Production: Industrial Machinery
  WPU114    — PPI: Industrial Machinery

Output: data/raw/robotics_signals/{series_id}.parquet
Schema: date (Date), value (Float64).

Free, no API key required (FRED_API_KEY env var increases rate limit).
FRED returns '.' for missing observations — converted to null.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_SERIES: tuple[str, ...] = ("NEWORDER", "NAPM", "IPG3331S", "WPU114")
_NAPM_FALLBACK = "USAPMI"  # if NAPM is unreachable, try this alternative

_SCHEMA = {"date": pl.Date, "value": pl.Float64}
_EMPTY_DF = pl.DataFrame(schema=_SCHEMA)


def fetch_fred_series(series_id: str, observation_start: str = "2010-01-01") -> pl.DataFrame:
    """Fetch a single FRED series. Returns empty DataFrame on failure."""
    api_key = os.getenv("FRED_API_KEY", "")
    params = {
        "series_id": series_id,
        "file_type": "json",
        "observation_start": observation_start,
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(_FRED_BASE, params=params, timeout=30)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
    except Exception as exc:  # noqa: BLE001 — fail soft per spec
        _LOG.warning("[FRED] %s: fetch failed (%s); returning empty", series_id, exc)
        return _EMPTY_DF.clone()

    if not observations:
        return _EMPTY_DF.clone()

    rows = []
    for obs in observations:
        try:
            d = date.fromisoformat(obs["date"])
        except (KeyError, ValueError):
            continue
        raw_val = obs.get("value", ".")
        val: float | None
        if raw_val == "." or raw_val is None:
            val = None
        else:
            try:
                val = float(raw_val)
            except ValueError:
                val = None
        rows.append({"date": d, "value": val})

    if not rows:
        return _EMPTY_DF.clone()
    return pl.DataFrame(rows, schema=_SCHEMA)


def save_robotics_signals(out_dir: Path, series_dfs: dict[str, pl.DataFrame]) -> None:
    """Write one snappy-compressed parquet per non-empty series."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for series_id, df in series_dfs.items():
        if df.is_empty():
            _LOG.warning("[FRED] %s: empty DataFrame — skipping write", series_id)
            continue
        df.write_parquet(out_dir / f"{series_id}.parquet", compression="snappy")


def fetch_all() -> dict[str, pl.DataFrame]:
    """Fetch all 4 FRED series (with NAPM fallback to USAPMI). Sleep 1s between calls."""
    out: dict[str, pl.DataFrame] = {}
    for series_id in _FRED_SERIES:
        df = fetch_fred_series(series_id)
        if df.is_empty() and series_id == "NAPM":
            _LOG.warning("[FRED] NAPM unreachable — trying fallback %s", _NAPM_FALLBACK)
            df = fetch_fred_series(_NAPM_FALLBACK)
        out[series_id] = df
        time.sleep(1)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ROOT = Path(__file__).parent.parent
    out_dir = _ROOT / "data" / "raw" / "robotics_signals"
    _LOG.info("Fetching %d FRED series...", len(_FRED_SERIES))
    rates = fetch_all()
    save_robotics_signals(out_dir, rates)
    for sid, df in rates.items():
        _LOG.info("  %s: %d rows", sid, len(df))
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/test_robotics_signals_ingestion.py -v`
Expected: 6 pass.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -m 'not integration' -q`
Expected: ~441 pass (435 baseline + 6 new), 0 failures.

- [ ] **Step 6: Commit**

```bash
git add ingestion/robotics_signals_ingestion.py tests/test_robotics_signals_ingestion.py
git commit -m "feat: add robotics_signals_ingestion.py for 4 FRED series"
```

---

### Task 2: Extend BLS JOLTS for NAICS 333 (Machinery Manufacturing)

**Files:**
- Modify: `ingestion/bls_jolts_ingestion.py`
- Modify: `tests/test_bls_jolts_ingestion.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_bls_jolts_ingestion.py`:

```python
def test_series_ids_include_naics_333():
    """_SERIES_IDS must contain both NAICS 51 (Information) and NAICS 333 (Machinery)."""
    from ingestion.bls_jolts_ingestion import _SERIES_IDS
    assert set(_SERIES_IDS) == {
        "JTS510000000000000JOL",
        "JTS333000000000000JOL",
    }


def test_fetch_jolts_returns_rows_for_both_series():
    """fetch_jolts returns rows for both NAICS 51 and 333 when both come back from BLS."""
    from unittest.mock import patch
    from ingestion.bls_jolts_ingestion import fetch_jolts

    fake_response = {
        "Results": {
            "series": [
                {
                    "seriesID": "JTS510000000000000JOL",
                    "data": [
                        {"year": "2025", "period": "M01", "value": "100.0"},
                    ],
                },
                {
                    "seriesID": "JTS333000000000000JOL",
                    "data": [
                        {"year": "2025", "period": "M01", "value": "50.0"},
                    ],
                },
            ]
        }
    }
    fake_resp = type("R", (), {
        "json": lambda self: fake_response,
        "raise_for_status": lambda self: None,
    })()

    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=fake_resp):
        df = fetch_jolts("2025-02-15")

    assert df["series_id"].to_list().count("JTS510000000000000JOL") == 1
    assert df["series_id"].to_list().count("JTS333000000000000JOL") == 1
```

- [ ] **Step 2: Run new tests — confirm they fail**

Run: `pytest tests/test_bls_jolts_ingestion.py::test_series_ids_include_naics_333 tests/test_bls_jolts_ingestion.py::test_fetch_jolts_returns_rows_for_both_series -v`
Expected: 2 fails (no `_SERIES_IDS` constant; existing module only iterates one series).

- [ ] **Step 3: Modify `ingestion/bls_jolts_ingestion.py` — replace single `_SERIES_ID` with `_SERIES_IDS` list**

Find the line:

```python
_SERIES_ID = "JTS510000000000000JOL"
```

Replace with:

```python
_SERIES_IDS: tuple[str, ...] = (
    "JTS510000000000000JOL",   # NAICS 51 — Information sector
    "JTS333000000000000JOL",   # NAICS 333 — Machinery Manufacturing (robotics-pillar adoption proxy)
)
```

- [ ] **Step 4: Update the payload in `fetch_jolts` to send both series IDs**

Find:

```python
payload: dict = {
    "seriesid": [_SERIES_ID],
    "startyear": str(today.year - 1),
    "endyear": str(today.year),
}
```

Replace with:

```python
payload: dict = {
    "seriesid": list(_SERIES_IDS),
    "startyear": str(today.year - 1),
    "endyear": str(today.year),
}
```

- [ ] **Step 5: Update the parsing loop to iterate ALL series, not just the first**

Find the block beginning `series_list = data.get("Results", {}).get("series", [])` and iterate every series. The existing code uses `series_list[0]` only. Replace the result-extraction block with a loop:

```python
series_list = data.get("Results", {}).get("series", [])
if not series_list:
    return pl.DataFrame(schema=_SCHEMA)

rows = []
for series in series_list:
    sid = series.get("seriesID", "")
    for entry in series.get("data", []):
        period = entry.get("period", "")
        if period not in _MONTHLY_PERIODS:
            continue
        try:
            year = int(entry["year"])
            month = int(period[1:])
            value = float(entry["value"])
        except (KeyError, ValueError):
            continue
        rows.append({
            "date": datetime.date.fromisoformat(date_str),
            "series_id": sid,
            "year": year,
            "period": period,
            "value": value,
        })

if not rows:
    return pl.DataFrame(schema=_SCHEMA)
return pl.DataFrame(rows, schema=_SCHEMA)
```

(If the existing parsing logic is structured differently — e.g., a comprehension — preserve its semantics but make it iterate over `series_list` instead of `series_list[0]`. The rest of the surrounding code, including `_same_month`, the staleness guard, save logic, and the `__main__` block, does not change.)

- [ ] **Step 6: Update module docstring**

Find the existing docstring (top of the file). Replace it with:

```python
"""BLS JOLTS sector job openings ingestion — Information (NAICS 51) + Machinery (NAICS 333).

Fetches both sectors from BLS API v2:
  JTS510000000000000JOL  — Information (NAICS 51)
  JTS333000000000000JOL  — Machinery Manufacturing (NAICS 333)

Output: data/raw/bls_jolts/date=YYYY-MM-DD/openings.parquet (one row per series_id × month)

Fetches current year + prior year (BLS API uses full calendar years, not rolling windows).
Feature module filters rows by period_date <= query_date and series_id at join time.

Staleness guard: skips re-download if existing snapshot is from the same calendar month
(BLS JOLTS publishes monthly data with ~6-week publication lag).
"""
```

- [ ] **Step 7: Run all BLS tests — confirm pass**

Run: `pytest tests/test_bls_jolts_ingestion.py -v`
Expected: all pass (existing + 2 new).

- [ ] **Step 8: Run full suite — confirm green**

Run: `pytest tests/ -m 'not integration' -q`
Expected: 0 failures. (The existing `processing/labor_features.py` filters its JOLTS query by series_id implicitly because the existing data is single-series — verify it still gets the NAICS 51 rows correctly. It uses `MAKE_DATE(year, ...)` and ROW_NUMBER, both of which still work; but if `labor_features.py` doesn't filter by series_id, the new NAICS 333 rows would pollute its `tech_job_openings_index`. Check this in step 9.)

- [ ] **Step 9: Verify `labor_features.py` filters by series_id**

Run: `grep -n series_id processing/labor_features.py`

If the SQL in `_load_jolts` or `join_labor_features` does NOT filter by `series_id = 'JTS510000000000000JOL'`, add the filter so existing labor features keep using only NAICS 51:

In the SQL where the JOLTS subquery is built, add `WHERE series_id = 'JTS510000000000000JOL'` to the `jolts_dated` CTE or wherever the JOLTS table is read. Then re-run `pytest tests/test_labor_features.py -v` to confirm.

- [ ] **Step 10: Commit**

```bash
git add ingestion/bls_jolts_ingestion.py tests/test_bls_jolts_ingestion.py
# add processing/labor_features.py only if Step 9 required edits there:
git add processing/labor_features.py 2>/dev/null || true
git commit -m "feat: extend BLS JOLTS to NAICS 333 machinery manufacturing"
```

---

### Task 3: Extend USPTO ingestion to a "physical-AI" mode for 6 CPC class buckets

**Files:**
- Modify: `ingestion/uspto_ingestion.py`
- Modify: `tests/test_uspto_ingestion.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_uspto_ingestion.py`:

```python
def test_physical_ai_buckets_definition():
    """_PHYSICAL_AI_BUCKETS maps each bucket to its CPC class prefixes."""
    from ingestion.uspto_ingestion import _PHYSICAL_AI_BUCKETS
    assert set(_PHYSICAL_AI_BUCKETS) == {
        "B25J", "B64", "B60W", "G05D1", "G05B19", "G06V",
    }
    assert set(_PHYSICAL_AI_BUCKETS["B64"]) == {"B64C", "B64U"}
    assert _PHYSICAL_AI_BUCKETS["B25J"] == ("B25J",)


def test_physical_ai_quarter_end_for_filing_date():
    """_quarter_end maps any filing_date to the last day of its calendar quarter."""
    import datetime
    from ingestion.uspto_ingestion import _quarter_end
    assert _quarter_end(datetime.date(2025, 1, 15)) == datetime.date(2025, 3, 31)
    assert _quarter_end(datetime.date(2025, 4, 1))  == datetime.date(2025, 6, 30)
    assert _quarter_end(datetime.date(2025, 7, 31)) == datetime.date(2025, 9, 30)
    assert _quarter_end(datetime.date(2025, 12, 31)) == datetime.date(2025, 12, 31)


def test_aggregate_physical_ai_groups_by_quarter_and_bucket():
    """_aggregate_physical_ai counts filings per (quarter_end, bucket)."""
    import datetime
    import polars as pl
    from ingestion.uspto_ingestion import _aggregate_physical_ai

    raw = pl.DataFrame({
        "filing_date": [
            datetime.date(2025, 1, 15),  # Q1, B25J
            datetime.date(2025, 2, 1),   # Q1, B25J
            datetime.date(2025, 4, 5),   # Q2, B64C → B64
            datetime.date(2025, 4, 6),   # Q2, B64U → B64
            datetime.date(2025, 5, 1),   # Q2, G06V
        ],
        "cpc_group": ["B25J9", "B25J11", "B64C39", "B64U10", "G06V20"],
    })

    agg = _aggregate_physical_ai(raw)
    rows = {(r["quarter_end"], r["cpc_class"]): r["filing_count"] for r in agg.to_dicts()}
    assert rows[(datetime.date(2025, 3, 31), "B25J")] == 2
    assert rows[(datetime.date(2025, 6, 30), "B64")] == 2
    assert rows[(datetime.date(2025, 6, 30), "G06V")] == 1


def test_aggregate_physical_ai_b64_combines_subclasses():
    """B64 bucket sums filings that match B64C* OR B64U*."""
    import datetime
    import polars as pl
    from ingestion.uspto_ingestion import _aggregate_physical_ai

    raw = pl.DataFrame({
        "filing_date": [datetime.date(2025, 1, 1)] * 3,
        "cpc_group": ["B64C39", "B64U10", "B64U99"],
    })
    agg = _aggregate_physical_ai(raw)
    assert agg.filter(pl.col("cpc_class") == "B64")["filing_count"][0] == 3


def test_aggregate_physical_ai_drops_non_target_classes():
    """Filings whose CPC group doesn't match any of the 6 buckets are dropped."""
    import datetime
    import polars as pl
    from ingestion.uspto_ingestion import _aggregate_physical_ai

    raw = pl.DataFrame({
        "filing_date": [datetime.date(2025, 1, 1)] * 3,
        "cpc_group": ["B25J9", "H01L21", "G06F8"],   # only B25J9 matches
    })
    agg = _aggregate_physical_ai(raw)
    total = agg["filing_count"].sum()
    assert total == 1
```

- [ ] **Step 2: Run new tests — confirm they fail**

Run: `pytest tests/test_uspto_ingestion.py -v -k "physical_ai or quarter_end"`
Expected: 5 fails (`ImportError` for `_PHYSICAL_AI_BUCKETS`, `_quarter_end`, `_aggregate_physical_ai`).

- [ ] **Step 3: Add the bucket definition + helpers to `ingestion/uspto_ingestion.py`**

Append after the existing `_CPC_CODES` constant:

```python
# Physical-AI mode: map bucket name → CPC class prefixes that count toward it.
# A patent's cpc_group_id must START WITH any of the prefixes to be counted in that bucket.
_PHYSICAL_AI_BUCKETS: dict[str, tuple[str, ...]] = {
    "B25J":   ("B25J",),
    "B64":    ("B64C", "B64U"),
    "B60W":   ("B60W",),
    "G05D1":  ("G05D1",),
    "G05B19": ("G05B19",),
    "G06V":   ("G06V",),
}

_PHYSICAL_AI_AGG_SCHEMA = {
    "quarter_end": pl.Date,
    "cpc_class":   pl.Utf8,
    "filing_count": pl.Int64,
}
```

Add the helper functions:

```python
def _quarter_end(d: "datetime.date") -> "datetime.date":
    """Return the last calendar day of d's quarter (Mar 31 / Jun 30 / Sep 30 / Dec 31)."""
    quarter = (d.month - 1) // 3 + 1
    last_month = quarter * 3
    if last_month == 3:
        return datetime.date(d.year, 3, 31)
    if last_month == 6:
        return datetime.date(d.year, 6, 30)
    if last_month == 9:
        return datetime.date(d.year, 9, 30)
    return datetime.date(d.year, 12, 31)


def _bucket_for_cpc(cpc_group: str) -> str | None:
    """Return the bucket name for a cpc_group_id, or None if it doesn't match any bucket."""
    for bucket, prefixes in _PHYSICAL_AI_BUCKETS.items():
        if any(cpc_group.startswith(p) for p in prefixes):
            return bucket
    return None


def _aggregate_physical_ai(raw: pl.DataFrame) -> pl.DataFrame:
    """Aggregate raw filings (filing_date, cpc_group) → (quarter_end, cpc_class, filing_count)."""
    if raw.is_empty():
        return pl.DataFrame(schema=_PHYSICAL_AI_AGG_SCHEMA)

    df = raw.with_columns([
        pl.col("filing_date").map_elements(_quarter_end, return_dtype=pl.Date).alias("quarter_end"),
        pl.col("cpc_group").map_elements(_bucket_for_cpc, return_dtype=pl.Utf8).alias("cpc_class"),
    ])
    df = df.filter(pl.col("cpc_class").is_not_null())
    if df.is_empty():
        return pl.DataFrame(schema=_PHYSICAL_AI_AGG_SCHEMA)
    return (
        df.group_by(["quarter_end", "cpc_class"])
          .agg(pl.len().alias("filing_count").cast(pl.Int64))
          .sort(["quarter_end", "cpc_class"])
    )
```

Add the public fetch function:

```python
def fetch_physical_ai_filings(date_str: str) -> pl.DataFrame:
    """Fetch all filings in the 365-day window ending date_str whose cpc_group matches
    any physical-AI bucket prefix. Returns aggregated (quarter_end, cpc_class, filing_count)."""
    start = _lookback_start(date_str)
    all_prefixes = sorted({p for prefixes in _PHYSICAL_AI_BUCKETS.values() for p in prefixes})

    records: list[dict] = []
    for prefix in all_prefixes:
        page = 1
        while page <= _MAX_PAGES:
            payload = {
                "q": {"_and": [
                    {"_gte": {"app_date": start}},
                    {"_lte": {"app_date": date_str}},
                    {"_begins": {"cpc_group_id": prefix}},
                ]},
                "f": ["app_id", "cpc_group_id", "app_date"],
                "o": {"per_page": _PER_PAGE, "page": page},
            }
            try:
                resp = requests.post(_APPS_URL, json=payload, timeout=30)
                resp.raise_for_status()
                page_data = resp.json().get("applications", [])
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("[USPTO] physical_ai prefix=%s page=%d: %s", prefix, page, exc)
                break
            if not page_data:
                break
            for entry in page_data:
                try:
                    fd = datetime.date.fromisoformat(entry["app_date"])
                except (KeyError, ValueError):
                    continue
                cpc = entry.get("cpc_group_id", "")
                if not cpc:
                    continue
                records.append({"filing_date": fd, "cpc_group": cpc})
            page += 1
            time.sleep(_SLEEP_BETWEEN_PAGES)

    raw = pl.DataFrame(records, schema={"filing_date": pl.Date, "cpc_group": pl.Utf8})
    return _aggregate_physical_ai(raw)


def save_physical_ai_filings(out_dir: Path, agg: pl.DataFrame) -> None:
    """Write one parquet per cpc_class bucket. Schema: (quarter_end, cpc_class, filing_count)."""
    if agg.is_empty():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for bucket in _PHYSICAL_AI_BUCKETS:
        sub = agg.filter(pl.col("cpc_class") == bucket)
        if sub.is_empty():
            continue
        bucket_dir = out_dir / f"cpc_class={bucket}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(bucket_dir / "filings.parquet", compression="snappy")
```

- [ ] **Step 4: Wire physical-AI mode into `__main__`**

In the `if __name__ == "__main__":` block (or the script's invocation entry point), append (do NOT replace) the existing apps/grants flow. Add immediately before the closing of the `__main__` block:

```python
    today = datetime.date.today().isoformat()
    physical_ai_dir = _ROOT / "data" / "raw" / "uspto" / "physical_ai"
    _LOG.info("Fetching physical-AI patent filings (6 CPC buckets)...")
    agg = fetch_physical_ai_filings(today)
    save_physical_ai_filings(physical_ai_dir, agg)
    _LOG.info("[USPTO] physical_ai: %d (quarter, bucket) rows written", len(agg))
```

(`_ROOT` already exists at the start of `__main__` per the existing pattern; if it doesn't, add `_ROOT = Path(__file__).parent.parent`.)

- [ ] **Step 5: Run new tests — confirm pass**

Run: `pytest tests/test_uspto_ingestion.py -v`
Expected: all pass (existing + 5 new).

- [ ] **Step 6: Run full suite**

Run: `pytest tests/ -m 'not integration' -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add ingestion/uspto_ingestion.py tests/test_uspto_ingestion.py
git commit -m "feat: extend USPTO patents ingestion to 6 physical-AI CPC classes"
```

---

### Task 4: New `physical_ai_features.py` module + `models/train.py` wiring

**Files:**
- Create: `processing/physical_ai_features.py`
- Create: `tests/test_physical_ai_features.py`
- Modify: `models/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Add failing tests for the feature module**

Create `tests/test_physical_ai_features.py`:

```python
"""Tests for physical AI feature engineering and join."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _write_fred_parquet(out_dir: Path, series_id: str, rows: list[tuple[date, float | None]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {"date": [r[0] for r in rows], "value": [r[1] for r in rows]},
        schema={"date": pl.Date, "value": pl.Float64},
    )
    df.write_parquet(out_dir / f"{series_id}.parquet", compression="snappy")


def _write_jolts_parquet(out_dir: Path, rows: list[tuple[date, str, int, str, float]]) -> None:
    snapshot_dir = out_dir / "date=2025-04-01"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "date":      [r[0] for r in rows],
            "series_id": [r[1] for r in rows],
            "year":      [r[2] for r in rows],
            "period":    [r[3] for r in rows],
            "value":     [r[4] for r in rows],
        },
        schema={"date": pl.Date, "series_id": pl.Utf8, "year": pl.Int32,
                "period": pl.Utf8, "value": pl.Float64},
    )
    df.write_parquet(snapshot_dir / "openings.parquet", compression="snappy")


def _write_patent_parquet(out_dir: Path, bucket: str, rows: list[tuple[date, int]]) -> None:
    bucket_dir = out_dir / f"cpc_class={bucket}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "quarter_end": [r[0] for r in rows],
            "cpc_class":   [bucket] * len(rows),
            "filing_count": [r[1] for r in rows],
        },
        schema={"quarter_end": pl.Date, "cpc_class": pl.Utf8, "filing_count": pl.Int64},
    )
    df.write_parquet(bucket_dir / "filings.parquet", compression="snappy")


def test_physical_ai_feature_cols_count_is_21():
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    assert len(PHYSICAL_AI_FEATURE_COLS) == 21


def test_physical_ai_feature_cols_exact_names():
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    assert set(PHYSICAL_AI_FEATURE_COLS) == {
        "phys_ai_capgoods_orders_level",
        "phys_ai_capgoods_orders_yoy",
        "phys_ai_pmi_level",
        "phys_ai_machinery_prod_level",
        "phys_ai_machinery_prod_yoy",
        "phys_ai_machinery_ppi_level",
        "phys_ai_machinery_ppi_yoy",
        "phys_ai_machinery_jobs_level",
        "phys_ai_machinery_jobs_yoy",
        "phys_ai_patents_manipulators_count",
        "phys_ai_patents_manipulators_yoy",
        "phys_ai_patents_aerial_count",
        "phys_ai_patents_aerial_yoy",
        "phys_ai_patents_avs_count",
        "phys_ai_patents_avs_yoy",
        "phys_ai_patents_motion_count",
        "phys_ai_patents_motion_yoy",
        "phys_ai_patents_progcontrol_count",
        "phys_ai_patents_progcontrol_yoy",
        "phys_ai_patents_vision_count",
        "phys_ai_patents_vision_yoy",
    }


def test_yoy_handles_zero_baseline():
    """yoy with prior period value = 0 returns null (no division by zero)."""
    from processing.physical_ai_features import _yoy
    assert _yoy(current=10.0, prior=0.0) is None
    assert _yoy(current=10.0, prior=None) is None
    assert _yoy(current=None, prior=10.0) is None
    assert _yoy(current=110.0, prior=100.0) == pytest.approx(0.10, rel=1e-6)


def test_join_macro_features_forward_fills_within_60d(tmp_path: Path):
    """FRED level value within 60 days of query date propagates."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    _write_fred_parquet(fred_dir, "NEWORDER",
        [(date(2025, 1, 1), 100.0), (date(2025, 2, 1), 105.0)])
    _write_fred_parquet(fred_dir, "NAPM",
        [(date(2025, 2, 1), 52.0)])
    _write_fred_parquet(fred_dir, "IPG3331S",
        [(date(2025, 2, 1), 110.0)])
    _write_fred_parquet(fred_dir, "WPU114",
        [(date(2025, 2, 1), 250.0)])

    jolts_dir = tmp_path / "bls_jolts"
    _write_jolts_parquet(jolts_dir, [
        (date(2025, 4, 1), "JTS333000000000000JOL", 2025, "M01", 75.0),
    ])

    patents_dir = tmp_path / "uspto" / "physical_ai"
    for bucket in ["B25J", "B64", "B60W", "G05D1", "G05B19", "G06V"]:
        _write_patent_parquet(patents_dir, bucket,
            [(date(2024, 12, 31), 100), (date(2025, 3, 31), 120)])

    spine = pl.DataFrame({
        "ticker": ["NVDA", "AAPL"],
        "date":   [date(2025, 3, 15), date(2025, 3, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)

    assert "phys_ai_capgoods_orders_level" in out.columns
    nvda = out.filter(pl.col("ticker") == "NVDA").row(0, named=True)
    assert nvda["phys_ai_capgoods_orders_level"] == 105.0
    assert nvda["phys_ai_pmi_level"] == 52.0


def test_join_macro_features_null_beyond_tolerance(tmp_path: Path):
    """FRED level beyond 60 days returns null."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    _write_fred_parquet(fred_dir, "NEWORDER",
        [(date(2024, 12, 1), 100.0)])  # >60d before query
    _write_fred_parquet(fred_dir, "NAPM", [(date(2024, 12, 1), 50.0)])
    _write_fred_parquet(fred_dir, "IPG3331S", [(date(2024, 12, 1), 100.0)])
    _write_fred_parquet(fred_dir, "WPU114", [(date(2024, 12, 1), 250.0)])

    jolts_dir = tmp_path / "bls_jolts"
    patents_dir = tmp_path / "uspto" / "physical_ai"

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2025, 3, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert out["phys_ai_capgoods_orders_level"][0] is None


def test_join_patent_features_quarterly_tolerance(tmp_path: Path):
    """Patent count from 120 days ago propagates; beyond returns null."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    jolts_dir = tmp_path / "bls_jolts"
    patents_dir = tmp_path / "uspto" / "physical_ai"

    # Q4-2024 ends Dec 31, 2024 — 105 days before April 15, 2025 (within 120d)
    for bucket in ["B25J", "B64", "B60W", "G05D1", "G05B19", "G06V"]:
        _write_patent_parquet(patents_dir, bucket, [(date(2024, 12, 31), 50)])

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2025, 4, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert out["phys_ai_patents_manipulators_count"][0] == 50.0


def test_phys_ai_features_apply_to_all_tickers(tmp_path: Path):
    """Every ticker in the spine gets a row with all 21 columns present."""
    from processing.physical_ai_features import (
        join_physical_ai_features, PHYSICAL_AI_FEATURE_COLS,
    )

    fred_dir = tmp_path / "robotics_signals"
    jolts_dir = tmp_path / "bls_jolts"
    patents_dir = tmp_path / "uspto" / "physical_ai"

    spine = pl.DataFrame({
        "ticker": ["NVDA", "TSLA", "ROK", "1683.HK"],
        "date":   [date(2025, 3, 15)] * 4,
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert len(out) == 4
    for col in PHYSICAL_AI_FEATURE_COLS:
        assert col in out.columns, f"{col} missing"


def test_jolts_filter_uses_naics_333_only(tmp_path: Path):
    """Phys-AI labor features must read NAICS 333 rows only, not NAICS 51."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    patents_dir = tmp_path / "uspto" / "physical_ai"
    jolts_dir = tmp_path / "bls_jolts"
    _write_jolts_parquet(jolts_dir, [
        (date(2025, 4, 1), "JTS510000000000000JOL", 2025, "M02", 999.0),  # NAICS 51 — must be ignored
        (date(2025, 4, 1), "JTS333000000000000JOL", 2025, "M02", 75.0),   # NAICS 333 — must be picked up
        (date(2025, 4, 1), "JTS333000000000000JOL", 2024, "M02", 60.0),   # 1y prior for yoy
    ])

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2025, 3, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert out["phys_ai_machinery_jobs_level"][0] == 75.0
    # yoy = (75 - 60) / 60 = 0.25
    assert out["phys_ai_machinery_jobs_yoy"][0] == pytest.approx(0.25, rel=1e-6)
```

- [ ] **Step 2: Run new tests — confirm they fail**

Run: `pytest tests/test_physical_ai_features.py -v`
Expected: 8 fails (`ImportError`).

- [ ] **Step 3: Create `processing/physical_ai_features.py`**

```python
"""Physical-AI feature engineering — macro + labor + patents (21 features).

Features (PHYSICAL_AI_FEATURE_COLS):
  Macro (FRED, monthly, 60-day forward-fill tolerance):
    phys_ai_capgoods_orders_level / _yoy        (NEWORDER)
    phys_ai_pmi_level                            (NAPM)
    phys_ai_machinery_prod_level / _yoy          (IPG3331S)
    phys_ai_machinery_ppi_level / _yoy           (WPU114)
  Labor (BLS JOLTS NAICS 333, monthly, 60-day tolerance):
    phys_ai_machinery_jobs_level / _yoy
  Patents (USPTO physical-AI, quarterly, 120-day tolerance):
    phys_ai_patents_{manipulators|aerial|avs|motion|progcontrol|vision}_count / _yoy

All 21 features apply uniformly to every ticker (model decides per-ticker weight).
Public entry: join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl

_LOG = logging.getLogger(__name__)

# Public name list — imported by models/train.py
PHYSICAL_AI_FEATURE_COLS: list[str] = [
    "phys_ai_capgoods_orders_level",
    "phys_ai_capgoods_orders_yoy",
    "phys_ai_pmi_level",
    "phys_ai_machinery_prod_level",
    "phys_ai_machinery_prod_yoy",
    "phys_ai_machinery_ppi_level",
    "phys_ai_machinery_ppi_yoy",
    "phys_ai_machinery_jobs_level",
    "phys_ai_machinery_jobs_yoy",
    "phys_ai_patents_manipulators_count",
    "phys_ai_patents_manipulators_yoy",
    "phys_ai_patents_aerial_count",
    "phys_ai_patents_aerial_yoy",
    "phys_ai_patents_avs_count",
    "phys_ai_patents_avs_yoy",
    "phys_ai_patents_motion_count",
    "phys_ai_patents_motion_yoy",
    "phys_ai_patents_progcontrol_count",
    "phys_ai_patents_progcontrol_yoy",
    "phys_ai_patents_vision_count",
    "phys_ai_patents_vision_yoy",
]

# (FRED series id) → (level column name, yoy column name | None)
_FRED_COL_MAP: dict[str, tuple[str, Optional[str]]] = {
    "NEWORDER": ("phys_ai_capgoods_orders_level", "phys_ai_capgoods_orders_yoy"),
    "NAPM":     ("phys_ai_pmi_level",             None),
    "IPG3331S": ("phys_ai_machinery_prod_level",  "phys_ai_machinery_prod_yoy"),
    "WPU114":   ("phys_ai_machinery_ppi_level",   "phys_ai_machinery_ppi_yoy"),
}

# (CPC bucket) → (count column name, yoy column name)
_PATENT_COL_MAP: dict[str, tuple[str, str]] = {
    "B25J":   ("phys_ai_patents_manipulators_count",  "phys_ai_patents_manipulators_yoy"),
    "B64":    ("phys_ai_patents_aerial_count",        "phys_ai_patents_aerial_yoy"),
    "B60W":   ("phys_ai_patents_avs_count",           "phys_ai_patents_avs_yoy"),
    "G05D1":  ("phys_ai_patents_motion_count",        "phys_ai_patents_motion_yoy"),
    "G05B19": ("phys_ai_patents_progcontrol_count",   "phys_ai_patents_progcontrol_yoy"),
    "G06V":   ("phys_ai_patents_vision_count",        "phys_ai_patents_vision_yoy"),
}

_JOLTS_NAICS_333 = "JTS333000000000000JOL"
_FRED_TOLERANCE_DAYS = 60
_PATENT_TOLERANCE_DAYS = 120


def _yoy(current: float | None, prior: float | None) -> float | None:
    """Year-over-year ratio. None if either value is missing or prior is 0."""
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / prior


def _load_fred_series(fred_dir: Path, series_id: str) -> pl.DataFrame:
    path = fred_dir / f"{series_id}.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64})
    return pl.read_parquet(path).sort("date")


def _load_jolts(jolts_dir: Path) -> pl.DataFrame:
    """Load all BLS JOLTS snapshots, filter to NAICS 333. Each row: period_date + value."""
    if not jolts_dir.exists():
        return pl.DataFrame(schema={"period_date": pl.Date, "value": pl.Float64})
    files = sorted(jolts_dir.glob("date=*/openings.parquet"))
    if not files:
        return pl.DataFrame(schema={"period_date": pl.Date, "value": pl.Float64})
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = df.filter(pl.col("series_id") == _JOLTS_NAICS_333)
    df = df.with_columns(
        pl.date(pl.col("year"), pl.col("period").str.slice(1, 2).cast(pl.Int32), 1).alias("period_date")
    )
    df = df.unique(subset=["period_date"], keep="last").sort("period_date")
    return df.select(["period_date", "value"])


def _load_patent_bucket(patents_dir: Path, bucket: str) -> pl.DataFrame:
    path = patents_dir / f"cpc_class={bucket}" / "filings.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"quarter_end": pl.Date, "filing_count": pl.Int64})
    df = pl.read_parquet(path).select(["quarter_end", "filing_count"]).sort("quarter_end")
    return df


def _value_at(df: pl.DataFrame, query_date: date, value_col: str, date_col: str,
              tolerance_days: int) -> float | None:
    """Most recent observation in df where date_col <= query_date and within tolerance.
    Returns None if no row satisfies the constraint."""
    if df.is_empty():
        return None
    eligible = df.filter(pl.col(date_col) <= query_date)
    if eligible.is_empty():
        return None
    row = eligible.tail(1).row(0, named=True)
    if (query_date - row[date_col]).days > tolerance_days:
        return None
    return row[value_col]


def _value_one_year_prior(df: pl.DataFrame, query_date: date, value_col: str, date_col: str,
                          tolerance_days: int) -> float | None:
    """Most recent observation where date <= (query_date - 365 days), within tolerance."""
    from datetime import timedelta
    target = query_date - timedelta(days=365)
    return _value_at(df, target, value_col, date_col, tolerance_days)


def join_physical_ai_features(
    spine: pl.DataFrame,
    fred_dir: Path,
    jolts_dir: Path,
    patents_dir: Path,
) -> pl.DataFrame:
    """Join the 21 physical-AI features onto spine. Spine must have 'ticker' (Utf8) and 'date' (Date)."""
    fred_data: dict[str, pl.DataFrame] = {
        sid: _load_fred_series(fred_dir, sid) for sid in _FRED_COL_MAP
    }
    jolts = _load_jolts(jolts_dir)
    patent_data: dict[str, pl.DataFrame] = {
        bucket: _load_patent_bucket(patents_dir, bucket) for bucket in _PATENT_COL_MAP
    }

    # For each unique date, compute the 21 column values once, then join back to spine.
    unique_dates = spine.select("date").unique().sort("date")["date"].to_list()
    rows: list[dict] = []
    for d in unique_dates:
        row: dict = {"date": d}
        # FRED
        for series_id, (level_col, yoy_col) in _FRED_COL_MAP.items():
            df = fred_data[series_id]
            level = _value_at(df, d, "value", "date", _FRED_TOLERANCE_DAYS)
            row[level_col] = level
            if yoy_col is not None:
                prior = _value_one_year_prior(df, d, "value", "date", _FRED_TOLERANCE_DAYS)
                row[yoy_col] = _yoy(level, prior)
        # JOLTS NAICS 333
        jolts_level = _value_at(jolts, d, "value", "period_date", _FRED_TOLERANCE_DAYS)
        jolts_prior = _value_one_year_prior(jolts, d, "value", "period_date", _FRED_TOLERANCE_DAYS)
        row["phys_ai_machinery_jobs_level"] = jolts_level
        row["phys_ai_machinery_jobs_yoy"] = _yoy(jolts_level, jolts_prior)
        # Patents
        for bucket, (count_col, yoy_col) in _PATENT_COL_MAP.items():
            df = patent_data[bucket]
            count = _value_at(df, d, "filing_count", "quarter_end", _PATENT_TOLERANCE_DAYS)
            prior = _value_one_year_prior(df, d, "filing_count", "quarter_end", _PATENT_TOLERANCE_DAYS)
            row[count_col] = float(count) if count is not None else None
            row[yoy_col] = _yoy(
                float(count) if count is not None else None,
                float(prior) if prior is not None else None,
            )
        rows.append(row)

    schema = {"date": pl.Date}
    for col in PHYSICAL_AI_FEATURE_COLS:
        schema[col] = pl.Float64
    feature_df = pl.DataFrame(rows, schema=schema)

    return spine.join(feature_df, on="date", how="left")
```

- [ ] **Step 4: Run feature module tests — confirm pass**

Run: `pytest tests/test_physical_ai_features.py -v`
Expected: 8 pass.

- [ ] **Step 5: Wire into `models/train.py`**

In `models/train.py`, find the imports section (around line 41–46) and add (preserving import order):

```python
from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS, join_physical_ai_features
```

Find the `FEATURE_COLS = (` block (lines 106–118). Replace the closing of the tuple to append `PHYSICAL_AI_FEATURE_COLS`:

Find:

```python
    + CENSUS_TRADE_FEATURE_COLS    # 82 → 88 features total
)
```

Replace with:

```python
    + CENSUS_TRADE_FEATURE_COLS    # 82 → 88 features total
    + PHYSICAL_AI_FEATURE_COLS     # 88 → 109 features total
)
```

Then find the existing tier registry (`TIER_FEATURE_COLS`). Physical-AI features are monthly/quarterly cadence, so they belong in **medium + long** tiers but NOT short. Locate the dict literal where short/medium/long lists are constructed and append `PHYSICAL_AI_FEATURE_COLS` to medium and long, but NOT short. Show the full file path with grep first to find the exact lines:

Run: `grep -n "TIER_FEATURE_COLS" models/train.py`

Then within the medium and long list constructions, append `+ PHYSICAL_AI_FEATURE_COLS` (alongside the other monthly cols like `LABOR_FEATURE_COLS`). The exact diff depends on the existing layout — preserve the existing structure and just include `PHYSICAL_AI_FEATURE_COLS` in the medium and long tier expressions.

Wire the join into the spine assembly. Find the join_labor_features call (around line 351):

```python
df = join_labor_features(df, usajobs_dir, jolts_dir)
```

Append immediately after it:

```python
df = join_physical_ai_features(
    df,
    fred_dir=_ROOT / "data" / "raw" / "robotics_signals",
    jolts_dir=jolts_dir,
    patents_dir=_ROOT / "data" / "raw" / "uspto" / "physical_ai",
)
```

(If `_ROOT` isn't already in scope at that point in the function, derive it from existing path variables that are — e.g., `jolts_dir.parent.parent` is `data/raw/`, so `jolts_dir.parent.parent.parent` is the project root. Use whatever path-rooting the surrounding function already uses.)

- [ ] **Step 6: Update test_train.py FEATURE_COLS count assertions**

In `tests/test_train.py`, find the four assertions of `len(FEATURE_COLS) == 88` (lines around 505, 556, 605, 658) and replace each with:

```python
    assert len(FEATURE_COLS) == 109
```

Update the message-string variants similarly:

```python
    assert len(FEATURE_COLS) == 109, f"Expected 109 features, got {len(FEATURE_COLS)}"
```

Then append a new test:

```python
def test_feature_cols_includes_physical_ai():
    """FEATURE_COLS must contain all 21 PHYSICAL_AI_FEATURE_COLS and total must be 109."""
    from models.train import FEATURE_COLS
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    assert len(PHYSICAL_AI_FEATURE_COLS) == 21
    for col in PHYSICAL_AI_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 109, f"Expected 109 features, got {len(FEATURE_COLS)}"


def test_physical_ai_cols_absent_from_short_tier():
    """Physical-AI cols must not appear in short tier — monthly/quarterly cadence too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in PHYSICAL_AI_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_physical_ai_cols_in_medium_tier():
    """Physical-AI cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    medium = set(TIER_FEATURE_COLS["medium"])
    for col in PHYSICAL_AI_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_physical_ai_cols_in_long_tier():
    """Physical-AI cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    long_cols = set(TIER_FEATURE_COLS["long"])
    for col in PHYSICAL_AI_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"
```

- [ ] **Step 7: Run tests — verify pass**

Run: `pytest tests/test_train.py tests/test_physical_ai_features.py -v`
Expected: all pass (existing test_train + 4 new test_train + 8 new physical_ai_features).

- [ ] **Step 8: Run full suite — confirm green**

Run: `pytest tests/ -m 'not integration' -q`
Expected: 0 failures, 0 errors. Test count ~457 (435 baseline + 6 from Task 1 + 2 from Task 2 + 5 from Task 3 + 8 + 4 from Task 4 = ~460, ±2).

- [ ] **Step 9: Commit**

```bash
git add processing/physical_ai_features.py tests/test_physical_ai_features.py models/train.py tests/test_train.py
git commit -m "feat: add physical_ai_features module, FEATURE_COLS 88→109"
```

---

### Task 5: Wire into `tools/run_refresh.sh` + final acceptance gate

**Files:**
- Modify: `tools/run_refresh.sh`

- [ ] **Step 1: Insert `robotics_signals_ingestion.py` step before BLS JOLTS**

In `tools/run_refresh.sh`, find:

```bash
echo "=== 15/16  BLS JOLTS tech sector job openings ==="
python ingestion/bls_jolts_ingestion.py
```

Insert immediately BEFORE the `15/16` block:

```bash
echo ""
echo "=== 15/17  Robotics macro signals (FRED) ==="
python ingestion/robotics_signals_ingestion.py
```

Then update the existing step numbering: `15/16 BLS JOLTS` → `16/17`, `16/16 Census trade` → `17/17`. Final structure:

```bash
echo "=== 14/17  USAJOBS federal AI/ML job postings ==="
python ingestion/usajobs_ingestion.py

echo ""
echo "=== 15/17  Robotics macro signals (FRED) ==="
python ingestion/robotics_signals_ingestion.py

echo ""
echo "=== 16/17  BLS JOLTS sector job openings (NAICS 51 + 333) ==="
python ingestion/bls_jolts_ingestion.py

echo ""
echo "=== 17/17  Census international trade (semiconductors + DC equipment) ==="
python ingestion/census_trade_ingestion.py
```

Also update the lower step numbers in the script for the steps `1/16` through `13/16` — change all `/16` to `/17`. Use a single sed-style replace if convenient:

```bash
sed -i.bak 's|/16  |/17  |g' tools/run_refresh.sh && rm tools/run_refresh.sh.bak
```

Then update the BLS JOLTS step's banner text from `tech sector job openings` to `sector job openings (NAICS 51 + 333)` to reflect the second series.

- [ ] **Step 2: Verify the script parses**

Run: `bash -n tools/run_refresh.sh`
Expected: no output (syntactically valid).

- [ ] **Step 3: Verify step count is 17**

Run: `grep -c '=== [0-9]\{1,2\}/17' tools/run_refresh.sh`
Expected: `17`

- [ ] **Step 4: Run full test suite — final acceptance**

Run: `pytest tests/ -m 'not integration' -q`
Expected: 0 failures, 0 errors. Test count ~457+.

- [ ] **Step 5: Sanity — feature count = 109**

Run: `python -c "from models.train import FEATURE_COLS; print(len(FEATURE_COLS))"`
Expected output: `109`

- [ ] **Step 6: Sanity — PHYSICAL_AI_FEATURE_COLS = 21**

Run: `python -c "from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS; print(len(PHYSICAL_AI_FEATURE_COLS))"`
Expected output: `21`

- [ ] **Step 7: Commit**

```bash
git add tools/run_refresh.sh
git commit -m "chore: wire physical-AI signals into run_refresh.sh (16→17 steps)"
```

---

## Self-review notes

**Spec coverage:**
- Spec §"Data sources / FRED" → Task 1 ✓
- Spec §"Data sources / BLS JOLTS" → Task 2 ✓
- Spec §"Data sources / USPTO" → Task 3 ✓
- Spec §"Modules / physical_ai_features.py + train.py wiring" → Task 4 ✓
- Spec §"Feature columns / 21 names" → Task 4 (exact-name test) + train.py FEATURE_COLS append ✓
- Spec §"Join semantics / 60d FRED + 120d patent + as-of join" → Task 4 (`_value_at` + 60/120 constants + tests) ✓
- Spec §"Error handling / fail-soft + null on missing" → Task 1 (FRED), Task 2 (existing JOLTS pattern), Task 3 (USPTO try/except) ✓
- Spec §"Tier assignment" → Task 4 step 5 + tier tests in step 6 ✓
- Spec §"Rollout / 5 commits" → Tasks 1–5 each end in one commit ✓
- Spec §"NAPM fallback" → Task 1 `fetch_all` retries `USAPMI` ✓
- Spec §"Refresh script wiring" → Task 5 ✓

**Type / name consistency:**
- `PHYSICAL_AI_FEATURE_COLS` (21 entries) defined once in Task 4 step 3, imported in train.py and tests.
- Bucket names `B25J / B64 / B60W / G05D1 / G05B19 / G06V` are identical across Task 3 (`_PHYSICAL_AI_BUCKETS`), Task 4 (`_PATENT_COL_MAP`), and the tests in both tasks.
- FRED series IDs `NEWORDER / NAPM / IPG3331S / WPU114` are identical across Task 1 (`_FRED_SERIES`) and Task 4 (`_FRED_COL_MAP`).
- Tolerance constants `_FRED_TOLERANCE_DAYS = 60` and `_PATENT_TOLERANCE_DAYS = 120` defined once.
- JOLTS NAICS 333 series id `JTS333000000000000JOL` consistent across Task 2 and Task 4.

**No placeholders.** Every step has complete code or exact commands.
