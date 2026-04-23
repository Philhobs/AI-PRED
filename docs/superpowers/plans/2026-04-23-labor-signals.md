# Labor Market Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add USAJOBS federal AI/ML job posting data and BLS JOLTS tech-sector openings, deriving 4 `LABOR_FEATURE_COLS` and growing FEATURE_COLS from 73 → 77.

**Architecture:** Two ingestion modules (USAJOBS GET API, BLS JOLTS POST API) write Hive-partitioned parquet with staleness guards. One feature module uses DuckDB cross-join rolling windows for USAJOBS and a window-function query for JOLTS. All 4 features are market-wide (joined on date only). Medium + long tiers only.

**Tech Stack:** Python 3.11+, Polars, DuckDB, requests, pytest with unittest.mock.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ingestion/usajobs_ingestion.py` | Create | USAJOBS GET API, 5 keyword queries, dedup, parquet output |
| `ingestion/bls_jolts_ingestion.py` | Create | BLS JOLTS POST API, 12-month lookback, parquet output |
| `processing/labor_features.py` | Create | 4 `LABOR_FEATURE_COLS` + `join_labor_features()` |
| `tests/test_usajobs_ingestion.py` | Create | 5 ingestion tests |
| `tests/test_bls_jolts_ingestion.py` | Create | 4 ingestion tests |
| `tests/test_labor_features.py` | Create | 8 feature tests |
| `models/train.py` | Modify | Import + FEATURE_COLS 73→77 + tier update + join call |
| `models/inference.py` | Modify | Import + join call |
| `tests/test_train.py` | Modify | Update count 73→77 + 5 new LABOR tests |
| `tools/run_refresh.sh` | Modify | Add steps 13/14 + 14/14, renumber prior to X/14 |

---

## Task 1: USAJOBS Ingestion

**Files:**
- Create: `ingestion/usajobs_ingestion.py`
- Create: `tests/test_usajobs_ingestion.py`

- [ ] **Step 1: Write the 5 failing tests**

Create `tests/test_usajobs_ingestion.py`:

```python
import datetime
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.usajobs_ingestion import _SCHEMA


def _make_item(position_id: str, title: str, pub_date: str) -> dict:
    return {
        "MatchedObjectId": position_id,
        "MatchedObjectDescriptor": {
            "PositionID": position_id,
            "PositionTitle": title,
            "PublicationStartDate": f"{pub_date}T00:00:00.0000000Z",
        },
    }


def _make_response(items: list[dict], total: int) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "SearchResult": {
            "SearchResultCount": len(items),
            "SearchResultCountAll": total,
            "SearchResultItems": items,
        }
    }
    return mock


_ONE_ITEM = _make_item("DOD-001", "AI Research Scientist", "2024-03-15")


def test_schema_correct():
    """fetch_postings returns a DataFrame matching _SCHEMA."""
    from ingestion.usajobs_ingestion import fetch_postings

    resp = _make_response([_ONE_ITEM], total=1)
    with patch("ingestion.usajobs_ingestion.requests.get", return_value=resp):
        with patch("ingestion.usajobs_ingestion.time.sleep"):
            df = fetch_postings("2024-04-01")

    assert df.schema == _SCHEMA
    assert len(df) == 1
    assert df["posting_id"][0] == "DOD-001"
    assert df["posted_date"][0] == datetime.date(2024, 3, 15)


def test_dedup_on_posting_id():
    """fetch_postings deduplicates rows with the same posting_id across keyword queries."""
    from ingestion.usajobs_ingestion import fetch_postings

    # Same posting_id returned by multiple keyword queries
    same_item = _make_item("DOD-001", "AI Researcher", "2024-03-15")
    resp = _make_response([same_item], total=1)

    with patch("ingestion.usajobs_ingestion.requests.get", return_value=resp):
        with patch("ingestion.usajobs_ingestion.time.sleep"):
            df = fetch_postings("2024-04-01")

    # 5 keywords × 1 item each = 5 raw rows → dedup to 1
    assert len(df) == 1


def test_same_week_snapshot_skipped(tmp_path):
    """ingest_usajobs skips re-download when existing parquet is from same ISO week."""
    from ingestion.usajobs_ingestion import ingest_usajobs

    # 2024-04-01 (Mon) and 2024-04-02 (Tue) are same ISO week
    existing = tmp_path / "date=2024-04-02"
    existing.mkdir()
    pl.DataFrame(schema=_SCHEMA).write_parquet(existing / "postings.parquet")

    with patch("ingestion.usajobs_ingestion.requests.get") as mock_get:
        ingest_usajobs("2024-04-01", tmp_path)

    mock_get.assert_not_called()


def test_empty_results_no_file_written(tmp_path):
    """When all keyword queries return 0 results, no parquet file is written."""
    from ingestion.usajobs_ingestion import ingest_usajobs

    resp = _make_response([], total=0)
    with patch("ingestion.usajobs_ingestion.requests.get", return_value=resp):
        with patch("ingestion.usajobs_ingestion.time.sleep"):
            ingest_usajobs("2024-04-01", tmp_path)

    assert not (tmp_path / "date=2024-04-01").exists()


def test_sleep_between_keyword_queries():
    """time.sleep(1.0) is called between keyword queries (not after the last one)."""
    from ingestion.usajobs_ingestion import fetch_postings, _KEYWORDS

    resp = _make_response([_ONE_ITEM], total=1)
    with patch("ingestion.usajobs_ingestion.requests.get", return_value=resp):
        with patch("ingestion.usajobs_ingestion.time.sleep") as mock_sleep:
            fetch_postings("2024-04-01")

    # sleep called len(_KEYWORDS) - 1 times (between queries, not after the last)
    assert mock_sleep.call_count == len(_KEYWORDS) - 1
    mock_sleep.assert_called_with(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_usajobs_ingestion.py -v 2>&1 | head -20
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'ingestion.usajobs_ingestion'`

- [ ] **Step 3: Implement `ingestion/usajobs_ingestion.py`**

Create `ingestion/usajobs_ingestion.py`:

```python
"""USAJOBS federal AI/ML job postings ingestion.

Fetches federal job postings across 5 AI/ML keyword terms.
Output: data/raw/usajobs/date=YYYY-MM-DD/postings.parquet

Staleness guard: skips re-download if existing snapshot is from the same ISO week.
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_BASE_URL = "https://data.usajobs.gov/api/search"
_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "GPU computing",
    "semiconductor",
]
_RESULTS_PER_PAGE = 500
_DATE_POSTED = 60  # 60-day lookback covers both the 30d and prior-30d windows
_SLEEP_BETWEEN_KEYWORDS = 1.0
_MAX_PAGES = 20  # safety cap: 20 × 500 = 10,000 results

_SCHEMA = {
    "date": pl.Date,
    "posting_id": pl.Utf8,
    "title": pl.Utf8,
    "posted_date": pl.Date,
    "keyword": pl.Utf8,
}


def _same_iso_week(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent parquet in existing_dir is from the same ISO week as today."""
    files = sorted(existing_dir.glob("date=*/postings.parquet"))
    if not files:
        return False
    last_date_str = files[-1].parent.name.replace("date=", "")
    try:
        last_date = datetime.date.fromisoformat(last_date_str)
        today = datetime.date.fromisoformat(today_str)
        last_iso = last_date.isocalendar()
        today_iso = today.isocalendar()
        return last_iso.week == today_iso.week and last_iso.year == today_iso.year
    except ValueError:
        return False


def _fetch_keyword(keyword: str, run_date: datetime.date, user_agent: str) -> list[dict]:
    """Fetch all USAJOBS postings for one keyword term across all pages."""
    headers = {"Host": "data.usajobs.gov", "User-Agent": user_agent}
    records: list[dict] = []
    page = 1

    while page <= _MAX_PAGES:
        params = {
            "Keyword": keyword,
            "DatePosted": str(_DATE_POSTED),
            "ResultsPerPage": str(_RESULTS_PER_PAGE),
            "Page": str(page),
        }
        resp = requests.get(_BASE_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("SearchResult", {}).get("SearchResultItems", [])
        if not items:
            break

        for item in items:
            desc = item.get("MatchedObjectDescriptor", {})
            pub_start = desc.get("PublicationStartDate", "")
            try:
                posted = datetime.date.fromisoformat(pub_start[:10])
            except (ValueError, TypeError):
                continue
            records.append({
                "date": run_date,
                "posting_id": desc.get("PositionID", ""),
                "title": desc.get("PositionTitle", ""),
                "posted_date": posted,
                "keyword": keyword,
            })

        total = data.get("SearchResult", {}).get("SearchResultCountAll", 0)
        if len(records) >= total or len(items) < _RESULTS_PER_PAGE:
            break
        page += 1

    return records


def fetch_postings(date_str: str) -> pl.DataFrame:
    """Fetch all federal AI/ML job postings, deduplicated on posting_id.

    Returns DataFrame with _SCHEMA. Empty DataFrame if no results.
    """
    user_agent = os.environ.get("USAJOBS_USER_AGENT", "ai-pred-research@example.com")
    run_date = datetime.date.fromisoformat(date_str)

    all_records: list[dict] = []
    for i, keyword in enumerate(_KEYWORDS):
        all_records.extend(_fetch_keyword(keyword, run_date, user_agent))
        if i < len(_KEYWORDS) - 1:
            time.sleep(_SLEEP_BETWEEN_KEYWORDS)

    if not all_records:
        return pl.DataFrame(schema=_SCHEMA)

    return (
        pl.DataFrame(all_records, schema=_SCHEMA)
        .unique(subset=["posting_id"], keep="first")
    )


def ingest_usajobs(date_str: str, output_dir: Path) -> None:
    """Fetch and persist USAJOBS postings for date_str.

    Skips download if same-ISO-week snapshot exists. Writes nothing when results are empty.
    """
    if _same_iso_week(output_dir, date_str):
        _LOG.info("USAJOBS: same ISO week snapshot exists — skipping for %s", date_str)
        return

    _LOG.info("USAJOBS: fetching AI/ML federal job postings for %s", date_str)
    df = fetch_postings(date_str)
    if not df.is_empty():
        out = output_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "postings.parquet", compression="snappy")
        _LOG.info("USAJOBS: wrote %d posting rows to %s", len(df), out)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ingest_usajobs(
        datetime.date.today().isoformat(),
        Path("data/raw/usajobs"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_usajobs_ingestion.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add ingestion/usajobs_ingestion.py tests/test_usajobs_ingestion.py
git commit -m "feat: add USAJOBS federal AI/ML job postings ingestion"
```

---

## Task 2: BLS JOLTS Ingestion

**Files:**
- Create: `ingestion/bls_jolts_ingestion.py`
- Create: `tests/test_bls_jolts_ingestion.py`

- [ ] **Step 1: Write the 4 failing tests**

Create `tests/test_bls_jolts_ingestion.py`:

```python
import datetime
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ingestion.bls_jolts_ingestion import _SCHEMA, _SERIES_ID


def _make_jolts_response(data_rows: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [{"seriesID": _SERIES_ID, "data": data_rows}]
        },
    }
    return mock


_ONE_ROW = {"year": "2024", "period": "M03", "periodName": "March", "value": "123.4", "footnotes": [{}]}


def test_schema_correct():
    """fetch_jolts returns a DataFrame matching _SCHEMA."""
    from ingestion.bls_jolts_ingestion import fetch_jolts

    resp = _make_jolts_response([_ONE_ROW])
    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=resp):
        df = fetch_jolts("2024-04-01")

    assert df.schema == _SCHEMA
    assert len(df) == 1
    assert df["year"][0] == 2024
    assert df["period"][0] == "M03"
    assert df["value"][0] == pytest.approx(123.4)


def test_period_stored_as_string():
    """period field is stored as a string 'M01'–'M12', not converted to a date."""
    from ingestion.bls_jolts_ingestion import fetch_jolts

    rows = [
        {"year": "2024", "period": "M01", "value": "100.0", "footnotes": []},
        {"year": "2024", "period": "M12", "value": "200.0", "footnotes": []},
    ]
    resp = _make_jolts_response(rows)
    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=resp):
        df = fetch_jolts("2024-04-01")

    assert df["period"].dtype == pl.Utf8
    assert set(df["period"].to_list()) == {"M01", "M12"}


def test_same_month_snapshot_skipped(tmp_path):
    """ingest_bls_jolts skips re-download when existing snapshot is from the same calendar month."""
    from ingestion.bls_jolts_ingestion import ingest_bls_jolts

    # Both 2024-04-01 and 2024-04-15 are in April 2024
    existing = tmp_path / "date=2024-04-01"
    existing.mkdir()
    pl.DataFrame(schema=_SCHEMA).write_parquet(existing / "openings.parquet")

    with patch("ingestion.bls_jolts_ingestion.requests.post") as mock_post:
        ingest_bls_jolts("2024-04-15", tmp_path)

    mock_post.assert_not_called()


def test_empty_series_no_file_written(tmp_path):
    """When API returns no data rows, no parquet file is written."""
    from ingestion.bls_jolts_ingestion import ingest_bls_jolts

    resp = _make_jolts_response([])
    with patch("ingestion.bls_jolts_ingestion.requests.post", return_value=resp):
        ingest_bls_jolts("2024-04-01", tmp_path)

    assert not (tmp_path / "date=2024-04-01").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_bls_jolts_ingestion.py -v 2>&1 | head -20
```

Expected: 4 failures with `ModuleNotFoundError: No module named 'ingestion.bls_jolts_ingestion'`

- [ ] **Step 3: Implement `ingestion/bls_jolts_ingestion.py`**

Create `ingestion/bls_jolts_ingestion.py`:

```python
"""BLS JOLTS tech sector job openings ingestion.

Fetches Computer & Electronic Products (NAICS 334) job openings from BLS API v2.
Series: JTS510000000000000JOL (job openings level, thousands, seasonally adjusted)
Output: data/raw/bls_jolts/date=YYYY-MM-DD/openings.parquet

Staleness guard: skips re-download if existing snapshot is from the same calendar month
(BLS JOLTS publishes monthly data with ~6-week publication lag).
"""
from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
_SERIES_ID = "JTS510000000000000JOL"

_SCHEMA = {
    "date": pl.Date,
    "series_id": pl.Utf8,
    "year": pl.Int32,
    "period": pl.Utf8,
    "value": pl.Float64,
}


def _same_month(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent parquet snapshot was taken in the same calendar month as today."""
    files = sorted(existing_dir.glob("date=*/openings.parquet"))
    if not files:
        return False
    last_date_str = files[-1].parent.name.replace("date=", "")
    try:
        last_date = datetime.date.fromisoformat(last_date_str)
        today = datetime.date.fromisoformat(today_str)
        return last_date.year == today.year and last_date.month == today.month
    except ValueError:
        return False


def fetch_jolts(date_str: str) -> pl.DataFrame:
    """Fetch BLS JOLTS tech sector openings for 12-month window ending date_str.

    Returns DataFrame with _SCHEMA. Empty DataFrame if no results.
    """
    today = datetime.date.fromisoformat(date_str)
    payload: dict = {
        "seriesid": [_SERIES_ID],
        "startyear": str(today.year - 1),
        "endyear": str(today.year),
    }
    api_key = os.environ.get("BLS_API_KEY", "")
    if api_key:
        payload["registrationkey"] = api_key

    resp = requests.post(_BLS_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    series_list = data.get("Results", {}).get("series", [])
    if not series_list:
        return pl.DataFrame(schema=_SCHEMA)

    rows = []
    for entry in series_list[0].get("data", []):
        period = entry.get("period", "")
        if not (period.startswith("M") and period != "M13"):
            continue
        try:
            year = int(entry["year"])
            value = float(entry["value"])
        except (KeyError, ValueError, TypeError):
            continue
        rows.append({
            "date": today,
            "series_id": _SERIES_ID,
            "year": year,
            "period": period,
            "value": value,
        })

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)

    return pl.DataFrame(rows, schema=_SCHEMA)


def ingest_bls_jolts(date_str: str, output_dir: Path) -> None:
    """Fetch and persist BLS JOLTS openings for date_str.

    Skips download if same-month snapshot exists. Writes nothing when results are empty.
    """
    if _same_month(output_dir, date_str):
        _LOG.info("BLS JOLTS: same month snapshot exists — skipping for %s", date_str)
        return

    _LOG.info("BLS JOLTS: fetching tech sector job openings for %s", date_str)
    df = fetch_jolts(date_str)
    if not df.is_empty():
        out = output_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out / "openings.parquet", compression="snappy")
        _LOG.info("BLS JOLTS: wrote %d rows to %s", len(df), out)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    ingest_bls_jolts(
        datetime.date.today().isoformat(),
        Path("data/raw/bls_jolts"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_bls_jolts_ingestion.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add ingestion/bls_jolts_ingestion.py tests/test_bls_jolts_ingestion.py
git commit -m "feat: add BLS JOLTS tech sector job openings ingestion"
```

---

## Task 3: Labor Features + Model Integration

**Files:**
- Create: `processing/labor_features.py`
- Create: `tests/test_labor_features.py`
- Modify: `models/train.py`
- Modify: `models/inference.py`
- Modify: `tests/test_train.py`
- Modify: `tools/run_refresh.sh`

- [ ] **Step 1: Write the 8 failing feature tests**

Create `tests/test_labor_features.py`:

```python
import datetime
import polars as pl
import pytest
from pathlib import Path

from ingestion.usajobs_ingestion import _SCHEMA as _POSTING_SCHEMA
from ingestion.bls_jolts_ingestion import _SCHEMA as _JOLTS_SCHEMA


def _write_postings(usajobs_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = usajobs_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_POSTING_SCHEMA).write_parquet(out / "postings.parquet")


def _write_jolts(jolts_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = jolts_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_JOLTS_SCHEMA).write_parquet(out / "openings.parquet")


def _query_df(date: datetime.date) -> pl.DataFrame:
    return pl.DataFrame(
        {"ticker": ["NVDA"], "date": [date]},
        schema={"ticker": pl.Utf8, "date": pl.Date},
    )


_QUERY_DATE = datetime.date(2024, 4, 1)
_RECENT = datetime.date(2024, 3, 20)    # 12 days ago — inside 30d window
_PRIOR = datetime.date(2024, 2, 20)     # 40 days ago — inside prior 30d window (31-60d)
_OLD = datetime.date(2024, 1, 1)        # 91 days ago — outside 60d window


def test_gov_ai_hiring_30d_window(tmp_path):
    """gov_ai_hiring_30d counts only postings within 30d, not older ones."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    _write_postings(usajobs_dir, [
        {"date": _QUERY_DATE, "posting_id": "P1", "title": "AI Engineer",
         "posted_date": _RECENT, "keyword": "artificial intelligence"},   # in window
        {"date": _QUERY_DATE, "posting_id": "P2", "title": "ML Scientist",
         "posted_date": _OLD, "keyword": "machine learning"},             # outside 60d
    ])

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["gov_ai_hiring_30d"][0] == 1.0


def test_gov_ai_hiring_momentum_positive(tmp_path):
    """gov_ai_hiring_momentum is positive when recent 30d count > prior 30d count."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    # 2 recent, 1 prior
    _write_postings(usajobs_dir, [
        {"date": _QUERY_DATE, "posting_id": "P1", "title": "AI A",
         "posted_date": _RECENT, "keyword": "artificial intelligence"},
        {"date": _QUERY_DATE, "posting_id": "P2", "title": "AI B",
         "posted_date": _RECENT, "keyword": "machine learning"},
        {"date": _QUERY_DATE, "posting_id": "P3", "title": "AI C",
         "posted_date": _PRIOR, "keyword": "deep learning"},
    ])

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["gov_ai_hiring_momentum"][0] == pytest.approx(1.0)


def test_tech_job_openings_index_most_recent_month(tmp_path):
    """tech_job_openings_index uses the most recent JOLTS month <= query date."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    # M02=100, M03=120, M04=150 (period_date=2024-04-01 = query date, included)
    # M05=200 (period_date=2024-05-01 > query date, excluded)
    _write_jolts(jolts_dir, [
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M02", "value": 100.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M03", "value": 120.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M04", "value": 150.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M05", "value": 200.0},
    ])

    # Use query date 2024-03-15 so M04 (2024-04-01) is excluded
    df = join_labor_features(_query_df(datetime.date(2024, 3, 15)), usajobs_dir, jolts_dir)
    assert df["tech_job_openings_index"][0] == pytest.approx(120.0)


def test_tech_job_openings_momentum(tmp_path):
    """tech_job_openings_momentum equals current minus previous month value."""
    from processing.labor_features import join_labor_features

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    _write_jolts(jolts_dir, [
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M01", "value": 90.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M02", "value": 100.0},
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M03", "value": 120.0},
    ])

    # Query date 2024-04-01: most recent ≤ date is M03=120, previous is M02=100
    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["tech_job_openings_momentum"][0] == pytest.approx(20.0)


def test_join_adds_exactly_4_columns(tmp_path):
    """join_labor_features adds exactly 4 new columns."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"
    input_df = _query_df(_QUERY_DATE)
    result = join_labor_features(input_df, usajobs_dir, jolts_dir)

    added = [c for c in result.columns if c not in input_df.columns]
    assert sorted(added) == sorted(LABOR_FEATURE_COLS)
    assert len(added) == 4


def test_missing_usajobs_dir_zero_fill(tmp_path):
    """Missing usajobs_dir returns zero-filled USAJOBS features."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "does_not_exist" / "usajobs"
    jolts_dir = tmp_path / "does_not_exist" / "bls_jolts"

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    for col in LABOR_FEATURE_COLS:
        assert df[col][0] == 0.0


def test_missing_jolts_dir_zero_fill(tmp_path):
    """Missing jolts_dir returns zero-filled JOLTS features."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "does_not_exist" / "usajobs"
    jolts_dir = tmp_path / "does_not_exist" / "bls_jolts"

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    for col in LABOR_FEATURE_COLS:
        assert df[col][0] == 0.0


def test_no_data_in_window_zero_fill(tmp_path):
    """All 4 features zero-filled when postings and JOLTS data are outside windows."""
    from processing.labor_features import join_labor_features, LABOR_FEATURE_COLS

    usajobs_dir = tmp_path / "usajobs"
    jolts_dir = tmp_path / "bls_jolts"

    # Posting is too old (outside 60d window)
    _write_postings(usajobs_dir, [
        {"date": _QUERY_DATE, "posting_id": "P1", "title": "Old Job",
         "posted_date": _OLD, "keyword": "semiconductor"},
    ])
    # JOLTS data is in the future (after query date)
    _write_jolts(jolts_dir, [
        {"date": _QUERY_DATE, "series_id": "JTS510000000000000JOL",
         "year": 2024, "period": "M12", "value": 999.0},  # 2024-12-01 > query date
    ])

    df = join_labor_features(_query_df(_QUERY_DATE), usajobs_dir, jolts_dir)
    assert df["gov_ai_hiring_30d"][0] == 0.0
    assert df["gov_ai_hiring_momentum"][0] == 0.0
    assert df["tech_job_openings_index"][0] == 0.0
    assert df["tech_job_openings_momentum"][0] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_labor_features.py -v 2>&1 | head -20
```

Expected: 8 failures with `ModuleNotFoundError: No module named 'processing.labor_features'`

- [ ] **Step 3: Implement `processing/labor_features.py`**

Create `processing/labor_features.py`:

```python
"""Labor market signal features.

Features (LABOR_FEATURE_COLS):
    gov_ai_hiring_30d        — count of federal AI/ML job postings in 30d rolling window
    gov_ai_hiring_momentum   — recent 30d postings minus prior 30d (government AI investment)
    tech_job_openings_index  — BLS JOLTS NAICS 334 job openings, most recent month (thousands)
    tech_job_openings_momentum — current month openings minus previous month (hiring acceleration)

All 4 features are market-wide (joined on date only).
All features zero-filled when data is absent.

Tier routing: medium + long only (monthly data too slow for 5d/20d horizons).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

_LOG = logging.getLogger(__name__)

LABOR_FEATURE_COLS: list[str] = [
    "gov_ai_hiring_30d",
    "gov_ai_hiring_momentum",
    "tech_job_openings_index",
    "tech_job_openings_momentum",
]

_POSTING_SCHEMA = {
    "date": pl.Date, "posting_id": pl.Utf8, "title": pl.Utf8,
    "posted_date": pl.Date, "keyword": pl.Utf8,
}
_JOLTS_SCHEMA = {
    "date": pl.Date, "series_id": pl.Utf8, "year": pl.Int32,
    "period": pl.Utf8, "value": pl.Float64,
}


def _load_postings(usajobs_dir: Path) -> pl.DataFrame:
    files = sorted(usajobs_dir.glob("date=*/postings.parquet")) if usajobs_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_POSTING_SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files])


def _load_jolts(jolts_dir: Path) -> pl.DataFrame:
    files = sorted(jolts_dir.glob("date=*/openings.parquet")) if jolts_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_JOLTS_SCHEMA)
    # Dedup: keep the most recent snapshot's value for each (year, period)
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .sort("date")
        .unique(subset=["year", "period"], keep="last")
    )


def join_labor_features(
    df: pl.DataFrame,
    usajobs_dir: Path,
    jolts_dir: Path,
) -> pl.DataFrame:
    """Left-join labor market features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'date' (Date) column.
        usajobs_dir: Root of data/raw/usajobs/ Hive tree.
        jolts_dir: Root of data/raw/bls_jolts/ Hive tree.

    Returns:
        df with LABOR_FEATURE_COLS appended (Float64). Zero-filled.
    """
    postings = _load_postings(usajobs_dir)
    jolts = _load_jolts(jolts_dir)
    query_dates = df.select(["date"]).unique()

    with duckdb.connect() as con:
        con.register("query_dates", query_dates.to_arrow())

        # USAJOBS: rolling 30d count and prior-30d momentum
        if not postings.is_empty():
            con.register("postings", postings.to_arrow())
            usajobs_result = con.execute("""
                SELECT
                    q.date,
                    COALESCE(CAST(SUM(CASE
                        WHEN p.posted_date >= q.date - INTERVAL 30 DAY
                             AND p.posted_date <= q.date
                        THEN 1 ELSE 0 END) AS DOUBLE), 0.0) AS gov_ai_hiring_30d,
                    COALESCE(CAST(SUM(CASE
                        WHEN p.posted_date >= q.date - INTERVAL 30 DAY
                             AND p.posted_date <= q.date
                        THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    - COALESCE(CAST(SUM(CASE
                        WHEN p.posted_date >= q.date - INTERVAL 60 DAY
                             AND p.posted_date < q.date - INTERVAL 30 DAY
                        THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS gov_ai_hiring_momentum
                FROM query_dates q
                LEFT JOIN postings p
                    ON p.posted_date <= q.date
                    AND p.posted_date >= q.date - INTERVAL 60 DAY
                GROUP BY q.date
            """).pl()
        else:
            usajobs_result = pl.DataFrame(schema={
                "date": pl.Date,
                "gov_ai_hiring_30d": pl.Float64,
                "gov_ai_hiring_momentum": pl.Float64,
            })

        # BLS JOLTS: most recent month <= query date (index) and prior month (momentum)
        if not jolts.is_empty():
            con.register("jolts", jolts.to_arrow())
            jolts_result = con.execute("""
                WITH jolts_dated AS (
                    SELECT
                        value,
                        MAKE_DATE(year, CAST(SUBSTR(period, 2) AS INTEGER), 1) AS period_date
                    FROM jolts
                ),
                ranked AS (
                    SELECT
                        q.date,
                        j.value,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY j.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN jolts_dated j
                    WHERE j.period_date <= q.date
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN value END), 0.0)
                        AS tech_job_openings_index,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN value END), 0.0)
                        - COALESCE(MAX(CASE WHEN rn = 2 THEN value END), 0.0)
                    AS tech_job_openings_momentum
                FROM ranked
                GROUP BY date
            """).pl()
        else:
            jolts_result = pl.DataFrame(schema={
                "date": pl.Date,
                "tech_job_openings_index": pl.Float64,
                "tech_job_openings_momentum": pl.Float64,
            })

    # Join both feature sets back to query_dates, then to original df
    result = query_dates.join(usajobs_result, on="date", how="left")
    result = result.join(jolts_result, on="date", how="left")
    df = df.join(result, on="date", how="left")

    # Zero-fill backstop: catches nulls from partial joins and empty windows
    for col in LABOR_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
```

- [ ] **Step 4: Run feature tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_labor_features.py -v
```

Expected: 8 passed

- [ ] **Step 5: Write 5 failing tests in `tests/test_train.py`**

Read the file to find the last test. Append these 5 tests after `test_tier_medium_equals_feature_cols_after_gov_integration`:

```python
def test_feature_cols_includes_labor():
    """FEATURE_COLS must contain all 4 LABOR_FEATURE_COLS and total must be 77."""
    from models.train import FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    assert len(LABOR_FEATURE_COLS) == 4
    for col in LABOR_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 77, f"Expected 77 features, got {len(FEATURE_COLS)}"


def test_labor_cols_absent_from_short_tier():
    """LABOR cols must not appear in short tier — monthly data too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in LABOR_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_labor_cols_in_medium_tier():
    """LABOR cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    medium = TIER_FEATURE_COLS["medium"]
    for col in LABOR_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_labor_cols_in_long_tier():
    """LABOR cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.labor_features import LABOR_FEATURE_COLS
    long_cols = TIER_FEATURE_COLS["long"]
    for col in LABOR_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_labor_col_names_correct():
    """LABOR_FEATURE_COLS must contain exactly the 4 expected column names."""
    from processing.labor_features import LABOR_FEATURE_COLS
    expected = {
        "gov_ai_hiring_30d",
        "gov_ai_hiring_momentum",
        "tech_job_openings_index",
        "tech_job_openings_momentum",
    }
    assert set(LABOR_FEATURE_COLS) == expected
```

Also update `test_feature_cols_includes_uspto_patent` — find its `assert len(FEATURE_COLS) == 73` line and change it to `assert len(FEATURE_COLS) == 77`.

- [ ] **Step 6: Run tests to verify the 5 new tests + updated count fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_train.py -k "test_feature_cols_includes_labor or test_labor or test_feature_cols_includes_uspto" -v 2>&1 | head -20
```

Expected: 6 failures

- [ ] **Step 7: Update `models/train.py`**

**Change A** — Add import after the `from processing.patent_features` line:
```python
from processing.labor_features import LABOR_FEATURE_COLS, join_labor_features
```

**Change B** — Append to FEATURE_COLS after `+ USPTO_PATENT_FEATURE_COLS    # 67 → 73 features total`:
```python
    + LABOR_FEATURE_COLS           # 73 → 77 features total
```

**Change C** — Update TIER_FEATURE_COLS:
- Change medium tier comment from `# all 73 features` to `# all 77 features`
- Add to long tier after `+ USPTO_PATENT_FEATURE_COLS    # patent grant cycles relevant at year+ horizons`:
```python
        + LABOR_FEATURE_COLS           # labor market cycles relevant at year+ horizons
```

**Change D** — Add join call in `build_training_dataset` after the USPTO join block (after `df = join_patent_features(df, patents_apps_dir, patents_grants_dir)`):
```python
    # Join labor market features (USAJOBS federal postings + BLS JOLTS tech openings)
    usajobs_dir = fundamentals_dir.parent.parent / "usajobs"
    jolts_dir = fundamentals_dir.parent.parent / "bls_jolts"
    df = join_labor_features(df, usajobs_dir, jolts_dir)
```

- [ ] **Step 8: Update `models/inference.py`**

**Change A** — Add import after `from processing.patent_features import join_patent_features`:
```python
from processing.labor_features import join_labor_features
```

**Change B** — Add join call after `df = join_patent_features(df, patents_apps_dir, patents_grants_dir)`:
```python
    usajobs_dir = data_dir / "usajobs"
    jolts_dir = data_dir / "bls_jolts"
    df = join_labor_features(df, usajobs_dir, jolts_dir)
```

- [ ] **Step 9: Update `tools/run_refresh.sh`**

Renumber all existing steps from `X/12` to `X/14`. Then replace the "Refresh complete" block:

Before:
```bash
echo ""
echo "=== 12/12  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
```

After:
```bash
echo ""
echo "=== 12/14  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== 13/14  USAJOBS federal AI/ML job postings ==="
python ingestion/usajobs_ingestion.py

echo ""
echo "=== 14/14  BLS JOLTS tech sector job openings ==="
python ingestion/bls_jolts_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
```

- [ ] **Step 10: Run the full test suite**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/ -m "not integration" -q --tb=short 2>&1 | tail -10
```

Expected: all tests pass including the 5 new LABOR tests in test_train.py

- [ ] **Step 11: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add processing/labor_features.py tests/test_labor_features.py \
    models/train.py models/inference.py tests/test_train.py tools/run_refresh.sh
git commit -m "feat: wire labor market features into model pipeline (FEATURE_COLS 73→77)"
```
