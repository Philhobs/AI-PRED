# USPTO Patent Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PatentsView v2 patent ingestion and derive 6 `USPTO_PATENT_FEATURE_COLS`, growing FEATURE_COLS from 67 → 73.

**Architecture:** One ingestion module polls PatentsView v2 (POST, no auth) for applications and grants, writing Hive-partitioned parquet with a same-ISO-week staleness guard. A feature module uses DuckDB cross-join rolling windows (identical pattern to `gov_behavioral_features.py`) and reuses `_TICKER_NAME_MAP` + difflib fuzzy match from that module. All 6 features route to medium + long tiers only.

**Tech Stack:** Python 3.11+, Polars, DuckDB, requests, difflib, pytest with unittest.mock.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ingestion/uspto_ingestion.py` | Create | PatentsView v2 fetch (apps + grants), staleness check, parquet output |
| `processing/patent_features.py` | Create | 6 `USPTO_PATENT_FEATURE_COLS` + `join_patent_features()` |
| `tests/test_uspto_ingestion.py` | Create | 6 ingestion tests |
| `tests/test_patent_features.py` | Create | 10 feature tests |
| `models/train.py` | Modify | Import + FEATURE_COLS 67→73 + tier update + join call |
| `models/inference.py` | Modify | Import + join call + docstring 67→73 |
| `tests/test_train.py` | Modify | Update count 67→73 + 6 new USPTO tests |
| `tools/run_refresh.sh` | Modify | Add step 12/12 USPTO ingestion |

---

## Task 1: USPTO Ingestion

**Files:**
- Create: `ingestion/uspto_ingestion.py`
- Create: `tests/test_uspto_ingestion.py`

- [ ] **Step 1: Write the 6 failing tests**

Create `tests/test_uspto_ingestion.py`:

```python
import datetime
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


_APP_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "app_id": pl.Utf8,
    "cpc_group": pl.Utf8, "filing_date": pl.Date,
}
_GRANT_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "patent_id": pl.Utf8,
    "cpc_group": pl.Utf8, "grant_date": pl.Date, "forward_citation_count": pl.Int32,
}

_ONE_APP = {
    "app_id": "APP001",
    "assignee_organization": "NVIDIA Corporation",
    "cpc_group_id": "G06N",
    "app_date": "2024-01-10",
}
_ONE_GRANT = {
    "patent_id": "US123456",
    "assignee_organization": "NVIDIA Corporation",
    "cpc_group_id": "G06N",
    "patent_date": "2024-01-10",
    "cited_by_count": 3,
}


def _make_response(records: list[dict], total: int, key: str) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {key: records, "total_patent_count": total, "total_app_count": total}
    return mock


def test_applications_schema_correct(tmp_path):
    """Parquet written by fetch_applications matches _APP_SCHEMA."""
    from ingestion.uspto_ingestion import fetch_applications

    resp = _make_response([_ONE_APP], total=1, key="applications")
    with patch("ingestion.uspto_ingestion.requests.post", return_value=resp):
        with patch("ingestion.uspto_ingestion.time.sleep"):
            df = fetch_applications("2024-01-15")

    assert df.schema == _APP_SCHEMA
    assert len(df) == 1
    assert df["assignee_name"][0] == "NVIDIA Corporation"
    assert df["cpc_group"][0] == "G06N"


def test_grants_schema_correct(tmp_path):
    """Parquet written by fetch_grants matches _GRANT_SCHEMA."""
    from ingestion.uspto_ingestion import fetch_grants

    resp = _make_response([_ONE_GRANT], total=1, key="patents")
    with patch("ingestion.uspto_ingestion.requests.post", return_value=resp):
        with patch("ingestion.uspto_ingestion.time.sleep"):
            df = fetch_grants("2024-01-15")

    assert df.schema == _GRANT_SCHEMA
    assert df["forward_citation_count"][0] == 3


def test_pagination_followed(tmp_path):
    """fetch_applications fetches all pages when total > per_page."""
    from ingestion.uspto_ingestion import fetch_applications

    page1 = [_ONE_APP.copy() for _ in range(100)]
    page2 = [_ONE_APP.copy() for _ in range(40)]

    resp1 = MagicMock()
    resp1.raise_for_status.return_value = None
    resp1.json.return_value = {"applications": page1, "total_app_count": 140}

    resp2 = MagicMock()
    resp2.raise_for_status.return_value = None
    resp2.json.return_value = {"applications": page2, "total_app_count": 140}

    with patch("ingestion.uspto_ingestion.requests.post", side_effect=[resp1, resp2]):
        with patch("ingestion.uspto_ingestion.time.sleep"):
            df = fetch_applications("2024-01-15")

    assert len(df) == 140


def test_same_week_snapshot_skipped(tmp_path):
    """ingest_uspto skips re-download when existing parquet is from same ISO week."""
    from ingestion.uspto_ingestion import ingest_uspto

    apps_dir = tmp_path / "patents" / "applications"
    grants_dir = tmp_path / "patents" / "grants"

    # Write a "same-week" snapshot — today is 2024-01-15 (week 3), use 2024-01-14 (also week 3)
    existing_app_dir = apps_dir / "date=2024-01-14"
    existing_app_dir.mkdir(parents=True)
    pl.DataFrame(schema=_APP_SCHEMA).write_parquet(existing_app_dir / "apps.parquet")

    existing_grant_dir = grants_dir / "date=2024-01-14"
    existing_grant_dir.mkdir(parents=True)
    pl.DataFrame(schema=_GRANT_SCHEMA).write_parquet(existing_grant_dir / "grants.parquet")

    with patch("ingestion.uspto_ingestion.requests.post") as mock_post:
        ingest_uspto("2024-01-15", apps_dir, grants_dir)

    mock_post.assert_not_called()


def test_empty_results_no_file_written(tmp_path):
    """When API returns 0 records, no parquet file is written."""
    from ingestion.uspto_ingestion import ingest_uspto

    apps_dir = tmp_path / "patents" / "applications"
    grants_dir = tmp_path / "patents" / "grants"

    empty_apps = MagicMock()
    empty_apps.raise_for_status.return_value = None
    empty_apps.json.return_value = {"applications": [], "total_app_count": 0}

    empty_grants = MagicMock()
    empty_grants.raise_for_status.return_value = None
    empty_grants.json.return_value = {"patents": [], "total_patent_count": 0}

    with patch("ingestion.uspto_ingestion.requests.post", side_effect=[empty_apps, empty_grants]):
        with patch("ingestion.uspto_ingestion.time.sleep"):
            ingest_uspto("2024-01-15", apps_dir, grants_dir)

    assert not (apps_dir / "date=2024-01-15").exists()
    assert not (grants_dir / "date=2024-01-15").exists()


def test_rate_limit_sleep_between_pages():
    """time.sleep(1.5) is called between pages when pagination is required."""
    from ingestion.uspto_ingestion import fetch_applications

    page1 = [_ONE_APP.copy() for _ in range(100)]
    page2 = [_ONE_APP.copy() for _ in range(10)]

    resp1 = MagicMock()
    resp1.raise_for_status.return_value = None
    resp1.json.return_value = {"applications": page1, "total_app_count": 110}

    resp2 = MagicMock()
    resp2.raise_for_status.return_value = None
    resp2.json.return_value = {"applications": page2, "total_app_count": 110}

    with patch("ingestion.uspto_ingestion.requests.post", side_effect=[resp1, resp2]):
        with patch("ingestion.uspto_ingestion.time.sleep") as mock_sleep:
            fetch_applications("2024-01-15")

    mock_sleep.assert_called_once_with(1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_uspto_ingestion.py -v 2>&1 | head -30
```

Expected: 6 failures with `ModuleNotFoundError: No module named 'ingestion.uspto_ingestion'`

- [ ] **Step 3: Implement `ingestion/uspto_ingestion.py`**

Create `ingestion/uspto_ingestion.py`:

```python
"""USPTO patent ingestion from PatentsView v2 API.

Fetches published patent applications and granted patents for AI/semiconductor CPC codes.
Output parquet files (snappy):
  data/raw/patents/applications/date=YYYY-MM-DD/apps.parquet
  data/raw/patents/grants/date=YYYY-MM-DD/grants.parquet

Staleness guard: skips re-download if existing snapshot is from the same ISO week
(PatentsView updates weekly).
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import polars as pl
import requests

_LOG = logging.getLogger(__name__)

_APPS_URL = "https://api.patentsview.org/applications/query"
_GRANTS_URL = "https://api.patentsview.org/patents/query"
_CPC_CODES = ["G06N", "H01L", "G06F", "G11C"]
_PER_PAGE = 100
_SLEEP_BETWEEN_PAGES = 1.5

_APP_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "app_id": pl.Utf8,
    "cpc_group": pl.Utf8, "filing_date": pl.Date,
}
_GRANT_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "patent_id": pl.Utf8,
    "cpc_group": pl.Utf8, "grant_date": pl.Date, "forward_citation_count": pl.Int32,
}


def _lookback_start(date_str: str) -> str:
    """Return the date 365 days before date_str as YYYY-MM-DD."""
    d = datetime.date.fromisoformat(date_str)
    return (d - datetime.timedelta(days=365)).isoformat()


def _same_iso_week(existing_dir: Path, today_str: str) -> bool:
    """True if the most recent existing parquet in existing_dir is from the same ISO week as today."""
    files = sorted(existing_dir.glob("date=*/apps.parquet")) + sorted(existing_dir.glob("date=*/grants.parquet"))
    if not files:
        return False
    # Extract date from parent directory name (date=YYYY-MM-DD)
    last_date_str = files[-1].parent.name.replace("date=", "")
    try:
        last_date = datetime.date.fromisoformat(last_date_str)
        today = datetime.date.fromisoformat(today_str)
        return last_date.isocalendar().week == today.isocalendar().week and last_date.year == today.year
    except ValueError:
        return False


def fetch_applications(date_str: str) -> pl.DataFrame:
    """Fetch all AI/semiconductor patent applications in the 365-day window ending date_str.

    Returns DataFrame with _APP_SCHEMA. Empty DataFrame if no results.
    """
    start = _lookback_start(date_str)
    records: list[dict] = []
    page = 1

    while True:
        payload = {
            "q": {"_and": [
                {"_gte": {"app_date": start}},
                {"_lte": {"app_date": date_str}},
                {"_or": [{"cpc_group_id": c} for c in _CPC_CODES]},
            ]},
            "f": ["app_id", "assignee_organization", "cpc_group_id", "app_date"],
            "o": {"per_page": _PER_PAGE, "page": page},
        }
        resp = requests.post(_APPS_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("applications", [])
        records.extend(batch)

        total = data.get("total_app_count", 0)
        if len(records) >= total or not batch:
            break
        time.sleep(_SLEEP_BETWEEN_PAGES)
        page += 1

    if not records:
        return pl.DataFrame(schema=_APP_SCHEMA)

    run_date = datetime.date.fromisoformat(date_str)
    rows = []
    for r in records:
        try:
            filing = datetime.date.fromisoformat(r["app_date"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "date": run_date,
            "assignee_name": r.get("assignee_organization") or "",
            "app_id": r.get("app_id") or "",
            "cpc_group": r.get("cpc_group_id") or "",
            "filing_date": filing,
        })

    return pl.DataFrame(rows, schema=_APP_SCHEMA)


def fetch_grants(date_str: str) -> pl.DataFrame:
    """Fetch all AI/semiconductor granted patents in the 365-day window ending date_str.

    Returns DataFrame with _GRANT_SCHEMA. Empty DataFrame if no results.
    """
    start = _lookback_start(date_str)
    records: list[dict] = []
    page = 1

    while True:
        payload = {
            "q": {"_and": [
                {"_gte": {"patent_date": start}},
                {"_lte": {"patent_date": date_str}},
                {"_or": [{"cpc_group_id": c} for c in _CPC_CODES]},
            ]},
            "f": ["patent_id", "assignee_organization", "cpc_group_id", "patent_date", "cited_by_count"],
            "o": {"per_page": _PER_PAGE, "page": page},
        }
        resp = requests.post(_GRANTS_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("patents", [])
        records.extend(batch)

        total = data.get("total_patent_count", 0)
        if len(records) >= total or not batch:
            break
        time.sleep(_SLEEP_BETWEEN_PAGES)
        page += 1

    if not records:
        return pl.DataFrame(schema=_GRANT_SCHEMA)

    run_date = datetime.date.fromisoformat(date_str)
    rows = []
    for r in records:
        try:
            grant_d = datetime.date.fromisoformat(r["patent_date"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "date": run_date,
            "assignee_name": r.get("assignee_organization") or "",
            "patent_id": r.get("patent_id") or "",
            "cpc_group": r.get("cpc_group_id") or "",
            "grant_date": grant_d,
            "forward_citation_count": int(r.get("cited_by_count") or 0),
        })

    return pl.DataFrame(rows, schema=_GRANT_SCHEMA)


def ingest_uspto(date_str: str, apps_dir: Path, grants_dir: Path) -> None:
    """Fetch and persist patent applications and grants for date_str.

    Skips download if both directories already have a same-ISO-week snapshot.
    Writes nothing when results are empty.
    """
    if _same_iso_week(apps_dir, date_str) and _same_iso_week(grants_dir, date_str):
        _LOG.info("USPTO: same ISO week snapshot exists — skipping download for %s", date_str)
        return

    _LOG.info("USPTO: fetching applications for %s (365d lookback)", date_str)
    apps_df = fetch_applications(date_str)
    if not apps_df.is_empty():
        out = apps_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        apps_df.write_parquet(out / "apps.parquet", compression="snappy")
        _LOG.info("USPTO: wrote %d application rows to %s", len(apps_df), out)

    _LOG.info("USPTO: fetching grants for %s (365d lookback)", date_str)
    grants_df = fetch_grants(date_str)
    if not grants_df.is_empty():
        out = grants_dir / f"date={date_str}"
        out.mkdir(parents=True, exist_ok=True)
        grants_df.write_parquet(out / "grants.parquet", compression="snappy")
        _LOG.info("USPTO: wrote %d grant rows to %s", len(grants_df), out)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    date_str = datetime.date.today().isoformat()
    base = Path("data/raw/patents")
    ingest_uspto(date_str, base / "applications", base / "grants")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_uspto_ingestion.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add ingestion/uspto_ingestion.py tests/test_uspto_ingestion.py
git commit -m "feat: add USPTO patent ingestion from PatentsView v2 API"
```

---

## Task 2: Patent Features

**Files:**
- Create: `processing/patent_features.py`
- Create: `tests/test_patent_features.py`

- [ ] **Step 1: Write the 10 failing tests**

Create `tests/test_patent_features.py`:

```python
import datetime
import polars as pl
import pytest
from pathlib import Path


_APP_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "app_id": pl.Utf8,
    "cpc_group": pl.Utf8, "filing_date": pl.Date,
}
_GRANT_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "patent_id": pl.Utf8,
    "cpc_group": pl.Utf8, "grant_date": pl.Date, "forward_citation_count": pl.Int32,
}


def _write_apps(apps_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = apps_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_APP_SCHEMA).write_parquet(out / "apps.parquet")


def _write_grants(grants_dir: Path, rows: list[dict]) -> None:
    date_str = rows[0]["date"].isoformat()
    out = grants_dir / f"date={date_str}"
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=_GRANT_SCHEMA).write_parquet(out / "grants.parquet")


def _query_df(ticker: str, date: datetime.date) -> pl.DataFrame:
    return pl.DataFrame({"ticker": [ticker], "date": [date]},
                        schema={"ticker": pl.Utf8, "date": pl.Date})


# Query date used throughout tests
_QUERY_DATE = datetime.date(2024, 4, 1)
# 70 days before query → inside 90d window
_RECENT = datetime.date(2024, 1, 22)
# 100 days before query → inside prior 90d window (91–180d ago)
_PRIOR = datetime.date(2023, 12, 22)
# 400 days before query → outside 365d window
_OLD = datetime.date(2023, 2, 25)


def test_patent_app_count_90d_window(tmp_path):
    """patent_app_count_90d counts only apps filed within 90d, not older ones."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},   # in window
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A2", "cpc_group": "G06N", "filing_date": _OLD},      # outside 365d
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_app_count_90d"][0] == 1.0


def test_patent_app_momentum_positive(tmp_path):
    """patent_app_momentum is positive when recent 90d count > prior 90d count."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    # 2 apps in recent 90d, 1 app in prior 90d (91–180d window)
    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A2", "cpc_group": "H01L", "filing_date": _RECENT},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A3", "cpc_group": "G06N", "filing_date": _PRIOR},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_app_momentum"][0] > 0


def test_patent_grant_count_365d(tmp_path):
    """patent_grant_count_365d sums grants over full 365d window."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    grant_date_in = datetime.date(2023, 4, 15)   # 351 days ago — inside 365d
    grant_date_out = datetime.date(2023, 3, 15)  # 382 days ago — outside 365d

    _write_grants(grants_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P1", "cpc_group": "G06N", "grant_date": grant_date_in,
         "forward_citation_count": 0},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P2", "cpc_group": "G06N", "grant_date": grant_date_out,
         "forward_citation_count": 0},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_grant_count_365d"][0] == 1.0


def test_patent_grant_rate_zero_safe(tmp_path):
    """patent_grant_rate_365d uses GREATEST(apps, 1) denominator — no division by zero."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    # 1 grant, 0 apps → rate = 1/1 = 1.0 (not Inf)
    _write_grants(grants_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P1", "cpc_group": "G06N",
         "grant_date": datetime.date(2024, 1, 1), "forward_citation_count": 0},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_grant_rate_365d"][0] == pytest.approx(1.0)


def test_patent_ai_cpc_share_isolates_g06n(tmp_path):
    """patent_ai_cpc_share_90d counts only G06N, excludes H01L/G06F/G11C."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},   # AI
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A2", "cpc_group": "H01L", "filing_date": _RECENT},   # not AI
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "app_id": "A3", "cpc_group": "G06F", "filing_date": _RECENT},   # not AI
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_ai_cpc_share_90d"][0] == pytest.approx(1.0 / 3.0)


def test_patent_citation_count_365d(tmp_path):
    """patent_citation_count_365d sums forward citations within 365d grant window."""
    from processing.patent_features import join_patent_features

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    _write_grants(grants_dir, [
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P1", "cpc_group": "G06N",
         "grant_date": datetime.date(2023, 12, 1), "forward_citation_count": 5},
        {"date": _QUERY_DATE, "assignee_name": "NVIDIA Corporation",
         "patent_id": "P2", "cpc_group": "H01L",
         "grant_date": datetime.date(2024, 1, 15), "forward_citation_count": 3},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    assert df["patent_citation_count_365d"][0] == 8.0


def test_join_adds_exactly_6_columns(tmp_path):
    """join_patent_features adds exactly 6 new columns to df."""
    from processing.patent_features import join_patent_features, USPTO_PATENT_FEATURE_COLS

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"
    input_df = _query_df("NVDA", _QUERY_DATE)
    result = join_patent_features(input_df, apps_dir, grants_dir)

    added = [c for c in result.columns if c not in input_df.columns]
    assert sorted(added) == sorted(USPTO_PATENT_FEATURE_COLS)
    assert len(added) == 6


def test_ticker_no_patents_zero_fill(tmp_path):
    """Ticker with no matching patents gets zero-filled for all 6 features."""
    from processing.patent_features import join_patent_features, USPTO_PATENT_FEATURE_COLS

    apps_dir = tmp_path / "applications"
    grants_dir = tmp_path / "grants"

    # Write data for a different company — NVDA gets no match
    _write_apps(apps_dir, [
        {"date": _QUERY_DATE, "assignee_name": "Some Unrelated Corp",
         "app_id": "A1", "cpc_group": "G06N", "filing_date": _RECENT},
    ])

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    for col in USPTO_PATENT_FEATURE_COLS:
        assert df[col][0] == 0.0, f"{col} should be 0.0 when no patents match"


def test_missing_apps_dir_zero_fill(tmp_path):
    """Missing apps_dir returns zero-filled features."""
    from processing.patent_features import join_patent_features, USPTO_PATENT_FEATURE_COLS

    apps_dir = tmp_path / "does_not_exist" / "applications"
    grants_dir = tmp_path / "does_not_exist" / "grants"

    df = join_patent_features(_query_df("NVDA", _QUERY_DATE), apps_dir, grants_dir)
    for col in USPTO_PATENT_FEATURE_COLS:
        assert df[col][0] == 0.0


def test_uspto_col_names_correct():
    """USPTO_PATENT_FEATURE_COLS contains exactly the 6 expected names."""
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    expected = {
        "patent_app_count_90d",
        "patent_app_momentum",
        "patent_grant_count_365d",
        "patent_grant_rate_365d",
        "patent_ai_cpc_share_90d",
        "patent_citation_count_365d",
    }
    assert set(USPTO_PATENT_FEATURE_COLS) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_patent_features.py -v 2>&1 | head -30
```

Expected: 10 failures with `ModuleNotFoundError: No module named 'processing.patent_features'`

- [ ] **Step 3: Implement `processing/patent_features.py`**

Create `processing/patent_features.py`:

```python
"""USPTO patent signal features.

Features (USPTO_PATENT_FEATURE_COLS):
    patent_app_count_90d      — count of AI/semiconductor patent applications in 90d
    patent_app_momentum       — recent 90d apps minus prior 90d apps (R&D acceleration)
    patent_grant_count_365d   — count of granted patents in 365d
    patent_grant_rate_365d    — grants_365d / max(apps_365d, 1) — IP portfolio quality
    patent_ai_cpc_share_90d   — fraction of 90d apps in G06N (AI/ML) vs all CPC codes
    patent_citation_count_365d— forward citations on patents granted in 365d window

Ticker matching: reuses _TICKER_NAME_MAP and _normalize_name from gov_behavioral_features.
All features zero-filled when data is absent.

Tier routing: medium + long only (patent cycles too slow for 5d/20d horizons).
"""
from __future__ import annotations

import difflib
import logging
from pathlib import Path

import duckdb
import polars as pl

from processing.gov_behavioral_features import _TICKER_NAME_MAP, _normalize_name

_LOG = logging.getLogger(__name__)

USPTO_PATENT_FEATURE_COLS: list[str] = [
    "patent_app_count_90d",
    "patent_app_momentum",
    "patent_grant_count_365d",
    "patent_grant_rate_365d",
    "patent_ai_cpc_share_90d",
    "patent_citation_count_365d",
]

_APP_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "app_id": pl.Utf8,
    "cpc_group": pl.Utf8, "filing_date": pl.Date,
}
_GRANT_SCHEMA = {
    "date": pl.Date, "assignee_name": pl.Utf8, "patent_id": pl.Utf8,
    "cpc_group": pl.Utf8, "grant_date": pl.Date, "forward_citation_count": pl.Int32,
}


def _load_apps(apps_dir: Path) -> pl.DataFrame:
    files = sorted(apps_dir.glob("date=*/apps.parquet")) if apps_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_APP_SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files])


def _load_grants(grants_dir: Path) -> pl.DataFrame:
    files = sorted(grants_dir.glob("date=*/grants.parquet")) if grants_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_GRANT_SCHEMA)
    return pl.concat([pl.read_parquet(f) for f in files])


def _build_assignee_to_ticker(assignee_names: list[str], tickers: list[str]) -> dict[str, str]:
    """Return {assignee_name: ticker} via normalize + difflib fuzzy match (cutoff 0.85)."""
    norm_to_assignee = {_normalize_name(n): n for n in assignee_names}
    result: dict[str, str] = {}
    for ticker in tickers:
        search_name = _TICKER_NAME_MAP.get(ticker)
        if not search_name:
            continue
        matches = difflib.get_close_matches(
            _normalize_name(search_name), norm_to_assignee.keys(), n=1, cutoff=0.85
        )
        if matches:
            result[norm_to_assignee[matches[0]]] = ticker
    return result


def join_patent_features(
    df: pl.DataFrame,
    apps_dir: Path,
    grants_dir: Path,
) -> pl.DataFrame:
    """Left-join USPTO patent features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'ticker' (Utf8) and 'date' (Date) columns.
        apps_dir: Root of data/raw/patents/applications/ Hive tree.
        grants_dir: Root of data/raw/patents/grants/ Hive tree.

    Returns:
        df with USPTO_PATENT_FEATURE_COLS appended (Float64). Zero-filled.
    """
    from ingestion.ticker_registry import TICKERS

    _empty_schema = {"ticker": pl.Utf8, "date": pl.Date, **{c: pl.Float64 for c in USPTO_PATENT_FEATURE_COLS}}

    raw_apps = _load_apps(apps_dir)
    raw_grants = _load_grants(grants_dir)

    all_assignees = set()
    if not raw_apps.is_empty():
        all_assignees.update(raw_apps["assignee_name"].unique().to_list())
    if not raw_grants.is_empty():
        all_assignees.update(raw_grants["assignee_name"].unique().to_list())

    if not all_assignees:
        for col in USPTO_PATENT_FEATURE_COLS:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        return df

    assignee_to_ticker = _build_assignee_to_ticker(list(all_assignees), TICKERS)
    if not assignee_to_ticker:
        for col in USPTO_PATENT_FEATURE_COLS:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
        return df

    mapping_df = pl.DataFrame(
        {"assignee_name": list(assignee_to_ticker.keys()),
         "ticker": list(assignee_to_ticker.values())},
        schema={"assignee_name": pl.Utf8, "ticker": pl.Utf8},
    )

    # Map assignee_name → ticker in both raw frames
    if not raw_apps.is_empty():
        apps = raw_apps.join(mapping_df, on="assignee_name", how="left").filter(
            pl.col("ticker").is_not_null()
        )
    else:
        apps = pl.DataFrame(schema={**_APP_SCHEMA, "ticker": pl.Utf8})

    if not raw_grants.is_empty():
        grants = raw_grants.join(mapping_df, on="assignee_name", how="left").filter(
            pl.col("ticker").is_not_null()
        )
    else:
        grants = pl.DataFrame(schema={**_GRANT_SCHEMA, "ticker": pl.Utf8})

    query_pairs = df.select(["ticker", "date"]).unique()

    with duckdb.connect() as con:
        con.register("apps", apps.to_arrow())
        con.register("grants", grants.to_arrow())
        con.register("query_pairs", query_pairs.to_arrow())

        result = con.execute("""
            SELECT
                q.ticker,
                q.date,

                -- patent_app_count_90d
                COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                         AND a.filing_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS patent_app_count_90d,

                -- patent_app_momentum = recent_90d - prior_90d (91-180d)
                COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                         AND a.filing_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                - COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 180 DAY
                         AND a.filing_date < q.date - INTERVAL 90 DAY
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS patent_app_momentum,

                -- patent_ai_cpc_share_90d = G06N apps / total apps (90d)
                COALESCE(
                    CAST(SUM(CASE
                        WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                             AND a.filing_date <= q.date
                             AND a.cpc_group = 'G06N'
                        THEN 1 ELSE 0 END) AS DOUBLE)
                    / GREATEST(CAST(SUM(CASE
                        WHEN a.filing_date >= q.date - INTERVAL 90 DAY
                             AND a.filing_date <= q.date
                        THEN 1 ELSE 0 END) AS DOUBLE), 1.0),
                0.0) AS patent_ai_cpc_share_90d,

                -- apps_365d (needed for grant rate denominator)
                COALESCE(CAST(SUM(CASE
                    WHEN a.filing_date >= q.date - INTERVAL 365 DAY
                         AND a.filing_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS _apps_365d

            FROM query_pairs q
            LEFT JOIN apps a
                ON a.ticker = q.ticker
                AND a.filing_date <= q.date
                AND a.filing_date >= q.date - INTERVAL 180 DAY
            GROUP BY q.ticker, q.date
        """).pl()

        grant_result = con.execute("""
            SELECT
                q.ticker,
                q.date,

                -- patent_grant_count_365d
                COALESCE(CAST(SUM(CASE
                    WHEN g.grant_date >= q.date - INTERVAL 365 DAY
                         AND g.grant_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS patent_grant_count_365d,

                -- patent_citation_count_365d
                COALESCE(SUM(CASE
                    WHEN g.grant_date >= q.date - INTERVAL 365 DAY
                         AND g.grant_date <= q.date
                    THEN CAST(g.forward_citation_count AS DOUBLE) ELSE 0.0 END), 0.0)
                    AS patent_citation_count_365d,

                -- grants_365d for rate calculation
                COALESCE(CAST(SUM(CASE
                    WHEN g.grant_date >= q.date - INTERVAL 365 DAY
                         AND g.grant_date <= q.date
                    THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS _grants_365d

            FROM query_pairs q
            LEFT JOIN grants g
                ON g.ticker = q.ticker
                AND g.grant_date <= q.date
                AND g.grant_date >= q.date - INTERVAL 365 DAY
            GROUP BY q.ticker, q.date
        """).pl()

    # Compute grant_rate_365d = grants_365d / max(apps_365d, 1)
    combined = result.join(grant_result, on=["ticker", "date"], how="left")
    combined = combined.with_columns(
        (pl.col("_grants_365d") / pl.col("_apps_365d").clip(lower_bound=1.0))
        .alias("patent_grant_rate_365d")
    ).drop(["_apps_365d", "_grants_365d"])

    # Left-join back to original df
    df = df.join(combined, on=["ticker", "date"], how="left")

    # Zero-fill backstop
    for col in USPTO_PATENT_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_patent_features.py -v
```

Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add processing/patent_features.py tests/test_patent_features.py
git commit -m "feat: add USPTO patent signal features (6 cols, DuckDB rolling windows)"
```

---

## Task 3: Model Integration

**Files:**
- Modify: `models/train.py`
- Modify: `models/inference.py`
- Modify: `tests/test_train.py`
- Modify: `tools/run_refresh.sh`

- [ ] **Step 1: Write the 6 failing tests in `tests/test_train.py`**

Append to `tests/test_train.py` (after the last test `test_tier_medium_equals_feature_cols_after_gov_integration`):

```python
def test_feature_cols_has_73_elements():
    """FEATURE_COLS must have exactly 73 elements after USPTO integration."""
    from models.train import FEATURE_COLS
    assert len(FEATURE_COLS) == 73, f"Expected 73 features, got {len(FEATURE_COLS)}"


def test_feature_cols_includes_uspto_patent():
    """FEATURE_COLS must contain all 6 USPTO_PATENT_FEATURE_COLS after integration."""
    from models.train import FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    assert len(USPTO_PATENT_FEATURE_COLS) == 6
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"
    assert len(FEATURE_COLS) == 73


def test_uspto_patent_cols_absent_from_short_tier():
    """USPTO cols must not appear in short tier — patent cycles too slow for 5d/20d."""
    from models.train import TIER_FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_uspto_patent_cols_in_medium_tier():
    """USPTO cols must be present in medium tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    medium = TIER_FEATURE_COLS["medium"]
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_uspto_patent_cols_in_long_tier():
    """USPTO cols must be present in long tier."""
    from models.train import TIER_FEATURE_COLS
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    long_cols = TIER_FEATURE_COLS["long"]
    for col in USPTO_PATENT_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_uspto_patent_col_names_correct():
    """USPTO_PATENT_FEATURE_COLS must contain exactly the 6 expected column names."""
    from processing.patent_features import USPTO_PATENT_FEATURE_COLS
    expected = {
        "patent_app_count_90d",
        "patent_app_momentum",
        "patent_grant_count_365d",
        "patent_grant_rate_365d",
        "patent_ai_cpc_share_90d",
        "patent_citation_count_365d",
    }
    assert set(USPTO_PATENT_FEATURE_COLS) == expected
```

Also update the existing `test_feature_cols_has_67_elements` test (line 393):

```python
def test_feature_cols_has_67_elements():
    """FEATURE_COLS must have exactly 73 elements after full feature set integration."""
    from models.train import FEATURE_COLS
    assert len(FEATURE_COLS) == 73, f"Expected 73 features, got {len(FEATURE_COLS)}"
```

- [ ] **Step 2: Run tests to verify the new 6 fail (old count test also fails)**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/test_train.py -k "test_feature_cols_has_67 or test_feature_cols_includes_uspto or test_uspto" -v 2>&1 | head -30
```

Expected: 7 failures (6 new USPTO tests + updated count test)

- [ ] **Step 3: Update `models/train.py`**

Add import line (after the `from processing.gov_behavioral_features` import at line 42):

```python
from processing.patent_features import USPTO_PATENT_FEATURE_COLS, join_patent_features
```

Update FEATURE_COLS block (after the `+ GOV_BEHAVIORAL_FEATURE_COLS` line):

```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS + ENERGY_FEATURE_COLS
    + SUPPLY_CHAIN_FEATURE_COLS + FX_FEATURE_COLS
    + CYBER_THREAT_FEATURE_COLS  # 48 → 55 features total
    + OPTIONS_FEATURE_COLS       # 55 → 61 features total
    + GOV_BEHAVIORAL_FEATURE_COLS  # 61 → 67 features total
    + USPTO_PATENT_FEATURE_COLS    # 67 → 73 features total
)
```

Update TIER_FEATURE_COLS (medium comment and long list):

```python
TIER_FEATURE_COLS: dict[str, list[str]] = {
    "short": (
        PRICE_FEATURE_COLS
        + SENTIMENT_FEATURE_COLS
        + INSIDER_FEATURE_COLS
        + SHORT_INTEREST_FEATURE_COLS
        + _CYBER_THREAT_SHORT_COLS   # 5 features: *_7d only
        + OPTIONS_FEATURE_COLS       # all 6 options features
    ),
    "medium": list(FEATURE_COLS),    # all 73 features (copy to avoid shared mutable reference)
    "long": (
        PRICE_FEATURE_COLS
        + FUND_FEATURE_COLS
        + EARNINGS_FEATURE_COLS
        + GRAPH_FEATURE_COLS
        + OWNERSHIP_FEATURE_COLS
        + ENERGY_FEATURE_COLS
        + SUPPLY_CHAIN_FEATURE_COLS
        + FX_FEATURE_COLS
        + GOV_BEHAVIORAL_FEATURE_COLS  # contract award cycles relevant at year+ horizons
        + USPTO_PATENT_FEATURE_COLS    # patent grant cycles relevant at year+ horizons
        # cyber threat features excluded — noise at year+ horizons
    ),
}
```

Add join call in `build_training_dataset` after the GOV join (after line 337):

```python
    # Join USPTO patent features (applications + grants from PatentsView v2)
    patents_apps_dir = fundamentals_dir.parent.parent / "patents" / "applications"
    patents_grants_dir = fundamentals_dir.parent.parent / "patents" / "grants"
    df = join_patent_features(df, patents_apps_dir, patents_grants_dir)
```

- [ ] **Step 4: Update `models/inference.py`**

Add import (after the `from processing.gov_behavioral_features import join_gov_behavioral_features` line):

```python
from processing.patent_features import join_patent_features
```

Update docstring in `_build_feature_df`:

```python
    """Build the 73-feature DataFrame for all tickers on date_str."""
```

Add join call after the GOV join (after `df = join_gov_behavioral_features(df, gov_contracts_dir, gov_ferc_dir)`):

```python
    patents_apps_dir = data_dir / "patents" / "applications"
    patents_grants_dir = data_dir / "patents" / "grants"
    df = join_patent_features(df, patents_apps_dir, patents_grants_dir)
```

- [ ] **Step 5: Update `tools/run_refresh.sh`**

Change the final echo block from:

```bash
echo ""
echo "=== 11/11  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
```

To:

```bash
echo ""
echo "=== 11/11  FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py

echo ""
echo "=== 12/12  USPTO patent applications + grants ==="
python ingestion/uspto_ingestion.py

echo ""
echo "=== Refresh complete at $(date) ==="
```

Also update all step counts from `X/11` to `X/12` in the file header echo lines.

- [ ] **Step 6: Run the full test suite**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
pytest tests/ -m "not integration" -v 2>&1 | tail -20
```

Expected: all tests pass including the 7 new USPTO tests in `test_train.py`

- [ ] **Step 7: Commit**

```bash
cd "/Users/phila/Documents/AI Projects/AI Market Prediction/AI-PRED"
git add models/train.py models/inference.py tests/test_train.py tools/run_refresh.sh
git commit -m "feat: wire USPTO patent features into model pipeline (FEATURE_COLS 67→73)"
```
