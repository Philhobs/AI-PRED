# Government Behavioral Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SAM.gov federal contract awards + FERC interconnection queue ingestion and derive 6 `GOV_BEHAVIORAL_FEATURE_COLS`, growing FEATURE_COLS from 61 → 67.

**Architecture:** Two new ingestion modules (Protocol pattern for SAM.gov, standalone for FERC) write Hive-partitioned parquet. A feature module uses DuckDB window functions for date-range rolling aggregations. Mixed join: 3 ticker-specific columns on (ticker, date), 3 market-wide on date. Medium + long tiers only.

**Tech Stack:** Python 3.11+, Polars, DuckDB, requests, pandas/openpyxl (FERC Excel parsing), difflib (ticker name fuzzy match), pytest with unittest.mock.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ingestion/sam_gov_ingestion.py` | Create | Protocol + SamGovSource + ingest_sam_gov() |
| `ingestion/ferc_queue_ingestion.py` | Create | Download LBL Excel + parse + staleness check |
| `processing/gov_behavioral_features.py` | Create | 6 feature columns + join_gov_behavioral_features() |
| `tests/test_sam_gov_ingestion.py` | Create | 7 tests for SAM.gov ingestion |
| `tests/test_ferc_queue_ingestion.py` | Create | 5 tests for FERC ingestion |
| `tests/test_gov_behavioral_features.py` | Create | 10 tests for feature computation |
| `models/train.py` | Modify | Import + FEATURE_COLS 61→67 + TIER_FEATURE_COLS + join call |
| `models/inference.py` | Modify | Import + join call + docstring update |
| `tests/test_train.py` | Modify | Update len assertion 61→67 + 6 new GOV tests |
| `tools/run_refresh.sh` | Modify | Add 2 ingestion lines |

---

## Task 1: SAM.gov Ingestion

**Files:**
- Create: `ingestion/sam_gov_ingestion.py`
- Create: `tests/test_sam_gov_ingestion.py`

- [ ] **Step 1: Write the 7 failing tests**

Create `tests/test_sam_gov_ingestion.py`:

```python
import datetime
import os
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_response(awards: list[dict], total: int) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"opportunitiesData": awards, "totalRecords": total}
    return mock


_ONE_AWARD = {
    "awardee": {"name": "NVIDIA Corporation", "ueiSAM": "ABC123"},
    "award": {"amount": 1_000_000},
    "naicsCode": "518210",
    "department": "Department of Defense",
    "organizationHierarchy": [{"name": "DOD"}],
}


def test_schema_correct(tmp_path):
    """Parquet written by ingest_sam_gov matches _CONTRACT_SCHEMA."""
    from ingestion.sam_gov_ingestion import ingest_sam_gov, _CONTRACT_SCHEMA

    class _FakeSource:
        def fetch(self, date_str: str) -> pl.DataFrame:
            return pl.DataFrame([{
                "date": datetime.date(2024, 1, 15),
                "awardee_name": "NVIDIA Corporation",
                "uei": "ABC123",
                "contract_value_usd": 1_000_000.0,
                "naics_code": "518210",
                "agency": "Department of Defense",
            }], schema=_CONTRACT_SCHEMA)

    output_dir = tmp_path / "gov_contracts"
    ingest_sam_gov("2024-01-15", output_dir, source=_FakeSource())

    parquet = output_dir / "date=2024-01-15" / "awards.parquet"
    assert parquet.exists()
    df = pl.read_parquet(parquet)
    assert df.schema == _CONTRACT_SCHEMA
    assert len(df) == 1


def test_pagination_followed(tmp_path):
    """SamGovSource fetches all pages when totalRecords > limit."""
    from ingestion.sam_gov_ingestion import SamGovSource

    page1 = [_ONE_AWARD.copy() for _ in range(100)]
    page2 = [_ONE_AWARD.copy() for _ in range(50)]

    responses = [
        _make_response(page1, total=150),
        _make_response(page2, total=150),
    ]

    with patch.dict(os.environ, {"SAM_GOV_API_KEY": "test-key"}):
        with patch("ingestion.sam_gov_ingestion.requests.get", side_effect=responses):
            with patch("ingestion.sam_gov_ingestion.time.sleep"):
                df = SamGovSource().fetch("2024-01-15")

    assert len(df) == 150


def test_rate_limit_sleep_called_between_pages(tmp_path):
    """time.sleep(6.0) is called exactly once when two pages are fetched."""
    from ingestion.sam_gov_ingestion import SamGovSource

    page1 = [_ONE_AWARD.copy() for _ in range(100)]
    page2 = [_ONE_AWARD.copy() for _ in range(10)]

    responses = [
        _make_response(page1, total=110),
        _make_response(page2, total=110),
    ]

    with patch.dict(os.environ, {"SAM_GOV_API_KEY": "test-key"}):
        with patch("ingestion.sam_gov_ingestion.requests.get", side_effect=responses):
            with patch("ingestion.sam_gov_ingestion.time.sleep") as mock_sleep:
                SamGovSource().fetch("2024-01-15")

    mock_sleep.assert_called_once_with(6.0)


def test_empty_awards_no_file_written(tmp_path):
    """When source returns empty DataFrame, no parquet file is written."""
    from ingestion.sam_gov_ingestion import ingest_sam_gov, _CONTRACT_SCHEMA

    class _EmptySource:
        def fetch(self, date_str: str) -> pl.DataFrame:
            return pl.DataFrame(schema=_CONTRACT_SCHEMA)

    output_dir = tmp_path / "gov_contracts"
    ingest_sam_gov("2024-01-15", output_dir, source=_EmptySource())

    assert not (output_dir / "date=2024-01-15").exists()


def test_missing_api_key_raises(tmp_path):
    """SamGovSource.fetch raises RuntimeError when SAM_GOV_API_KEY is not set."""
    from ingestion.sam_gov_ingestion import SamGovSource

    env_without_key = {k: v for k, v in os.environ.items() if k != "SAM_GOV_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True):
        with pytest.raises(RuntimeError, match="SAM_GOV_API_KEY"):
            SamGovSource().fetch("2024-01-15")


def test_naics_filter_in_request(tmp_path):
    """SamGovSource includes all 5 NAICS codes in the request params."""
    from ingestion.sam_gov_ingestion import SamGovSource, _NAICS_CODES

    with patch.dict(os.environ, {"SAM_GOV_API_KEY": "test-key"}):
        with patch("ingestion.sam_gov_ingestion.requests.get") as mock_get:
            mock_get.return_value = _make_response([], total=0)
            with patch("ingestion.sam_gov_ingestion.time.sleep"):
                SamGovSource().fetch("2024-01-15")

    call_params = mock_get.call_args[1]["params"]
    assert call_params["naicsCode"] == _NAICS_CODES
    assert "541511" in call_params["naicsCode"]
    assert "334413" in call_params["naicsCode"]


def test_date_range_covers_90_days(tmp_path):
    """awardDateRange parameter spans exactly 90 days ending on date_str."""
    from ingestion.sam_gov_ingestion import SamGovSource

    with patch.dict(os.environ, {"SAM_GOV_API_KEY": "test-key"}):
        with patch("ingestion.sam_gov_ingestion.requests.get") as mock_get:
            mock_get.return_value = _make_response([], total=0)
            with patch("ingestion.sam_gov_ingestion.time.sleep"):
                SamGovSource().fetch("2024-04-01")

    call_params = mock_get.call_args[1]["params"]
    date_range = call_params["awardDateRange"]
    start_str, end_str = date_range.split(",")
    start = datetime.date.fromisoformat(start_str.strip())
    end = datetime.date.fromisoformat(end_str.strip())
    assert (end - start).days == 90
    assert end == datetime.date(2024, 4, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sam_gov_ingestion.py -v
```

Expected: 7 × `ModuleNotFoundError: No module named 'ingestion.sam_gov_ingestion'`

- [ ] **Step 3: Write the implementation**

Create `ingestion/sam_gov_ingestion.py`:

```python
"""
Ingest federal contract award data from SAM.gov for AI/datacenter NAICS codes.

Raw storage: data/raw/gov_contracts/date=YYYY-MM-DD/awards.parquet
Schema: (date, awardee_name, uei, contract_value_usd, naics_code, agency)

NAICS codes: 541511, 541512, 541519 (IT services), 518210 (hosting), 334413 (semiconductors)

Requires: SAM_GOV_API_KEY environment variable (free key at sam.gov)
Rate limit: 10 requests/minute → time.sleep(6.0) between pages.

Usage:
    python ingestion/sam_gov_ingestion.py               # fetch today (rolling 90-day window)
    python ingestion/sam_gov_ingestion.py --date 2024-01-15
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

_LOG = logging.getLogger(__name__)

_SAM_GOV_BASE_URL = "https://api.sam.gov/opportunities/v2/search"
_NAICS_CODES = "541511,541512,541519,518210,334413"

_CONTRACT_SCHEMA = {
    "date": pl.Date,
    "awardee_name": pl.Utf8,
    "uei": pl.Utf8,
    "contract_value_usd": pl.Float64,
    "naics_code": pl.Utf8,
    "agency": pl.Utf8,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_CONTRACT_SCHEMA)


@runtime_checkable
class GovContractSource(Protocol):
    def fetch(self, date_str: str) -> pl.DataFrame:
        """Return AI/DC NAICS contract awards for rolling 90-day window ending date_str.

        Returns DataFrame matching _CONTRACT_SCHEMA.
        Returns empty DataFrame if no awards found.
        """
        ...


class SamGovSource:
    """Fetch contract awards from SAM.gov API (requires SAM_GOV_API_KEY)."""

    def fetch(self, date_str: str) -> pl.DataFrame:
        import requests

        api_key = os.environ.get("SAM_GOV_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SAM_GOV_API_KEY not set. Get a free key at https://sam.gov/content/duns-sam"
            )

        as_of = datetime.date.fromisoformat(date_str)
        start = as_of - datetime.timedelta(days=90)

        rows: list[dict] = []
        offset = 0
        limit = 100

        while True:
            params = {
                "api_key": api_key,
                "limit": limit,
                "offset": offset,
                "awardDateRange": f"{start},{as_of}",
                "naicsCode": _NAICS_CODES,
            }
            resp = requests.get(_SAM_GOV_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            awards = data.get("opportunitiesData") or data.get("results") or []
            for award in awards:
                awardee = award.get("awardee") or {}
                award_info = award.get("award") or {}
                org = (award.get("organizationHierarchy") or [{}])[0]
                rows.append({
                    "date": as_of,
                    "awardee_name": awardee.get("name") or "",
                    "uei": awardee.get("ueiSAM") or "",
                    "contract_value_usd": float(award_info.get("amount") or 0),
                    "naics_code": award.get("naicsCode") or "",
                    "agency": award.get("department") or org.get("name") or "",
                })

            total = int(data.get("totalRecords") or 0)
            offset += len(awards)

            if not awards or offset >= total:
                break

            time.sleep(6.0)  # 10 req/min rate limit

        if not rows:
            return _empty()

        return pl.DataFrame(rows, schema=_CONTRACT_SCHEMA)


def ingest_sam_gov(
    date_str: str,
    output_dir: Path,
    source: GovContractSource | None = None,
) -> None:
    """Fetch contract awards and write Hive-partitioned parquet.

    Dates with no awards produce no file (silently skipped).
    """
    if source is None:
        source = SamGovSource()

    df = source.fetch(date_str)
    if not df.is_empty():
        out_dir = output_dir / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / "awards.parquet", compression="snappy")
        _LOG.info("Wrote %d contract awards for %s", len(df), date_str)
    else:
        _LOG.debug("No contract awards for %s — skipping", date_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest SAM.gov contract awards")
    parser.add_argument(
        "--date",
        default=str(datetime.date.today()),
        help="As-of date (YYYY-MM-DD). Fetches rolling 90-day window. Defaults to today.",
    )
    args = parser.parse_args()

    output_dir = Path("data/raw/gov_contracts")
    _LOG.info("Fetching SAM.gov awards as of %s", args.date)
    ingest_sam_gov(args.date, output_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_sam_gov_ingestion.py -v
```

Expected: 7 × PASSED

- [ ] **Step 5: Commit**

```bash
git add ingestion/sam_gov_ingestion.py tests/test_sam_gov_ingestion.py
git commit -m "feat: add SAM.gov contract awards ingestion

GovContractSource Protocol + SamGovSource (paged, 6s sleep between pages).
Writes data/raw/gov_contracts/date=YYYY-MM-DD/awards.parquet.
7 tests covering schema, pagination, rate-limit, missing key, NAICS filter, date range.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: FERC Queue Ingestion

**Files:**
- Create: `ingestion/ferc_queue_ingestion.py`
- Create: `tests/test_ferc_queue_ingestion.py`

- [ ] **Step 1: Write the 5 failing tests**

Create `tests/test_ferc_queue_ingestion.py`:

```python
import datetime
import io
import polars as pl
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_excel_bytes(rows: list[dict]) -> bytes:
    """Create in-memory Excel file from list of dicts."""
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _mock_download(content: bytes) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.content = content
    return mock


_VA_ROW = {
    "Queue Date": "2023-06-01",
    "Project Name": "Solar Farm VA",
    "MW": 250.0,
    "State": "VA",
    "Fuel": "Solar",
    "Status": "Active",
    "ISO": "PJM",
}

_FL_ROW = {
    "Queue Date": "2023-06-01",
    "Project Name": "Wind Farm FL",
    "MW": 100.0,
    "State": "FL",   # NOT a DC power state
    "Fuel": "Wind",
    "Status": "Active",
    "ISO": "SERC",
}


def test_schema_correct(tmp_path):
    """Output parquet matches _FERC_SCHEMA."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue, _FERC_SCHEMA

    content = _make_excel_bytes([_VA_ROW])
    with patch("ingestion.ferc_queue_ingestion.requests.get",
               return_value=_mock_download(content)):
        ingest_ferc_queue("2024-01-15", tmp_path)

    df = pl.read_parquet(tmp_path / "date=2024-01-15" / "queue.parquet")
    assert df.schema == _FERC_SCHEMA
    assert len(df) == 1


def test_same_half_year_skips_download(tmp_path):
    """Download is skipped when existing parquet has same half-year snapshot_date."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue

    existing_dir = tmp_path / "date=2024-01-10"
    existing_dir.mkdir()
    pl.DataFrame([{
        "snapshot_date": datetime.date(2024, 1, 10),
        "queue_date": datetime.date(2023, 6, 1),
        "project_name": "Solar VA",
        "mw": 250.0,
        "state": "VA",
        "fuel": "Solar",
        "status": "Active",
        "iso": "PJM",
    }]).write_parquet(existing_dir / "queue.parquet")

    with patch("ingestion.ferc_queue_ingestion.requests.get") as mock_get:
        ingest_ferc_queue("2024-03-01", tmp_path)   # same Jan–Jun half-year
        mock_get.assert_not_called()


def test_state_filter_keeps_only_dc_states(tmp_path):
    """Only rows with DC power states (VA, TX, OH, AZ, NV, OR, GA, WA) are stored."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue

    content = _make_excel_bytes([_VA_ROW, _FL_ROW])
    with patch("ingestion.ferc_queue_ingestion.requests.get",
               return_value=_mock_download(content)):
        ingest_ferc_queue("2024-01-15", tmp_path)

    df = pl.read_parquet(tmp_path / "date=2024-01-15" / "queue.parquet")
    assert len(df) == 1
    assert df["state"][0] == "VA"


def test_empty_sheet_no_file_written(tmp_path):
    """Empty Excel sheet (or all non-DC rows) produces no parquet file."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue

    content = _make_excel_bytes([_FL_ROW])   # only non-DC row
    with patch("ingestion.ferc_queue_ingestion.requests.get",
               return_value=_mock_download(content)):
        ingest_ferc_queue("2024-01-15", tmp_path)

    assert not (tmp_path / "date=2024-01-15").exists()


def test_bad_url_raises_runtime_error(tmp_path):
    """RuntimeError raised when download fails."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue
    import requests as _requests

    with patch("ingestion.ferc_queue_ingestion.requests.get") as mock_get:
        mock_get.side_effect = _requests.RequestException("connection refused")
        with pytest.raises(RuntimeError, match="Failed to download"):
            ingest_ferc_queue("2024-01-15", tmp_path, ferc_url="http://bad-url/")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ferc_queue_ingestion.py -v
```

Expected: 5 × `ModuleNotFoundError: No module named 'ingestion.ferc_queue_ingestion'`

- [ ] **Step 3: Write the implementation**

Create `ingestion/ferc_queue_ingestion.py`:

```python
"""
Ingest FERC interconnection queue data from Lawrence Berkeley National Lab.

Source: LBL "Queued Up" semi-annual Excel dataset
URL: https://emp.lbl.gov/sites/default/files/queued_up.xlsx
     Override via FERC_QUEUE_URL environment variable.

Raw storage: data/raw/ferc_queue/date=YYYY-MM-DD/queue.parquet
Schema: (snapshot_date, queue_date, project_name, mw, state, fuel, status, iso)
Filtered to DC power states only: VA, TX, OH, AZ, NV, OR, GA, WA

Staleness: LBL updates in January and July. Re-download is skipped when an existing
parquet has a snapshot_date in the same half-year (Jan–Jun or Jul–Dec).

Usage:
    python ingestion/ferc_queue_ingestion.py               # snapshot today
    python ingestion/ferc_queue_ingestion.py --date 2024-01-15
"""
from __future__ import annotations

import argparse
import datetime
import io
import logging
import os
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)

_FERC_QUEUE_DEFAULT_URL = "https://emp.lbl.gov/sites/default/files/queued_up.xlsx"

_DC_STATES = {"VA", "TX", "OH", "AZ", "NV", "OR", "GA", "WA"}

_FERC_SCHEMA = {
    "snapshot_date": pl.Date,
    "queue_date": pl.Date,
    "project_name": pl.Utf8,
    "mw": pl.Float64,
    "state": pl.Utf8,
    "fuel": pl.Utf8,
    "status": pl.Utf8,
    "iso": pl.Utf8,
}

# Flexible column mapping — LBL uses slightly different names across versions
_COLUMN_MAP: dict[str, str] = {
    "Queue Date": "queue_date",
    "Date Entered Queue": "queue_date",
    "Project Name": "project_name",
    "Project": "project_name",
    "MW": "mw",
    "MW (DC)": "mw",
    "Capacity (MW)": "mw",
    "State": "state",
    "State/Province/Territory": "state",
    "Fuel": "fuel",
    "Fuel Type": "fuel",
    "Type": "fuel",
    "Status": "status",
    "Interconnection Queue Status": "status",
    "ISO": "iso",
    "ISO/RTO": "iso",
    "Balancing Authority": "iso",
}


def _is_same_half_year(d1: datetime.date, d2: datetime.date) -> bool:
    """Return True if both dates fall in the same half-year."""
    def _half(d: datetime.date) -> tuple[int, int]:
        return (d.year, 1 if d.month <= 6 else 2)
    return _half(d1) == _half(d2)


def _parse_excel(content: bytes, snapshot_date: datetime.date) -> pl.DataFrame:
    """Parse LBL Excel bytes into a DC-state-filtered DataFrame."""
    import pandas as pd

    try:
        raw = pd.read_excel(io.BytesIO(content), engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(f"Failed to parse FERC queue Excel: {exc}") from exc

    if raw.empty:
        return pl.DataFrame(schema=_FERC_SCHEMA)

    # Apply column mapping — first match wins for each target column
    rename_map: dict[str, str] = {}
    target_used: set[str] = set()
    for src_col in raw.columns:
        target = _COLUMN_MAP.get(str(src_col))
        if target and target not in target_used:
            rename_map[src_col] = target
            target_used.add(target)

    raw = raw.rename(columns=rename_map)

    for col in ["queue_date", "project_name", "mw", "state", "fuel", "status", "iso"]:
        if col not in raw.columns:
            raw[col] = "" if col != "mw" else 0.0

    # Filter to DC power states only
    raw = raw[raw["state"].astype(str).isin(_DC_STATES)].copy()
    if raw.empty:
        return pl.DataFrame(schema=_FERC_SCHEMA)

    raw["snapshot_date"] = snapshot_date
    raw["queue_date"] = pd.to_datetime(raw["queue_date"], errors="coerce").dt.date
    raw["mw"] = pd.to_numeric(raw["mw"], errors="coerce").fillna(0.0)
    for col in ["project_name", "state", "fuel", "status", "iso"]:
        raw[col] = raw[col].fillna("").astype(str)

    df = pl.from_pandas(
        raw[["snapshot_date", "queue_date", "project_name",
             "mw", "state", "fuel", "status", "iso"]]
    )
    return df.cast(_FERC_SCHEMA)


def ingest_ferc_queue(
    date_str: str,
    output_dir: Path,
    ferc_url: str | None = None,
) -> None:
    """Download LBL FERC queue Excel and write Hive-partitioned parquet.

    Skips download if existing data is from the same half-year (Jan/Jul update cycle).
    """
    import requests

    url = ferc_url or os.environ.get("FERC_QUEUE_URL", _FERC_QUEUE_DEFAULT_URL)
    snapshot_date = datetime.date.fromisoformat(date_str)

    # Staleness check
    existing_files = sorted(output_dir.glob("date=*/queue.parquet"))
    if existing_files:
        try:
            existing = pl.read_parquet(existing_files[-1])
            if not existing.is_empty() and "snapshot_date" in existing.columns:
                existing_snap = existing["snapshot_date"][0]
                if _is_same_half_year(existing_snap, snapshot_date):
                    _LOG.info("FERC queue is current for %s half-year — skipping", date_str)
                    return
        except Exception:
            pass  # Re-download on any read error

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Failed to download FERC queue from {url}: {exc}") from exc

    df = _parse_excel(resp.content, snapshot_date)

    if df.is_empty():
        _LOG.debug("No DC-state FERC queue entries for %s — skipping", date_str)
        return

    out_dir = output_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "queue.parquet", compression="snappy")
    _LOG.info("Wrote %d FERC queue entries for %s", len(df), date_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest FERC interconnection queue from LBL")
    parser.add_argument(
        "--date",
        default=str(datetime.date.today()),
        help="Snapshot date (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    output_dir = Path("data/raw/ferc_queue")
    _LOG.info("Fetching FERC queue snapshot for %s", args.date)
    ingest_ferc_queue(args.date, output_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ferc_queue_ingestion.py -v
```

Expected: 5 × PASSED

- [ ] **Step 5: Commit**

```bash
git add ingestion/ferc_queue_ingestion.py tests/test_ferc_queue_ingestion.py
git commit -m "feat: add FERC interconnection queue ingestion

Downloads LBL Queued Up Excel; filters to DC power states (VA/TX/OH/AZ/NV/OR/GA/WA).
Staleness check skips re-download in same Jan-Jun or Jul-Dec half-year.
Writes data/raw/ferc_queue/date=YYYY-MM-DD/queue.parquet.
5 tests covering schema, staleness, state filter, empty sheet, bad URL.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Government Behavioral Features

**Files:**
- Create: `processing/gov_behavioral_features.py`
- Create: `tests/test_gov_behavioral_features.py`

- [ ] **Step 1: Write the 10 failing tests**

Create `tests/test_gov_behavioral_features.py`:

```python
import datetime
import pytest
import polars as pl
from pathlib import Path


def _write_contracts(contracts_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    date_str = rows[0]["date"].isoformat()
    out_dir = contracts_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={
        "date": pl.Date, "awardee_name": pl.Utf8, "uei": pl.Utf8,
        "contract_value_usd": pl.Float64, "naics_code": pl.Utf8, "agency": pl.Utf8,
    }).write_parquet(out_dir / "awards.parquet")


def _write_ferc(ferc_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    date_str = rows[0]["snapshot_date"].isoformat()
    out_dir = ferc_dir / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema={
        "snapshot_date": pl.Date, "queue_date": pl.Date,
        "project_name": pl.Utf8, "mw": pl.Float64,
        "state": pl.Utf8, "fuel": pl.Utf8, "status": pl.Utf8, "iso": pl.Utf8,
    }).write_parquet(out_dir / "queue.parquet")


def _input_df(tickers: list[str], dates: list[datetime.date]) -> pl.DataFrame:
    rows = [{"ticker": t, "date": d} for t in tickers for d in dates]
    return pl.DataFrame(rows, schema={"ticker": pl.Utf8, "date": pl.Date})


def test_gov_contract_value_90d_correct(tmp_path):
    """gov_contract_value_90d sums contract_value_usd over rolling 90-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d1 = datetime.date(2024, 1, 1)
    d2 = datetime.date(2024, 1, 15)

    _write_contracts(contracts_dir, [
        {"date": d1, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
    ])
    _write_contracts(contracts_dir, [
        {"date": d2, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 500.0, "naics_code": "518210", "agency": "DOD"},
    ])

    result = join_gov_behavioral_features(_input_df(["NVDA"], [d2]), contracts_dir, ferc_dir)
    val = result.filter(pl.col("ticker") == "NVDA")["gov_contract_value_90d"][0]
    assert val == pytest.approx(1500.0)   # d1 (1000) + d2 (500)


def test_gov_contract_count_90d_correct(tmp_path):
    """gov_contract_count_90d counts individual award rows in rolling 90-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d = datetime.date(2024, 1, 15)

    _write_contracts(contracts_dir, [
        {"date": d, "awardee_name": "Microsoft Corporation", "uei": "U2",
         "contract_value_usd": 500.0, "naics_code": "541511", "agency": "GSA"},
        {"date": d, "awardee_name": "Microsoft Corporation", "uei": "U2",
         "contract_value_usd": 250.0, "naics_code": "541512", "agency": "GSA"},
    ])

    result = join_gov_behavioral_features(_input_df(["MSFT"], [d]), contracts_dir, ferc_dir)
    count = result.filter(pl.col("ticker") == "MSFT")["gov_contract_count_90d"][0]
    assert count == pytest.approx(2.0)


def test_gov_contract_momentum_positive_when_recent_exceeds_prior(tmp_path):
    """gov_contract_momentum > 0 when recent 30d awards exceed prior 60d awards."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    today = datetime.date(2024, 3, 1)
    old = datetime.date(2024, 1, 10)    # 51 days ago — in prior 60d window
    recent = datetime.date(2024, 2, 20) # 10 days ago — in 30d window

    _write_contracts(contracts_dir, [
        {"date": old, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 100.0, "naics_code": "518210", "agency": "DOD"},
    ])
    _write_contracts(contracts_dir, [
        {"date": recent, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
    ])

    result = join_gov_behavioral_features(_input_df(["NVDA"], [today]), contracts_dir, ferc_dir)
    momentum = result.filter(pl.col("ticker") == "NVDA")["gov_contract_momentum"][0]
    assert momentum > 0


def test_gov_ai_spend_30d_sums_all_awardees(tmp_path):
    """gov_ai_spend_30d is market-wide: sums all awardee contract values in 30-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d = datetime.date(2024, 1, 15)

    _write_contracts(contracts_dir, [
        {"date": d, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
        {"date": d, "awardee_name": "Some Unmatched Company LLC", "uei": "U9",
         "contract_value_usd": 2000.0, "naics_code": "541511", "agency": "DHS"},
    ])

    result = join_gov_behavioral_features(_input_df(["NVDA"], [d]), contracts_dir, ferc_dir)
    spend = result["gov_ai_spend_30d"][0]
    assert spend == pytest.approx(3000.0)   # both awardees counted


def test_ferc_queue_mw_30d_sums_dc_state_mw(tmp_path):
    """ferc_queue_mw_30d sums MW for DC power states in rolling 30-day window."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    snapshot = datetime.date(2024, 1, 1)
    queue_d = datetime.date(2024, 1, 10)   # within 30d of query_date Jan 20

    _write_ferc(ferc_dir, [
        {"snapshot_date": snapshot, "queue_date": queue_d, "project_name": "Solar VA",
         "mw": 300.0, "state": "VA", "fuel": "Solar", "status": "Active", "iso": "PJM"},
        {"snapshot_date": snapshot, "queue_date": queue_d, "project_name": "Wind TX",
         "mw": 200.0, "state": "TX", "fuel": "Wind", "status": "Active", "iso": "ERCOT"},
    ])

    query_date = datetime.date(2024, 1, 20)
    result = join_gov_behavioral_features(_input_df(["NVDA"], [query_date]), contracts_dir, ferc_dir)
    mw = result["ferc_queue_mw_30d"][0]
    assert mw == pytest.approx(500.0)   # 300 + 200


def test_ferc_grid_constraint_score_above_one_when_spike(tmp_path):
    """ferc_grid_constraint_score > 1 when recent 30d MW well exceeds 12-month monthly average."""
    from processing.gov_behavioral_features import join_gov_behavioral_features
    import datetime as _dt

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    snapshot = _dt.date(2023, 1, 1)

    # 12 months of modest baseline: 10 MW/month
    baseline = [
        {"snapshot_date": snapshot,
         "queue_date": snapshot + _dt.timedelta(days=30 * i),
         "project_name": f"Base{i}", "mw": 10.0,
         "state": "VA", "fuel": "Solar", "status": "Active", "iso": "PJM"}
        for i in range(12)
    ]
    # Large spike in final month
    baseline.append({
        "snapshot_date": snapshot,
        "queue_date": _dt.date(2024, 1, 10),
        "project_name": "Spike", "mw": 500.0,
        "state": "VA", "fuel": "Solar", "status": "Active", "iso": "PJM",
    })
    _write_ferc(ferc_dir, baseline)

    query_date = _dt.date(2024, 1, 20)
    result = join_gov_behavioral_features(_input_df(["NVDA"], [query_date]), contracts_dir, ferc_dir)
    score = result["ferc_grid_constraint_score"][0]
    assert score > 1.0


def test_join_adds_exactly_six_columns(tmp_path):
    """join_gov_behavioral_features adds exactly 6 new columns."""
    from processing.gov_behavioral_features import join_gov_behavioral_features, GOV_BEHAVIORAL_FEATURE_COLS

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    df = _input_df(["NVDA"], [datetime.date(2024, 1, 15)])

    result = join_gov_behavioral_features(df, contracts_dir, ferc_dir)
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in result.columns, f"Missing column: {col}"
    assert len(result.columns) == len(df.columns) + 6


def test_ticker_with_no_contracts_zero_fills(tmp_path):
    """Ticker absent from contract awards gets 0.0 for all gov_contract_* columns."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"
    d = datetime.date(2024, 1, 15)
    _write_contracts(contracts_dir, [
        {"date": d, "awardee_name": "NVIDIA Corporation", "uei": "U1",
         "contract_value_usd": 1000.0, "naics_code": "518210", "agency": "DOD"},
    ])

    result = join_gov_behavioral_features(_input_df(["MSFT"], [d]), contracts_dir, ferc_dir)
    row = result.filter(pl.col("ticker") == "MSFT")
    assert row["gov_contract_value_90d"][0] == pytest.approx(0.0)
    assert row["gov_contract_count_90d"][0] == pytest.approx(0.0)
    assert row["gov_contract_momentum"][0] == pytest.approx(0.0)


def test_no_ferc_data_zero_fills(tmp_path):
    """Missing ferc_dir produces 0.0 for ferc_queue_mw_30d and ferc_grid_constraint_score."""
    from processing.gov_behavioral_features import join_gov_behavioral_features

    contracts_dir = tmp_path / "gov_contracts"
    ferc_dir = tmp_path / "ferc_queue"   # empty — no files

    result = join_gov_behavioral_features(
        _input_df(["NVDA"], [datetime.date(2024, 1, 15)]), contracts_dir, ferc_dir
    )
    assert result["ferc_queue_mw_30d"][0] == pytest.approx(0.0)
    assert result["ferc_grid_constraint_score"][0] == pytest.approx(0.0)


def test_no_contracts_dir_zero_fills_all_six(tmp_path):
    """Missing contracts_dir produces 0.0 for all 6 GOV_BEHAVIORAL_FEATURE_COLS."""
    from processing.gov_behavioral_features import join_gov_behavioral_features, GOV_BEHAVIORAL_FEATURE_COLS

    contracts_dir = tmp_path / "gov_contracts"   # does not exist
    ferc_dir = tmp_path / "ferc_queue"           # does not exist

    result = join_gov_behavioral_features(
        _input_df(["NVDA"], [datetime.date(2024, 1, 15)]), contracts_dir, ferc_dir
    )
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert result[col][0] == pytest.approx(0.0), f"{col} should be 0.0"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_gov_behavioral_features.py -v
```

Expected: 10 × `ModuleNotFoundError: No module named 'processing.gov_behavioral_features'`

- [ ] **Step 3: Write the implementation**

Create `processing/gov_behavioral_features.py`:

```python
"""Government behavioral data features: SAM.gov contracts + FERC interconnection queue.

Features (GOV_BEHAVIORAL_FEATURE_COLS):
    gov_contract_value_90d    — rolling 90-day USD awards sum for the ticker's company
    gov_contract_count_90d    — rolling 90-day award count for the ticker's company
    gov_contract_momentum     — (2 * 30d_sum) - 90d_sum  (positive = accelerating)
    gov_ai_spend_30d          — market-wide rolling 30-day AI/DC NAICS award total
    ferc_queue_mw_30d         — rolling 30-day MW filed in DC power states
    ferc_grid_constraint_score — ferc_queue_mw_30d / (365d_total / 12), clipped at 1.0 floor

Ticker-specific features joined on (ticker, date); market-wide joined on date only.
All features zero-filled when data is absent.

DC power states: VA, TX, OH, AZ, NV, OR, GA, WA
"""
from __future__ import annotations

import difflib
import logging
from pathlib import Path

import duckdb
import polars as pl

_LOG = logging.getLogger(__name__)

GOV_BEHAVIORAL_FEATURE_COLS: list[str] = [
    "gov_contract_value_90d",
    "gov_contract_count_90d",
    "gov_contract_momentum",
    "gov_ai_spend_30d",
    "ferc_queue_mw_30d",
    "ferc_grid_constraint_score",
]

# Tickers whose symbol gives no hint of their SAM.gov awardee name
GOV_TICKER_OVERRIDE_MAP: dict[str, str] = {
    "GOOGL": "Alphabet",
    "META":  "Meta Platforms",
    "MSFT":  "Microsoft",
    "AMZN":  "Amazon Web Services",
    "TSM":   "Taiwan Semiconductor",
}

# Full search-name map for fuzzy matching against SAM.gov awardee_name strings
_TICKER_NAME_MAP: dict[str, str] = {
    **GOV_TICKER_OVERRIDE_MAP,
    "NVDA": "NVIDIA",
    "AMD":  "Advanced Micro Devices",
    "INTC": "Intel",
    "ORCL": "Oracle",
    "IBM":  "International Business Machines",
    "DELL": "Dell Technologies",
    "HPE":  "Hewlett Packard Enterprise",
    "CSCO": "Cisco",
    "ACN":  "Accenture",
    "BAH":  "Booz Allen Hamilton",
    "SAIC": "Science Applications International",
    "LEIDOS": "Leidos",
    "CDW":  "CDW",
}

_LEGAL_SUFFIXES = (
    " corporation", " corp", " inc", " incorporated",
    " llc", " ltd", " limited", " co", " company",
    " holdings", " technologies", " systems", " solutions",
)

_TICKER_SCHEMA_EMPTY = {
    "ticker": pl.Utf8, "date": pl.Date,
    "gov_contract_value_90d": pl.Float64,
    "gov_contract_count_90d": pl.Float64,
    "gov_contract_momentum": pl.Float64,
}
_MARKET_SCHEMA_EMPTY = {
    "date": pl.Date,
    "gov_ai_spend_30d": pl.Float64,
    "ferc_queue_mw_30d": pl.Float64,
    "ferc_grid_constraint_score": pl.Float64,
}


def _normalize_name(name: str) -> str:
    n = name.lower().strip()
    for suffix in _LEGAL_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _build_awardee_to_ticker(awardee_names: list[str], tickers: list[str]) -> dict[str, str]:
    """Return {awardee_name: ticker} via normalize + difflib fuzzy match (cutoff 0.85)."""
    norm_to_awardee = {_normalize_name(n): n for n in awardee_names}
    result: dict[str, str] = {}
    for ticker in tickers:
        search_name = _TICKER_NAME_MAP.get(ticker)
        if not search_name:
            continue
        matches = difflib.get_close_matches(
            _normalize_name(search_name), norm_to_awardee.keys(), n=1, cutoff=0.85
        )
        if matches:
            result[norm_to_awardee[matches[0]]] = ticker
    return result


def _load_contracts(contracts_dir: Path) -> pl.DataFrame:
    files = sorted(contracts_dir.glob("date=*/awards.parquet")) if contracts_dir.exists() else []
    if not files:
        return pl.DataFrame(schema={
            "date": pl.Date, "awardee_name": pl.Utf8, "uei": pl.Utf8,
            "contract_value_usd": pl.Float64, "naics_code": pl.Utf8, "agency": pl.Utf8,
        })
    return pl.concat([pl.read_parquet(f) for f in files])


def _load_ferc(ferc_dir: Path) -> pl.DataFrame:
    files = sorted(ferc_dir.glob("date=*/queue.parquet")) if ferc_dir.exists() else []
    if not files:
        return pl.DataFrame(schema={
            "snapshot_date": pl.Date, "queue_date": pl.Date, "project_name": pl.Utf8,
            "mw": pl.Float64, "state": pl.Utf8, "fuel": pl.Utf8,
            "status": pl.Utf8, "iso": pl.Utf8,
        })
    raw = pl.concat([pl.read_parquet(f) for f in files])
    return raw.sort("snapshot_date").unique(subset=["project_name", "queue_date"], keep="last")


def _build_ticker_features(contracts_dir: Path, tickers: list[str]) -> pl.DataFrame:
    raw = _load_contracts(contracts_dir)
    if raw.is_empty():
        return pl.DataFrame(schema=_TICKER_SCHEMA_EMPTY)

    awardee_to_ticker = _build_awardee_to_ticker(
        raw["awardee_name"].unique().to_list(), tickers
    )
    if not awardee_to_ticker:
        return pl.DataFrame(schema=_TICKER_SCHEMA_EMPTY)

    matched = raw.with_columns(
        pl.col("awardee_name")
        .map_elements(lambda n: awardee_to_ticker.get(n), return_dtype=pl.Utf8)
        .alias("ticker")
    ).filter(pl.col("ticker").is_not_null())

    if matched.is_empty():
        return pl.DataFrame(schema=_TICKER_SCHEMA_EMPTY)

    daily = (
        matched
        .group_by(["ticker", "date"])
        .agg(
            pl.col("contract_value_usd").sum().alias("daily_value"),
            pl.len().alias("daily_count"),
        )
        .sort(["ticker", "date"])
    )

    con = duckdb.connect()
    con.register("daily", daily.to_arrow())
    result = con.execute("""
        SELECT
            ticker,
            date,
            SUM(daily_value) OVER w90  AS gov_contract_value_90d,
            CAST(SUM(daily_count) OVER w90 AS DOUBLE) AS gov_contract_count_90d,
            2.0 * SUM(daily_value) OVER w30 - SUM(daily_value) OVER w90
                AS gov_contract_momentum
        FROM daily
        WINDOW
            w90 AS (PARTITION BY ticker ORDER BY date
                    RANGE BETWEEN INTERVAL '90 days' PRECEDING AND CURRENT ROW),
            w30 AS (PARTITION BY ticker ORDER BY date
                    RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW)
    """).pl()
    con.close()
    return result


def _build_market_features(contracts_dir: Path, ferc_dir: Path) -> pl.DataFrame:
    raw_contracts = _load_contracts(contracts_dir)
    raw_ferc = _load_ferc(ferc_dir)

    con = duckdb.connect()

    # Market-wide SAM.gov 30-day rolling sum
    if not raw_contracts.is_empty():
        market_daily = (
            raw_contracts
            .group_by("date")
            .agg(pl.col("contract_value_usd").sum().alias("daily_total"))
            .sort("date")
        )
        con.register("market_daily", market_daily.to_arrow())
        sam_df = con.execute("""
            SELECT date,
                SUM(daily_total) OVER (
                    ORDER BY date RANGE BETWEEN INTERVAL '30 days' PRECEDING AND CURRENT ROW
                ) AS gov_ai_spend_30d
            FROM market_daily
        """).pl()
    else:
        sam_df = pl.DataFrame(schema={"date": pl.Date, "gov_ai_spend_30d": pl.Float64})

    # FERC rolling window features
    ferc_df: pl.DataFrame
    if not raw_ferc.is_empty() and "queue_date" in raw_ferc.columns:
        ferc_valid = raw_ferc.filter(pl.col("queue_date").is_not_null())
        if not ferc_valid.is_empty():
            ferc_daily = (
                ferc_valid
                .rename({"queue_date": "date"})
                .group_by("date")
                .agg(pl.col("mw").sum().alias("daily_mw"))
                .sort("date")
            )
            con.register("ferc_daily", ferc_daily.to_arrow())
            raw_ferc_result = con.execute("""
                SELECT date,
                    SUM(daily_mw) OVER w30  AS ferc_queue_mw_30d,
                    SUM(daily_mw) OVER w365 AS _mw_365d
                FROM ferc_daily
                WINDOW
                    w30  AS (ORDER BY date
                             RANGE BETWEEN INTERVAL '30 days'  PRECEDING AND CURRENT ROW),
                    w365 AS (ORDER BY date
                             RANGE BETWEEN INTERVAL '365 days' PRECEDING AND CURRENT ROW)
            """).pl()
            ferc_df = raw_ferc_result.with_columns(
                (pl.col("ferc_queue_mw_30d")
                 / (pl.col("_mw_365d") / 12.0).clip(lower_bound=1.0))
                .alias("ferc_grid_constraint_score")
            ).drop("_mw_365d")
        else:
            ferc_df = pl.DataFrame(schema={
                "date": pl.Date,
                "ferc_queue_mw_30d": pl.Float64,
                "ferc_grid_constraint_score": pl.Float64,
            })
    else:
        ferc_df = pl.DataFrame(schema={
            "date": pl.Date,
            "ferc_queue_mw_30d": pl.Float64,
            "ferc_grid_constraint_score": pl.Float64,
        })

    con.close()

    if sam_df.is_empty() and ferc_df.is_empty():
        return pl.DataFrame(schema=_MARKET_SCHEMA_EMPTY)

    if sam_df.is_empty():
        return ferc_df.with_columns(pl.lit(0.0).alias("gov_ai_spend_30d")).select(
            ["date", "gov_ai_spend_30d", "ferc_queue_mw_30d", "ferc_grid_constraint_score"]
        )
    if ferc_df.is_empty():
        return sam_df.with_columns([
            pl.lit(0.0).alias("ferc_queue_mw_30d"),
            pl.lit(0.0).alias("ferc_grid_constraint_score"),
        ])

    return sam_df.join(ferc_df, on="date", how="outer_coalesce").sort("date")


def join_gov_behavioral_features(
    df: pl.DataFrame,
    contracts_dir: Path,
    ferc_dir: Path,
) -> pl.DataFrame:
    """Left-join government behavioral features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'ticker' (Utf8) and 'date' (Date) columns.
        contracts_dir: Root of data/raw/gov_contracts/ Hive tree.
        ferc_dir: Root of data/raw/ferc_queue/ Hive tree.

    Returns:
        df with GOV_BEHAVIORAL_FEATURE_COLS appended (Float64). Zero-filled.
    """
    from ingestion.ticker_registry import TICKERS

    ticker_df = _build_ticker_features(contracts_dir, TICKERS)
    market_df = _build_market_features(contracts_dir, ferc_dir)

    if not ticker_df.is_empty():
        df = df.join(ticker_df, on=["ticker", "date"], how="left")
    else:
        for col in ["gov_contract_value_90d", "gov_contract_count_90d", "gov_contract_momentum"]:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))

    if not market_df.is_empty():
        df = df.join(market_df, on="date", how="left")
    else:
        for col in ["gov_ai_spend_30d", "ferc_queue_mw_30d", "ferc_grid_constraint_score"]:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))

    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_gov_behavioral_features.py -v
```

Expected: 10 × PASSED

- [ ] **Step 5: Commit**

```bash
git add processing/gov_behavioral_features.py tests/test_gov_behavioral_features.py
git commit -m "feat: add government behavioral feature module

6 GOV_BEHAVIORAL_FEATURE_COLS from SAM.gov (ticker-specific 90d/30d rolling sums,
market-wide 30d spend) and FERC queue (MW_30d, grid_constraint_score).
DuckDB window functions for date-range rolling aggregations.
Ticker matching via difflib fuzzy match (cutoff 0.85) + GOV_TICKER_OVERRIDE_MAP.
10 tests covering all features, zero-fill paths, and missing data.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Model Integration

**Files:**
- Modify: `models/train.py`
- Modify: `models/inference.py`
- Modify: `tests/test_train.py`
- Modify: `tools/run_refresh.sh`

- [ ] **Step 1: Write 6 failing tests — append to `tests/test_train.py`**

Append these tests to the end of `tests/test_train.py`:

```python
# ── Government behavioral signals integration ─────────────────────────────────

def test_feature_cols_has_67_elements():
    """FEATURE_COLS must have exactly 67 elements after adding GOV_BEHAVIORAL_FEATURE_COLS."""
    from models.train import FEATURE_COLS
    assert len(FEATURE_COLS) == 67, f"Expected 67 features, got {len(FEATURE_COLS)}"


def test_gov_behavioral_feature_cols_in_feature_cols():
    """All 6 GOV_BEHAVIORAL_FEATURE_COLS must appear in FEATURE_COLS."""
    from models.train import FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"


def test_gov_behavioral_feature_cols_not_in_short_tier():
    """GOV_BEHAVIORAL_FEATURE_COLS must NOT be in TIER_FEATURE_COLS['short']."""
    from models.train import TIER_FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    short = set(TIER_FEATURE_COLS["short"])
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col not in short, f"{col} must not be in short tier"


def test_gov_behavioral_feature_cols_in_medium_tier():
    """All 6 GOV_BEHAVIORAL_FEATURE_COLS must appear in TIER_FEATURE_COLS['medium']."""
    from models.train import TIER_FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    medium = set(TIER_FEATURE_COLS["medium"])
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in medium, f"{col} missing from medium tier"


def test_gov_behavioral_feature_cols_in_long_tier():
    """All 6 GOV_BEHAVIORAL_FEATURE_COLS must appear in TIER_FEATURE_COLS['long']."""
    from models.train import TIER_FEATURE_COLS
    from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS
    long_cols = set(TIER_FEATURE_COLS["long"])
    for col in GOV_BEHAVIORAL_FEATURE_COLS:
        assert col in long_cols, f"{col} missing from long tier"


def test_medium_tier_equals_feature_cols_67():
    """TIER_FEATURE_COLS['medium'] must equal FEATURE_COLS (67 elements) and be a copy."""
    from models.train import FEATURE_COLS, TIER_FEATURE_COLS
    assert TIER_FEATURE_COLS["medium"] == FEATURE_COLS
    assert TIER_FEATURE_COLS["medium"] is not FEATURE_COLS
    assert len(TIER_FEATURE_COLS["medium"]) == 67
```

Also update the two existing assertions that check for 61 features.

In `test_feature_cols_includes_cyber_threat` (around line 369), change:
```python
    assert len(FEATURE_COLS) == 61, f"Expected 61 features, got {len(FEATURE_COLS)}"
```
to:
```python
    assert len(FEATURE_COLS) == 67, f"Expected 67 features, got {len(FEATURE_COLS)}"
```

In `test_feature_cols_has_61_elements` (around line 394–397), change to:
```python
def test_feature_cols_has_61_elements():
    """FEATURE_COLS must have exactly 67 elements after adding OPTIONS_FEATURE_COLS and GOV_BEHAVIORAL_FEATURE_COLS."""
    from models.train import FEATURE_COLS
    assert len(FEATURE_COLS) == 67, f"Expected 67 features, got {len(FEATURE_COLS)}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py -k "gov_behavioral or has_67" -v
```

Expected: 6 new tests × FAILED (`AssertionError: Expected 67 features, got 61`)

- [ ] **Step 3: Update `models/train.py`**

**3a — Add import** (after the `options_features` import on line 42):

```python
from processing.gov_behavioral_features import GOV_BEHAVIORAL_FEATURE_COLS, join_gov_behavioral_features
```

**3b — Update FEATURE_COLS** (change the comment block starting at line 107):

```python
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS + ENERGY_FEATURE_COLS
    + SUPPLY_CHAIN_FEATURE_COLS + FX_FEATURE_COLS
    + CYBER_THREAT_FEATURE_COLS    # 48 → 55 features total
    + OPTIONS_FEATURE_COLS         # 55 → 61 features total
    + GOV_BEHAVIORAL_FEATURE_COLS  # 61 → 67 features total
)
```

**3c — Update TIER_FEATURE_COLS** (replace the existing dict starting at line 133):

```python
TIER_FEATURE_COLS: dict[str, list[str]] = {
    "short": (
        PRICE_FEATURE_COLS
        + SENTIMENT_FEATURE_COLS
        + INSIDER_FEATURE_COLS
        + SHORT_INTEREST_FEATURE_COLS
        + _CYBER_THREAT_SHORT_COLS   # 5 features: *_7d only
        + OPTIONS_FEATURE_COLS       # all 6 options features
        # gov behavioral excluded — contract award cycles too slow for 5d/20d horizons
    ),
    "medium": list(FEATURE_COLS),    # all 67 features (copy to avoid shared mutable reference)
    "long": (
        PRICE_FEATURE_COLS
        + FUND_FEATURE_COLS
        + EARNINGS_FEATURE_COLS
        + GRAPH_FEATURE_COLS
        + OWNERSHIP_FEATURE_COLS
        + ENERGY_FEATURE_COLS
        + SUPPLY_CHAIN_FEATURE_COLS
        + FX_FEATURE_COLS
        + GOV_BEHAVIORAL_FEATURE_COLS  # government contracts relevant at year+ horizons
        # cyber threat + options excluded — noise at year+ horizons
    ),
}
```

**3d — Add join call in `build_training_dataset`** (after the options join, around line 329):

```python
    # Join government behavioral features (ticker-specific SAM.gov + market-wide FERC)
    gov_contracts_dir = fundamentals_dir.parent.parent / "gov_contracts"
    ferc_dir = fundamentals_dir.parent.parent / "ferc_queue"
    df = join_gov_behavioral_features(df, gov_contracts_dir, ferc_dir)
```

Also update the docstring comment on `train_single_layer` from `"all 61 FEATURE_COLS"` to `"all 67 FEATURE_COLS"`.

- [ ] **Step 4: Update `models/inference.py`**

**4a — Add import** (after `from processing.options_features import join_options_features` on line 45):

```python
from processing.gov_behavioral_features import join_gov_behavioral_features
```

**4b — Add join call in `_build_feature_df`** (after the options join on line 125):

```python
    gov_contracts_dir = data_dir / "gov_contracts"
    ferc_dir = data_dir / "ferc_queue"
    df = join_gov_behavioral_features(df, gov_contracts_dir, ferc_dir)
```

**4c — Update docstring** on `_build_feature_df` (line 57):

```python
    """Build the 67-feature DataFrame for all tickers on date_str."""
```

- [ ] **Step 5: Update `tools/run_refresh.sh`**

Add two lines after the last existing ingestion step (the `=== 9/9  Ownership features ===` block) and before the final echo. Change the count from 9 to 11 and add two new steps:

```bash
echo ""
echo "=== 10/11 SAM.gov contract awards ==="
python ingestion/sam_gov_ingestion.py

echo ""
echo "=== 11/11 FERC interconnection queue ==="
python ingestion/ferc_queue_ingestion.py
```

Also update the first echo line from `"=== 1/9"` through `"=== 9/9"` to `"=== 1/11"` through `"=== 9/11"`, and update the final echo to `"=== Refresh complete at $(date) ==="`.

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -m "not integration" -v
```

Expected: All tests PASSED, including the 6 new GOV integration tests and the 2 updated len-61→67 tests.

- [ ] **Step 7: Commit**

```bash
git add models/train.py models/inference.py tests/test_train.py tools/run_refresh.sh
git commit -m "feat: wire gov behavioral features into model (61→67 FEATURE_COLS)

- train.py: add GOV_BEHAVIORAL_FEATURE_COLS import, extend FEATURE_COLS 61→67,
  add to long tier, exclude from short tier, add join call in build_training_dataset
- inference.py: add join call + update docstring to 67-feature
- test_train.py: update len assertions 61→67, add 6 GOV integration tests
- run_refresh.sh: add sam_gov and ferc_queue ingestion steps (9→11 steps)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Success Criteria

- [ ] `pytest tests/ -m "not integration"` passes with 0 failures
- [ ] `len(FEATURE_COLS) == 67`
- [ ] `GOV_BEHAVIORAL_FEATURE_COLS` absent from short tier, present in medium + long
- [ ] `TIER_FEATURE_COLS["medium"] == FEATURE_COLS` (equality) and `is not FEATURE_COLS` (identity)
- [ ] `join_gov_behavioral_features` zero-fills when either data directory is missing
- [ ] `ingest_sam_gov` raises `RuntimeError` when `SAM_GOV_API_KEY` not set
- [ ] FERC ingestion skips re-download in same half-year
