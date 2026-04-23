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

    # Write a "same-week" snapshot — today is 2024-01-15 (week 3), use 2024-01-16 (also week 3)
    existing_app_dir = apps_dir / "date=2024-01-16"
    existing_app_dir.mkdir(parents=True)
    pl.DataFrame(schema=_APP_SCHEMA).write_parquet(existing_app_dir / "apps.parquet")

    existing_grant_dir = grants_dir / "date=2024-01-16"
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
