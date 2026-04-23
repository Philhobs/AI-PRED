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


def test_pagination_followed():
    """fetch_postings requests page 2 when page 1 has fewer items than SearchResultCountAll."""
    from ingestion.usajobs_ingestion import fetch_postings, _KEYWORDS

    item1 = _make_item("DOD-001", "AI Engineer", "2024-03-10")
    item2 = _make_item("DOD-002", "ML Scientist", "2024-03-11")

    # For each of the 5 keywords: page 1 returns 1 item with total=2, page 2 returns 1 item with total=2
    page1 = _make_response([item1], total=2)
    page2 = _make_response([item2], total=2)
    side_effects = [page1, page2] * len(_KEYWORDS)

    with patch("ingestion.usajobs_ingestion.requests.get", side_effect=side_effects):
        with patch("ingestion.usajobs_ingestion.time.sleep"):
            df = fetch_postings("2024-04-01")

    # 2 unique items (DOD-001 and DOD-002) after dedup across 5 keywords
    assert len(df) == 2
    assert set(df["posting_id"].to_list()) == {"DOD-001", "DOD-002"}
