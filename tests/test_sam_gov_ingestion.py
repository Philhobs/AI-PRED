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
