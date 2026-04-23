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
