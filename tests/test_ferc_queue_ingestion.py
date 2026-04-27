import datetime
import io
import polars as pl
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_excel_bytes(rows: list[dict]) -> bytes:
    """Create in-memory Excel file from list of dicts (legacy single-sheet, header on row 1)."""
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


# All URL-path tests pass `local_path=tmp_path / "absent.xlsx"` to disable the
# (now-default) local-file lookup and force the URL fallback path.

def test_schema_correct(tmp_path):
    """Output parquet matches _FERC_SCHEMA when fed via URL fetch."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue, _FERC_SCHEMA

    content = _make_excel_bytes([_VA_ROW])
    with patch("ingestion.ferc_queue_ingestion.requests.get",
               return_value=_mock_download(content)):
        ingest_ferc_queue("2024-01-15", tmp_path,
                          local_path=tmp_path / "absent.xlsx")

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
        ingest_ferc_queue("2024-03-01", tmp_path,
                          local_path=tmp_path / "absent.xlsx")
        mock_get.assert_not_called()


def test_state_filter_keeps_only_dc_states(tmp_path):
    """Only rows with DC power states (VA, TX, OH, AZ, NV, OR, GA, WA) are stored."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue

    content = _make_excel_bytes([_VA_ROW, _FL_ROW])
    with patch("ingestion.ferc_queue_ingestion.requests.get",
               return_value=_mock_download(content)):
        ingest_ferc_queue("2024-01-15", tmp_path,
                          local_path=tmp_path / "absent.xlsx")

    df = pl.read_parquet(tmp_path / "date=2024-01-15" / "queue.parquet")
    assert len(df) == 1
    assert df["state"][0] == "VA"


def test_empty_sheet_no_file_written(tmp_path):
    """Empty Excel sheet (or all non-DC rows) produces no parquet file."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue

    content = _make_excel_bytes([_FL_ROW])   # only non-DC row
    with patch("ingestion.ferc_queue_ingestion.requests.get",
               return_value=_mock_download(content)):
        ingest_ferc_queue("2024-01-15", tmp_path,
                          local_path=tmp_path / "absent.xlsx")

    assert not (tmp_path / "date=2024-01-15").exists()


def test_bad_url_logs_and_returns_empty(tmp_path, caplog):
    """Download failure logs a warning and returns without raising — fail-soft."""
    import logging
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue
    import requests as _requests

    with patch("ingestion.ferc_queue_ingestion.requests.get") as mock_get:
        mock_get.side_effect = _requests.RequestException("connection refused")
        with caplog.at_level(logging.WARNING, logger="ingestion.ferc_queue_ingestion"):
            ingest_ferc_queue("2024-01-15", tmp_path,
                              ferc_url="http://bad-url/",
                              local_path=tmp_path / "absent.xlsx")

    # No parquet should be written when the download fails.
    assert not (tmp_path / "date=2024-01-15").exists()
    assert any("download failed" in rec.message for rec in caplog.records)


def test_http_404_logs_and_returns_empty(tmp_path, caplog):
    """A 404 from the upstream URL also fail-softs, no exception bubbles up."""
    import logging
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue

    mock = MagicMock()
    mock.raise_for_status.side_effect = Exception("404 Not Found")
    with patch("ingestion.ferc_queue_ingestion.requests.get", return_value=mock):
        with caplog.at_level(logging.WARNING, logger="ingestion.ferc_queue_ingestion"):
            ingest_ferc_queue("2024-01-15", tmp_path,
                              ferc_url="http://bad-url/",
                              local_path=tmp_path / "absent.xlsx")
    assert not (tmp_path / "date=2024-01-15").exists()
    assert any("download failed" in rec.message for rec in caplog.records)


# ── New: local-file path coverage ────────────────────────────────────────────

def test_local_file_preferred_over_url(tmp_path):
    """When a local file exists, it's read instead of fetching the URL."""
    from ingestion.ferc_queue_ingestion import ingest_ferc_queue

    local = tmp_path / "lbnl_local.xlsx"
    local.write_bytes(_make_excel_bytes([_VA_ROW]))

    with patch("ingestion.ferc_queue_ingestion.requests.get") as mock_get:
        ingest_ferc_queue("2024-01-15", tmp_path, local_path=local)
        mock_get.assert_not_called()

    df = pl.read_parquet(tmp_path / "date=2024-01-15" / "queue.parquet")
    assert df["state"][0] == "VA"
    assert df["project_name"][0] == "Solar Farm VA"


def test_excel_serial_date_conversion_logic():
    """Document the parser's Excel→date conversion (1899-12-30 epoch + days)."""
    converted = (pd.Timestamp("1899-12-30") + pd.to_timedelta(43511, unit="D")).date()
    assert converted == datetime.date(2019, 2, 15)
    # And a known boundary: serial 1 = 1899-12-31
    boundary = (pd.Timestamp("1899-12-30") + pd.to_timedelta(1, unit="D")).date()
    assert boundary == datetime.date(1899, 12, 31)
