import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pyarrow.parquet as pq
import requests as req_lib


def test_fetch_arrivals_returns_normalised_list():
    """fetch_arrivals_at_airport returns list with callsign whitespace stripped."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "icao24": "abc123",
            "callsign": "FDX1234 ",
            "lastSeen": 1700000000,
            "estDepartureAirport": "RCTP",
        }
    ]
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.flight_ingestion import fetch_arrivals_at_airport
        results = fetch_arrivals_at_airport("KLAX", 1699913600, 1700000000)

    assert len(results) == 1
    assert results[0]["icao24"] == "abc123"
    assert results[0]["callsign"] == "FDX1234"  # Whitespace stripped


def test_fetch_arrivals_returns_empty_on_404():
    """OpenSky returns 404 when no flights exist — must return [] not raise."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = req_lib.HTTPError(response=mock_resp)

    with patch("requests.get", return_value=mock_resp):
        from ingestion.flight_ingestion import fetch_arrivals_at_airport
        results = fetch_arrivals_at_airport("KLAX", 1699913600, 1700000000)

    assert results == []


def test_run_daily_cargo_scan_writes_parquet(tmp_path):
    """run_daily_cargo_scan writes cargo.parquet to flights/date=.../."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {
            "icao24": "b748xx",
            "callsign": "CCA001 ",
            "lastSeen": 1700000000,
            "estDepartureAirport": "RCTP",
        }
    ]
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        with patch("time.sleep"):  # Skip rate-limit sleep in tests
            from ingestion.flight_ingestion import run_daily_cargo_scan
            run_daily_cargo_scan(tmp_path)

    parquet_files = list(tmp_path.glob("flights/date=*/cargo.parquet"))
    assert len(parquet_files) == 1
    table = pq.read_table(parquet_files[0])
    assert table.num_rows > 0
    assert "icao24" in table.schema.names
    assert "signal_type" in table.schema.names
