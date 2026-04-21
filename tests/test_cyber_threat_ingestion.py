"""Tests for cyber_threat_ingestion.py — all HTTP calls are mocked."""
import datetime
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


# ── Schema helper ─────────────────────────────────────────────────────────────

_EXPECTED_SCHEMA = {"date": pl.Date, "source": pl.Utf8, "metric": pl.Utf8, "value": pl.Float64}


def _check_schema(df: pl.DataFrame) -> None:
    assert set(df.columns) == set(_EXPECTED_SCHEMA), f"Unexpected columns: {df.columns}"
    for col, dtype in _EXPECTED_SCHEMA.items():
        assert df[col].dtype == dtype, f"Column {col}: expected {dtype}, got {df[col].dtype}"


# ── NVDSource ─────────────────────────────────────────────────────────────────

def _nvd_response(score: float, published: str = "2024-01-15T12:00:00.000") -> dict:
    """Build a minimal NVD API v2 response with a single CVE."""
    return {
        "totalResults": 1,
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-0001",
                "published": published,
                "metrics": {
                    "cvssMetricV31": [{
                        "cvssData": {"baseScore": score}
                    }]
                }
            }
        }]
    }


def test_nvd_source_critical_cve():
    """NVDSource returns metric='cve_critical' for CVSS >= 9.0."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nvd_response(score=9.8)
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "cve_critical"
    assert df["value"][0] == 1.0
    assert df["source"][0] == "nvd"


def test_nvd_source_high_cve():
    """NVDSource returns metric='cve_high' for 7.0 <= CVSS < 9.0."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nvd_response(score=7.5)
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "cve_high"


def test_nvd_source_below_threshold_excluded():
    """NVDSource drops CVEs with CVSS < 7.0."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _nvd_response(score=5.5)
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert len(df) == 0


def test_nvd_source_network_error_returns_empty():
    """NVDSource returns empty DataFrame on network error (no crash)."""
    with patch("requests.get", side_effect=Exception("timeout")), \
         patch("time.sleep"):
        from ingestion.cyber_threat_ingestion import NVDSource
        source = NVDSource()
        df = source.fetch("2024-01-15", "2024-01-15")

    _check_schema(df)
    assert df.is_empty()


# ── CISASource ────────────────────────────────────────────────────────────────

def _cisa_response(date_added: str = "2024-01-15") -> dict:
    return {
        "vulnerabilities": [{
            "cveID": "CVE-2021-44228",
            "dateAdded": date_added,
            "vendorProject": "Apache",
        }]
    }


def test_cisa_source_returns_kev_rows():
    """CISASource returns metric='cisa_kev', one row per KEV entry in range."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _cisa_response("2024-01-15")
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.cyber_threat_ingestion import CISASource
        source = CISASource()
        df = source.fetch("2024-01-01", "2024-01-31")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "cisa_kev"
    assert df["source"][0] == "cisa"
    assert str(df["date"][0]) == "2024-01-15"


def test_cisa_source_filters_by_date_range():
    """CISASource drops KEVs outside the requested date range."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"vulnerabilities": [
        {"cveID": "CVE-A", "dateAdded": "2024-01-15"},
        {"cveID": "CVE-B", "dateAdded": "2024-02-01"},  # outside range
    ]}
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.cyber_threat_ingestion import CISASource
        source = CISASource()
        df = source.fetch("2024-01-01", "2024-01-31")

    assert len(df) == 1
    assert str(df["date"][0]) == "2024-01-15"


# ── OTXSource ─────────────────────────────────────────────────────────────────

def test_otx_source_no_key_returns_empty():
    """OTXSource returns empty DataFrame when OTX_API_KEY is not set."""
    import os
    env_backup = os.environ.pop("OTX_API_KEY", None)
    try:
        from ingestion.cyber_threat_ingestion import OTXSource
        source = OTXSource()
        df = source.fetch("2024-01-01", "2024-01-31")
        _check_schema(df)
        assert df.is_empty()
    finally:
        if env_backup is not None:
            os.environ["OTX_API_KEY"] = env_backup


def test_otx_source_with_key_returns_pulse_rows(monkeypatch):
    """OTXSource returns metric='otx_pulse' rows when key is present."""
    monkeypatch.setenv("OTX_API_KEY", "test-key-abc")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [{"created": "2024-01-15T10:00:00.000000", "name": "Log4Shell"}],
        "next": None,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp), \
         patch("time.sleep"):
        import importlib
        from ingestion import cyber_threat_ingestion
        importlib.reload(cyber_threat_ingestion)
        source = cyber_threat_ingestion.OTXSource()
        df = source.fetch("2024-01-01", "2024-01-31")

    _check_schema(df)
    assert len(df) == 1
    assert df["metric"][0] == "otx_pulse"
    assert df["source"][0] == "otx"


# ── ingest_cyber_threats ──────────────────────────────────────────────────────

def test_ingest_cyber_threats_writes_parquet(tmp_path):
    """ingest_cyber_threats writes threats.parquet partitioned by date."""

    class _FakeSource:
        def fetch(self, start_date: str, end_date: str) -> pl.DataFrame:
            return pl.DataFrame({
                "date": [datetime.date(2024, 1, 15)],
                "source": ["test"],
                "metric": ["cve_critical"],
                "value": [1.0],
            })

    from ingestion.cyber_threat_ingestion import ingest_cyber_threats
    ingest_cyber_threats(
        start_date="2024-01-15",
        end_date="2024-01-15",
        output_dir=tmp_path,
        sources=[_FakeSource()],
    )

    expected = tmp_path / "date=2024-01-15" / "threats.parquet"
    assert expected.exists(), f"Expected parquet at {expected}"
    df = pl.read_parquet(expected)
    assert "metric" in df.columns
    assert len(df) >= 1
