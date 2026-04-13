import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pyarrow.parquet as pq


def test_fetch_edgar_xbrl_returns_filing_records():
    """fetch_edgar_xbrl returns list of quarterly financial records, excluding 8-K."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "units": {
            "USD": [
                {
                    "val": 5_000_000_000,
                    "start": "2023-01-01",
                    "end": "2023-03-31",
                    "form": "10-Q",
                    "filed": "2023-04-30",
                    "accn": "0001234567890",
                },
                {
                    # Should be excluded — wrong form type
                    "val": 1_000_000,
                    "end": "2023-03-31",
                    "form": "8-K",
                    "filed": "2023-04-01",
                    "accn": "0001234567892",
                },
            ]
        }
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.financial_ingestion import fetch_edgar_xbrl
        results = fetch_edgar_xbrl(
            "0000789019",
            "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"
        )

    assert len(results) == 1  # 8-K excluded
    assert results[0]["value"] == 5_000_000_000
    assert results[0]["form"] == "10-Q"
    assert results[0]["cik"] == "0000789019"
    assert results[0]["concept"] == "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment"


def test_fetch_edgar_xbrl_sends_user_agent_header():
    """SEC EDGAR fair-use policy requires a descriptive User-Agent header."""
    captured_headers = {}

    def mock_get(url, headers=None, timeout=None, **kwargs):
        captured_headers.update(headers or {})
        resp = MagicMock()
        resp.json.return_value = {"units": {"USD": []}}
        resp.raise_for_status = MagicMock()
        return resp

    with patch("requests.get", side_effect=mock_get):
        from ingestion.financial_ingestion import fetch_edgar_xbrl
        fetch_edgar_xbrl("0000789019", "us-gaap:Revenues")

    assert "User-Agent" in captured_headers


@patch("ingestion.financial_ingestion.time.sleep")
def test_fetch_fred_energy_indicators_returns_named_series(mock_sleep):
    """fetch_fred_energy_indicators returns dict with henry_hub_gas and other keys."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "observations": [
            {"date": "2024-01-15", "value": "2.85"},
            {"date": "2024-01-08", "value": "2.90"},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.financial_ingestion import fetch_fred_energy_indicators
        result = fetch_fred_energy_indicators()

    assert isinstance(result, dict)
    assert "henry_hub_gas" in result
    assert "electricity_retail_price" in result
    assert len(result["henry_hub_gas"]) == 2
    assert result["henry_hub_gas"][0]["date"] == "2024-01-15"
    assert result["henry_hub_gas"][0]["value"] == pytest.approx(2.85)


@patch("ingestion.financial_ingestion.time.sleep")
def test_fetch_fred_handles_missing_dot_values(mock_sleep):
    """FRED uses '.' for missing values — must become None, not raise ValueError."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "observations": [{"date": "2024-01-01", "value": "."}]
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.financial_ingestion import fetch_fred_energy_indicators
        result = fetch_fred_energy_indicators()

    assert result["henry_hub_gas"][0]["value"] is None


@patch("ingestion.financial_ingestion.time.sleep")
def test_fetch_all_hyperscaler_capex_writes_parquet(mock_sleep, tmp_path):
    """fetch_all_hyperscaler_capex writes capex_history.parquet with ticker column."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "units": {
            "USD": [
                {
                    "val": 10_000_000_000,
                    "start": "2023-01-01",
                    "end": "2023-03-31",
                    "form": "10-Q",
                    "filed": "2023-04-30",
                    "accn": "0001234567890",
                }
            ]
        }
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        from ingestion.financial_ingestion import fetch_all_hyperscaler_capex
        fetch_all_hyperscaler_capex(tmp_path)

    parquet_path = tmp_path / "financials" / "capex_history.parquet"
    assert parquet_path.exists()
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    assert len(df) > 0
    assert "ticker" in df.columns
    assert "value" in df.columns


def test_fetch_all_hyperscaler_capex_falls_back_on_404(tmp_path):
    """fetch_all_hyperscaler_capex tries the next concept when the first returns 404."""
    call_count = {"n": 0}

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if call_count["n"] == 0:
            # First concept: 404
            call_count["n"] += 1
            from requests.exceptions import HTTPError
            resp.raise_for_status.side_effect = HTTPError("404 Not Found")
        else:
            # Second concept: success
            resp.raise_for_status.side_effect = None
            resp.json.return_value = {
                "units": {
                    "USD": [
                        {"val": 1_000_000, "start": "2023-01-01", "end": "2023-03-31",
                         "form": "10-Q", "filed": "2023-04-30", "accn": "0001"},
                    ]
                }
            }
        return resp

    with patch("ingestion.financial_ingestion.requests.get", side_effect=mock_get), \
         patch("ingestion.financial_ingestion.time.sleep"):
        from ingestion.financial_ingestion import fetch_all_hyperscaler_capex
        # Use a single-entry CIK_MAP subset to limit calls
        import ingestion.financial_ingestion as fi
        with patch.object(fi, "CIK_MAP", {"MSFT": "0000789019"}):
            fetch_all_hyperscaler_capex(tmp_path)

    parquet_path = tmp_path / "financials" / "capex_history.parquet"
    assert parquet_path.exists(), "capex_history.parquet should exist after successful fallback"
