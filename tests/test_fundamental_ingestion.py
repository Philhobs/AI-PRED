import datetime
import pytest
import pandas as pd
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock
from pathlib import Path


def _make_mock_ticker():
    """
    Mock yf.Ticker with two quarters of financial data.
    quarterly_financials: rows=metric names, cols=period-end Timestamps.
    """
    mock = MagicMock()

    mock.quarterly_financials = pd.DataFrame({
        pd.Timestamp("2024-03-31"): {
            "Total Revenue":       44_000_000_000.0,
            "Gross Profit":        32_000_000_000.0,
            "Operating Income":    20_000_000_000.0,
            "Capital Expenditure": -6_000_000_000.0,
        },
        pd.Timestamp("2023-03-31"): {
            "Total Revenue":       36_000_000_000.0,
            "Gross Profit":        27_000_000_000.0,
            "Operating Income":    16_000_000_000.0,
            "Capital Expenditure": -5_000_000_000.0,
        },
    })

    mock.quarterly_balance_sheet = pd.DataFrame({
        pd.Timestamp("2024-03-31"): {
            "Stockholders Equity": 100_000_000_000.0,
            "Total Debt":           50_000_000_000.0,
            "Current Assets":       80_000_000_000.0,
            "Current Liabilities":  30_000_000_000.0,
        },
        pd.Timestamp("2023-03-31"): {
            "Stockholders Equity":  90_000_000_000.0,
            "Total Debt":           45_000_000_000.0,
            "Current Assets":       70_000_000_000.0,
            "Current Liabilities":  28_000_000_000.0,
        },
    })

    mock.info = {
        "trailingPE": 35.0,
        "priceToSalesTrailing12Months": 12.5,
        "priceToBook": 10.2,
    }

    return mock


def test_fetch_fundamentals_returns_one_record_per_quarter():
    """fetch_fundamentals returns one dict per quarter in quarterly_financials."""
    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=_make_mock_ticker()):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        records = fetch_fundamentals("NVDA")

    assert len(records) == 2
    assert all(r["ticker"] == "NVDA" for r in records)
    assert all(isinstance(r["period_end"], datetime.date) for r in records)


def test_fetch_fundamentals_computes_gross_margin():
    """gross_margin = Gross Profit / Total Revenue for each quarter."""
    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=_make_mock_ticker()):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        records = fetch_fundamentals("NVDA")

    records.sort(key=lambda r: r["period_end"], reverse=True)
    assert records[0]["gross_margin"] == pytest.approx(32 / 44, rel=1e-4)
    assert records[1]["gross_margin"] == pytest.approx(27 / 36, rel=1e-4)


def test_fetch_fundamentals_valuation_ratios_only_on_most_recent():
    """pe_ratio_trailing, price_to_sales, price_to_book are non-null only for most recent quarter."""
    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=_make_mock_ticker()):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        records = fetch_fundamentals("NVDA")

    records.sort(key=lambda r: r["period_end"], reverse=True)
    assert records[0]["pe_ratio_trailing"] == pytest.approx(35.0)
    assert records[0]["price_to_sales"] == pytest.approx(12.5)
    assert records[0]["price_to_book"] == pytest.approx(10.2)
    assert records[1]["pe_ratio_trailing"] is None
    assert records[1]["price_to_sales"] is None
    assert records[1]["price_to_book"] is None


def test_fetch_fundamentals_returns_empty_for_empty_financials():
    """fetch_fundamentals returns [] when quarterly_financials is empty."""
    mock = MagicMock()
    mock.quarterly_financials = pd.DataFrame()
    mock.quarterly_balance_sheet = pd.DataFrame()
    mock.info = {}

    with patch("ingestion.fundamental_ingestion.yf.Ticker", return_value=mock):
        from ingestion.fundamental_ingestion import fetch_fundamentals
        assert fetch_fundamentals("UNKNOWN") == []


def test_save_fundamentals_writes_parquet_with_correct_schema(tmp_path):
    """save_fundamentals writes snappy Parquet at the expected path with correct columns."""
    records = [
        {
            "ticker": "NVDA",
            "period_end": datetime.date(2024, 3, 31),
            "pe_ratio_trailing": 35.0,
            "price_to_sales": 12.5,
            "price_to_book": 10.2,
            "revenue_growth_yoy": 0.22,
            "gross_margin": 0.727,
            "operating_margin": 0.455,
            "capex_to_revenue": 0.136,
            "debt_to_equity": 0.5,
            "current_ratio": 2.67,
        }
    ]

    from ingestion.fundamental_ingestion import save_fundamentals
    save_fundamentals(records, "NVDA", tmp_path)

    path = tmp_path / "financials" / "fundamentals" / "NVDA" / "quarterly.parquet"
    assert path.exists()

    table = pq.read_table(str(path))
    assert "ticker" in table.schema.names
    assert "period_end" in table.schema.names
    assert "gross_margin" in table.schema.names
    assert table.num_rows == 1
