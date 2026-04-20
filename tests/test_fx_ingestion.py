"""Tests for FX rate ingestion."""
from __future__ import annotations
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest


def _mock_yf_response(rates: list[float], dates: list[str]) -> pd.DataFrame:
    """Build a fake yfinance download response."""
    return pd.DataFrame(
        {"Close": rates},
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_fetch_fx_rates_schema():
    """fetch_fx_rates returns {pair: DataFrame} with date (Date) and rate (Float64)."""
    from ingestion.fx_ingestion import fetch_fx_rates

    mock_data = _mock_yf_response([1.08, 1.09], ["2025-01-01", "2025-01-02"])
    with patch("ingestion.fx_ingestion.yf.download", return_value=mock_data):
        result = fetch_fx_rates(["EURUSD"], years=1)

    assert "EURUSD" in result
    df = result["EURUSD"]
    assert "date" in df.columns
    assert "rate" in df.columns
    assert df["rate"].dtype == pl.Float64
    assert df["date"].dtype == pl.Date
    assert len(df) == 2
    assert df["date"][0] == date(2025, 1, 1)


def test_fetch_fx_rates_sorted_ascending():
    """Returned DataFrame is sorted by date ascending."""
    from ingestion.fx_ingestion import fetch_fx_rates

    mock_data = _mock_yf_response(
        [1.09, 1.08],
        ["2025-01-02", "2025-01-01"],  # deliberately reversed
    )
    with patch("ingestion.fx_ingestion.yf.download", return_value=mock_data):
        result = fetch_fx_rates(["EURUSD"], years=1)

    df = result["EURUSD"]
    assert df["date"][0] == date(2025, 1, 1)
    assert df["date"][1] == date(2025, 1, 2)


def test_fetch_fx_rates_returns_empty_on_error():
    """Connection errors return empty schema DataFrame — pipeline does not crash."""
    from ingestion.fx_ingestion import fetch_fx_rates

    with patch("ingestion.fx_ingestion.yf.download", side_effect=Exception("network error")):
        result = fetch_fx_rates(["EURUSD"], years=1)

    df = result["EURUSD"]
    assert df.is_empty()
    assert "date" in df.columns
    assert "rate" in df.columns


def test_save_fx_rates_creates_parquet(tmp_path):
    """save_fx_rates writes {pair}.parquet with correct schema."""
    from ingestion.fx_ingestion import save_fx_rates

    data = {"EURUSD": pl.DataFrame({
        "date": [date(2025, 1, 1), date(2025, 1, 2)],
        "rate": [1.08, 1.09],
    })}
    save_fx_rates(tmp_path, data)

    parquet_path = tmp_path / "EURUSD.parquet"
    assert parquet_path.exists()
    df = pl.read_parquet(parquet_path)
    assert list(df.columns) == ["date", "rate"]
    assert df["rate"].dtype == pl.Float64


def test_save_fx_rates_deduplicates(tmp_path):
    """Re-saving overlapping dates does not create duplicate rows."""
    from ingestion.fx_ingestion import save_fx_rates

    first = {"EURUSD": pl.DataFrame({
        "date": [date(2025, 1, 1), date(2025, 1, 2)],
        "rate": [1.08, 1.09],
    })}
    second = {"EURUSD": pl.DataFrame({
        "date": [date(2025, 1, 2), date(2025, 1, 3)],  # Jan 2 is duplicate
        "rate": [1.09, 1.10],
    })}

    save_fx_rates(tmp_path, first)
    save_fx_rates(tmp_path, second)

    df = pl.read_parquet(tmp_path / "EURUSD.parquet")
    assert len(df) == 3  # Jan 1, 2, 3 — no duplicate Jan 2
    assert df["date"].to_list() == [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
