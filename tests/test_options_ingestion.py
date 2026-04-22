"""Tests for ingestion/options_ingestion.py — yfinance calls are mocked."""
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

_EXPECTED_SCHEMA = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "expiry": pl.Date,
    "option_type": pl.Utf8,
    "strike": pl.Float64,
    "iv": pl.Float64,
    "oi": pl.Int64,
    "volume": pl.Int64,
}


def _check_schema(df: pl.DataFrame) -> None:
    assert set(df.columns) == set(_EXPECTED_SCHEMA), f"Unexpected columns: {df.columns}"
    for col, dtype in _EXPECTED_SCHEMA.items():
        assert df[col].dtype == dtype, f"Column {col}: expected {dtype}, got {df[col].dtype}"


def _make_options_df(strikes: list[float], ivs: list[float]) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame({
        "strike": strikes,
        "impliedVolatility": ivs,
        "openInterest": [100] * len(strikes),
        "volume": [50] * len(strikes),
    })


def _mock_ticker(expiry_dates: list[str], strikes: list[float], ivs: list[float]):
    """Build a mock yf.Ticker that returns a known options chain."""
    chain = MagicMock()
    chain.calls = _make_options_df(strikes, ivs)
    chain.puts = _make_options_df(strikes, ivs)

    mock_t = MagicMock()
    mock_t.options = expiry_dates
    mock_t.option_chain.return_value = chain
    return mock_t


def test_yfinance_source_returns_correct_schema():
    """YFinanceOptionsSource.fetch returns all 8 schema columns with correct dtypes."""
    mock_t = _mock_ticker(
        expiry_dates=["2024-02-16"],
        strikes=[100.0, 105.0, 110.0],
        ivs=[0.30, 0.28, 0.32],
    )
    with patch("yfinance.Ticker", return_value=mock_t), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("NVDA", "2024-01-15")

    _check_schema(df)
    assert len(df) > 0
    assert set(df["option_type"].unique().to_list()) == {"call", "put"}
    assert df["ticker"][0] == "NVDA"
    assert df["date"][0] == datetime.date(2024, 1, 15)
    assert df["expiry"][0] == datetime.date(2024, 2, 16)


def test_yfinance_source_captures_all_strikes():
    """All strikes from the options chain are stored (ATM/OTM selection happens downstream)."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    mock_t = _mock_ticker(expiry_dates=["2024-02-16"], strikes=strikes, ivs=[0.3] * 5)
    with patch("yfinance.Ticker", return_value=mock_t), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("NVDA", "2024-01-15")

    stored_strikes = sorted(df["strike"].unique().to_list())
    assert stored_strikes == strikes, f"Expected all strikes stored, got: {stored_strikes}"


def test_yfinance_source_empty_chain_returns_empty():
    """Empty options chain returns empty DataFrame with correct schema — no crash."""
    mock_t = MagicMock()
    mock_t.options = []
    with patch("yfinance.Ticker", return_value=mock_t), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("DARK.L", "2024-01-15")

    _check_schema(df)
    assert len(df) == 0


def test_yfinance_source_exception_returns_empty():
    """If yfinance raises, return empty DataFrame — no crash."""
    with patch("yfinance.Ticker", side_effect=Exception("network error")), patch("time.sleep"):
        from ingestion.options_ingestion import YFinanceOptionsSource
        source = YFinanceOptionsSource()
        df = source.fetch("NVDA", "2024-01-15")

    _check_schema(df)
    assert len(df) == 0


def test_ingest_options_writes_parquet_at_correct_path(tmp_path):
    """ingest_options writes ticker.parquet under date=YYYY-MM-DD/ partition."""
    mock_source = MagicMock()
    mock_source.fetch.return_value = pl.DataFrame(
        [{
            "ticker": "NVDA",
            "date": datetime.date(2024, 1, 15),
            "expiry": datetime.date(2024, 2, 16),
            "option_type": "call",
            "strike": 100.0,
            "iv": 0.30,
            "oi": 100,
            "volume": 50,
        }],
        schema={
            "ticker": pl.Utf8, "date": pl.Date, "expiry": pl.Date,
            "option_type": pl.Utf8, "strike": pl.Float64,
            "iv": pl.Float64, "oi": pl.Int64, "volume": pl.Int64,
        },
    )
    from ingestion.options_ingestion import ingest_options
    ingest_options(["NVDA"], "2024-01-15", tmp_path, source=mock_source)

    expected_path = tmp_path / "date=2024-01-15" / "NVDA.parquet"
    assert expected_path.exists(), f"Expected parquet at {expected_path}"
    result = pl.read_parquet(expected_path)
    assert result["ticker"][0] == "NVDA"
    assert result["date"][0] == datetime.date(2024, 1, 15)


def test_ingest_options_skips_empty_chain(tmp_path):
    """ingest_options does not write a file when the source returns empty data."""
    mock_source = MagicMock()
    mock_source.fetch.return_value = pl.DataFrame(schema={
        "ticker": pl.Utf8, "date": pl.Date, "expiry": pl.Date,
        "option_type": pl.Utf8, "strike": pl.Float64,
        "iv": pl.Float64, "oi": pl.Int64, "volume": pl.Int64,
    })
    from ingestion.options_ingestion import ingest_options
    ingest_options(["DARK.L"], "2024-01-15", tmp_path, source=mock_source)

    assert not (tmp_path / "date=2024-01-15").exists()
