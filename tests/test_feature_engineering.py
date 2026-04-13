import duckdb
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime


def _write_ohlcv_parquet(tmp_path: Path, ticker: str = "NVDA", n: int = 30):
    """Write minimal OHLCV parquet for a ticker."""
    path = tmp_path / "financials" / "ohlcv" / ticker / "2024.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "ticker": [ticker] * n,
        "date": pd.date_range("2024-01-01", periods=n),
        "close_price": [100.0 + i for i in range(n)],
        "volume": [1_000_000] * n,
    })
    df.to_parquet(path)


def test_build_daily_feature_matrix_returns_polars_df(tmp_path):
    """build_daily_feature_matrix returns a Polars DataFrame."""
    _write_ohlcv_parquet(tmp_path, "NVDA")
    con = duckdb.connect()

    from processing.feature_engineering import build_daily_feature_matrix
    result = build_daily_feature_matrix(con, "2024-01-20", data_dir=tmp_path)

    assert isinstance(result, pl.DataFrame)


def test_feature_matrix_contains_return_columns(tmp_path):
    """Feature matrix includes return_1d and return_5d columns when price data exists."""
    _write_ohlcv_parquet(tmp_path, "NVDA")
    con = duckdb.connect()

    from processing.feature_engineering import build_daily_feature_matrix
    result = build_daily_feature_matrix(con, "2024-01-20", data_dir=tmp_path)

    assert not result.is_empty(), "Expected non-empty DataFrame for date 2024-01-20 within fixture data range"
    assert "return_1d" in result.columns
    assert "return_5d" in result.columns
    assert "taiwan_cargo_ratio" in result.columns
    assert "sentiment_mean" in result.columns


def test_build_daily_feature_matrix_returns_empty_for_missing_date(tmp_path):
    """Returns empty DataFrame when no price data exists for the requested date."""
    _write_ohlcv_parquet(tmp_path, "NVDA")
    con = duckdb.connect()

    from processing.feature_engineering import build_daily_feature_matrix
    # Request a date outside the 30-day parquet range
    result = build_daily_feature_matrix(con, "2020-01-01", data_dir=tmp_path)

    assert isinstance(result, pl.DataFrame)
    assert result.is_empty()
