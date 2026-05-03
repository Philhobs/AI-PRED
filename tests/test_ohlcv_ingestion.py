import pytest
from unittest.mock import patch
from pathlib import Path
import pandas as pd
import datetime
import pyarrow as pa
import pyarrow.parquet as pq


def _make_mock_df():
    """Minimal yfinance-style DataFrame for one ticker."""
    df = pd.DataFrame(
        {
            "Open":   [500.0, 510.0],
            "High":   [510.0, 520.0],
            "Low":    [495.0, 505.0],
            "Close":  [505.0, 515.0],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.to_datetime(["2024-01-15", "2024-01-16"]),
    )
    df.index.name = "Date"
    return df


def test_fetch_ohlcv_returns_records_with_close_price():
    """fetch_ohlcv maps yfinance 'Close' → 'close_price' and returns list of dicts."""
    with patch("ingestion.ohlcv_ingestion.yf.download", return_value=_make_mock_df()):
        from ingestion.ohlcv_ingestion import fetch_ohlcv
        results = fetch_ohlcv("NVDA", period="5d")

    assert len(results) == 2
    assert results[0]["ticker"] == "NVDA"
    assert results[0]["close_price"] == 505.0
    assert isinstance(results[0]["date"], datetime.date)
    assert "open" in results[0]
    assert "volume" in results[0]


def test_fetch_ohlcv_returns_empty_for_empty_download():
    """fetch_ohlcv returns [] when yfinance returns an empty DataFrame."""
    with patch("ingestion.ohlcv_ingestion.yf.download", return_value=pd.DataFrame()):
        from ingestion.ohlcv_ingestion import fetch_ohlcv
        assert fetch_ohlcv("INVALID", period="5d") == []


def test_save_ohlcv_writes_partitioned_parquet(tmp_path):
    """save_ohlcv writes snappy Parquet to data/raw/financials/ohlcv/<TICKER>/<YEAR>.parquet."""
    records = [
        {
            "ticker": "NVDA",
            "date": datetime.date(2024, 1, 15),
            "open": 500.0, "high": 510.0, "low": 495.0,
            "close_price": 505.0, "volume": 1_000_000,
        },
        {
            "ticker": "NVDA",
            "date": datetime.date(2024, 6, 15),
            "open": 800.0, "high": 820.0, "low": 790.0,
            "close_price": 810.0, "volume": 2_000_000,
        },
    ]

    from ingestion.ohlcv_ingestion import save_ohlcv
    save_ohlcv(records, "NVDA", tmp_path)

    parquet_files = sorted(tmp_path.glob("financials/ohlcv/NVDA/*.parquet"))
    assert len(parquet_files) == 1  # Both rows are year 2024 → written to one partition file

    table = pq.read_table(parquet_files[0])
    assert "close_price" in table.schema.names
    assert "ticker" in table.schema.names
    assert table.num_rows == 2
    assert table.schema.field("close_price").type.id == pa.float64().id  # float64 (double)


def _make_mock_df_multiindex():
    """Matches actual yfinance 1.2.1 single-ticker output structure."""
    arrays = [
        ["Close", "High", "Low", "Open", "Volume"],
        ["NVDA",  "NVDA", "NVDA", "NVDA", "NVDA"],
    ]
    df = pd.DataFrame(
        [[505.0, 510.0, 495.0, 500.0, 1_000_000],
         [515.0, 520.0, 505.0, 510.0, 1_100_000]],
        columns=pd.MultiIndex.from_arrays(arrays),
        index=pd.to_datetime(["2024-01-15", "2024-01-16"]),
    )
    df.index.name = "Date"
    return df


def test_fetch_ohlcv_handles_multiindex_columns():
    """fetch_ohlcv correctly flattens MultiIndex columns from yfinance >= 0.2.38."""
    with patch("ingestion.ohlcv_ingestion.yf.download", return_value=_make_mock_df_multiindex()):
        from ingestion.ohlcv_ingestion import fetch_ohlcv
        results = fetch_ohlcv("NVDA", period="5d")

    assert len(results) == 2
    assert results[0]["close_price"] == 505.0
    assert results[0]["open"] == 500.0
    assert results[0]["volume"] == 1_000_000


def test_save_ohlcv_appends_to_existing_year_file(tmp_path):
    """Daily refresh (small fetch) must MERGE into existing year file, not truncate it.

    Regression: prior to 2026-05-03 the function overwrote the file unconditionally,
    so a `--period 5d` daily refresh would shrink the year's history to 5 rows.
    """
    from ingestion.ohlcv_ingestion import save_ohlcv
    import pyarrow.parquet as pq

    # First save: 5 historical days
    historical = [
        {"ticker": "NVDA", "date": datetime.date(2024, 1, 2 + i),
         "open": 500.0 + i, "high": 510.0 + i, "low": 495.0 + i,
         "close_price": 505.0 + i, "volume": 1_000_000}
        for i in range(5)
    ]
    save_ohlcv(historical, "NVDA", tmp_path)
    path = tmp_path / "financials" / "ohlcv" / "NVDA" / "2024.parquet"
    assert pq.read_table(path).num_rows == 5

    # Daily refresh: 2 records (one overlaps, one fresh)
    refresh = [
        {"ticker": "NVDA", "date": datetime.date(2024, 1, 6),  # overlaps with historical
         "open": 999.0, "high": 999.0, "low": 999.0,
         "close_price": 999.0, "volume": 9_999_999},          # newer values should win
        {"ticker": "NVDA", "date": datetime.date(2024, 1, 8),  # net-new day
         "open": 600.0, "high": 610.0, "low": 595.0,
         "close_price": 605.0, "volume": 1_500_000},
    ]
    save_ohlcv(refresh, "NVDA", tmp_path)
    merged = pq.read_table(path).to_pandas().sort_values("date").reset_index(drop=True)

    # 5 original + 1 net-new = 6 rows (overlap deduped)
    assert len(merged) == 6
    # Conflicting date Jan 6: refresh value (999) wins
    jan6 = merged[merged["date"] == datetime.date(2024, 1, 6)].iloc[0]
    assert jan6["close_price"] == 999.0
    # Original Jan 2 still present
    assert datetime.date(2024, 1, 2) in merged["date"].values


def test_save_ohlcv_partitions_across_years(tmp_path):
    """Records spanning year boundaries produce one file per year."""
    records = [
        {"ticker": "NVDA", "date": datetime.date(2023, 12, 31),
         "open": 495.0, "high": 500.0, "low": 490.0,
         "close_price": 498.0, "volume": 900_000},
        {"ticker": "NVDA", "date": datetime.date(2024, 1, 2),
         "open": 500.0, "high": 510.0, "low": 495.0,
         "close_price": 505.0, "volume": 1_000_000},
    ]
    from ingestion.ohlcv_ingestion import save_ohlcv
    save_ohlcv(records, "NVDA", tmp_path)

    files = sorted(tmp_path.glob("financials/ohlcv/NVDA/*.parquet"))
    assert len(files) == 2
    assert files[0].stem == "2023"
    assert files[1].stem == "2024"
