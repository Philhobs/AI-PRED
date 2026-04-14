"""Shared price feature computation via DuckDB windowed SQL."""
from pathlib import Path

import duckdb
import polars as pl

_PRICE_SQL = """
    WITH price AS (
        SELECT
            ticker,
            date::date AS date,
            close_price / NULLIF(LAG(close_price, 1) OVER w, 0) - 1  AS return_1d,
            close_price / NULLIF(LAG(close_price, 5) OVER w, 0) - 1  AS return_5d,
            close_price / NULLIF(LAG(close_price, 20) OVER w, 0) - 1 AS return_20d,
            close_price / NULLIF(AVG(close_price) OVER (
                PARTITION BY ticker ORDER BY date
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            ), 0) - 1                                                 AS sma_20_deviation,
            STDDEV(close_price) OVER (
                PARTITION BY ticker ORDER BY date
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            ) / NULLIF(AVG(close_price) OVER (
                PARTITION BY ticker ORDER BY date
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            ), 0)                                                     AS volatility_20d,
            volume / NULLIF(AVG(CAST(volume AS DOUBLE)) OVER (
                PARTITION BY ticker ORDER BY date
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            ), 0)                                                     AS volume_ratio
        FROM read_parquet('{glob}')
        WINDOW w AS (PARTITION BY ticker ORDER BY date)
    )
    SELECT * FROM price{where}
"""


def build_price_features(
    ohlcv_dir: Path,
    filter_date: str | None = None,
) -> pl.DataFrame:
    """
    Compute windowed price features for all ticker×date rows.

    Args:
        ohlcv_dir: Directory containing <TICKER>/*.parquet files.
        filter_date: If provided (YYYY-MM-DD), only return rows for that date.
                     If None, return all dates (for training).

    Returns:
        DataFrame with columns: ticker, date, return_1d, return_5d, return_20d,
        sma_20_deviation, volatility_20d, volume_ratio.
    """
    ohlcv_glob = str(ohlcv_dir / "*" / "*.parquet")
    where = f"\n    WHERE date = DATE '{filter_date}'" if filter_date else ""
    sql = _PRICE_SQL.format(glob=ohlcv_glob, where=where)

    con = duckdb.connect()
    try:
        df = con.execute(sql).pl()
    finally:
        con.close()

    return df.with_columns(pl.col("date").cast(pl.Date))
