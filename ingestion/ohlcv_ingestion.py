import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

from ingestion.ticker_registry import TICKERS

SCHEMA = pa.schema([
    pa.field("ticker", pa.string()),
    pa.field("date", pa.date32()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close_price", pa.float64()),  # matches feature_engineering.py column name
    pa.field("volume", pa.int64()),
])


def fetch_ohlcv(ticker: str, period: str = "2y") -> list[dict]:
    """
    Download OHLCV history for one ticker via yfinance (unofficial Yahoo Finance).
    Uses auto_adjust=True so splits and dividends are pre-adjusted.
    Returns list of dicts with close_price (not 'close') to match the feature schema.
    Returns [] if yfinance returns an empty DataFrame.
    """
    import pandas as pd
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    records = []
    for dt, row in df.iterrows():
        records.append({
            "ticker": ticker,
            "date": dt.date(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close_price": float(row["Close"]),
            "volume": int(row["Volume"]),
        })
    return records


def save_ohlcv(records: list[dict], ticker: str, output_dir: Path) -> None:
    """
    Write OHLCV records to Hive-style Parquet partitioned by year.
    Path: <output_dir>/financials/ohlcv/<TICKER>/<YEAR>.parquet
    All records for the same ticker+year are written to a single file.
    """
    if not records:
        return

    by_year: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_year[r["date"].year].append(r)

    for year, year_records in by_year.items():
        path = output_dir / "financials" / "ohlcv" / ticker / f"{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(year_records, schema=SCHEMA)
        pq.write_table(table, path, compression="snappy")
        print(f"[OHLCV] {ticker}/{year}: {len(year_records)} rows → {path}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download OHLCV price data for all watchlist tickers.")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Download maximum available history (~20y). Omit for daily 5-day refresh.",
    )
    args = parser.parse_args()

    period = "max" if args.bootstrap else "5d"
    output_dir = Path("data/raw")

    for ticker in TICKERS:
        print(f"[OHLCV] Downloading {ticker} (period={period})...")
        records = fetch_ohlcv(ticker, period=period)
        save_ohlcv(records, ticker, output_dir)
        time.sleep(1)  # Rate limit — Yahoo Finance fair use
    print(f"[OHLCV] Done. {len(TICKERS)} tickers written to {output_dir}/financials/ohlcv/")
