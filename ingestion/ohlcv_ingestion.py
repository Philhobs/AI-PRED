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
    Merge OHLCV records into Hive-style Parquet partitioned by year.

    Path: <output_dir>/financials/ohlcv/<TICKER>/<YEAR>.parquet
    All records for the same ticker+year are merged into a single file. When a
    file already exists for that year, this *unions* the new records with the
    existing rows (deduplicating by date, keeping the new value on conflict).

    This append-merge behavior was added 2026-05-03 — previously the function
    overwrote the file unconditionally, which truncated the year's history to
    whatever fetch period the caller used (e.g. period='5d' would shrink the
    year to 5 rows). The fix preserves history across daily refreshes while
    still letting --bootstrap rewrite from a fresh download.
    """
    if not records:
        return

    by_year: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_year[r["date"].year].append(r)

    for year, year_records in by_year.items():
        path = output_dir / "financials" / "ohlcv" / ticker / f"{year}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        new_table = pa.Table.from_pylist(year_records, schema=SCHEMA)

        if path.exists():
            existing = pq.read_table(path)
            # Concatenate, then dedupe by date keeping the LAST occurrence
            # (i.e. the freshly-fetched value wins over the stored copy).
            combined = pa.concat_tables([existing, new_table])
            df = combined.to_pandas().drop_duplicates(subset=["date"], keep="last").sort_values("date")
            merged = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
            pq.write_table(merged, path, compression="snappy")
            print(f"[OHLCV] {ticker}/{year}: merged {len(year_records)} new rows "
                  f"→ {merged.num_rows} total → {path}")
        else:
            pq.write_table(new_table, path, compression="snappy")
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
