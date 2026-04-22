"""
Ingest raw options chain data for all registered tickers via yfinance (or pluggable paid source).

Raw storage: data/raw/options/date=YYYY-MM-DD/{ticker}.parquet
Schema: (ticker, date, expiry, option_type, strike, iv, oi, volume)

Adding a paid source (Tradier, CBOE): implement OptionsSource.fetch() in a new class
and pass it to ingest_options(source=...).

Usage:
    python ingestion/options_ingestion.py               # fetch today
    python ingestion/options_ingestion.py --date 2024-01-15
"""
from __future__ import annotations

import argparse
import datetime
import logging
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
import polars as pl

from ingestion.ticker_registry import TICKERS

_LOG = logging.getLogger(__name__)

_SCHEMA = {
    "ticker": pl.Utf8,
    "date": pl.Date,
    "expiry": pl.Date,
    "option_type": pl.Utf8,
    "strike": pl.Float64,
    "iv": pl.Float64,
    "oi": pl.Int64,
    "volume": pl.Int64,
}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=_SCHEMA)


@runtime_checkable
class OptionsSource(Protocol):
    def fetch(self, ticker: str, date: str) -> pl.DataFrame:
        """Return raw options contracts for ticker on date.

        Returns DataFrame with schema matching _SCHEMA.
        Returns empty DataFrame if no options data is available.
        """
        ...


class YFinanceOptionsSource:
    """Fetch the full options chain via yfinance (free, no API key required)."""

    def fetch(self, ticker: str, date: str) -> pl.DataFrame:
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            expiry_dates = t.options
        except Exception as exc:
            _LOG.debug("No options data for %s: %s", ticker, exc)
            return _empty()

        if not expiry_dates:
            return _empty()

        fetch_date = datetime.date.fromisoformat(date)
        rows: list[dict] = []

        for expiry_str in expiry_dates:
            try:
                expiry_date = datetime.date.fromisoformat(expiry_str)
                chain = t.option_chain(expiry_str)
                for opt_type, opts_df in [("call", chain.calls), ("put", chain.puts)]:
                    for _, row in opts_df.iterrows():
                        rows.append({
                            "ticker": ticker,
                            "date": fetch_date,
                            "expiry": expiry_date,
                            "option_type": opt_type,
                            "strike": 0.0 if pd.isna(row.get("strike")) else float(row["strike"]),
                            "iv": 0.0 if pd.isna(row.get("impliedVolatility")) else float(row["impliedVolatility"]),
                            "oi": 0 if pd.isna(row.get("openInterest")) else int(row["openInterest"]),
                            "volume": 0 if pd.isna(row.get("volume")) else int(row["volume"]),
                        })
            except Exception as exc:
                _LOG.debug("Error fetching expiry %s for %s: %s", expiry_str, ticker, exc)
                continue

        if not rows:
            return _empty()

        return pl.DataFrame(rows, schema=_SCHEMA)


def ingest_options(
    tickers: list[str],
    date_str: str,
    output_dir: Path,
    source: OptionsSource | None = None,
) -> None:
    """Fetch options chain for all tickers and write Hive-partitioned parquet.

    Tickers with no options data (e.g. DARK.L) are silently skipped — no file written.
    time.sleep(1.0) is called between every ticker to respect yfinance rate limits.
    """
    if source is None:
        source = YFinanceOptionsSource()

    for ticker in tickers:
        df = source.fetch(ticker, date_str)
        if not df.is_empty():
            out_dir = output_dir / f"date={date_str}"
            out_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_dir / f"{ticker}.parquet", compression="snappy")
            _LOG.info("Wrote %d contracts for %s on %s", len(df), ticker, date_str)
        else:
            _LOG.debug("No options data for %s on %s — skipping", ticker, date_str)
        time.sleep(1.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest raw options chain data")
    parser.add_argument(
        "--date",
        default=str(datetime.date.today()),
        help="Date to fetch (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    output_dir = Path("data/raw/options")
    _LOG.info("Fetching options for %d tickers on %s", len(TICKERS), args.date)
    ingest_options(TICKERS, args.date, output_dir)
