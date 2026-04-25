"""Daily FX rate ingestion via yfinance.

Fetches 9 currency pairs needed for USD-normalizing non-USD tickers.
Saves to data/raw/financials/fx/{pair}.parquet (e.g., EURUSD.parquet).
Schema: date (pl.Date), rate (pl.Float64).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import polars as pl
import yfinance as yf

_LOG = logging.getLogger(__name__)

# Map from pair name to yfinance symbol
_FX_SYMBOLS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "CHFUSD": "CHFUSD=X",
    "JPYUSD": "JPYUSD=X",
    "DKKUSD": "DKKUSD=X",
    "SEKUSD": "SEKUSD=X",
    "NOKUSD": "NOKUSD=X",
    "GBPUSD": "GBPUSD=X",
    "HKDUSD": "HKDUSD=X",
    "KRWUSD": "KRWUSD=X",
}

# All currency codes covered by _FX_SYMBOLS (used externally for validation)
SUPPORTED_CURRENCIES: frozenset[str] = frozenset(
    {"EUR", "CHF", "JPY", "DKK", "SEK", "NOK", "GBP", "HKD", "KRW"}
)

# Map currency ISO code → pair name
CURRENCY_TO_PAIR: dict[str, str] = {
    "EUR": "EURUSD",
    "CHF": "CHFUSD",
    "JPY": "JPYUSD",
    "DKK": "DKKUSD",
    "SEK": "SEKUSD",
    "NOK": "NOKUSD",
    "GBP": "GBPUSD",
    "HKD": "HKDUSD",
    "KRW": "KRWUSD",
}

_EMPTY_SCHEMA = {"date": pl.Date, "rate": pl.Float64}


def fetch_fx_rates(
    pairs: list[str] | None = None,
    years: int = 5,
) -> dict[str, pl.DataFrame]:
    """Fetch daily closing FX rates from yfinance.

    Args:
        pairs: Subset of pair names to fetch (default: all 9).
        years: History to fetch in years.

    Returns:
        Dict mapping pair name → DataFrame(date, rate). Empty DataFrame on failure.
    """
    if pairs is None:
        pairs = list(_FX_SYMBOLS.keys())

    result: dict[str, pl.DataFrame] = {}
    for pair in pairs:
        if pair not in _FX_SYMBOLS:
            _LOG.warning("[FX] Unknown pair %s — skipping", pair)
            result[pair] = pl.DataFrame(schema=_EMPTY_SCHEMA)
            continue
        yf_symbol = _FX_SYMBOLS[pair]
        try:
            raw = yf.download(
                yf_symbol,
                period=f"{years}y",
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                _LOG.warning("[FX] No data returned for %s", pair)
                result[pair] = pl.DataFrame(schema=_EMPTY_SCHEMA)
                continue

            # Flatten MultiIndex columns (yfinance >= 0.2.38 default for single ticker)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"]
            df = (
                pl.from_pandas(close.reset_index())
                .rename({"Date": "date", "Close": "rate"})
                .with_columns(pl.col("date").cast(pl.Date))
                .sort("date")
                .select(["date", "rate"])
            )
            df = df.with_columns(pl.col("rate").cast(pl.Float64))
            result[pair] = df
            time.sleep(1)

        except Exception as exc:
            _LOG.warning("[FX] Failed to fetch %s (%s): %s", pair, yf_symbol, exc)
            result[pair] = pl.DataFrame(schema=_EMPTY_SCHEMA)

    return result


def save_fx_rates(fx_dir: Path, pairs: dict[str, pl.DataFrame]) -> None:
    """Append-and-deduplicate FX rates into data/raw/financials/fx/{pair}.parquet.

    Args:
        fx_dir: Destination directory (created if absent).
        pairs: Dict of pair name → DataFrame(date, rate) from fetch_fx_rates().
    """
    fx_dir.mkdir(parents=True, exist_ok=True)
    for pair, df in pairs.items():
        if df.is_empty():
            continue
        path = fx_dir / f"{pair}.parquet"
        if path.exists():
            try:
                existing = pl.read_parquet(path)
                df = pl.concat([existing, df]).unique("date", keep="last").sort("date")
            except Exception as exc:
                _LOG.warning("[FX] Failed to read existing %s, overwriting: %s", path.name, exc)
        df.write_parquet(path, compression="snappy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ROOT = Path(__file__).parent.parent
    fx_dir = _ROOT / "data" / "raw" / "financials" / "fx"
    _LOG.info("Fetching FX rates for %d pairs...", len(_FX_SYMBOLS))
    rates = fetch_fx_rates()
    save_fx_rates(fx_dir, rates)
    for pair, df in rates.items():
        _LOG.info("  %s: %d rows", pair, len(df))
