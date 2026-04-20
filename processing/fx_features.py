"""FX-adjusted return features for non-USD tickers.

Provides:
  build_usd_close_matrix() — convert non-USD close prices to USD via daily FX rates
  join_fx_features()       — add fx_adjusted_return_20d to training spine

Called by models/train.py via join_fx_features().
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ingestion.fx_ingestion import CURRENCY_TO_PAIR
from ingestion.ticker_registry import TICKER_CURRENCY

_LOG = logging.getLogger(__name__)


def _load_fx_rate(pair: str, fx_dir: Path) -> pl.DataFrame:
    """Load FX rate parquet. Returns empty DataFrame(date, rate) if not found."""
    path = fx_dir / f"{pair}.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"date": pl.Date, "rate": pl.Float64})
    return pl.read_parquet(path)


def build_usd_close_matrix(
    close_wide: pl.DataFrame,
    fx_dir: Path,
) -> pl.DataFrame:
    """Convert non-USD ticker columns to USD using daily FX rates.

    USD tickers pass through unchanged.
    Non-USD tickers: close_price * fx_rate_on_date. Missing rate → null.

    Args:
        close_wide: Wide close-price DataFrame with date column and one column per ticker.
        fx_dir: Directory containing {pair}.parquet FX rate files.

    Returns:
        Same shape as close_wide with non-USD columns converted to USD.
    """
    result = close_wide.clone()
    ticker_cols = [c for c in close_wide.columns if c != "date"]

    # Group non-USD tickers by FX pair to minimise file reads
    pair_to_tickers: dict[str, list[str]] = {}
    for ticker in ticker_cols:
        currency = TICKER_CURRENCY.get(ticker, "USD")
        if currency == "USD":
            continue
        pair = CURRENCY_TO_PAIR.get(currency)
        if pair is None:
            _LOG.warning("[FX] No pair for currency %s (ticker %s) — skipping", currency, ticker)
            continue
        pair_to_tickers.setdefault(pair, []).append(ticker)

    for pair, tickers in pair_to_tickers.items():
        fx_df = _load_fx_rate(pair, fx_dir)
        if fx_df.is_empty():
            # Null out all tickers for this pair
            result = result.with_columns([
                pl.lit(None).cast(pl.Float64).alias(t) for t in tickers
            ])
            continue

        rate_col = f"_fx_{pair}"
        result = result.join(fx_df.rename({"rate": rate_col}), on="date", how="left")
        for ticker in tickers:
            result = result.with_columns(
                (pl.col(ticker) * pl.col(rate_col)).alias(ticker)
            )
        result = result.drop(rate_col)

    return result


def _build_close_matrix_for_tickers(
    ohlcv_dir: Path,
    tickers: list[str],
    min_date: date,
    max_date: date,
    extra_days: int = 100,
) -> pl.DataFrame:
    """Build wide close-price matrix for a specific list of tickers.

    Mirrors supply_chain_features._build_close_matrix but scoped to given tickers
    to avoid a circular import between fx_features and supply_chain_features.
    """
    load_start = min_date - timedelta(days=extra_days)

    frames = []
    for ticker in tickers:
        ticker_dir = ohlcv_dir / ticker
        if not ticker_dir.exists():
            continue
        year_files = sorted(
            f for f in ticker_dir.glob("*.parquet")
            if f.stem.isdigit() and int(f.stem) >= load_start.year
        )
        if not year_files:
            continue
        df = pl.concat([pl.read_parquet(f) for f in year_files])
        df = (
            df.filter((pl.col("date") >= load_start) & (pl.col("date") <= max_date))
            .sort("date")
            .select(["date", pl.col("close_price").alias(ticker)])
        )
        frames.append(df)

    if not frames:
        return pl.DataFrame({"date": pl.Series([], dtype=pl.Date)})

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="date", how="full", coalesce=True)
    return result.sort("date")


def join_fx_features(
    df: pl.DataFrame,
    fx_dir: Path | None = None,
    ohlcv_dir: Path | None = None,
) -> pl.DataFrame:
    """Add fx_adjusted_return_20d: 20-day cumulative return on USD-normalised close prices.

    - Non-USD tickers: populated Float64.
    - USD tickers: null (their price features already capture USD returns).

    Args:
        df: Training spine with columns [ticker, date, ...].
        fx_dir: Path to data/raw/financials/fx/ (default: resolved from __file__).
        ohlcv_dir: Path to data/raw/financials/ohlcv/ (default: resolved from __file__).

    Returns:
        df with 1 new Float64 column. Missing data → null (not 0).
    """
    _ROOT = Path(__file__).parent.parent
    if fx_dir is None:
        fx_dir = _ROOT / "data" / "raw" / "financials" / "fx"
    if ohlcv_dir is None:
        ohlcv_dir = _ROOT / "data" / "raw" / "financials" / "ohlcv"

    null_col = pl.lit(None).cast(pl.Float64).alias("fx_adjusted_return_20d")

    if not fx_dir.exists():
        _LOG.warning("[FX] No fx_dir found — fx_adjusted_return_20d will be null")
        return df.with_columns(null_col)

    # Only compute for non-USD tickers that appear in the spine
    spine_tickers = df["ticker"].unique().to_list()
    intl_tickers = [t for t in spine_tickers if TICKER_CURRENCY.get(t, "USD") != "USD"]

    if not intl_tickers:
        return df.with_columns(null_col)

    min_date = df["date"].min()
    max_date = df["date"].max()

    close_wide = _build_close_matrix_for_tickers(ohlcv_dir, intl_tickers, min_date, max_date)
    if close_wide.is_empty() or close_wide.height < 21:
        return df.with_columns(null_col)

    usd_close = build_usd_close_matrix(close_wide, fx_dir)

    ticker_cols = [c for c in usd_close.columns if c != "date"]
    ret_20d_usd = usd_close.select(
        ["date"] + [
            (pl.col(t) / pl.col(t).shift(20) - 1).alias(t)
            for t in ticker_cols
        ]
    )

    # Melt wide ret_20d_usd → long, then left-join onto spine
    ret_long = ret_20d_usd.unpivot(
        index="date", variable_name="ticker", value_name="fx_adjusted_return_20d"
    )
    return df.join(ret_long, on=["ticker", "date"], how="left")
