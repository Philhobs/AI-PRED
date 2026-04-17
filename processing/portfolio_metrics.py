"""
Portfolio metrics post-processor.

Enriches raw predictions with:
  market_cap_b        — market cap in billions (yfinance, cached daily)
  is_liquid           — True if market_cap_b >= 1.0
  model_agreement     — fraction of sub-models agreeing with ensemble direction
  peer_correlation_90d — avg pairwise 90d return correlation with other top-10 picks

Usage:
  python processing/portfolio_metrics.py 2026-04-15   # enrich specific date
  python processing/portfolio_metrics.py              # enrich today
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yfinance as yf

_LOG = logging.getLogger(__name__)
_LIQUIDITY_THRESHOLD_B = 1.0  # $1B minimum market cap


def _model_agreement(df: pl.DataFrame) -> pl.Series:
    """Fraction of sub-models (lgbm, rf, ridge) agreeing with ensemble direction.

    1.0 = all three point same way as ensemble.
    0.67 = two of three agree.
    0.33 = one of three agrees.
    0.0 = none agree.
    """
    ensemble_sign = np.sign(df["expected_annual_return"].to_numpy())
    agreements = []
    for col in ["lgbm_return", "rf_return", "ridge_return"]:
        sub_sign = np.sign(df[col].to_numpy())
        agreements.append((sub_sign == ensemble_sign).astype(float))
    mean_agreement = (agreements[0] + agreements[1] + agreements[2]) / 3.0
    return pl.Series("model_agreement", mean_agreement.tolist())


def _apply_liquidity(df: pl.DataFrame, caps: dict[str, float | None]) -> pl.DataFrame:
    """Add market_cap_b and is_liquid columns using preloaded caps dict."""
    market_cap_b = pl.Series(
        "market_cap_b",
        [caps.get(t) for t in df["ticker"].to_list()],
        dtype=pl.Float64,
    )
    is_liquid = pl.Series(
        "is_liquid",
        [
            (c is not None and c >= _LIQUIDITY_THRESHOLD_B)
            for c in market_cap_b.to_list()
        ],
        dtype=pl.Boolean,
    )
    return df.with_columns([market_cap_b, is_liquid])


def _get_market_caps(tickers: list[str], as_of: date) -> dict[str, float | None]:
    """
    Return {ticker: market_cap_billions} for all tickers.

    Uses a daily cache at data/raw/financials/market_caps.parquet.
    Fetches from yfinance only for tickers missing from today's cache.
    """
    cache_path = Path("data/raw/financials/market_caps.parquet")

    cached: dict[str, float | None] = {}
    if cache_path.exists():
        try:
            cache_df = pl.read_parquet(cache_path).filter(pl.col("date") == as_of)
            valid = cache_df.filter(pl.col("market_cap_b").is_not_null())
            cached = dict(zip(valid["ticker"].to_list(), valid["market_cap_b"].to_list()))
        except Exception as exc:
            _LOG.warning("Failed to read market cap cache: %s", exc)

    missing = [t for t in tickers if t not in cached]
    if not missing:
        return {t: cached.get(t) for t in tickers}

    fetched: dict[str, float | None] = {}
    for ticker in missing:
        try:
            info = yf.Ticker(ticker).fast_info
            market_cap = getattr(info, "market_cap", None)
            fetched[ticker] = round(market_cap / 1e9, 3) if market_cap else None
        except Exception as exc:
            _LOG.debug("Failed to fetch market cap for %s: %s", ticker, exc)
            fetched[ticker] = None
        time.sleep(0.1)

    if fetched:
        new_rows = pl.DataFrame({
            "ticker": list(fetched.keys()),
            "date": [as_of] * len(fetched),
            "market_cap_b": list(fetched.values()),
        })
        if cache_path.exists():
            existing = pl.read_parquet(cache_path).filter(pl.col("date") != as_of)
            pl.concat([existing, new_rows]).write_parquet(cache_path, compression="snappy")
        else:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            new_rows.write_parquet(cache_path, compression="snappy")

    all_caps = {**cached, **fetched}
    return {t: all_caps.get(t) for t in tickers}


def _peer_correlation(
    df: pl.DataFrame,
    ohlcv_dir: Path,
    as_of: date,
    top_n: int = 10,
) -> pl.Series:
    """
    For each ticker, compute its average pairwise 90-day return correlation
    with the other top-(top_n-1) tickers by rank.

    Returns null for tickers ranked > top_n.
    High value (>0.65) means highly correlated with other basket members.
    """
    top_tickers = df.sort("rank").head(top_n)["ticker"].to_list()
    all_tickers = df["ticker"].to_list()
    window_start = as_of - timedelta(days=90)

    prices: dict[str, np.ndarray] = {}
    for ticker in top_tickers:
        ticker_parquets = list(ohlcv_dir.glob(f"{ticker}/*.parquet"))
        if not ticker_parquets:
            continue
        try:
            p = (
                pl.scan_parquet([str(x) for x in ticker_parquets])
                .filter(
                    (pl.col("date") >= window_start) & (pl.col("date") <= as_of)
                )
                .select(["date", "close_price"])
                .collect()
                .sort("date")
            )
            if len(p) > 5:
                returns = p["close_price"].pct_change().drop_nulls().to_numpy()
                if len(returns) > 5:
                    prices[ticker] = returns
        except Exception as exc:
            _LOG.debug("Failed to load OHLCV for %s: %s", ticker, exc)

    corr_map: dict[str, float | None] = {}
    for ticker in top_tickers:
        if ticker not in prices:
            corr_map[ticker] = None
            continue
        peers = [t for t in top_tickers if t != ticker and t in prices]
        if not peers:
            corr_map[ticker] = 0.0
            continue
        peer_corrs = []
        for peer in peers:
            min_len = min(len(prices[ticker]), len(prices[peer]))
            if min_len < 10:
                continue
            c = float(np.corrcoef(prices[ticker][-min_len:], prices[peer][-min_len:])[0, 1])
            if not np.isnan(c):
                peer_corrs.append(c)
        corr_map[ticker] = round(float(np.mean(peer_corrs)), 4) if peer_corrs else None

    return pl.Series(
        "peer_correlation_90d",
        [corr_map.get(t) for t in all_tickers],
        dtype=pl.Float64,
    )


def enrich(date_str: str, predictions_dir: Path | None = None) -> pl.DataFrame:
    """
    Enrich predictions for a given date with portfolio metrics.

    Reads: data/predictions/date={date_str}/predictions.parquet
    Writes: data/predictions/date={date_str}/predictions_enriched.parquet
    Returns: enriched DataFrame
    """
    if predictions_dir is None:
        predictions_dir = Path("data/predictions")

    pred_path = predictions_dir / f"date={date_str}" / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions at {pred_path}. Run inference first.")

    df = pl.read_parquet(pred_path)
    as_of = date.fromisoformat(date_str)

    # 1. Market cap + liquidity
    caps = _get_market_caps(df["ticker"].to_list(), as_of)
    df = _apply_liquidity(df, caps)

    # 2. Model agreement
    df = df.with_columns(_model_agreement(df))

    # 3. Peer correlation (top-10 vs each other)
    ohlcv_dir = Path("data/raw/financials/ohlcv")
    df = df.with_columns(_peer_correlation(df, ohlcv_dir, as_of))

    # Write enriched output
    out_path = predictions_dir / f"date={date_str}" / "predictions_enriched.parquet"
    df.write_parquet(out_path, compression="snappy")

    liquid = df.filter(pl.col("is_liquid")).sort("rank")
    _LOG.info(
        "[PortfolioMetrics] %d liquid picks (≥$1B market cap) out of %d total",
        len(liquid), len(df),
    )
    print(f"\nLiquid picks for {date_str} (market cap ≥ $1B):\n")
    print(
        liquid.select([
            "rank", "ticker", "layer",
            "expected_annual_return", "market_cap_b",
            "model_agreement", "peer_correlation_90d",
        ])
    )
    illiquid = df.filter(~pl.col("is_liquid"))["ticker"].to_list()
    if illiquid:
        print(f"\nExcluded (< $1B): {', '.join(illiquid)}")

    return df


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    date_str = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    enrich(date_str)
