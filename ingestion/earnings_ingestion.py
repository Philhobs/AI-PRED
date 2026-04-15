"""
Earnings surprise ingestion via yfinance.

For each watchlist ticker, fetches quarterly EPS actuals vs analyst consensus
estimates using yfinance.Ticker.earnings_history.

Output schema per row:
  ticker (str), quarter_end (date), eps_actual (float), eps_estimate (float),
  eps_surprise (float), eps_surprise_pct (float)

where eps_surprise = eps_actual - eps_estimate
      eps_surprise_pct = (eps_actual - eps_estimate) / abs(eps_estimate) when estimate != 0
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import polars as pl
import yfinance as yf

_LOG = logging.getLogger(__name__)


def fetch_earnings_surprises(tickers: list[str]) -> pl.DataFrame:
    """Fetch earnings surprise history for each ticker via yfinance.

    Returns DataFrame with columns:
      ticker, quarter_end, eps_actual, eps_estimate, eps_surprise, eps_surprise_pct
    """
    all_rows: list[dict] = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.earnings_history  # pandas DataFrame; None if unavailable
        except Exception as exc:
            _LOG.debug("%s: yfinance error: %s", ticker, exc)
            continue

        if hist is None or hist.empty:
            _LOG.debug("%s: no earnings history available", ticker)
            continue

        for quarter, row in hist.iterrows():
            eps_actual = row.get("epsActual")
            eps_estimate = row.get("epsEstimate")
            if eps_actual is None or eps_estimate is None:
                continue

            try:
                quarter_date = quarter.date() if hasattr(quarter, "date") else None
            except Exception:
                quarter_date = None
            if quarter_date is None:
                continue

            surprise = float(eps_actual) - float(eps_estimate)
            denom = abs(float(eps_estimate))
            surprise_pct = surprise / denom if denom > 1e-9 else None

            all_rows.append({
                "ticker": ticker,
                "quarter_end": quarter_date,
                "eps_actual": float(eps_actual),
                "eps_estimate": float(eps_estimate),
                "eps_surprise": surprise,
                "eps_surprise_pct": surprise_pct,
            })

        time.sleep(0.5)

    if not all_rows:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "quarter_end": pl.Date,
            "eps_actual": pl.Float64,
            "eps_estimate": pl.Float64,
            "eps_surprise": pl.Float64,
            "eps_surprise_pct": pl.Float64,
        })

    return (
        pl.DataFrame(all_rows)
        .with_columns(pl.col("quarter_end").cast(pl.Date))
        .sort(["ticker", "quarter_end"])
    )


def save_earnings_surprises(df: pl.DataFrame, output_dir: Path) -> None:
    """Write to output_dir/earnings_surprises.parquet."""
    out_path = output_dir / "earnings_surprises.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="snappy")
    _LOG.info("Saved %d earnings surprise rows to %s", len(df), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from ingestion.insider_trading_ingestion import CIK_MAP

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "raw" / "financials" / "earnings"

    tickers = list(CIK_MAP.keys())
    _LOG.info("Fetching earnings surprises for %d tickers...", len(tickers))
    df = fetch_earnings_surprises(tickers)
    _LOG.info("Fetched %d rows for %d tickers", len(df), df["ticker"].n_unique() if len(df) > 0 else 0)
    save_earnings_surprises(df, output_dir)
