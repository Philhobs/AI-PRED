"""Hyperscaler raw capex + revenue ingestion (Sequoia ratio inputs).

Fetches quarterly raw revenue and capex for the hyperscaler subset (US cloud
layer) so we can compute aggregate ratios that EDGAR's derived-ratio output
doesn't expose. Powers two macro features in processing/ai_economics_features.py:

  ai_capex_coverage_ratio       — TTM hyperscaler capex / TTM hyperscaler revenue
  hyperscaler_capex_aggregate   — TTM sum capex (USD billions)
  hyperscaler_capex_yoy         — yoy growth in TTM capex

Output: data/raw/financials/ai_economics/hyperscalers_quarterly.parquet
Schema: ticker (Utf8), period_end (Date), revenue (Float64), capex (Float64).

Free, no API key required (yfinance). Fail-soft per ticker.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import polars as pl
import yfinance as yf

_LOG = logging.getLogger(__name__)

# Hyperscalers — US cloud layer minus IBM (legacy), foreign (SAP/CAP/OVH).
# Sequoia's $600B framing is specifically about these big-3-plus capex spenders.
HYPERSCALER_TICKERS: tuple[str, ...] = (
    "MSFT", "AMZN", "GOOGL", "META", "ORCL",
)

_SCHEMA = {
    "ticker": pl.Utf8,
    "period_end": pl.Date,
    "revenue": pl.Float64,
    "capex": pl.Float64,
}


def fetch_hyperscaler_quarterly(ticker: str) -> pl.DataFrame:
    """Pull the most recent ~4-8 quarters of revenue + capex for one ticker.

    Returns empty DataFrame on any failure (fail-soft). yfinance returns:
      ticker.quarterly_financials  → DataFrame indexed by metric, columns = period_end
      ticker.quarterly_cashflow    → same shape
    Capex sign convention: yfinance reports it as negative (cash outflow); we
    take abs() for downstream arithmetic.
    """
    try:
        t = yf.Ticker(ticker)
        fin = t.quarterly_financials   # rows include "Total Revenue"
        cash = t.quarterly_cashflow    # rows include "Capital Expenditure"
    except Exception as exc:  # noqa: BLE001 — fail-soft per project convention
        _LOG.warning("[ai_economics] %s: yfinance fetch failed (%s); returning empty", ticker, exc)
        return pl.DataFrame(schema=_SCHEMA)

    if fin is None or fin.empty or cash is None or cash.empty:
        _LOG.warning("[ai_economics] %s: empty financials or cashflow", ticker)
        return pl.DataFrame(schema=_SCHEMA)

    # yfinance row labels are case-insensitive but exact match preferred
    revenue_row = next(
        (r for r in ("Total Revenue", "TotalRevenue", "Revenue") if r in fin.index),
        None,
    )
    capex_row = next(
        (r for r in ("Capital Expenditure", "CapitalExpenditure", "Capex") if r in cash.index),
        None,
    )
    if revenue_row is None or capex_row is None:
        _LOG.warning(
            "[ai_economics] %s: missing revenue/capex row (rev=%s, capex=%s)",
            ticker, revenue_row, capex_row,
        )
        return pl.DataFrame(schema=_SCHEMA)

    rev_series = fin.loc[revenue_row]
    capex_series = cash.loc[capex_row]
    common_dates = sorted(set(rev_series.index) & set(capex_series.index))
    if not common_dates:
        return pl.DataFrame(schema=_SCHEMA)

    rows = []
    for d in common_dates:
        try:
            rev = float(rev_series.loc[d])
            capex_raw = float(capex_series.loc[d])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "ticker": ticker,
            "period_end": d.date() if hasattr(d, "date") else d,
            "revenue": rev,
            "capex": abs(capex_raw),
        })

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.DataFrame(rows, schema=_SCHEMA).sort(["ticker", "period_end"])


def fetch_all() -> pl.DataFrame:
    """Fetch all hyperscalers, concatenated. Sleep 1s between calls."""
    parts: list[pl.DataFrame] = []
    for ticker in HYPERSCALER_TICKERS:
        df = fetch_hyperscaler_quarterly(ticker)
        if not df.is_empty():
            parts.append(df)
        time.sleep(1)
    if not parts:
        return pl.DataFrame(schema=_SCHEMA)
    return pl.concat(parts).sort(["ticker", "period_end"])


def save_hyperscaler_quarterly(out_dir: Path, df: pl.DataFrame) -> None:
    """Write snappy parquet to data/raw/financials/ai_economics/hyperscalers_quarterly.parquet."""
    if df.is_empty():
        _LOG.warning("[ai_economics] empty DataFrame — skipping write")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "hyperscalers_quarterly.parquet", compression="snappy")
    _LOG.info("[ai_economics] wrote %d rows for %d tickers",
              len(df), df["ticker"].n_unique())


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _ROOT = Path(__file__).parent.parent
    out_dir = _ROOT / "data" / "raw" / "financials" / "ai_economics"
    _LOG.info("Fetching hyperscaler quarterly capex+revenue for %d tickers...",
              len(HYPERSCALER_TICKERS))
    df = fetch_all()
    save_hyperscaler_quarterly(out_dir, df)
