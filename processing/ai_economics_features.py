"""AI economics features — Sequoia ratio + hyperscaler capex aggregates.

Three macro features applied uniformly to all tickers (model decides per-ticker
weight). Inputs come from ingestion/ai_economics_ingestion.py:

  ai_capex_coverage_ratio       — TTM hyperscaler capex / TTM hyperscaler revenue.
                                  This is the Sequoia "$600B question" ratio:
                                  historically ~10-15%; sustained 25%+ is the
                                  AI-bubble warning sign.
  hyperscaler_capex_aggregate   — TTM aggregate hyperscaler capex (USD billions).
                                  Direct read of AI-infra demand strength.
  hyperscaler_capex_yoy         — yoy growth of TTM aggregate capex. Rate-of-
                                  change signal (turning point indicator).

All three are forward-filled within 120-day tolerance (matches quarterly cadence).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)

AI_ECONOMICS_FEATURE_COLS: list[str] = [
    "ai_capex_coverage_ratio",
    "hyperscaler_capex_aggregate",
    "hyperscaler_capex_yoy",
]

_TOLERANCE_DAYS = 120


def _load_quarterly(raw_path: Path) -> pl.DataFrame:
    if not raw_path.exists():
        return pl.DataFrame(schema={
            "ticker": pl.Utf8, "period_end": pl.Date,
            "revenue": pl.Float64, "capex": pl.Float64,
        })
    return pl.read_parquet(raw_path).sort(["ticker", "period_end"])


def _ttm_aggregate(df: pl.DataFrame, as_of: date) -> tuple[float | None, float | None]:
    """TTM (last 4 quarters) aggregate revenue and capex across all tickers as of as_of.

    Returns (None, None) if fewer than 4 distinct period_end dates fall within the
    TTM window for any ticker — the aggregate would be misleading.
    """
    cutoff = as_of - timedelta(days=400)   # 4 quarters + buffer for late filers
    eligible = df.filter(
        (pl.col("period_end") <= as_of) & (pl.col("period_end") >= cutoff)
    )
    if eligible.is_empty():
        return None, None

    # Per-ticker: keep the 4 most recent periods
    per_ticker = (
        eligible.sort(["ticker", "period_end"], descending=[False, True])
        .group_by("ticker", maintain_order=True)
        .head(4)
    )
    # Need at least 4 quarters per ticker to have a real TTM
    counts = per_ticker.group_by("ticker").len()
    if counts.filter(pl.col("len") < 4).height > 0:
        return None, None

    rev = float(per_ticker["revenue"].sum())
    capex = float(per_ticker["capex"].sum())
    return rev, capex


def _yoy(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / prior


def join_ai_economics_features(
    spine: pl.DataFrame,
    raw_path: Path = Path("data/raw/financials/ai_economics/hyperscalers_quarterly.parquet"),
) -> pl.DataFrame:
    """Join the 3 AI economics features onto spine (must have 'ticker', 'date' Date)."""
    raw = _load_quarterly(raw_path)
    unique_dates = spine.select("date").unique().sort("date")["date"].to_list()
    rows: list[dict] = []
    for d in unique_dates:
        rev, capex = _ttm_aggregate(raw, d)
        rev_prior, capex_prior = _ttm_aggregate(raw, d - timedelta(days=365))

        ratio = (capex / rev) if (rev and capex is not None and rev > 0) else None
        capex_b = (capex / 1e9) if capex is not None else None
        yoy = _yoy(capex, capex_prior)

        rows.append({
            "date": d,
            "ai_capex_coverage_ratio": ratio,
            "hyperscaler_capex_aggregate": capex_b,
            "hyperscaler_capex_yoy": yoy,
        })

    schema: dict = {"date": pl.Date}
    for col in AI_ECONOMICS_FEATURE_COLS:
        schema[col] = pl.Float64
    feature_df = pl.DataFrame(rows, schema=schema)
    return spine.join(feature_df, on="date", how="left")
