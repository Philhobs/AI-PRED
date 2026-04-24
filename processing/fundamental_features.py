from pathlib import Path

import polars as pl

FUNDAMENTAL_FEATURE_COLS: list[str] = [
    # Existing 9
    "pe_ratio_trailing",
    "price_to_sales",
    "price_to_book",
    "revenue_growth_yoy",
    "gross_margin",
    "operating_margin",
    "capex_to_revenue",
    "debt_to_equity",
    "current_ratio",
    # 5 new TTM-based metrics
    "net_income_margin",
    "free_cash_flow_margin",
    "capex_growth_yoy",
    "revenue_growth_accel",
    "research_to_revenue",
]

_FUND_SCHEMA = {
    "ticker": pl.Utf8,
    "period_end": pl.Date,
    "pe_ratio_trailing": pl.Float64,
    "price_to_sales": pl.Float64,
    "price_to_book": pl.Float64,
    "revenue_growth_yoy": pl.Float64,
    "gross_margin": pl.Float64,
    "operating_margin": pl.Float64,
    "capex_to_revenue": pl.Float64,
    "debt_to_equity": pl.Float64,
    "current_ratio": pl.Float64,
    "net_income_margin": pl.Float64,
    "free_cash_flow_margin": pl.Float64,
    "capex_growth_yoy": pl.Float64,
    "revenue_growth_accel": pl.Float64,
    "research_to_revenue": pl.Float64,
}


def join_fundamentals(
    price_df: pl.DataFrame,
    fundamentals_dir: Path = Path("data/raw/financials/fundamentals"),
) -> pl.DataFrame:
    """
    Join the most recent available quarterly fundamental snapshot onto each
    price_df row using a backward asof join on date/period_end, per ticker.

    For each (ticker, date) row: selects the fundamental row where
    period_end <= date and period_end is maximised (most recent quarter).

    If no fundamentals exist for a ticker (or at all), fundamental columns
    are null — LightGBM handles nulls natively; Ridge/RF use saved imputation
    medians at training time.

    Args:
        price_df: DataFrame with at minimum columns [ticker, date].
        fundamentals_dir: Root directory containing <TICKER>/quarterly.parquet files.

    Returns:
        price_df with 14 additional fundamental columns appended.
    """
    null_fund_cols = [pl.lit(None).cast(pl.Float64).alias(c) for c in FUNDAMENTAL_FEATURE_COLS]

    glob = str(fundamentals_dir / "*" / "quarterly.parquet")

    try:
        fund_df = (
            pl.scan_parquet(glob, schema=_FUND_SCHEMA)
            .with_columns(pl.col("period_end").cast(pl.Date))
            .collect()
            .sort(["ticker", "period_end"])
        )
    except FileNotFoundError:
        return price_df.with_columns(null_fund_cols)

    if fund_df.is_empty():
        return price_df.with_columns(null_fund_cols)

    price_sorted = (
        price_df
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["ticker", "date"])
    )

    joined = price_sorted.join_asof(
        fund_df.select(["ticker", "period_end"] + FUNDAMENTAL_FEATURE_COLS),
        left_on="date",
        right_on="period_end",
        by="ticker",
        strategy="backward",
        check_sortedness=False,
    )

    # Drop the right join key (period_end) — keep output schema consistent with null-fallback path
    return joined.drop("period_end").sort(["ticker", "date"])
