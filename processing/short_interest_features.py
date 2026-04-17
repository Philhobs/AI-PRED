"""
Short interest feature computation.

Source: FINRA daily short sale volume (short_vol_ratio = ShortVolume / TotalVolume)

Features per ticker per OHLCV date:
  short_vol_ratio_10d   — mean short_vol_ratio over 10-day window
  short_vol_ratio_30d   — mean short_vol_ratio over 30-day window
  short_ratio_momentum  — short_vol_ratio_10d minus prior 10-day mean (days 11-20)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

_LOG = logging.getLogger(__name__)

# Maximum age of a short interest observation to propagate forward (1 trading week).
_SI_STALE_DAYS = 7


# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers — used in unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _compute_short_ratio_10d(
    si_df: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 10,
) -> float | None:
    """Mean short_vol_ratio for ticker in [as_of-window, as_of]. None if no data."""
    if si_df.is_empty():
        return None
    with duckdb.connect() as con:
        con.register("si", si_df)
        row = con.execute("""
            SELECT AVG(short_vol_ratio)
            FROM si
            WHERE ticker = ?
              AND date BETWEEN CAST(? AS DATE) - (? * INTERVAL '1 DAY') AND CAST(? AS DATE)
        """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    return row[0] if row and row[0] is not None else None


def _compute_short_ratio_30d(
    si_df: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 30,
) -> float | None:
    """Mean short_vol_ratio for ticker in [as_of-30d, as_of]. None if no data."""
    return _compute_short_ratio_10d(si_df, ticker, as_of, window_days)


def _compute_short_ratio_momentum(
    si_df: pl.DataFrame,
    ticker: str,
    as_of: date,
) -> float | None:
    """short_vol_ratio_10d minus prior 10-day mean (days 11-20). None if either window empty."""
    if si_df.is_empty():
        return None
    with duckdb.connect() as con:
        con.register("si", si_df)
        row = con.execute("""
            SELECT
                AVG(CASE WHEN date BETWEEN CAST(? AS DATE) - 10 * INTERVAL '1 DAY'
                                       AND CAST(? AS DATE)
                         THEN short_vol_ratio END) AS current_mean,
                AVG(CASE WHEN date BETWEEN CAST(? AS DATE) - 20 * INTERVAL '1 DAY'
                                       AND CAST(? AS DATE) - 11 * INTERVAL '1 DAY'
                         THEN short_vol_ratio END) AS prior_mean
            FROM si
            WHERE ticker = ?
        """, [str(as_of), str(as_of), str(as_of), str(as_of), ticker]).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    return row[0] - row[1]


# ─────────────────────────────────────────────────────────────────────────────
# Bulk DuckDB computation over OHLCV spine
# ─────────────────────────────────────────────────────────────────────────────

def compute_short_interest_features(
    short_interest_path: Path,
    ohlcv_dir: Path,
) -> pl.DataFrame:
    """Compute 3 rolling short interest features for all tickers × OHLCV dates.

    Returns DataFrame: [ticker, date, short_vol_ratio_10d, short_vol_ratio_30d,
                        short_ratio_momentum]
    """
    ohlcv_parquets = list(ohlcv_dir.glob("**/*.parquet"))
    if not ohlcv_parquets:
        _LOG.error("No OHLCV parquets found in %s", ohlcv_dir)
        return pl.DataFrame()

    with duckdb.connect() as con:
        ohlcv_glob = str(ohlcv_dir / "**" / "*.parquet")
        con.execute("""
            CREATE TEMP TABLE spine AS
            SELECT DISTINCT ticker, date
            FROM read_parquet(?)
            ORDER BY ticker, date
        """, [ohlcv_glob])

        if short_interest_path.exists():
            con.execute("""
                CREATE TEMP TABLE si AS
                SELECT date, ticker, short_vol_ratio
                FROM read_parquet(?)
                WHERE short_vol_ratio IS NOT NULL
            """, [str(short_interest_path)])
        else:
            _LOG.warning(
                "No short interest parquet at %s — features will be null",
                short_interest_path,
            )
            con.execute("""
                CREATE TEMP TABLE si (
                    date DATE,
                    ticker VARCHAR,
                    short_vol_ratio DOUBLE
                )
            """)

        result_df = con.execute("""
            WITH agg10 AS (
                SELECT s.ticker, s.date,
                    AVG(si10.short_vol_ratio) AS short_vol_ratio_10d
                FROM spine s
                LEFT JOIN si si10
                    ON si10.ticker = s.ticker
                   AND si10.date BETWEEN s.date - INTERVAL 10 DAY AND s.date
                GROUP BY s.ticker, s.date
            ),
            agg30 AS (
                SELECT s.ticker, s.date,
                    AVG(si30.short_vol_ratio) AS short_vol_ratio_30d
                FROM spine s
                LEFT JOIN si si30
                    ON si30.ticker = s.ticker
                   AND si30.date BETWEEN s.date - INTERVAL 30 DAY AND s.date
                GROUP BY s.ticker, s.date
            ),
            agg_prior AS (
                SELECT s.ticker, s.date,
                    AVG(sip.short_vol_ratio) AS prior_ratio_10d
                FROM spine s
                LEFT JOIN si sip
                    ON sip.ticker = s.ticker
                   AND sip.date BETWEEN s.date - INTERVAL 20 DAY
                       AND s.date - INTERVAL 11 DAY
                GROUP BY s.ticker, s.date
            )
            SELECT
                a10.ticker,
                a10.date,
                a10.short_vol_ratio_10d,
                a30.short_vol_ratio_30d,
                a10.short_vol_ratio_10d - ap.prior_ratio_10d AS short_ratio_momentum
            FROM agg10 a10
            LEFT JOIN agg30 a30 USING (ticker, date)
            LEFT JOIN agg_prior ap USING (ticker, date)
            ORDER BY a10.ticker, a10.date
        """).pl()

    _LOG.info(
        "Computed short interest features: %d rows for %d tickers",
        len(result_df),
        result_df["ticker"].n_unique(),
    )
    return result_df


def save_short_interest_features(df: pl.DataFrame, output_dir: Path) -> None:
    """Write per-ticker daily parquets to output_dir/<TICKER>/si_daily.parquet."""
    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        out_path = output_dir / ticker / "si_daily.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ticker_df.write_parquet(out_path, compression="snappy")
    _LOG.info(
        "Saved short interest features for %d tickers to %s",
        df["ticker"].n_unique(),
        output_dir,
    )


def join_short_interest_features(df: pl.DataFrame, si_features_dir: Path) -> pl.DataFrame:
    """Backward asof join short interest features onto training DataFrame on (ticker, date).

    Adds null columns for tickers with no data.
    """
    feature_cols = ["short_vol_ratio_10d", "short_vol_ratio_30d", "short_ratio_momentum"]
    parquet_glob = str(si_features_dir / "*/si_daily.parquet")
    if not any(si_features_dir.glob("*/si_daily.parquet")):
        _LOG.warning(
            "No short interest feature parquets in %s — features will be null",
            si_features_dir,
        )
        for col in feature_cols:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        return df

    # collect() is intentional: join_asof requires a materialised, sorted DataFrame.
    features = pl.scan_parquet(parquet_glob).sort(["ticker", "date"]).collect()
    features_renamed = features.rename({"date": "si_date"})

    df_sorted = df.sort(["ticker", "date"])
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="si_date",
        by="ticker",
        strategy="backward",
        tolerance=timedelta(days=_SI_STALE_DAYS),
    )

    non_null = result["short_vol_ratio_10d"].drop_nulls().len()
    _LOG.info(
        "Joined short interest features: %d rows, %d non-null short_vol_ratio_10d values",
        len(result),
        non_null,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — compute and save features
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    si_path = project_root / "data" / "raw" / "financials" / "short_interest" / "short_interest_daily.parquet"
    ohlcv_dir = project_root / "data" / "raw" / "financials" / "ohlcv"
    output_dir = project_root / "data" / "raw" / "financials" / "short_interest_features"

    df = compute_short_interest_features(si_path, ohlcv_dir)
    if len(df) == 0:
        _LOG.error(
            "No features computed. Run: python ingestion/short_interest_ingestion.py"
        )
        sys.exit(1)

    save_short_interest_features(df, output_dir)
    _LOG.info("Done. Retrain with: python models/train.py")
