"""
Earnings surprise feature computation.

Source: yfinance quarterly EPS actual vs estimate (earnings_ingestion.py)

Features per ticker per OHLCV date:
  eps_surprise_last     — eps_surprise_pct of the most recent earnings release
  eps_surprise_mean_4q  — mean eps_surprise_pct over last 4 quarters (trailing year)
  eps_beat_streak       — count of consecutive quarters with positive eps_surprise
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

_LOG = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers — used in unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _compute_eps_surprise_last(
    earnings_df: pl.DataFrame,
    ticker: str,
    as_of: date,
) -> float | None:
    """Most recent eps_surprise_pct for ticker with quarter_end <= as_of.
    Returns None if no qualifying row.
    """
    if earnings_df.is_empty():
        return None
    with duckdb.connect() as con:
        con.register("earn", earnings_df)
        row = con.execute("""
            SELECT eps_surprise_pct
            FROM earn
            WHERE ticker = ?
              AND quarter_end <= CAST(? AS DATE)
            ORDER BY quarter_end DESC
            LIMIT 1
        """, [ticker, str(as_of)]).fetchone()
    return row[0] if row and row[0] is not None else None


def _compute_eps_surprise_mean_4q(
    earnings_df: pl.DataFrame,
    ticker: str,
    as_of: date,
) -> float | None:
    """Mean eps_surprise_pct over last 4 quarters with quarter_end <= as_of.
    Returns None if fewer than 1 qualifying quarter.
    """
    if earnings_df.is_empty():
        return None
    with duckdb.connect() as con:
        con.register("earn", earnings_df)
        row = con.execute("""
            SELECT AVG(eps_surprise_pct)
            FROM (
                SELECT eps_surprise_pct
                FROM earn
                WHERE ticker = ?
                  AND quarter_end <= CAST(? AS DATE)
                  AND eps_surprise_pct IS NOT NULL
                ORDER BY quarter_end DESC
                LIMIT 4
            ) t
        """, [ticker, str(as_of)]).fetchone()
    return row[0] if row and row[0] is not None else None


def _compute_eps_beat_streak(
    earnings_df: pl.DataFrame,
    ticker: str,
    as_of: date,
) -> int:
    """Count consecutive quarters with eps_surprise > 0 ending at most recent quarter.
    Returns 0 if no quarters or most recent was a miss.
    """
    if earnings_df.is_empty():
        return 0
    with duckdb.connect() as con:
        con.register("earn", earnings_df)
        rows = con.execute("""
            SELECT eps_surprise
            FROM earn
            WHERE ticker = ?
              AND quarter_end <= CAST(? AS DATE)
              AND eps_surprise IS NOT NULL
            ORDER BY quarter_end DESC
            LIMIT 8
        """, [ticker, str(as_of)]).fetchall()
    streak = 0
    for (surprise,) in rows:
        if surprise > 0:
            streak += 1
        else:
            break
    return streak


# ─────────────────────────────────────────────────────────────────────────────
# Bulk DuckDB computation over OHLCV spine
# ─────────────────────────────────────────────────────────────────────────────

def compute_earnings_features(
    earnings_path: Path,
    ohlcv_dir: Path,
) -> pl.DataFrame:
    """Compute 3 earnings surprise features for all tickers × OHLCV dates.

    For each (ticker, date) row in the OHLCV spine, looks up the most recent
    quarterly earnings with quarter_end <= date (backward asof join pattern).

    Returns DataFrame: [ticker, date, eps_surprise_last, eps_surprise_mean_4q,
                        eps_beat_streak]
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

        if earnings_path.exists():
            con.execute("""
                CREATE TEMP TABLE earn AS
                SELECT ticker, quarter_end, eps_surprise, eps_surprise_pct
                FROM read_parquet(?)
            """, [str(earnings_path)])
        else:
            _LOG.warning(
                "No earnings parquet at %s — features will be null",
                earnings_path,
            )
            con.execute("""
                CREATE TEMP TABLE earn (
                    ticker VARCHAR,
                    quarter_end DATE,
                    eps_surprise DOUBLE,
                    eps_surprise_pct DOUBLE
                )
            """)

        # For each spine row, get the most recent quarter_end <= date (backward asof)
        result_df = con.execute("""
            WITH ranked AS (
                SELECT
                    s.ticker,
                    s.date,
                    e.eps_surprise_pct,
                    e.eps_surprise,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.ticker, s.date
                        ORDER BY e.quarter_end DESC
                    ) AS rn
                FROM spine s
                LEFT JOIN earn e
                    ON e.ticker = s.ticker
                   AND e.quarter_end <= s.date
            ),
            last_q AS (
                SELECT ticker, date, eps_surprise_pct AS eps_surprise_last
                FROM ranked
                WHERE rn = 1
            ),
            mean_4q AS (
                SELECT s.ticker, s.date,
                    AVG(e4.eps_surprise_pct) AS eps_surprise_mean_4q
                FROM spine s
                LEFT JOIN (
                    SELECT ticker, quarter_end, eps_surprise_pct,
                        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY quarter_end DESC) AS qr
                    FROM earn
                ) e4
                    ON e4.ticker = s.ticker
                   AND e4.quarter_end <= s.date
                   AND e4.qr <= 4
                GROUP BY s.ticker, s.date
            )
            SELECT
                lq.ticker,
                lq.date,
                lq.eps_surprise_last,
                m4.eps_surprise_mean_4q
            FROM last_q lq
            LEFT JOIN mean_4q m4 USING (ticker, date)
            ORDER BY lq.ticker, lq.date
        """).pl()

    # eps_beat_streak: compute via polars (window function with consecutive run logic)
    # Load earnings data and compute streak per (ticker, date) efficiently.
    if earnings_path.exists():
        earn_df = pl.read_parquet(earnings_path).sort(["ticker", "quarter_end"])
        streak_rows: list[dict] = []

        spine_df = result_df.select(["ticker", "date"])
        for ticker in spine_df["ticker"].unique().to_list():
            ticker_earn = earn_df.filter(pl.col("ticker") == ticker)
            ticker_spine = spine_df.filter(pl.col("ticker") == ticker).sort("date")

            for row_date in ticker_spine["date"].to_list():
                past = ticker_earn.filter(
                    pl.col("quarter_end") <= row_date
                ).sort("quarter_end", descending=True)
                streak = 0
                for surprise in past["eps_surprise"].to_list():
                    if surprise is not None and surprise > 0:
                        streak += 1
                    else:
                        break
                streak_rows.append({"ticker": ticker, "date": row_date, "eps_beat_streak": streak})

        streak_df = pl.DataFrame(streak_rows).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("eps_beat_streak").cast(pl.Int32),
        )
        result_df = result_df.join(streak_df, on=["ticker", "date"], how="left")
    else:
        result_df = result_df.with_columns(pl.lit(None).cast(pl.Int32).alias("eps_beat_streak"))

    _LOG.info(
        "Computed earnings features: %d rows for %d tickers",
        len(result_df),
        result_df["ticker"].n_unique(),
    )
    return result_df


def save_earnings_features(df: pl.DataFrame, output_dir: Path) -> None:
    """Write per-ticker daily parquets to output_dir/<TICKER>/earnings_daily.parquet."""
    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        out_path = output_dir / ticker / "earnings_daily.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ticker_df.write_parquet(out_path, compression="snappy")
    _LOG.info(
        "Saved earnings features for %d tickers to %s",
        df["ticker"].n_unique(),
        output_dir,
    )


def join_earnings_features(df: pl.DataFrame, earnings_features_dir: Path) -> pl.DataFrame:
    """Backward asof join earnings features onto training DataFrame on (ticker, date).

    Adds null columns for tickers with no data.
    """
    feature_cols = ["eps_surprise_last", "eps_surprise_mean_4q", "eps_beat_streak"]
    parquet_glob = str(earnings_features_dir / "*/earnings_daily.parquet")
    if not any(earnings_features_dir.glob("*/earnings_daily.parquet")):
        _LOG.warning(
            "No earnings feature parquets in %s — features will be null",
            earnings_features_dir,
        )
        for col in feature_cols:
            dtype = pl.Int32 if col == "eps_beat_streak" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df

    # collect() is intentional: join_asof requires a materialised, sorted DataFrame.
    features = pl.scan_parquet(parquet_glob).sort(["ticker", "date"]).collect()
    features_renamed = features.rename({"date": "earn_date"})

    df_sorted = df.sort(["ticker", "date"])
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="earn_date",
        by="ticker",
        strategy="backward",
    )

    non_null = result["eps_surprise_last"].drop_nulls().len()
    _LOG.info(
        "Joined earnings features: %d rows, %d non-null eps_surprise_last values",
        len(result),
        non_null,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# __main__
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    earnings_path = project_root / "data" / "raw" / "financials" / "earnings" / "earnings_surprises.parquet"
    ohlcv_dir = project_root / "data" / "raw" / "financials" / "ohlcv"
    output_dir = project_root / "data" / "raw" / "financials" / "earnings_features"

    df = compute_earnings_features(earnings_path, ohlcv_dir)
    if len(df) == 0:
        _LOG.error(
            "No features computed. Run: python ingestion/earnings_ingestion.py"
        )
        sys.exit(1)

    save_earnings_features(df, output_dir)
    _LOG.info("Done. Retrain with: python models/train.py")
