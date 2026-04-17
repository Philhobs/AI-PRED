"""
News sentiment feature computation.

Computes 5 rolling-window per-ticker features from FinBERT-scored news articles.
Articles are tagged with mentioned_tickers at ingest time (news_ingestion.py).
All aggregations use DuckDB SQL against the OHLCV date spine.

Features:
  sentiment_mean_7d         — mean net_sentiment for articles mentioning ticker (7d)
  sentiment_std_7d          — std dev of net_sentiment (7d); null if <2 articles
  article_count_7d          — count of articles mentioning ticker (7d); 0 not null
  sentiment_momentum_14d    — sentiment_mean_7d minus prior 7-day mean (days 8-14)
  ticker_vs_market_7d       — sentiment_mean_7d minus market-wide daily mean (7d)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

_LOG = logging.getLogger(__name__)

# Maximum age of a sentiment observation to be propagated forward (~1 trading month).
# Sentiment signal degrades quickly; beyond this window, null is preferable to stale data.
_SENTIMENT_STALE_DAYS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers — used directly in unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sentiment_mean_7d(
    articles: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 7,
) -> float | None:
    """Mean net_sentiment for articles mentioning ticker in [as_of-window, as_of].
    Returns None if no matching articles in window.
    """
    if articles.is_empty():
        return None
    with duckdb.connect() as con:
        con.register("arts", articles)
        row = con.execute("""
            SELECT AVG(net_sentiment)
            FROM arts
            WHERE list_contains(mentioned_tickers, ?)
              AND article_date BETWEEN CAST(? AS DATE) - (? * INTERVAL '1 DAY')
                  AND CAST(? AS DATE)
        """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    return row[0] if row and row[0] is not None else None


def _compute_sentiment_momentum_14d(
    articles: pl.DataFrame,
    ticker: str,
    as_of: date,
) -> float | None:
    """Sentiment momentum: mean(days 0-7) minus mean(days 8-14).
    Returns None if either half-window has no articles.
    """
    if articles.is_empty():
        return None
    with duckdb.connect() as con:
        con.register("arts", articles)
        row = con.execute("""
            SELECT
                AVG(CASE WHEN article_date BETWEEN CAST(? AS DATE) - 7 * INTERVAL '1 DAY'
                                                AND CAST(? AS DATE)
                         THEN net_sentiment END) AS current_mean,
                AVG(CASE WHEN article_date BETWEEN CAST(? AS DATE) - 14 * INTERVAL '1 DAY'
                                                AND CAST(? AS DATE) - 8 * INTERVAL '1 DAY'
                         THEN net_sentiment END) AS prior_mean
            FROM arts
            WHERE list_contains(mentioned_tickers, ?)
        """, [str(as_of), str(as_of), str(as_of), str(as_of), ticker]).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    return row[0] - row[1]


def _compute_article_count_7d(
    articles: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 7,
) -> int:
    """Count of articles mentioning ticker in [as_of-window, as_of].
    Returns 0 (not None) when no articles match.
    """
    if articles.is_empty():
        return 0
    with duckdb.connect() as con:
        con.register("arts", articles)
        row = con.execute("""
            SELECT COUNT(*)
            FROM arts
            WHERE list_contains(mentioned_tickers, ?)
              AND article_date BETWEEN CAST(? AS DATE) - (? * INTERVAL '1 DAY')
                  AND CAST(? AS DATE)
        """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    return int(row[0]) if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# Bulk DuckDB computation over OHLCV spine
# ─────────────────────────────────────────────────────────────────────────────

def compute_ticker_sentiment_features(
    scored_news_dir: Path,
    ohlcv_dir: Path,
) -> pl.DataFrame:
    """Compute 5 rolling-window sentiment features for all tickers × OHLCV dates.

    Uses DuckDB UNNEST to expand mentioned_tickers list into per-ticker rows,
    then LEFT JOINs two time windows and market-daily mean.

    Returns DataFrame: [ticker, date, sentiment_mean_7d, sentiment_std_7d,
                        article_count_7d, sentiment_momentum_14d, ticker_vs_market_7d]
    """
    ohlcv_parquets = list(ohlcv_dir.glob("**/*.parquet"))
    if not ohlcv_parquets:
        _LOG.error("No OHLCV parquets found in %s", ohlcv_dir)
        return pl.DataFrame()

    scored_parquets = list(scored_news_dir.glob("date=*/data.parquet"))

    with duckdb.connect() as con:
        ohlcv_glob = str(ohlcv_dir / "**" / "*.parquet")
        con.execute("""
            CREATE TEMP TABLE spine AS
            SELECT DISTINCT ticker, date
            FROM read_parquet(?)
            ORDER BY ticker, date
        """, [ohlcv_glob])

        if scored_parquets:
            news_glob = str(scored_news_dir / "date=*" / "data.parquet")
            con.execute("""
                CREATE TEMP TABLE raw_articles AS
                SELECT
                    CAST(timestamp AS DATE) AS article_date,
                    mentioned_tickers,
                    net_sentiment
                FROM read_parquet(?)
                WHERE mentioned_tickers IS NOT NULL
            """, [news_glob])
        else:
            _LOG.warning(
                "No scored news parquets in %s — sentiment features will be null",
                scored_news_dir,
            )
            con.execute("""
                CREATE TEMP TABLE raw_articles (
                    article_date DATE,
                    mentioned_tickers VARCHAR[],
                    net_sentiment DOUBLE
                )
            """)

        # Materialize per-ticker rows to avoid repeated UNNEST in JOIN
        con.execute("""
            CREATE TEMP TABLE ticker_articles AS
            SELECT
                article_date,
                UNNEST(mentioned_tickers) AS ticker,
                net_sentiment
            FROM raw_articles
            WHERE len(mentioned_tickers) > 0
        """)

        # Market-wide daily mean for ticker_vs_market_7d baseline
        con.execute("""
            CREATE TEMP TABLE market_daily AS
            SELECT
                article_date,
                AVG(net_sentiment) AS market_mean
            FROM raw_articles
            GROUP BY article_date
        """)

        result_df = con.execute("""
            WITH agg7 AS (
                SELECT s.ticker, s.date,
                    AVG(ta7.net_sentiment)         AS sentiment_mean_7d,
                    STDDEV_SAMP(ta7.net_sentiment) AS sentiment_std_7d,
                    COUNT(ta7.net_sentiment)        AS article_count_7d
                FROM spine s
                LEFT JOIN ticker_articles ta7
                    ON ta7.ticker = s.ticker
                   AND ta7.article_date BETWEEN s.date - INTERVAL 7 DAY AND s.date
                GROUP BY s.ticker, s.date
            ),
            agg14 AS (
                SELECT s.ticker, s.date,
                    AVG(ta14.net_sentiment) AS prior_mean_7d
                FROM spine s
                LEFT JOIN ticker_articles ta14
                    ON ta14.ticker = s.ticker
                   AND ta14.article_date BETWEEN s.date - INTERVAL 14 DAY
                       AND s.date - INTERVAL 8 DAY
                GROUP BY s.ticker, s.date
            ),
            market_agg AS (
                SELECT s.ticker, s.date,
                    AVG(md.market_mean) AS avg_market_mean
                FROM spine s
                LEFT JOIN market_daily md
                    ON md.article_date BETWEEN s.date - INTERVAL 7 DAY AND s.date
                GROUP BY s.ticker, s.date
            )
            SELECT
                a7.ticker,
                a7.date,
                a7.sentiment_mean_7d,
                a7.sentiment_std_7d,
                a7.article_count_7d,
                a7.sentiment_mean_7d - a14.prior_mean_7d  AS sentiment_momentum_14d,
                a7.sentiment_mean_7d - m.avg_market_mean  AS ticker_vs_market_7d
            FROM agg7 a7
            LEFT JOIN agg14 a14 USING (ticker, date)
            LEFT JOIN market_agg m  USING (ticker, date)
            ORDER BY a7.ticker, a7.date
        """).pl()

    _LOG.info(
        "Computed sentiment features: %d rows for %d tickers",
        len(result_df),
        result_df["ticker"].n_unique(),
    )
    return result_df


def save_sentiment_features(df: pl.DataFrame, output_dir: Path) -> None:
    """Write per-ticker daily parquets to output_dir/<TICKER>/daily.parquet."""
    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        out_path = output_dir / ticker / "daily.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ticker_df.write_parquet(out_path, compression="snappy")
    _LOG.info(
        "Saved sentiment features for %d tickers to %s",
        df["ticker"].n_unique(),
        output_dir,
    )


def join_sentiment_features(df: pl.DataFrame, sentiment_features_dir: Path) -> pl.DataFrame:
    """Backward asof join sentiment features onto training DataFrame on (ticker, date).

    For each (ticker, date) row in df, finds the most recent feature row where
    feature_date <= training_date. Adds null columns for tickers with no data.
    """
    feature_cols = [
        "sentiment_mean_7d",
        "sentiment_std_7d",
        "article_count_7d",
        "sentiment_momentum_14d",
        "ticker_vs_market_7d",
    ]
    parquet_glob = str(sentiment_features_dir / "*/daily.parquet")
    if not any(sentiment_features_dir.glob("*/daily.parquet")):
        _LOG.warning(
            "No sentiment feature parquets in %s — features will be null",
            sentiment_features_dir,
        )
        for col in feature_cols:
            dtype = pl.Int64 if col == "article_count_7d" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df

    # collect() is intentional: join_asof requires a materialised, sorted DataFrame.
    # Per-ticker parquets are small (≤50 tickers × daily rows), so memory is bounded.
    features = pl.scan_parquet(parquet_glob).sort(["ticker", "date"]).collect()
    features_renamed = features.rename({"date": "feature_date"})

    df_sorted = df.sort(["ticker", "date"])
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="feature_date",
        by="ticker",
        strategy="backward",
        tolerance=timedelta(days=_SENTIMENT_STALE_DAYS),
    )

    non_null = result["sentiment_mean_7d"].drop_nulls().len()
    _LOG.info(
        "Joined sentiment features: %d rows, %d non-null sentiment_mean_7d values",
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
    scored_news_dir = project_root / "data" / "raw" / "news" / "scored"
    ohlcv_dir = project_root / "data" / "raw" / "financials" / "ohlcv"
    output_dir = project_root / "data" / "raw" / "news" / "sentiment_features"

    df = compute_ticker_sentiment_features(scored_news_dir, ohlcv_dir)
    if len(df) == 0:
        _LOG.error(
            "No features computed. Run: python ingestion/news_ingestion.py && "
            "python processing/nlp_pipeline.py"
        )
        sys.exit(1)

    save_sentiment_features(df, output_dir)
    _LOG.info("Done. Retrain with: python models/train.py")
