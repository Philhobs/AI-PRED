"""
Insider trading feature computation.

Computes 5 rolling-window features from raw insider trade and congressional
trade parquets, using OHLCV dates as the spine. All aggregations use DuckDB SQL.

Features:
  insider_cluster_buy_90d     — distinct insiders with code P in last 90 days
  insider_net_value_30d       — (purchase value - sale value) / 1M over 30 days
  insider_buy_sell_ratio_90d  — purchase count / total count over 90 days
  congress_net_buy_90d        — (purchase amount - sale amount) / 1M over 90 days
  congress_trade_count_90d    — total congressional trades over 90 days

Null (not zero) when the window has no qualifying trades.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

_LOG = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar helpers — used directly in unit tests
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cluster_buy_90d(
    insider_trades: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 90,
) -> int | None:
    """Count distinct insiders with code='P' in [as_of - window_days, as_of].
    Returns None if no purchases in window.
    """
    if insider_trades.is_empty():
        return None
    con = duckdb.connect()
    con.register("trades", insider_trades)
    row = con.execute("""
        SELECT COUNT(DISTINCT insider_name) AS cnt
        FROM trades
        WHERE ticker = ?
          AND transaction_code = 'P'
          AND transaction_date >= CAST(? AS DATE) - INTERVAL (?) DAY
          AND transaction_date <= CAST(? AS DATE)
    """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    count = row[0] if row else 0
    return int(count) if count and count > 0 else None


def _compute_net_value_30d(
    insider_trades: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 30,
) -> float | None:
    """Net insider value (purchases - sales) in millions over window.
    Returns None if no trades in window.
    """
    if insider_trades.is_empty():
        return None
    con = duckdb.connect()
    con.register("trades", insider_trades)
    count_row = con.execute("""
        SELECT COUNT(*) FROM trades
        WHERE ticker = ?
          AND transaction_date >= CAST(? AS DATE) - INTERVAL (?) DAY
          AND transaction_date <= CAST(? AS DATE)
    """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    if not count_row or count_row[0] == 0:
        return None
    row = con.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN transaction_code = 'P' THEN value ELSE 0 END), 0.0)
            - COALESCE(SUM(CASE WHEN transaction_code = 'S' THEN value ELSE 0 END), 0.0)
        FROM trades
        WHERE ticker = ?
          AND transaction_date >= CAST(? AS DATE) - INTERVAL (?) DAY
          AND transaction_date <= CAST(? AS DATE)
    """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    return row[0] / 1_000_000.0 if row and row[0] is not None else None


def _compute_buy_sell_ratio_90d(
    insider_trades: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 90,
) -> float | None:
    """Purchase count / total count in window. Returns None if no trades."""
    if insider_trades.is_empty():
        return None
    con = duckdb.connect()
    con.register("trades", insider_trades)
    row = con.execute("""
        SELECT
            SUM(CASE WHEN transaction_code = 'P' THEN 1 ELSE 0 END) AS buys,
            COUNT(*) AS total
        FROM trades
        WHERE ticker = ?
          AND transaction_date >= CAST(? AS DATE) - INTERVAL (?) DAY
          AND transaction_date <= CAST(? AS DATE)
    """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    if not row or row[1] == 0:
        return None
    return row[0] / row[1]


def _compute_congress_net_buy_90d(
    congressional_trades: pl.DataFrame,
    ticker: str,
    as_of: date,
    window_days: int = 90,
) -> float | None:
    """Congressional net buy (purchases - sales) in millions over window.
    Returns None if no trades in window.
    """
    if congressional_trades.is_empty():
        return None
    con = duckdb.connect()
    con.register("congress", congressional_trades)
    count_row = con.execute("""
        SELECT COUNT(*) FROM congress
        WHERE ticker = ?
          AND trade_date >= CAST(? AS DATE) - INTERVAL (?) DAY
          AND trade_date <= CAST(? AS DATE)
    """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    if not count_row or count_row[0] == 0:
        return None
    row = con.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN transaction_type = 'purchase' THEN amount_mid ELSE 0 END), 0.0)
            - COALESCE(SUM(CASE WHEN transaction_type = 'sale' THEN amount_mid ELSE 0 END), 0.0)
        FROM congress
        WHERE ticker = ?
          AND trade_date >= CAST(? AS DATE) - INTERVAL (?) DAY
          AND trade_date <= CAST(? AS DATE)
    """, [ticker, str(as_of), window_days, str(as_of)]).fetchone()
    return row[0] / 1_000_000.0 if row and row[0] is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Bulk DuckDB computation over OHLCV spine
# ─────────────────────────────────────────────────────────────────────────────

def compute_insider_features(
    insider_trades_dir: Path,
    congressional_trades_path: Path,
    ohlcv_dir: Path,
) -> pl.DataFrame:
    """Compute 5 rolling-window insider features for all tickers and all OHLCV dates.

    Uses DuckDB SQL correlated subqueries for windowed aggregation.

    Returns DataFrame: [ticker, date, insider_cluster_buy_90d, insider_net_value_30d,
                        insider_buy_sell_ratio_90d, congress_net_buy_90d, congress_trade_count_90d]
    """
    con = duckdb.connect()

    # Load insider trades
    insider_parquets = list(insider_trades_dir.glob("*/transactions.parquet"))
    if insider_parquets:
        paths_repr = repr([str(p) for p in insider_parquets])
        con.execute(f"CREATE TABLE insider AS SELECT * FROM read_parquet({paths_repr})")
    else:
        con.execute("""
            CREATE TABLE insider (
                ticker VARCHAR, transaction_date DATE,
                insider_name VARCHAR, transaction_code VARCHAR, value DOUBLE
            )
        """)
        _LOG.warning("No insider trade parquets found in %s", insider_trades_dir)

    # Load congressional trades
    if congressional_trades_path.exists():
        con.execute(f"CREATE TABLE congress AS SELECT * FROM read_parquet('{congressional_trades_path}')")
    else:
        con.execute("""
            CREATE TABLE congress (
                ticker VARCHAR, trade_date DATE,
                transaction_type VARCHAR, amount_mid DOUBLE
            )
        """)
        _LOG.warning("No congressional trades parquet found at %s", congressional_trades_path)

    # Build OHLCV spine
    ohlcv_parquets = list(ohlcv_dir.glob("**/*.parquet"))
    if not ohlcv_parquets:
        _LOG.error("No OHLCV parquets found in %s", ohlcv_dir)
        return pl.DataFrame()

    ohlcv_repr = repr([str(p) for p in ohlcv_parquets])
    con.execute(f"""
        CREATE TABLE spine AS
        SELECT DISTINCT ticker, date
        FROM read_parquet({ohlcv_repr})
        ORDER BY ticker, date
    """)

    result_df = con.execute("""
        SELECT
            s.ticker,
            s.date,

            -- insider_cluster_buy_90d: distinct buyers in last 90 days
            NULLIF((
                SELECT COUNT(DISTINCT i.insider_name)
                FROM insider i
                WHERE i.ticker = s.ticker
                  AND i.transaction_code = 'P'
                  AND i.transaction_date BETWEEN s.date - INTERVAL 90 DAY AND s.date
            ), 0) AS insider_cluster_buy_90d,

            -- insider_net_value_30d: net buy value in millions over 30 days
            CASE WHEN (
                SELECT COUNT(*) FROM insider i
                WHERE i.ticker = s.ticker
                  AND i.transaction_date BETWEEN s.date - INTERVAL 30 DAY AND s.date
            ) = 0 THEN NULL
            ELSE (
                SELECT (
                    COALESCE(SUM(CASE WHEN i.transaction_code = 'P' THEN i.value ELSE 0 END), 0.0)
                    - COALESCE(SUM(CASE WHEN i.transaction_code = 'S' THEN i.value ELSE 0 END), 0.0)
                ) / 1000000.0
                FROM insider i
                WHERE i.ticker = s.ticker
                  AND i.transaction_date BETWEEN s.date - INTERVAL 30 DAY AND s.date
            ) END AS insider_net_value_30d,

            -- insider_buy_sell_ratio_90d
            CASE WHEN (
                SELECT COUNT(*) FROM insider i
                WHERE i.ticker = s.ticker
                  AND i.transaction_date BETWEEN s.date - INTERVAL 90 DAY AND s.date
            ) = 0 THEN NULL
            ELSE (
                SELECT CAST(SUM(CASE WHEN i.transaction_code = 'P' THEN 1 ELSE 0 END) AS DOUBLE)
                       / COUNT(*)
                FROM insider i
                WHERE i.ticker = s.ticker
                  AND i.transaction_date BETWEEN s.date - INTERVAL 90 DAY AND s.date
            ) END AS insider_buy_sell_ratio_90d,

            -- congress_net_buy_90d
            CASE WHEN (
                SELECT COUNT(*) FROM congress c
                WHERE c.ticker = s.ticker
                  AND c.trade_date BETWEEN s.date - INTERVAL 90 DAY AND s.date
            ) = 0 THEN NULL
            ELSE (
                SELECT (
                    COALESCE(SUM(CASE WHEN c.transaction_type = 'purchase' THEN c.amount_mid ELSE 0 END), 0.0)
                    - COALESCE(SUM(CASE WHEN c.transaction_type = 'sale' THEN c.amount_mid ELSE 0 END), 0.0)
                ) / 1000000.0
                FROM congress c
                WHERE c.ticker = s.ticker
                  AND c.trade_date BETWEEN s.date - INTERVAL 90 DAY AND s.date
            ) END AS congress_net_buy_90d,

            -- congress_trade_count_90d
            NULLIF((
                SELECT COUNT(*)
                FROM congress c
                WHERE c.ticker = s.ticker
                  AND c.trade_date BETWEEN s.date - INTERVAL 90 DAY AND s.date
            ), 0) AS congress_trade_count_90d

        FROM spine s
    """).pl()

    _LOG.info(
        "Computed insider features: %d rows for %d tickers",
        len(result_df),
        result_df["ticker"].n_unique(),
    )
    return result_df


def save_insider_features(df: pl.DataFrame, output_dir: Path) -> None:
    """Write per-ticker daily parquets to output_dir/<TICKER>/daily.parquet."""
    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        out_path = output_dir / ticker / "daily.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ticker_df.write_parquet(out_path, compression="snappy")
    _LOG.info("Saved insider features for %d tickers to %s", df["ticker"].n_unique(), output_dir)


def join_insider_features(df: pl.DataFrame, insider_features_dir: Path) -> pl.DataFrame:
    """Backward asof join insider features onto training DataFrame on (ticker, date).

    For each (ticker, date) row in df, finds the most recent feature row where
    feature_date <= training_date (same pattern as join_fundamentals in fundamental_features.py).
    """
    feature_cols = [
        "insider_cluster_buy_90d",
        "insider_net_value_30d",
        "insider_buy_sell_ratio_90d",
        "congress_net_buy_90d",
        "congress_trade_count_90d",
    ]
    parquets = list(insider_features_dir.glob("*/daily.parquet"))
    if not parquets:
        _LOG.warning(
            "No insider feature parquets found in %s — features will be null", insider_features_dir
        )
        for col in feature_cols:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
        return df

    features = pl.concat([pl.read_parquet(p) for p in parquets]).sort(["ticker", "date"])
    features_renamed = features.rename({"date": "feature_date"})

    df_sorted = df.sort(["ticker", "date"])
    result = df_sorted.join_asof(
        features_renamed,
        left_on="date",
        right_on="feature_date",
        by="ticker",
        strategy="backward",
    )

    non_null = result["insider_cluster_buy_90d"].drop_nulls().len()
    _LOG.info("Joined insider features: %d rows, %d non-null cluster_buy values", len(result), non_null)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — compute and save features
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    insider_trades_dir = project_root / "data" / "raw" / "financials" / "insider_trades"
    congressional_trades_path = (
        project_root / "data" / "raw" / "financials" / "congressional_trades" / "all_transactions.parquet"
    )
    ohlcv_dir = project_root / "data" / "raw" / "financials" / "ohlcv"
    output_dir = project_root / "data" / "raw" / "financials" / "insider_features"

    df = compute_insider_features(insider_trades_dir, congressional_trades_path, ohlcv_dir)
    if len(df) == 0:
        _LOG.error("No features computed — run insider_trading_ingestion.py first")
        sys.exit(1)

    save_insider_features(df, output_dir)
    _LOG.info("Done. Retrain with: python models/train.py")
