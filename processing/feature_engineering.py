import duckdb
import polars as pl
from pathlib import Path
from datetime import datetime, timezone

WATCHLIST = {
    "hyperscalers": ["MSFT", "AMZN", "GOOGL", "META"],
    "ai_chips": ["NVDA", "AMD", "AVGO", "MRVL", "TSM"],
    "foundry_equipment": ["ASML", "AMAT", "LRCX", "KLAC"],
    "ai_infrastructure": ["VRT", "SMCI", "DELL", "HPE"],
    "data_center_reits": ["EQIX", "DLR", "AMT"],
    "power_nuclear": ["CEG", "VST", "NRG", "TLN"],
}

FEATURE_RATIONALE = {
    "taiwan_cargo_ratio": "Cargo vessel count Taiwan Strait (weekly) — GPU shipment leading indicator",
    "construction_score_current": "Sentinel-2 NDBI at data center sites — capex deployment pace",
    "sentiment_momentum_7d": "FinBERT net sentiment 7-day momentum — narrative shift signal",
    "henry_hub_gas_price": "Natural gas spot — power cost leading indicator for utility margins",
    "return_5d": "5-day price return — momentum signal",
    "volume_ratio": "Volume vs 20-day MA — institutional activity signal",
}


def build_daily_feature_matrix(
    con: duckdb.DuckDBPyConnection,
    date_str: str,
    data_dir: Path = Path("data/raw"),
) -> pl.DataFrame:
    """
    Build the complete feature matrix for a given date.
    Returns Polars DataFrame: one row per ticker × all signal features.
    Returns empty DataFrame if no price data exists for the date.
    """
    ohlcv_glob = str(data_dir / "financials" / "ohlcv" / "*" / "*.parquet")

    try:
        price_features = con.execute(f"""
            WITH price AS (
                SELECT
                    ticker,
                    date,
                    close_price,
                    volume,
                    close_price / NULLIF(LAG(close_price, 1) OVER w, 0) - 1  AS return_1d,
                    close_price / NULLIF(LAG(close_price, 5) OVER w, 0) - 1  AS return_5d,
                    close_price / NULLIF(LAG(close_price, 20) OVER w, 0) - 1 AS return_20d,
                    close_price / NULLIF(AVG(close_price) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ), 0) - 1 AS sma_20_deviation,
                    STDDEV(close_price) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ) / NULLIF(AVG(close_price) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ), 0) AS volatility_20d,
                    volume / NULLIF(AVG(volume) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ), 0) AS volume_ratio
                FROM read_parquet('{ohlcv_glob}')
                WINDOW w AS (PARTITION BY ticker ORDER BY date)
            )
            SELECT * FROM price
            WHERE date::date = DATE '{date_str}'
        """).pl()
    except Exception as e:
        print(f"[Features] No price data for {date_str}: {e}")
        return pl.DataFrame()

    if price_features.is_empty():
        return price_features

    # AIS signal: Taiwan Strait weekly cargo count (1.0 if no data yet)
    try:
        ais_row = con.execute(f"""
            SELECT COUNT(*) / 7.0 AS taiwan_cargo_ratio
            FROM v_ais
            WHERE timestamp >= DATE '{date_str}' - INTERVAL '7 days'
                AND timestamp < DATE '{date_str}'
                AND corridor = 'taiwan_strait'
                AND vessel_type BETWEEN 70 AND 89
        """).fetchone()
        taiwan_cargo_ratio = float(ais_row[0] or 1.0)
    except Exception:
        taiwan_cargo_ratio = 1.0

    # Sentiment signal: 7-day mean (0.0 if no NLP data yet)
    try:
        sent_row = con.execute(f"""
            SELECT AVG(net_sentiment) AS mean_sentiment
            FROM v_news
            WHERE timestamp >= DATE '{date_str}' - INTERVAL '7 days'
                AND timestamp < DATE '{date_str}'
        """).fetchone()
        mean_sentiment = float(sent_row[0] or 0.0)
    except Exception:
        mean_sentiment = 0.0

    return price_features.with_columns([
        pl.lit(taiwan_cargo_ratio).alias("taiwan_cargo_ratio"),
        pl.lit(mean_sentiment).alias("sentiment_mean"),
        pl.lit(date_str).alias("feature_date"),
    ])


def save_feature_matrix(df: pl.DataFrame, date_str: str, output_dir: Path = Path("data/features")):
    """Save feature matrix to Hive-partitioned Parquet."""
    path = output_dir / "daily" / f"date={date_str}" / "features.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path), compression="snappy")
    print(f"[Features] Saved {len(df)} rows × {len(df.columns)} features → {path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    from storage import get_db_connection, create_views
    load_dotenv()

    con = get_db_connection()
    create_views(con)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[Features] Building feature matrix for {date_str}...")

    df = build_daily_feature_matrix(con, date_str)

    if df.is_empty():
        print("[Features] No data yet. Run financial_ingestion.py and add OHLCV data first.")
    else:
        save_feature_matrix(df, date_str)
        print(f"[Features] Done. {len(df)} rows × {len(df.columns)} features")

    con.close()
