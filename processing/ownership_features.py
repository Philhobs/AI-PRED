# processing/ownership_features.py
"""
13F Institutional Ownership Feature Computation

Aggregates raw per-filer 13F holdings into 5 per-ticker quarterly features:
  inst_ownership_pct       — total inst shares / shares_outstanding × 100
  inst_net_shares_qoq      — QoQ change in shares held / shares_outstanding
  inst_holder_count        — distinct institution count
  inst_concentration_top10 — top-10 filers' shares / total inst shares
  inst_momentum_2q         — 2-quarter change in inst_ownership_pct

Storage: data/raw/financials/13f_holdings/features/<TICKER>/quarterly.parquet

Run: python processing/ownership_features.py
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)

_FEATURE_COLS = [
    "inst_ownership_pct",
    "inst_net_shares_qoq",
    "inst_holder_count",
    "inst_concentration_top10",
    "inst_momentum_2q",
]


def _load_shares_outstanding(ohlcv_dir: Path) -> dict[str, int]:
    """
    Fetch current shares outstanding for each ticker via yfinance.
    Uses the most recent value as a constant approximation for all periods.
    Falls back to 0 for tickers where yfinance fails.
    """
    import yfinance as yf
    from ingestion.ticker_registry import TICKERS

    result: dict[str, int] = {}
    for ticker in TICKERS:
        try:
            shares = yf.Ticker(ticker).fast_info.shares
            if shares and shares > 0:
                result[ticker] = int(shares)
        except Exception:
            pass
        time.sleep(0.3)
    _LOG.info("Fetched shares outstanding for %d tickers", len(result))
    return result


def compute_ownership_features(
    raw_dir: Path,
    shares_map: dict[str, int] | None = None,
    ohlcv_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Aggregate raw 13F holdings into per-ticker quarterly features.

    Args:
        raw_dir: path to 13f_holdings/raw/ — contains <YYYYQQ>/<CIK>.parquet files
        shares_map: ticker → shares_outstanding (if None, fetched via yfinance)
        ohlcv_dir: used only if shares_map is None (passed to _load_shares_outstanding)

    Returns DataFrame with columns:
        ticker, quarter, period_end, inst_ownership_pct, inst_net_shares_qoq,
        inst_holder_count, inst_concentration_top10, inst_momentum_2q
    """
    parquets = list(raw_dir.glob("*/*.parquet"))
    if not parquets:
        _LOG.warning("No raw 13F parquets found in %s", raw_dir)
        return pl.DataFrame()

    if shares_map is None and ohlcv_dir is None:
        raise ValueError(
            "Either shares_map or ohlcv_dir must be provided to compute inst_ownership_pct"
        )

    raw = pl.concat([pl.read_parquet(p) for p in parquets])

    # Backwards compat: raw parquets ingested before the available_date column
    # was added (sec_13f_ingestion v2 onward) won't have it. Synthesize one as
    # period_end + 45 days (SEC Rule 13f-1 max).
    if "available_date" not in raw.columns:
        raw = raw.with_columns(
            (pl.col("period_end") + pl.duration(days=45)).cast(pl.Date).alias("available_date")
        )
    else:
        # Fill any nulls with the same fallback rule.
        raw = raw.with_columns(
            pl.when(pl.col("available_date").is_null())
            .then((pl.col("period_end") + pl.duration(days=45)).cast(pl.Date))
            .otherwise(pl.col("available_date"))
            .alias("available_date")
        )

    if shares_map is None:
        shares_map = _load_shares_outstanding(ohlcv_dir)

    shares_series = pl.DataFrame({
        "ticker":             list(shares_map.keys()),
        "shares_outstanding": [int(v) for v in shares_map.values()],
    })

    # Aggregate per (ticker, quarter, period_end). available_date = the LATEST
    # filer's filed date for that reporting quarter — this is when the
    # aggregate becomes fully observable to the public. Using MAX (rather than
    # MEAN/MEDIAN) is the conservative point-in-time choice: the aggregated
    # ownership_pct is accurate only after every contributing filer has
    # publicly disclosed their position.
    agg = (
        raw
        .group_by(["ticker", "quarter", "period_end"])
        .agg([
            pl.col("shares_held").sum().alias("total_inst_shares"),
            pl.col("cik").n_unique().alias("inst_holder_count"),
            pl.col("available_date").max().alias("available_date"),
        ])
    )

    # Top-10 concentration: sum of top-10 filers' shares per (ticker, quarter)
    top10 = (
        raw
        .group_by(["ticker", "quarter"])
        .agg(
            pl.col("shares_held")
            .sort(descending=True)
            .head(10)
            .sum()
            .alias("top10_shares")
        )
    )

    features = (
        agg
        .join(top10, on=["ticker", "quarter"], how="left")
        .join(shares_series, on="ticker", how="left")
        .with_columns([
            # inst_ownership_pct
            pl.when(pl.col("shares_outstanding") > 0)
            .then(
                pl.col("total_inst_shares") / pl.col("shares_outstanding") * 100.0
            )
            .otherwise(None)
            .alias("inst_ownership_pct"),

            # inst_concentration_top10
            pl.when(pl.col("total_inst_shares") > 0)
            .then(pl.col("top10_shares") / pl.col("total_inst_shares"))
            .otherwise(None)
            .alias("inst_concentration_top10"),
        ])
        .sort(["ticker", "period_end"])
    )

    # QoQ and 2Q momentum computed as lags within each ticker's time series
    features = features.with_columns([
        # inst_net_shares_qoq = (shares_q - shares_q-1) / shares_outstanding
        pl.when(pl.col("shares_outstanding") > 0)
        .then(
            (pl.col("total_inst_shares") - pl.col("total_inst_shares").shift(1).over("ticker"))
            / pl.col("shares_outstanding")
        )
        .otherwise(None)
        .alias("inst_net_shares_qoq"),

        # inst_momentum_2q = ownership_pct - ownership_pct 2 quarters ago
        (
            pl.col("inst_ownership_pct")
            - pl.col("inst_ownership_pct").shift(2).over("ticker")
        ).alias("inst_momentum_2q"),
    ])

    return features.select([
        "ticker", "quarter", "period_end", "available_date",
        "inst_ownership_pct", "inst_net_shares_qoq",
        pl.col("inst_holder_count").cast(pl.Int32),
        "inst_concentration_top10", "inst_momentum_2q",
    ])


def save_ownership_features(df: pl.DataFrame, output_dir: Path) -> None:
    """Write per-ticker quarterly.parquet files to output_dir/<TICKER>/quarterly.parquet."""
    for ticker in df["ticker"].unique().to_list():
        out = output_dir / ticker / "quarterly.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.filter(pl.col("ticker") == ticker).write_parquet(out, compression="snappy")
    _LOG.info("Saved ownership features for %d tickers", df["ticker"].n_unique())


def join_ownership_features(df: pl.DataFrame, features_dir: Path) -> pl.DataFrame:
    """
    Backward-asof join ownership features onto training/inference DataFrame.

    For each (ticker, date) row in df, attaches the most recent quarterly
    features where available_date <= date. available_date is when the SEC
    publicly published the slowest-filing constituent of the (ticker, quarter)
    aggregate — using period_end here would be a 45-day lookahead.

    Backwards compat: parquets generated before the available_date column was
    added are treated as available_date = period_end + 45 days (Rule 13f-1 max).

    Same pattern as join_graph_features() in processing/graph_features.py.
    """
    parquets = list(features_dir.glob("*/quarterly.parquet"))
    if not parquets:
        _LOG.warning("No ownership feature parquets in %s — features will be null", features_dir)
        for col in _FEATURE_COLS:
            dtype = pl.Int32 if col == "inst_holder_count" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        return df

    raw = pl.concat([pl.read_parquet(p) for p in parquets])
    if "available_date" not in raw.columns:
        raw = raw.with_columns(
            (pl.col("period_end") + pl.duration(days=45)).cast(pl.Date).alias("available_date")
        )
    else:
        raw = raw.with_columns(
            pl.when(pl.col("available_date").is_null())
            .then((pl.col("period_end") + pl.duration(days=45)).cast(pl.Date))
            .otherwise(pl.col("available_date"))
            .alias("available_date")
        )

    features = (
        raw.select(["ticker", "available_date"] + _FEATURE_COLS)
        .sort(["ticker", "available_date"])
    )
    features_renamed = features.rename({"available_date": "ownership_date"})

    result = df.sort(["ticker", "date"]).join_asof(
        features_renamed,
        left_on="date",
        right_on="ownership_date",
        by="ticker",
        strategy="backward",
    )
    non_null = result["inst_ownership_pct"].drop_nulls().len()
    _LOG.info(
        "Joined ownership features: %d rows, %d non-null inst_ownership_pct",
        len(result), non_null,
    )
    return result


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    raw_dir      = project_root / "data" / "raw" / "financials" / "13f_holdings" / "raw"
    ohlcv_dir    = project_root / "data" / "raw" / "financials" / "ohlcv"
    features_dir = project_root / "data" / "raw" / "financials" / "13f_holdings" / "features"

    df = compute_ownership_features(raw_dir, ohlcv_dir=ohlcv_dir)
    if df.is_empty():
        print("No data — run: python ingestion/sec_13f_ingestion.py --bootstrap first")
    else:
        save_ownership_features(df, features_dir)
        print(f"Ownership features saved for {df['ticker'].n_unique()} tickers")
        print(df.group_by("ticker").len().sort("len", descending=True).head(10))
