from pathlib import Path

import polars as pl

_EMPTY_SCHEMA = {"ticker": pl.String, "date": pl.Date, "label_return_1y": pl.Float64}


def build_labels(
    ohlcv_dir: Path = Path("data/raw/financials/ohlcv"),
) -> pl.DataFrame:
    """
    Compute 1-year forward annualized return for each ticker×date row.

    label_return_1y = close_price[t+252] / close_price[t] - 1

    Uses row-offset shift within each ticker partition (Polars shift(-252).over("ticker")).
    Rows where the 252-day forward price is unavailable are dropped — this prevents
    look-ahead leakage in model training. A ticker with ≤252 rows returns no labeled rows.

    Returns DataFrame with columns: ticker (String), date (Date), label_return_1y (Float64).
    Returns empty DataFrame with the same schema when no data exists.
    """
    glob = str(ohlcv_dir / "*" / "*.parquet")

    try:
        df = (
            pl.scan_parquet(glob)
            .select(["ticker", "date", "close_price"])
            .collect()
        )
    except (FileNotFoundError, pl.exceptions.ComputeError):
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    if df.is_empty():
        return pl.DataFrame(schema=_EMPTY_SCHEMA)

    result = (
        df
        .with_columns(pl.col("date").cast(pl.Date))
        .sort(["ticker", "date"])
        .with_columns(
            pl.col("close_price")
            .shift(-252)
            .over("ticker")
            .alias("future_price")
        )
        .filter(pl.col("future_price").is_not_null())
        .with_columns(
            (pl.col("future_price") / pl.col("close_price") - 1).alias("label_return_1y")
        )
        .select(["ticker", "date", "label_return_1y"])
    )

    return result


def build_multi_horizon_labels(
    ohlcv_dir: Path = Path("data/raw/financials/ohlcv"),
    horizons: dict[str, int] | None = None,
) -> pl.DataFrame:
    """
    Compute multi-horizon forward returns for each ticker×date row.

    horizons: mapping of tag → shift_days, e.g. {"5d": 5, "252d": 252}.
    Defaults to the 8-horizon set from HORIZON_CONFIGS in models/train.py.

    Returns DataFrame: ticker (String), date (Date), label_return_{tag} (Float64)...
    Rows are NOT filtered — each column has nulls where the forward window is unavailable.
    Returns empty DataFrame with ticker/date schema when no data exists.
    """
    if horizons is None:
        horizons = {
            "5d": 5, "20d": 20, "65d": 65, "252d": 252,
            "756d": 756, "1260d": 1260, "2520d": 2520, "5040d": 5040,
        }

    glob = str(ohlcv_dir / "*" / "*.parquet")

    try:
        df = (
            pl.scan_parquet(glob)
            .select(["ticker", "date", "close_price"])
            .collect()
        )
    except (FileNotFoundError, pl.exceptions.ComputeError):
        schema = {"ticker": pl.String, "date": pl.Date}
        schema.update({f"label_return_{tag}": pl.Float64 for tag in horizons})
        return pl.DataFrame(schema=schema)

    if df.is_empty():
        schema = {"ticker": pl.String, "date": pl.Date}
        schema.update({f"label_return_{tag}": pl.Float64 for tag in horizons})
        return pl.DataFrame(schema=schema)

    df = df.with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])

    label_exprs = [
        (
            pl.col("close_price").shift(-shift).over("ticker")
            / pl.col("close_price") - 1
        ).alias(f"label_return_{tag}")
        for tag, shift in horizons.items()
    ]

    return df.with_columns(label_exprs).select(
        ["ticker", "date"] + [f"label_return_{tag}" for tag in horizons]
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    ohlcv_dir = Path("data/raw/financials/ohlcv")
    print("[Labels] Building 1-year forward return labels...")
    df = build_labels(ohlcv_dir=ohlcv_dir)

    if df.is_empty():
        print("[Labels] No OHLCV data found. Run ingestion/ohlcv_ingestion.py --bootstrap first.")
    else:
        out_path = Path("data/raw/financials/labels.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(out_path), compression="snappy")
        print(f"[Labels] {len(df)} labeled rows → {out_path}")
