from datetime import date
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
    max_date: "date | None" = None,
) -> pl.DataFrame:
    """
    Compute multi-horizon forward returns for each ticker×date row.

    horizons: mapping of tag → shift_days, e.g. {"5d": 5, "252d": 252}.
    Defaults to the 8-horizon set from HORIZON_CONFIGS in models/train.py.

    max_date: if given, drops OHLCV rows with date > max_date BEFORE computing
    labels. This is the walk-forward cutoff: labels for spine dates near
    max_date will naturally be null (because the forward shift can't find a
    future price), and downstream filtering on label-not-null produces a
    training spine that hasn't peeked past max_date.

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
    if max_date is not None:
        df = df.filter(pl.col("date") <= max_date)

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


def build_multi_horizon_excess_labels(
    ohlcv_dir: Path = Path("data/raw/financials/ohlcv"),
    horizons: dict[str, int] | None = None,
    ticker_layers: dict[str, str] | None = None,
    max_date: "date | None" = None,
) -> pl.DataFrame:
    """
    Compute layer-residualized (sector-neutral) forward returns.

    For each (ticker, date, horizon):
        label_excess_<tag> = ticker_return_<tag>
                           - mean(ticker_return_<tag> over tickers in same layer)

    Why: predicting raw returns lets the model exploit "the cloud layer rallied"
    rather than the more useful "this cloud ticker outperformed its peers."
    Residualizing against the layer benchmark forces the model to learn
    cross-sectional alpha within a layer.

    Args:
        ohlcv_dir: Path to per-ticker OHLCV parquets.
        horizons: tag → forward shift days. Defaults to the 8 horizon tags.
        ticker_layers: ticker → layer_name. If None, loaded from
            ingestion.ticker_registry.TICKER_LAYERS.

    Returns:
        DataFrame [ticker (String), date (Date), label_excess_<tag>...].
        Rows where any per-ticker forward return is null receive null in
        label_excess_<tag> for that horizon. Layer benchmarks are computed
        only over the tickers in `ticker_layers`; tickers not in the map
        are excluded entirely.
    """
    if horizons is None:
        horizons = {
            "5d": 5, "20d": 20, "65d": 65, "252d": 252,
            "756d": 756, "1260d": 1260, "2520d": 2520, "5040d": 5040,
        }

    if ticker_layers is None:
        from ingestion.ticker_registry import TICKER_LAYERS
        ticker_layers = TICKER_LAYERS

    raw = build_multi_horizon_labels(ohlcv_dir, horizons=horizons, max_date=max_date)
    if raw.is_empty():
        schema = {"ticker": pl.String, "date": pl.Date}
        schema.update({f"label_excess_{tag}": pl.Float64 for tag in horizons})
        return pl.DataFrame(schema=schema)

    layers_df = pl.DataFrame(
        {"ticker": list(ticker_layers.keys()),
         "layer":  list(ticker_layers.values())},
        schema={"ticker": pl.String, "layer": pl.String},
    )
    enriched = raw.join(layers_df, on="ticker", how="inner")  # drops unmapped tickers

    # For each (date, layer): the layer benchmark is the mean of label_return_<tag>.
    layer_means = (
        enriched
        .group_by(["date", "layer"])
        .agg([
            pl.col(f"label_return_{tag}").mean().alias(f"_layer_mean_{tag}")
            for tag in horizons
        ])
    )

    result = enriched.join(layer_means, on=["date", "layer"], how="left")
    excess_exprs = [
        (pl.col(f"label_return_{tag}") - pl.col(f"_layer_mean_{tag}"))
        .alias(f"label_excess_{tag}")
        for tag in horizons
    ]
    result = result.with_columns(excess_exprs)
    return result.select(
        ["ticker", "date"] + [f"label_excess_{tag}" for tag in horizons]
    ).sort(["ticker", "date"])


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
