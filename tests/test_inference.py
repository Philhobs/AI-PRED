import datetime
import json
import pickle

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pathlib import Path


TICKERS_FIXTURE = ["NVDA", "MSFT", "AMZN", "GOOGL", "META"]
N_DAYS = 500


def _write_ohlcv_fixture(ohlcv_dir: Path, tickers: list[str], n_days: int) -> None:
    start = datetime.date(2020, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(n_days)]
    for ticker in tickers:
        path = ohlcv_dir / ticker / "all.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(hash(ticker) % (2**31))
        prices = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_days))
        table = pa.table({
            "ticker": [ticker] * n_days,
            "date": pa.array(dates, type=pa.date32()),
            "open": prices.tolist(),
            "high": (prices * 1.01).tolist(),
            "low": (prices * 0.99).tolist(),
            "close_price": prices.tolist(),
            "volume": [1_000_000] * n_days,
        })
        pq.write_table(table, str(path))


def _write_fundamentals_fixture(fund_dir: Path, tickers: list[str]) -> None:
    for ticker in tickers:
        path = fund_dir / ticker / "quarterly.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([{
            "ticker": ticker,
            "period_end": datetime.date(2019, 12, 31),
            "pe_ratio_trailing": 25.0,
            "price_to_sales": 8.0,
            "price_to_book": 3.0,
            "revenue_growth_yoy": 0.15,
            "gross_margin": 0.60,
            "operating_margin": 0.25,
            "capex_to_revenue": 0.08,
            "debt_to_equity": 0.5,
            "current_ratio": 1.8,
        }]).write_parquet(str(path))


@pytest.fixture(scope="module")
def trained_env(tmp_path_factory):
    """Train per-layer models on fixture data; return (data_dir, artifacts_dir, date_str)."""
    base = tmp_path_factory.mktemp("inference")
    ohlcv_dir = base / "financials" / "ohlcv"
    fund_dir = base / "financials" / "fundamentals"
    artifacts_dir = base / "artifacts"

    # Write fixture data for the 5 test tickers
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    # Train per-layer models. Only cloud and compute layers have data for TICKERS_FIXTURE.
    from models.train import train_single_layer, build_training_dataset, FEATURE_COLS
    from ingestion.ticker_registry import LAYER_IDS, tickers_in_layer, layers as all_layers
    import numpy as np

    df_all = build_training_dataset(ohlcv_dir, fund_dir)

    for layer in all_layers():
        layer_tickers = tickers_in_layer(layer)
        layer_df = df_all.filter(pl.col("ticker").is_in(layer_tickers))
        if layer_df.is_empty():
            continue
        layer_id = LAYER_IDS[layer]
        layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"
        train_single_layer(layer_df, layer_dir, label_col="label_return_1y")

    # Day 300 of the fixture: has 20+ days price history
    date_str = (datetime.date(2020, 1, 1) + datetime.timedelta(days=300)).isoformat()

    return base, artifacts_dir, date_str


def test_run_inference_returns_correct_schema(trained_env):
    """run_inference returns DataFrame with all required output columns."""
    data_dir, artifacts_dir, date_str = trained_env

    from models.inference import run_inference
    result = run_inference(
        date_str=date_str,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        output_dir=data_dir / "predictions",
    )

    required_cols = {
        "ticker", "rank", "layer", "expected_annual_return",
        "confidence_low", "confidence_high",
        "lgbm_return", "rf_return", "ridge_return", "as_of_date",
    }
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )
    assert len(result) >= 1


def test_run_inference_ranks_are_unique_and_sequential(trained_env):
    """Ranks are unique integers 1..n_tickers with no gaps."""
    data_dir, artifacts_dir, date_str = trained_env

    from models.inference import run_inference
    result = run_inference(
        date_str=date_str,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        output_dir=data_dir / "predictions",
    )

    ranks = sorted(result["rank"].to_list())
    n = len(result)
    assert ranks == list(range(1, n + 1)), f"Ranks: {ranks}"


def test_run_inference_writes_parquet(trained_env):
    """run_inference writes predictions.parquet to output_dir/date=<date_str>/."""
    data_dir, artifacts_dir, date_str = trained_env
    output_dir = data_dir / "predictions"

    from models.inference import run_inference
    run_inference(
        date_str=date_str,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
    )

    parquet_path = output_dir / f"date={date_str}" / "predictions.parquet"
    assert parquet_path.exists(), f"Parquet not found at {parquet_path}"

    saved = pl.read_parquet(str(parquet_path))
    assert len(saved) >= 1
    assert "expected_annual_return" in saved.columns
