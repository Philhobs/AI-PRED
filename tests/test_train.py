import datetime
import json

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pathlib import Path


# ── Fixture helpers ──────────────────────────────────────────────────────────

TICKERS_FIXTURE = ["NVDA", "MSFT", "AMZN", "GOOGL", "META"]
N_DAYS = 500  # 500 total days → 500-252=248 labeled days per ticker


def _write_ohlcv_fixture(ohlcv_dir: Path, tickers: list[str], n_days: int) -> None:
    """Write n_days of synthetic random-walk OHLCV for each ticker."""
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
    """Write a single quarterly snapshot per ticker starting before the OHLCV range."""
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


# Shared reduced hyperparams for test speed
_LGBM_TEST = {"n_estimators": 10, "learning_rate": 0.1, "num_leaves": 8,
              "min_child_samples": 5, "n_jobs": 1, "random_state": 42}
_RF_TEST    = {"n_estimators": 10, "max_features": "sqrt",
               "min_samples_leaf": 2, "n_jobs": 1, "random_state": 42}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_training_dataset_returns_labeled_rows(tmp_path):
    """build_training_dataset returns (N_DAYS-252)*n_tickers rows with all feature cols."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import build_training_dataset, FEATURE_COLS
    df = build_training_dataset(ohlcv_dir, fund_dir)

    assert len(df) == (N_DAYS - 252) * len(TICKERS_FIXTURE)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"
    assert "label_return_1y" in df.columns
    assert df["label_return_1y"].null_count() == 0


def test_train_writes_all_artifacts(tmp_path):
    """train() writes all 9 required artifact files to artifacts_dir."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train
    train(
        ohlcv_dir=ohlcv_dir,
        fundamentals_dir=fund_dir,
        artifacts_dir=artifacts_dir,
        lgbm_params=_LGBM_TEST,
        rf_params=_RF_TEST,
        val_window_days=50,
    )

    expected = [
        "lgbm_q10.pkl", "lgbm_q50.pkl", "lgbm_q90.pkl",
        "rf_model.pkl", "ridge_model.pkl", "feature_scaler.pkl",
        "imputation_medians.json", "feature_names.json", "ensemble_weights.json",
    ]
    for name in expected:
        assert (artifacts_dir / name).exists(), f"Missing artifact: {name}"


def test_ensemble_weights_sum_to_one(tmp_path):
    """ensemble_weights.json values sum to 1.0 within floating-point tolerance."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, N_DAYS)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train
    train(
        ohlcv_dir=ohlcv_dir,
        fundamentals_dir=fund_dir,
        artifacts_dir=artifacts_dir,
        lgbm_params=_LGBM_TEST,
        rf_params=_RF_TEST,
        val_window_days=50,
    )

    weights = json.loads((artifacts_dir / "ensemble_weights.json").read_text())
    total = weights["lgbm"] + weights["rf"] + weights["ridge"]
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"
    assert weights["lgbm"] >= 0
    assert weights["rf"] >= 0
    assert weights["ridge"] >= 0


def test_train_raises_on_insufficient_data(tmp_path):
    """train() raises ValueError when data has fewer unique dates than 3*val_window+50."""
    ohlcv_dir = tmp_path / "financials" / "ohlcv"
    fund_dir = tmp_path / "financials" / "fundamentals"
    artifacts_dir = tmp_path / "artifacts"
    # 300 days → 300-252=48 labeled days; need 252*3+50=806 → should raise
    _write_ohlcv_fixture(ohlcv_dir, TICKERS_FIXTURE, 300)
    _write_fundamentals_fixture(fund_dir, TICKERS_FIXTURE)

    from models.train import train
    with pytest.raises(ValueError, match="Insufficient unique dates"):
        train(
            ohlcv_dir=ohlcv_dir,
            fundamentals_dir=fund_dir,
            artifacts_dir=artifacts_dir,
            val_window_days=252,  # production default — 300 days is way too few
        )


def test_feature_cols_contains_fx():
    from models.train import FEATURE_COLS, FX_FEATURE_COLS
    assert all(c in FEATURE_COLS for c in FX_FEATURE_COLS)


def test_tier_feature_cols_medium_equals_feature_cols():
    """TIER_FEATURE_COLS['medium'] must be identical to FEATURE_COLS (48 features)."""
    from models.train import FEATURE_COLS, TIER_FEATURE_COLS
    assert TIER_FEATURE_COLS["medium"] == FEATURE_COLS


def test_horizon_configs_has_all_eight_horizons():
    """HORIZON_CONFIGS contains exactly the 8 expected horizon tags."""
    from models.train import HORIZON_CONFIGS
    expected = {"5d", "20d", "65d", "252d", "756d", "1260d", "2520d", "5040d"}
    assert set(HORIZON_CONFIGS.keys()) == expected


def test_horizon_configs_tiers_are_valid():
    """Every HORIZON_CONFIGS entry has a tier that exists in TIER_FEATURE_COLS."""
    from models.train import HORIZON_CONFIGS, TIER_FEATURE_COLS
    for tag, cfg in HORIZON_CONFIGS.items():
        assert cfg["tier"] in TIER_FEATURE_COLS, f"Invalid tier for horizon {tag}"


def test_impute_uses_feature_cols_param(tmp_path):
    """_impute fills NaNs only for the provided feature_cols, not global FEATURE_COLS."""
    from models.train import _impute
    import numpy as np

    feature_cols = ["a", "b"]
    X = np.array([[1.0, np.nan], [np.nan, 3.0]])
    medians = {"a": 10.0, "b": 20.0}
    result = _impute(X, medians, feature_cols=feature_cols)

    assert result[0, 1] == pytest.approx(20.0)
    assert result[1, 0] == pytest.approx(10.0)
