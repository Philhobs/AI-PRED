"""
Walk-forward backtest for the ensemble model.

Methodology:
  1. Build the full labeled dataset (features + 1-year forward labels) via build_training_dataset.
  2. Time-based split: train on dates before TRAIN_CUTOFF, test on dates from TEST_START to TEST_END.
  3. Fit ensemble on training split (same pipeline as train.py, but here as a standalone function).
  4. Predict on test split.
  5. Report metrics on monthly rebalance dates to avoid overlapping-label bias:
       - IC  : Spearman rank correlation (predicted rank vs actual 1-year return) per month,
               mean IC and t-statistic over test period.
       - Hit rate: fraction of months where top-half predicted tickers beat bottom-half.
       - Return spread: mean actual return of top-5 ranked tickers minus bottom-5, per month.

Results written to data/backtest/walk_forward_results.json.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

_LOG = logging.getLogger(__name__)

# Walk-forward config
TRAIN_CUTOFF = "2022-12-31"   # everything before this date is training data
TEST_START   = "2023-01-01"   # start of out-of-sample test period
TEST_END     = "2024-03-31"   # end of test period (labels need 252d horizon, ~2025-03-31 prices OK)
REBALANCE_DAY = 1             # use first trading day of each month for IC calculation


def _fit_ensemble(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]):
    """Fit LightGBM + RF ensemble on training data. Returns (lgbm_q50, rf, weights)."""
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import nnls

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)

    lgbm_q50 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5,
        n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=20,
        random_state=42, verbose=-1,
    )
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    ridge = Ridge(alpha=1.0)

    import pandas as pd
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    lgbm_q50.fit(X_train_df, y_train)
    rf.fit(X_train, y_train)
    ridge.fit(X_sc, y_train)

    lgbm_pred = lgbm_q50.predict(X_train_df)
    rf_pred   = rf.predict(X_train)
    ridge_pred = ridge.predict(X_sc)

    A = np.column_stack([lgbm_pred, rf_pred, ridge_pred])
    weights, _ = nnls(A, y_train)
    total = weights.sum()
    weights = weights / total if total > 0 else np.array([0.5, 0.5, 0.0])

    return lgbm_q50, rf, ridge, scaler, weights


def _predict(lgbm_q50, rf, ridge, scaler, weights, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    import pandas as pd
    X_df = pd.DataFrame(X, columns=feature_names)
    X_sc = scaler.transform(X)
    pred = (
        weights[0] * lgbm_q50.predict(X_df)
        + weights[1] * rf.predict(X)
        + weights[2] * ridge.predict(X_sc)
    )
    return pred


def run_backtest(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    output_dir: Path = Path("data/backtest"),
) -> dict:
    """
    Run walk-forward backtest. Returns dict with summary metrics.

    Steps:
    1. Build full labeled dataset.
    2. Train/test split by date.
    3. Impute NaN with training-set medians.
    4. Fit ensemble on training set.
    5. For each month in test period (first date of month), compute cross-sectional IC.
    6. Report mean IC, IC t-stat, hit rate, top5-bottom5 spread.
    """
    from models.train import build_training_dataset, FEATURE_COLS

    _LOG.info("Building full labeled dataset...")
    df = build_training_dataset(ohlcv_dir, fundamentals_dir)
    if df.is_empty():
        _LOG.error("No labeled data available — run ingestion pipeline first")
        return {}

    _LOG.info("Full dataset: %d rows, %d tickers, dates %s to %s",
              len(df), df["ticker"].n_unique(),
              df["date"].min(), df["date"].max())

    train_df = df.filter(pl.col("date") <= pl.lit(TRAIN_CUTOFF).str.to_date())
    test_df  = df.filter(
        (pl.col("date") >= pl.lit(TEST_START).str.to_date()) &
        (pl.col("date") <= pl.lit(TEST_END).str.to_date())
    )

    _LOG.info("Train rows: %d | Test rows: %d", len(train_df), len(test_df))

    if len(train_df) < 100 or len(test_df) < 10:
        _LOG.error("Insufficient data for backtest")
        return {}

    X_train = train_df.select(FEATURE_COLS).to_numpy().astype(float)
    y_train = train_df["label_return_1y"].to_numpy().astype(float)
    X_test  = test_df.select(FEATURE_COLS).to_numpy().astype(float)

    # Impute NaN with training-set medians
    medians = {}
    for i, col in enumerate(FEATURE_COLS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmedian(X_train[:, i])
        medians[col] = 0.0 if np.isnan(v) else float(v)

    def _impute(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for i, col in enumerate(FEATURE_COLS):
            mask = np.isnan(X[:, i])
            if mask.any():
                X[mask, i] = medians[col]
        return X

    X_train_imp = _impute(X_train)
    X_test_imp  = _impute(X_test)

    _LOG.info("Fitting ensemble on %d training rows...", len(X_train_imp))
    lgbm_q50, rf, ridge, scaler, weights = _fit_ensemble(X_train_imp, y_train, FEATURE_COLS)
    _LOG.info("Ensemble weights: lgbm=%.3f rf=%.3f ridge=%.3f", *weights)

    predictions = _predict(lgbm_q50, rf, ridge, scaler, weights, X_test_imp, FEATURE_COLS)

    # Attach predictions to test DataFrame
    test_with_pred = test_df.with_columns(
        pl.Series("predicted_return", predictions)
    )

    # Monthly rebalance: pick first trading date of each month
    monthly_dates = (
        test_with_pred
        .with_columns(
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.year().alias("year"),
        )
        .group_by(["year", "month"])
        .agg(pl.col("date").min().alias("rebalance_date"))
        .sort(["year", "month"])
        ["rebalance_date"]
        .to_list()
    )

    ics: list[float] = []
    hit_rates: list[float] = []
    spreads: list[float] = []

    for rebalance_date in monthly_dates:
        month_df = test_with_pred.filter(pl.col("date") == rebalance_date)
        if len(month_df) < 4:
            continue

        actual = month_df["label_return_1y"].to_numpy()
        predicted = month_df["predicted_return"].to_numpy()

        # Cross-sectional IC (Spearman)
        rho, _ = spearmanr(predicted, actual)
        if not np.isnan(rho):
            ics.append(rho)

        # Hit rate: top-half predicted beats bottom-half predicted (by actual return)
        median_pred = np.median(predicted)
        top_mask = predicted >= median_pred
        bottom_mask = predicted < median_pred
        if top_mask.sum() > 0 and bottom_mask.sum() > 0:
            top_mean = actual[top_mask].mean()
            bottom_mean = actual[bottom_mask].mean()
            hit_rates.append(1.0 if top_mean > bottom_mean else 0.0)
            spreads.append(top_mean - bottom_mean)

    if not ics:
        _LOG.error("No IC values computed — test period may be too short")
        return {}

    ics_arr = np.array(ics)
    mean_ic = float(np.mean(ics_arr))
    ic_std  = float(np.std(ics_arr, ddof=1))
    ic_tstat = float(mean_ic / (ic_std / np.sqrt(len(ics_arr)))) if ic_std > 0 else 0.0
    hit_rate = float(np.mean(hit_rates)) if hit_rates else 0.0
    mean_spread = float(np.mean(spreads)) if spreads else 0.0

    results = {
        "train_cutoff": TRAIN_CUTOFF,
        "test_start":   TEST_START,
        "test_end":     TEST_END,
        "train_rows":   len(train_df),
        "test_rows":    len(test_df),
        "monthly_rebalance_dates": len(ics),
        "ensemble_weights": {"lgbm": float(weights[0]), "rf": float(weights[1]), "ridge": float(weights[2])},
        "mean_ic":       round(mean_ic, 4),
        "ic_std":        round(ic_std, 4),
        "ic_tstat":      round(ic_tstat, 4),
        "hit_rate":      round(hit_rate, 4),
        "mean_spread":   round(mean_spread, 4),
        "feature_count": len(FEATURE_COLS),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "walk_forward_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    _LOG.info("Backtest results written to %s", out_path)

    return results


def _print_report(results: dict) -> None:
    print("\n" + "═" * 50)
    print("  Walk-Forward Backtest Results")
    print("═" * 50)
    print(f"  Train period:  up to {results['train_cutoff']}")
    print(f"  Test period:   {results['test_start']} → {results['test_end']}")
    print(f"  Train rows:    {results['train_rows']:,}")
    print(f"  Test rows:     {results['test_rows']:,}")
    print(f"  Rebalance months: {results['monthly_rebalance_dates']}")
    print(f"  Features:      {results['feature_count']}")
    print()
    print(f"  Mean IC:       {results['mean_ic']:.4f}  (t-stat {results['ic_tstat']:.2f})")
    print(f"  IC std:        {results['ic_std']:.4f}")
    print(f"  Hit rate:      {results['hit_rate']:.1%}")
    print(f"  Mean spread:   {results['mean_spread']:.2%}  (top-half vs bottom-half annual return)")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    ohlcv_dir      = project_root / "data" / "raw" / "financials" / "ohlcv"
    fundamentals_dir = project_root / "data" / "raw" / "financials" / "fundamentals"
    output_dir       = project_root / "data" / "backtest"

    results = run_backtest(ohlcv_dir, fundamentals_dir, output_dir)
    if results:
        _print_report(results)
