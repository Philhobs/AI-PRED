"""
Ensemble model training: LightGBM (q10/q50/q90) + Random Forest + Ridge.

Artifacts saved to models/artifacts/:
  lgbm_q10.pkl, lgbm_q50.pkl, lgbm_q90.pkl  — LightGBM quantile regressors
  rf_model.pkl                                 — RandomForestRegressor
  ridge_model.pkl                              — Ridge regressor
  feature_scaler.pkl                           — StandardScaler (fit on all training data)
  imputation_medians.json                      — per-feature median for null imputation
  feature_names.json                           — ordered feature list (guards column drift)
  ensemble_weights.json                        — {lgbm, rf, ridge} NNLS weights summing to 1
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd  # noqa: F401 — used for named-feature predict calls
import polars as pl
from scipy.optimize import nnls
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from processing.fundamental_features import join_fundamentals
from processing.label_builder import build_labels
from processing.price_features import build_price_features

# ── Feature columns ───────────────────────────────────────────────────────────
# Order is locked — inference.py and feature_names.json must match.
PRICE_FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "sma_20_deviation", "volatility_20d", "volume_ratio",
]
FUND_FEATURE_COLS = [
    "pe_ratio_trailing", "price_to_sales", "price_to_book",
    "revenue_growth_yoy", "gross_margin", "operating_margin",
    "capex_to_revenue", "debt_to_equity", "current_ratio",
]
FEATURE_COLS = PRICE_FEATURE_COLS + FUND_FEATURE_COLS  # 15 features total


# ── Data assembly ─────────────────────────────────────────────────────────────

def build_training_dataset(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
) -> pl.DataFrame:
    """
    Assemble the full labeled training dataset.

    Returns DataFrame with columns: ticker, date, FEATURE_COLS..., label_return_1y.
    Returns empty DataFrame if no OHLCV data exists.
    """
    labels = build_labels(ohlcv_dir)
    if labels.is_empty():
        return pl.DataFrame()

    price_df = build_price_features(ohlcv_dir)
    price_features = price_df.select(["ticker", "date"] + PRICE_FEATURE_COLS)

    # Inner join: keep only rows with complete 252-day forward labels
    df = price_features.join(labels, on=["ticker", "date"], how="inner")

    # Backward asof join: attach most recent quarterly fundamentals per row
    df = join_fundamentals(df, fundamentals_dir)

    return (
        df.select(["ticker", "date"] + FEATURE_COLS + ["label_return_1y"])
        .sort(["date", "ticker"])
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _impute(X: np.ndarray, medians: dict[str, float]) -> np.ndarray:
    """Fill NaN values column-wise using pre-computed per-feature medians."""
    X = X.copy()
    for i, name in enumerate(FEATURE_COLS):
        mask = np.isnan(X[:, i])
        if mask.any():
            X[mask, i] = medians.get(name, 0.0)
    return X


def _compute_medians(X: np.ndarray) -> dict[str, float]:
    """Compute per-feature nanmedian over the training set. Never uses validation data."""
    return {
        name: float(np.nanmedian(X[:, i]))
        for i, name in enumerate(FEATURE_COLS)
    }


# ── Main training entry point ─────────────────────────────────────────────────

def train(
    ohlcv_dir: Path = Path("data/raw/financials/ohlcv"),
    fundamentals_dir: Path = Path("data/raw/financials/fundamentals"),
    artifacts_dir: Path = Path("models/artifacts"),
    lgbm_params: dict | None = None,
    rf_params: dict | None = None,
    val_window_days: int = 252,
) -> None:
    """
    Train the full ensemble and save 9 artifacts to artifacts_dir.

    Walk-forward CV (3 folds) learns ensemble weights via NNLS on validation
    predictions. Final models are retrained on all available data.

    Args:
        lgbm_params: Override LightGBM hyperparameters (useful for testing).
        rf_params:   Override RF hyperparameters (useful for testing).
        val_window_days: Validation window size in days (default 252 = 1 year).
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = build_training_dataset(ohlcv_dir, fundamentals_dir)
    if df.is_empty():
        raise ValueError("No training data found. Run ingestion pipeline first.")

    dates_sorted = df["date"].cast(pl.Date).unique().sort().to_list()
    n_dates = len(dates_sorted)

    if n_dates < val_window_days * 3 + 50:
        raise ValueError(
            f"Insufficient unique dates: {n_dates} "
            f"(need ≥{val_window_days * 3 + 50} for 3-fold walk-forward CV)"
        )

    X_all = df.select(FEATURE_COLS).to_numpy().astype(float)
    y_all = df["label_return_1y"].to_numpy().astype(float)

    # Default production hyperparameters
    lgbm_base = lgbm_params or {
        "n_estimators": 500, "learning_rate": 0.05,
        "num_leaves": 31, "min_child_samples": 20,
        "n_jobs": -1, "random_state": 42,
    }
    rf_base = rf_params or {
        "n_estimators": 300, "max_features": "sqrt",
        "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42,
    }

    # ── Walk-forward CV: collect validation predictions for NNLS ─────────────
    # 3 split points: train on dates[0:split], validate on dates[split:split+val_window]
    # Splits are evenly spaced so validation windows don't overlap.
    splits = [
        n_dates - 3 * val_window_days,
        n_dates - 2 * val_window_days,
        n_dates - 1 * val_window_days,
    ]

    val_lgbm_preds, val_rf_preds, val_ridge_preds, val_y_all = [], [], [], []

    for split in splits:
        train_dates = dates_sorted[:split]
        val_dates = dates_sorted[split: split + val_window_days]

        train_mask = df["date"].is_in(train_dates)
        val_mask = df["date"].is_in(val_dates)

        X_tr = df.filter(train_mask).select(FEATURE_COLS).to_numpy().astype(float)
        y_tr = df.filter(train_mask)["label_return_1y"].to_numpy().astype(float)
        X_val = df.filter(val_mask).select(FEATURE_COLS).to_numpy().astype(float)
        y_val = df.filter(val_mask)["label_return_1y"].to_numpy().astype(float)

        medians_fold = _compute_medians(X_tr)
        X_tr_imp = _impute(X_tr, medians_fold)
        X_val_imp = _impute(X_val, medians_fold)

        scaler_fold = StandardScaler().fit(X_tr_imp)
        X_tr_sc = scaler_fold.transform(X_tr_imp)
        X_val_sc = scaler_fold.transform(X_val_imp)

        # LightGBM q50 (point estimate) — handles NaN natively (raw features)
        lgbm_fold = lgb.LGBMRegressor(
            objective="quantile", alpha=0.50, verbose=-1, **lgbm_base
        ).fit(X_tr, y_tr, feature_name=FEATURE_COLS)

        rf_fold = RandomForestRegressor(**rf_base).fit(X_tr_imp, y_tr)
        ridge_fold = Ridge(alpha=1.0).fit(X_tr_sc, y_tr)

        val_lgbm_preds.append(lgbm_fold.predict(pd.DataFrame(X_val, columns=FEATURE_COLS)))
        val_rf_preds.append(rf_fold.predict(X_val_imp))
        val_ridge_preds.append(ridge_fold.predict(X_val_sc))
        val_y_all.append(y_val)

    # Stack all fold predictions and solve NNLS for ensemble weights
    pred_matrix = np.column_stack([
        np.concatenate(val_lgbm_preds),
        np.concatenate(val_rf_preds),
        np.concatenate(val_ridge_preds),
    ])
    y_stacked = np.concatenate(val_y_all)
    raw_weights, _ = nnls(pred_matrix, y_stacked)
    w_sum = raw_weights.sum()
    weights = raw_weights / w_sum if w_sum > 0 else np.array([1 / 3, 1 / 3, 1 / 3])

    # ── Final models: retrain on ALL data ────────────────────────────────────
    imputation_medians = _compute_medians(X_all)
    X_all_imp = _impute(X_all, imputation_medians)
    scaler = StandardScaler().fit(X_all_imp)
    X_all_sc = scaler.transform(X_all_imp)

    lgbm_q10 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.10, verbose=-1, **lgbm_base
    ).fit(X_all, y_all, feature_name=FEATURE_COLS)
    lgbm_q50 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.50, verbose=-1, **lgbm_base
    ).fit(X_all, y_all, feature_name=FEATURE_COLS)
    lgbm_q90 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.90, verbose=-1, **lgbm_base
    ).fit(X_all, y_all, feature_name=FEATURE_COLS)
    rf_model = RandomForestRegressor(**rf_base).fit(X_all_imp, y_all)
    ridge_model = Ridge(alpha=1.0).fit(X_all_sc, y_all)

    # ── Save artifacts ────────────────────────────────────────────────────────
    def _pkl(obj: object, name: str) -> None:
        with open(artifacts_dir / name, "wb") as f:
            pickle.dump(obj, f)

    _pkl(lgbm_q10, "lgbm_q10.pkl")
    _pkl(lgbm_q50, "lgbm_q50.pkl")
    _pkl(lgbm_q90, "lgbm_q90.pkl")
    _pkl(rf_model, "rf_model.pkl")
    _pkl(ridge_model, "ridge_model.pkl")
    _pkl(scaler, "feature_scaler.pkl")

    (artifacts_dir / "imputation_medians.json").write_text(
        json.dumps(imputation_medians, indent=2)
    )
    (artifacts_dir / "feature_names.json").write_text(
        json.dumps(FEATURE_COLS, indent=2)
    )
    (artifacts_dir / "ensemble_weights.json").write_text(
        json.dumps({"lgbm": float(weights[0]), "rf": float(weights[1]),
                    "ridge": float(weights[2])}, indent=2)
    )

    print(f"[Train] Artifacts → {artifacts_dir}")
    print(f"[Train] Weights: lgbm={weights[0]:.3f} rf={weights[1]:.3f} ridge={weights[2]:.3f}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    train()
