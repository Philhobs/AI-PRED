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
import logging
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
from processing.insider_features import join_insider_features
from processing.label_builder import build_labels
from processing.price_features import build_price_features
from processing.earnings_features import join_earnings_features
from processing.sentiment_features import join_sentiment_features
from processing.short_interest_features import join_short_interest_features
from ingestion.ticker_registry import LAYER_IDS, tickers_in_layer, layers as all_layers
from processing.graph_features import join_graph_features
from processing.ownership_features import join_ownership_features

_LOG = logging.getLogger(__name__)

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
INSIDER_FEATURE_COLS = [
    "insider_cluster_buy_90d",
    "insider_net_value_30d",
    "insider_buy_sell_ratio_90d",
    "congress_net_buy_90d",
    "congress_trade_count_90d",
]
SENTIMENT_FEATURE_COLS = [
    "sentiment_mean_7d",
    "sentiment_std_7d",
    "article_count_7d",
    "sentiment_momentum_14d",
    "ticker_vs_market_7d",
]
SHORT_INTEREST_FEATURE_COLS = [
    "short_vol_ratio_10d",
    "short_vol_ratio_30d",
    "short_ratio_momentum",
]
EARNINGS_FEATURE_COLS = [
    "eps_surprise_last",
    "eps_surprise_mean_4q",
    "eps_beat_streak",
]
GRAPH_FEATURE_COLS = [
    "graph_partner_momentum_30d",
    "graph_deal_count_90d",
    "graph_hops_to_hyperscaler",
]
OWNERSHIP_FEATURE_COLS = [
    "inst_ownership_pct",
    "inst_net_shares_qoq",
    "inst_holder_count",
    "inst_concentration_top10",
    "inst_momentum_2q",
]
FEATURE_COLS = (
    PRICE_FEATURE_COLS + FUND_FEATURE_COLS + INSIDER_FEATURE_COLS
    + SENTIMENT_FEATURE_COLS + SHORT_INTEREST_FEATURE_COLS
    + EARNINGS_FEATURE_COLS + GRAPH_FEATURE_COLS
    + OWNERSHIP_FEATURE_COLS  # 34 → 39 features total
)


# ── Data assembly ─────────────────────────────────────────────────────────────

def build_training_dataset(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    layer: str | None = None,
) -> pl.DataFrame:
    """
    Assemble the full labeled training dataset, optionally filtered to one layer.

    layer: if given, filters to tickers in that layer only (for per-layer training).
    Returns DataFrame with columns: ticker, date, FEATURE_COLS..., label_return_1y.
    Returns empty DataFrame if no OHLCV data exists.
    """
    labels = build_labels(ohlcv_dir)
    if labels.is_empty():
        return pl.DataFrame()

    # Filter to layer tickers if specified
    if layer is not None:
        layer_tickers = tickers_in_layer(layer)
        labels = labels.filter(pl.col("ticker").is_in(layer_tickers))
        if labels.is_empty():
            return pl.DataFrame()

    price_df = build_price_features(ohlcv_dir)
    if layer is not None:
        price_df = price_df.filter(pl.col("ticker").is_in(layer_tickers))
    price_features = price_df.select(["ticker", "date"] + PRICE_FEATURE_COLS)

    # Inner join: keep only rows with complete 252-day forward labels
    df = price_features.join(labels, on=["ticker", "date"], how="inner")

    # Backward asof join: attach most recent quarterly fundamentals per row
    df = join_fundamentals(df, fundamentals_dir)

    # Join insider signal features (backward asof join on ticker, date)
    insider_features_dir = fundamentals_dir.parent / "insider_features"
    if insider_features_dir.exists():
        df = join_insider_features(df, insider_features_dir)
    else:
        _LOG.warning(
            "Insider features directory not found at %s — "
            "insider columns will be null. Run: python processing/insider_features.py",
            insider_features_dir,
        )
        for col in INSIDER_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Join sentiment signal features (backward asof join on ticker, date)
    # fundamentals_dir = data/raw/financials/fundamentals → .parent.parent = data/raw
    sentiment_features_dir = fundamentals_dir.parent.parent / "news" / "sentiment_features"
    if sentiment_features_dir.exists():
        df = join_sentiment_features(df, sentiment_features_dir)
    else:
        _LOG.warning(
            "Sentiment features directory not found at %s — "
            "sentiment columns will be null. Run: python processing/sentiment_features.py",
            sentiment_features_dir,
        )
        for col in SENTIMENT_FEATURE_COLS:
            dtype = pl.Int64 if col == "article_count_7d" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Join short interest features (backward asof join on ticker, date)
    si_features_dir = fundamentals_dir.parent / "short_interest_features"
    if si_features_dir.exists():
        df = join_short_interest_features(df, si_features_dir)
    else:
        _LOG.warning(
            "Short interest features directory not found at %s — "
            "short interest columns will be null. Run: python processing/short_interest_features.py",
            si_features_dir,
        )
        for col in SHORT_INTEREST_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Join earnings surprise features (backward asof join on ticker, date)
    earnings_features_dir = fundamentals_dir.parent / "earnings_features"
    if earnings_features_dir.exists():
        df = join_earnings_features(df, earnings_features_dir)
    else:
        _LOG.warning(
            "Earnings features directory not found at %s — "
            "earnings columns will be null. Run: python processing/earnings_features.py",
            earnings_features_dir,
        )
        for col in EARNINGS_FEATURE_COLS:
            dtype = pl.Int32 if col == "eps_beat_streak" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Join graph features (backward asof join on ticker, date)
    graph_features_dir = fundamentals_dir.parent / "graph" / "features"
    if graph_features_dir.exists():
        df = join_graph_features(df, graph_features_dir)
    else:
        for col in GRAPH_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Join 13F institutional ownership features (backward asof join on ticker, date)
    ownership_features_dir = fundamentals_dir.parent / "13f_holdings" / "features"
    if ownership_features_dir.exists():
        df = join_ownership_features(df, ownership_features_dir)
    else:
        _LOG.warning(
            "Ownership features not found at %s — columns will be null. "
            "Run: python ingestion/sec_13f_ingestion.py --bootstrap "
            "then python processing/ownership_features.py",
            ownership_features_dir,
        )
        for col in OWNERSHIP_FEATURE_COLS:
            dtype = pl.Int32 if col == "inst_holder_count" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

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
    """Compute per-feature nanmedian over the training set. Never uses validation data.
    Falls back to 0.0 for columns that are entirely NaN (e.g. fundamentals not yet available)."""
    result = {}
    for i, name in enumerate(FEATURE_COLS):
        v = np.nanmedian(X[:, i])
        result[name] = 0.0 if np.isnan(v) else float(v)
    return result


# ── Per-layer training ────────────────────────────────────────────────────────

def train_single_layer(df: pl.DataFrame, artifacts_dir: Path) -> None:
    """Fit the ensemble on df and save all artifacts to artifacts_dir.

    df must have columns: ticker, date, FEATURE_COLS..., label_return_1y.
    """
    if len(df) < 50:
        _LOG.warning("Only %d rows — skipping layer (too few samples)", len(df))
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    X = df.select(FEATURE_COLS).to_numpy().astype(float)
    y = df["label_return_1y"].to_numpy().astype(float)

    medians = _compute_medians(X)
    X_imp = _impute(X, medians)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    X_df = pd.DataFrame(X_imp, columns=FEATURE_COLS)

    lgbm_q10 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.1, n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
    )
    lgbm_q50 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
    )
    lgbm_q90 = lgb.LGBMRegressor(
        objective="quantile", alpha=0.9, n_estimators=400, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
    )
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    ridge = Ridge(alpha=1.0)

    lgbm_q10.fit(X_df, y)
    lgbm_q50.fit(X_df, y)
    lgbm_q90.fit(X_df, y)
    rf.fit(X_imp, y)
    ridge.fit(X_sc, y)

    preds = np.column_stack([
        lgbm_q50.predict(X_df),
        rf.predict(X_imp),
        ridge.predict(X_sc),
    ])
    weights, _ = nnls(preds, y)
    total = weights.sum()
    weights = weights / total if total > 1e-9 else np.array([0.5, 0.5, 0.0])

    def _pkl(obj: object, name: str) -> None:
        with open(artifacts_dir / name, "wb") as f:
            pickle.dump(obj, f)

    _pkl(lgbm_q10, "lgbm_q10.pkl")
    _pkl(lgbm_q50, "lgbm_q50.pkl")
    _pkl(lgbm_q90, "lgbm_q90.pkl")
    _pkl(rf, "rf_model.pkl")
    _pkl(ridge, "ridge_model.pkl")
    _pkl(scaler, "feature_scaler.pkl")

    (artifacts_dir / "imputation_medians.json").write_text(json.dumps(medians))
    (artifacts_dir / "feature_names.json").write_text(json.dumps(FEATURE_COLS))
    (artifacts_dir / "ensemble_weights.json").write_text(
        json.dumps({"lgbm": float(weights[0]), "rf": float(weights[1]), "ridge": float(weights[2])})
    )
    _LOG.info(
        "[%s] Trained on %d rows. Weights: lgbm=%.3f rf=%.3f ridge=%.3f",
        artifacts_dir.name, len(df), *weights,
    )


def train_all_layers(
    ohlcv_dir: Path,
    fundamentals_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Train one ensemble per supply chain layer and save artifacts."""
    for layer in all_layers():
        layer_id = LAYER_IDS[layer]
        layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"
        _LOG.info("Training layer %02d: %s", layer_id, layer)

        df = build_training_dataset(ohlcv_dir, fundamentals_dir, layer=layer)
        if df.is_empty():
            _LOG.warning("No data for layer %s — skipping", layer)
            continue

        train_single_layer(df, layer_dir)


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).parent.parent
    ohlcv_dir        = project_root / "data" / "raw" / "financials" / "ohlcv"
    fundamentals_dir = project_root / "data" / "raw" / "financials" / "fundamentals"
    artifacts_dir    = project_root / "models" / "artifacts"

    _LOG.info("Training per-layer ensembles for 10 supply chain layers...")
    train_all_layers(ohlcv_dir, fundamentals_dir, artifacts_dir)
    _LOG.info("[Train] All layer artifacts → %s", artifacts_dir)
    print(f"[Train] Artifacts → {artifacts_dir}")
