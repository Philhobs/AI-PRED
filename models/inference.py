"""
Per-layer ensemble inference: run one model per supply chain layer, merge, rank globally.

Output schema per ticker:
  ticker, rank, layer, expected_annual_return, confidence_low, confidence_high,
  lgbm_return, rf_return, ridge_return, as_of_date

Written to: <output_dir>/date={date_str}/predictions.parquet
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import pickle
from pathlib import Path

_LOG = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import polars as pl

from ingestion.ticker_registry import (
    LAYER_IDS, TICKERS, tickers_in_layer, layers as all_layers,
)
from models.train import (
    FEATURE_COLS, INSIDER_FEATURE_COLS, SENTIMENT_FEATURE_COLS,
    SHORT_INTEREST_FEATURE_COLS, EARNINGS_FEATURE_COLS, GRAPH_FEATURE_COLS,
    OWNERSHIP_FEATURE_COLS, ENERGY_FEATURE_COLS,
)
from processing.earnings_features import join_earnings_features
from processing.energy_geo_features import join_energy_geo_features
from processing.fundamental_features import join_fundamentals
from processing.graph_features import join_graph_features
from processing.insider_features import join_insider_features
from processing.ownership_features import join_ownership_features
from processing.price_features import build_price_features
from processing.sentiment_features import join_sentiment_features
from processing.short_interest_features import join_short_interest_features


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _impute(X: np.ndarray, medians: dict[str, float]) -> np.ndarray:
    X = X.copy()
    for i, name in enumerate(FEATURE_COLS):
        mask = np.isnan(X[:, i])
        if mask.any():
            X[mask, i] = medians.get(name, 0.0)
    return X


def _build_feature_df(
    date_str: str,
    data_dir: Path,
) -> pl.DataFrame:
    """Build the 43-feature DataFrame for all tickers on date_str."""
    ohlcv_dir        = data_dir / "financials" / "ohlcv"
    fundamentals_dir = data_dir / "financials" / "fundamentals"

    price_df = build_price_features(ohlcv_dir, filter_date=date_str)
    if price_df.is_empty():
        raise RuntimeError(
            f"No price data for {date_str}. Run ohlcv_ingestion.py to refresh."
        )

    df = join_fundamentals(price_df, fundamentals_dir)

    insider_features_dir = data_dir / "financials" / "insider_features"
    if insider_features_dir.exists():
        df = join_insider_features(df, insider_features_dir)
    else:
        for col in INSIDER_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    sentiment_features_dir = data_dir / "news" / "sentiment_features"
    if sentiment_features_dir.exists():
        df = join_sentiment_features(df, sentiment_features_dir)
    else:
        for col in SENTIMENT_FEATURE_COLS:
            dtype = pl.Int64 if col == "article_count_7d" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    si_features_dir = data_dir / "financials" / "short_interest_features"
    if si_features_dir.exists():
        df = join_short_interest_features(df, si_features_dir)
    else:
        for col in SHORT_INTEREST_FEATURE_COLS:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    earnings_features_dir = data_dir / "financials" / "earnings_features"
    if earnings_features_dir.exists():
        df = join_earnings_features(df, earnings_features_dir)
    else:
        for col in EARNINGS_FEATURE_COLS:
            dtype = pl.Int32 if col == "eps_beat_streak" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    graph_features_dir = data_dir / "graph" / "features"
    if graph_features_dir.exists():
        df = join_graph_features(df, graph_features_dir)
    else:
        # join_graph_features also produces energy_deal_mw_90d and hyperscaler_ppa_count_90d
        for col in GRAPH_FEATURE_COLS + ["energy_deal_mw_90d", "hyperscaler_ppa_count_90d"]:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    ownership_features_dir = data_dir / "financials" / "13f_holdings" / "features"
    if ownership_features_dir.exists():
        df = join_ownership_features(df, ownership_features_dir)
    else:
        for col in OWNERSHIP_FEATURE_COLS:
            dtype = pl.Int32 if col == "inst_holder_count" else pl.Float64
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))

    # Join energy geography features — adds us_power_moat_score and geo_weighted_tailwind_score.
    df = join_energy_geo_features(df)

    return df


def _predict_layer(
    feature_df: pl.DataFrame,
    layer: str,
    artifacts_dir: Path,
) -> pl.DataFrame | None:
    """Run one layer model on the tickers belonging to that layer.

    Returns DataFrame with [ticker, layer, expected_annual_return,
    confidence_low, confidence_high, lgbm_return, rf_return, ridge_return]
    or None if artifacts missing.
    """
    layer_id = LAYER_IDS[layer]
    layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"

    if not (layer_dir / "feature_names.json").exists():
        return None

    feature_names_saved = json.loads((layer_dir / "feature_names.json").read_text())
    if feature_names_saved != FEATURE_COLS:
        raise ValueError(
            f"Layer {layer}: feature mismatch. Retrain with models/train.py."
        )

    layer_tickers = tickers_in_layer(layer)
    layer_df = feature_df.filter(pl.col("ticker").is_in(layer_tickers))
    if layer_df.is_empty():
        return None

    tickers = layer_df["ticker"].to_list()
    medians = json.loads((layer_dir / "imputation_medians.json").read_text())
    weights = json.loads((layer_dir / "ensemble_weights.json").read_text())

    X_raw = layer_df.select(FEATURE_COLS).to_numpy().astype(float)
    X_imp = _impute(X_raw, medians)
    scaler = _load_pickle(layer_dir / "feature_scaler.pkl")
    X_sc = scaler.transform(X_imp)
    X_df = pd.DataFrame(X_imp, columns=FEATURE_COLS)

    lgbm_q10 = _load_pickle(layer_dir / "lgbm_q10.pkl")
    lgbm_q50 = _load_pickle(layer_dir / "lgbm_q50.pkl")
    lgbm_q90 = _load_pickle(layer_dir / "lgbm_q90.pkl")
    rf_model  = _load_pickle(layer_dir / "rf_model.pkl")
    ridge_model = _load_pickle(layer_dir / "ridge_model.pkl")

    q10_preds = lgbm_q10.predict(X_df)
    q50_preds = lgbm_q50.predict(X_df)
    q90_preds = lgbm_q90.predict(X_df)
    rf_preds    = rf_model.predict(X_imp)
    ridge_preds = ridge_model.predict(X_sc)

    expected = (
        weights["lgbm"] * q50_preds
        + weights["rf"]  * rf_preds
        + weights["ridge"] * ridge_preds
    )

    return pl.DataFrame({
        "ticker": tickers,
        "layer": [layer] * len(tickers),
        "expected_annual_return": expected.tolist(),
        "confidence_low":  q10_preds.tolist(),
        "confidence_high": q90_preds.tolist(),
        "lgbm_return":  q50_preds.tolist(),
        "rf_return":    rf_preds.tolist(),
        "ridge_return": ridge_preds.tolist(),
    })


def run_inference(
    date_str: str,
    data_dir: Path = Path("data/raw"),
    artifacts_dir: Path = Path("models/artifacts"),
    output_dir: Path = Path("data/predictions"),
) -> pl.DataFrame:
    """Run all 10 layer models and return globally ranked predictions.

    Raises ValueError if date_str is a weekend.
    Raises RuntimeError if no price data exists for date_str.
    """
    as_of = dt.date.fromisoformat(date_str)
    if as_of.weekday() >= 5:
        raise ValueError(f"{date_str} is a weekend. Skip inference on non-trading days.")

    print(f"[Inference] Running for {date_str}...")

    feature_df = _build_feature_df(date_str, data_dir)

    all_preds: list[pl.DataFrame] = []
    for layer in all_layers():
        layer_preds = _predict_layer(feature_df, layer, artifacts_dir)
        if layer_preds is not None:
            all_preds.append(layer_preds)

    if not all_preds:
        raise RuntimeError(
            f"No layer artifacts found in {artifacts_dir}. Run models/train.py first."
        )

    combined = pl.concat(all_preds).sort("expected_annual_return", descending=True)
    combined = combined.with_columns(
        pl.Series("rank", list(range(1, len(combined) + 1)), dtype=pl.Int32),
        pl.lit(as_of).cast(pl.Date).alias("as_of_date"),
    )

    out_path = output_dir / f"date={date_str}" / "predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="snappy")

    n_tickers = len(combined)
    n_layers = combined["layer"].n_unique()
    print(f"[Inference] {n_tickers} tickers across {n_layers} layers → {out_path}")
    print(combined.select(["rank", "ticker", "layer", "expected_annual_return"]).head(10))

    # Enrich predictions with portfolio metrics (liquidity, agreement, correlation)
    try:
        from processing.portfolio_metrics import enrich
        enrich(date_str, predictions_dir=output_dir)
    except Exception as exc:
        _LOG.warning("Portfolio metrics enrichment failed (non-fatal): %s", exc, exc_info=True)

    return combined


if __name__ == "__main__":
    import sys
    today = dt.date.today()
    if today.weekday() >= 5:
        print(f"[Inference] {today} is a weekend — skipping.")
        sys.exit(0)
    run_inference(date_str=today.isoformat())
