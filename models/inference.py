"""
Ensemble inference: load artifacts, build today's feature vector, predict, rank.

Output schema per ticker:
  ticker (string), rank (int32), expected_annual_return (float64),
  confidence_low (float64), confidence_high (float64),
  lgbm_return (float64), rf_return (float64), ridge_return (float64),
  as_of_date (date32)

Written to: <output_dir>/date={date_str}/predictions.parquet
"""
import datetime as dt
import json
import pickle
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from models.train import FEATURE_COLS
from processing.fundamental_features import join_fundamentals


def _load_pickle(artifacts_dir: Path, name: str):
    with open(artifacts_dir / name, "rb") as f:
        return pickle.load(f)


def _impute(X: np.ndarray, medians: dict[str, float]) -> np.ndarray:
    """Fill NaN in feature matrix using training-set medians (never refit at inference time)."""
    X = X.copy()
    for i, name in enumerate(FEATURE_COLS):
        mask = np.isnan(X[:, i])
        if mask.any():
            X[mask, i] = medians.get(name, 0.0)
    return X


def run_inference(
    date_str: str,
    data_dir: Path = Path("data/raw"),
    artifacts_dir: Path = Path("models/artifacts"),
    output_dir: Path = Path("data/predictions"),
) -> pl.DataFrame:
    """
    Build predictions for all tickers on date_str and write to Parquet.

    Steps:
    1. Compute price features for all tickers via DuckDB windowed SQL, filtered to date_str.
    2. Join the most recent quarterly fundamentals (backward asof join).
    3. Load all artifacts; verify feature_names.json matches FEATURE_COLS.
    4. Apply imputation_medians + feature_scaler (never refit at inference time).
    5. LightGBM q10/q50/q90 on raw features; RF and Ridge on imputed+scaled.
    6. Ensemble: expected_annual_return = w_lgbm*q50 + w_rf*rf + w_ridge*ridge.
    7. Rank by expected_annual_return descending (rank 1 = highest expected return).
    8. Write <output_dir>/date={date_str}/predictions.parquet.
    9. Return the predictions DataFrame.

    Raises:
        FileNotFoundError: if any artifact is missing (run models/train.py first).
        ValueError: if feature_names.json doesn't match current FEATURE_COLS.
        RuntimeError: if no price data exists for date_str.
    """
    ohlcv_dir = data_dir / "financials" / "ohlcv"
    fundamentals_dir = data_dir / "financials" / "fundamentals"

    # ── Step 1: Price features for today ─────────────────────────────────────
    ohlcv_glob = str(ohlcv_dir / "*" / "*.parquet")
    con = duckdb.connect()
    try:
        price_df = con.execute(f"""
            WITH price AS (
                SELECT
                    ticker,
                    date::date AS date,
                    close_price / NULLIF(LAG(close_price, 1) OVER w, 0) - 1  AS return_1d,
                    close_price / NULLIF(LAG(close_price, 5) OVER w, 0) - 1  AS return_5d,
                    close_price / NULLIF(LAG(close_price, 20) OVER w, 0) - 1 AS return_20d,
                    close_price / NULLIF(AVG(close_price) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ), 0) - 1                                                 AS sma_20_deviation,
                    STDDEV(close_price) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ) / NULLIF(AVG(close_price) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ), 0)                                                     AS volatility_20d,
                    volume / NULLIF(AVG(CAST(volume AS DOUBLE)) OVER (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ), 0)                                                     AS volume_ratio
                FROM read_parquet('{ohlcv_glob}')
                WINDOW w AS (PARTITION BY ticker ORDER BY date)
            )
            SELECT * FROM price
            WHERE date = DATE '{date_str}'
        """).pl()
    finally:
        con.close()

    if price_df.is_empty():
        raise RuntimeError(
            f"No price data found for {date_str}. "
            "Run ohlcv_ingestion.py to refresh data."
        )

    price_df = price_df.with_columns(pl.col("date").cast(pl.Date))

    # ── Step 2: Fundamental features ─────────────────────────────────────────
    feature_df = join_fundamentals(price_df, fundamentals_dir)

    # ── Step 3: Load and validate artifacts ──────────────────────────────────
    feature_names_saved = json.loads(
        (artifacts_dir / "feature_names.json").read_text()
    )
    if feature_names_saved != FEATURE_COLS:
        raise ValueError(
            f"Feature mismatch: artifacts have {feature_names_saved}, "
            f"code expects {FEATURE_COLS}. Retrain with models/train.py."
        )

    lgbm_q10   = _load_pickle(artifacts_dir, "lgbm_q10.pkl")
    lgbm_q50   = _load_pickle(artifacts_dir, "lgbm_q50.pkl")
    lgbm_q90   = _load_pickle(artifacts_dir, "lgbm_q90.pkl")
    rf_model   = _load_pickle(artifacts_dir, "rf_model.pkl")
    ridge_model = _load_pickle(artifacts_dir, "ridge_model.pkl")
    scaler     = _load_pickle(artifacts_dir, "feature_scaler.pkl")
    imputation_medians = json.loads(
        (artifacts_dir / "imputation_medians.json").read_text()
    )
    weights = json.loads(
        (artifacts_dir / "ensemble_weights.json").read_text()
    )

    # ── Step 4: Prepare feature matrix ───────────────────────────────────────
    tickers = feature_df["ticker"].to_list()
    X_raw = feature_df.select(FEATURE_COLS).to_numpy().astype(float)
    X_imp = _impute(X_raw, imputation_medians)
    X_sc  = scaler.transform(X_imp)

    # ── Step 5: Predict ───────────────────────────────────────────────────────
    lgbm_q10_preds = lgbm_q10.predict(X_raw)
    lgbm_q50_preds = lgbm_q50.predict(X_raw)
    lgbm_q90_preds = lgbm_q90.predict(X_raw)
    rf_preds       = rf_model.predict(X_imp)
    ridge_preds    = ridge_model.predict(X_sc)

    # ── Step 6: Ensemble ──────────────────────────────────────────────────────
    expected_return = (
        weights["lgbm"]  * lgbm_q50_preds
        + weights["rf"]  * rf_preds
        + weights["ridge"] * ridge_preds
    )

    # ── Step 7: Rank and build output DataFrame ───────────────────────────────
    as_of = dt.date.fromisoformat(date_str)

    result = (
        pl.DataFrame({
            "ticker":                 tickers,
            "expected_annual_return": expected_return.tolist(),
            "confidence_low":         lgbm_q10_preds.tolist(),
            "confidence_high":        lgbm_q90_preds.tolist(),
            "lgbm_return":            lgbm_q50_preds.tolist(),
            "rf_return":              rf_preds.tolist(),
            "ridge_return":           ridge_preds.tolist(),
            "as_of_date":             [as_of] * len(tickers),
        })
        .sort("expected_annual_return", descending=True)
        .with_columns(
            pl.Series("rank", list(range(1, len(tickers) + 1)), dtype=pl.Int32)
        )
        .select([
            "ticker", "rank", "expected_annual_return",
            "confidence_low", "confidence_high",
            "lgbm_return", "rf_return", "ridge_return", "as_of_date",
        ])
    )

    # ── Step 8: Write Parquet ─────────────────────────────────────────────────
    out_path = output_dir / f"date={date_str}" / "predictions.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(out_path), compression="snappy")

    print(f"[Inference] {len(result)} tickers → {out_path}")

    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    date_str = dt.date.today().isoformat()
    print(f"[Inference] Running for {date_str}...")
    df = run_inference(date_str)
    print(df.select(["rank", "ticker", "expected_annual_return"]))
