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
    OWNERSHIP_FEATURE_COLS, ENERGY_FEATURE_COLS, SUPPLY_CHAIN_FEATURE_COLS,
    FX_FEATURE_COLS, _impute,
)
from processing.earnings_features import join_earnings_features
from processing.energy_geo_features import join_energy_geo_features
from processing.fundamental_features import join_fundamentals
from processing.fx_features import join_fx_features
from processing.graph_features import join_graph_features
from processing.insider_features import join_insider_features
from processing.ownership_features import join_ownership_features
from processing.price_features import build_price_features
from processing.sentiment_features import join_sentiment_features
from processing.short_interest_features import join_short_interest_features
from processing.supply_chain_features import join_supply_chain_features
from processing.cyber_threat_features import join_cyber_threat_features
from processing.options_features import join_options_features
from processing.gov_behavioral_features import join_gov_behavioral_features
from processing.patent_features import join_patent_features


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_feature_df(
    date_str: str,
    data_dir: Path,
) -> pl.DataFrame:
    """Build the 73-feature DataFrame for all tickers on date_str."""
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

    df = join_supply_chain_features(df, ohlcv_dir=ohlcv_dir, fx_dir=ohlcv_dir.parent / "fx")
    df = join_fx_features(df, ohlcv_dir=ohlcv_dir)

    cyber_threat_dir = data_dir / "cyber_threat"
    df = join_cyber_threat_features(df, cyber_threat_dir)

    options_dir = data_dir / "options"
    df = join_options_features(df, options_dir, ohlcv_dir)

    gov_contracts_dir = data_dir / "gov_contracts"
    gov_ferc_dir = data_dir / "ferc_queue"
    df = join_gov_behavioral_features(df, gov_contracts_dir, gov_ferc_dir)

    patents_apps_dir = data_dir / "patents" / "applications"
    patents_grants_dir = data_dir / "patents" / "grants"
    df = join_patent_features(df, patents_apps_dir, patents_grants_dir)

    return df


def _predict_layer(
    feature_df: pl.DataFrame,
    layer: str,
    artifacts_dir: Path,
    horizon_tag: str = "252d",
) -> pl.DataFrame | None:
    """Run one layer model on the tickers belonging to that layer for a given horizon.

    Looks for artifacts at layer_dir/horizon_{horizon_tag}/. Falls back to layer_dir/
    directly for legacy flat-structure 252d artifacts (backward compat).

    Returns DataFrame with [ticker, layer, horizon, expected_annual_return,
    confidence_low, confidence_high, lgbm_return, rf_return, ridge_return]
    or None if artifacts missing.
    """
    layer_id = LAYER_IDS[layer]
    layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"
    horizon_dir = layer_dir / f"horizon_{horizon_tag}"

    # Backward compat: fall back to flat structure for legacy 252d artifacts
    if not horizon_dir.exists() and horizon_tag == "252d":
        horizon_dir = layer_dir

    if not (horizon_dir / "feature_names.json").exists():
        _LOG.debug("No feature_names.json for layer %s horizon %s — skipping", layer, horizon_tag)
        return None

    feature_names_saved: list[str] = json.loads(
        (horizon_dir / "feature_names.json").read_text()
    )

    layer_tickers = tickers_in_layer(layer)
    layer_df = feature_df.filter(pl.col("ticker").is_in(layer_tickers))
    if layer_df.is_empty():
        _LOG.debug("No tickers in layer %s for horizon %s — skipping", layer, horizon_tag)
        return None

    tickers = layer_df["ticker"].to_list()
    medians = json.loads((horizon_dir / "imputation_medians.json").read_text())
    weights = json.loads((horizon_dir / "ensemble_weights.json").read_text())

    X_raw = layer_df.select(feature_names_saved).to_numpy().astype(float)
    X_imp = _impute(X_raw, medians, feature_names_saved)
    scaler = _load_pickle(horizon_dir / "feature_scaler.pkl")
    X_sc = scaler.transform(X_imp)
    X_df = pd.DataFrame(X_imp, columns=feature_names_saved)

    lgbm_q10 = _load_pickle(horizon_dir / "lgbm_q10.pkl")
    lgbm_q50 = _load_pickle(horizon_dir / "lgbm_q50.pkl")
    lgbm_q90 = _load_pickle(horizon_dir / "lgbm_q90.pkl")
    rf_model  = _load_pickle(horizon_dir / "rf_model.pkl")
    ridge_model = _load_pickle(horizon_dir / "ridge_model.pkl")

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
        "horizon": [horizon_tag] * len(tickers),
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
    horizon_tag: str | None = None,
) -> pl.DataFrame:
    """Run all trained layer models and return globally ranked predictions.

    horizon_tag: if given, runs only that horizon. If None, runs all horizons
                 that have at least one trained layer artifact.

    Returns the primary horizon's (252d if available, else first found) combined
    DataFrame for backward compatibility.

    Raises ValueError if date_str is a weekend.
    Raises RuntimeError if no price data exists for date_str.
    """
    as_of = dt.date.fromisoformat(date_str)
    if as_of.weekday() >= 5:
        raise ValueError(f"{date_str} is a weekend. Skip inference on non-trading days.")

    from models.train import HORIZON_CONFIGS
    horizons_to_run = [horizon_tag] if horizon_tag else list(HORIZON_CONFIGS.keys())

    print(f"[Inference] Running for {date_str}, horizon(s): {horizon_tag or 'all'}...")

    feature_df = _build_feature_df(date_str, data_dir)

    primary_combined: pl.DataFrame | None = None

    for h_tag in horizons_to_run:
        all_preds: list[pl.DataFrame] = []
        for layer in all_layers():
            layer_preds = _predict_layer(feature_df, layer, artifacts_dir, h_tag)
            if layer_preds is not None:
                all_preds.append(layer_preds)

        if not all_preds:
            _LOG.debug("No artifacts for horizon %s — skipping", h_tag)
            continue

        combined = pl.concat(all_preds).sort("expected_annual_return", descending=True)
        combined = combined.with_columns(
            pl.Series("rank", list(range(1, len(combined) + 1)), dtype=pl.Int32),
            pl.lit(as_of).cast(pl.Date).alias("as_of_date"),
        )

        # Write horizon-partitioned output
        out_path = output_dir / f"date={date_str}" / f"horizon={h_tag}" / "predictions.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(out_path, compression="snappy")

        # Backward-compat alias for 252d
        if h_tag == "252d":
            compat_path = output_dir / f"date={date_str}" / "predictions.parquet"
            combined.write_parquet(compat_path, compression="snappy")

        n_tickers = len(combined)
        n_layers = combined["layer"].n_unique()
        print(f"[Inference] [{h_tag}] {n_tickers} tickers × {n_layers} layers → {out_path}")

        # Track primary result (252d preferred, else first)
        if primary_combined is None or h_tag == "252d":
            primary_combined = combined

    if primary_combined is None:
        raise RuntimeError(
            f"No layer artifacts found in {artifacts_dir} for any requested horizon. "
            "Run models/train.py first."
        )

    # Enrich primary predictions with portfolio metrics
    try:
        from processing.portfolio_metrics import enrich
        enrich(date_str, predictions_dir=output_dir)
    except Exception as exc:
        _LOG.warning("Portfolio metrics enrichment failed (non-fatal): %s", exc, exc_info=True)

    return primary_combined


if __name__ == "__main__":
    import argparse
    today = dt.date.today()
    if today.weekday() >= 5:
        print(f"[Inference] {today} is a weekend — skipping.")
        import sys; sys.exit(0)

    parser = argparse.ArgumentParser(description="Run multi-horizon inference.")
    parser.add_argument(
        "--horizon", default=None,
        help="Single horizon tag, e.g. '5d' or '252d'. Default: all trained horizons.",
    )
    args = parser.parse_args()
    run_inference(date_str=today.isoformat(), horizon_tag=args.horizon)
