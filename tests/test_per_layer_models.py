# tests/test_per_layer_models.py
import pytest
import polars as pl
import numpy as np
from datetime import date as _date
from pathlib import Path


def _make_layer_df(layer: str, n_rows: int = 100) -> pl.DataFrame:
    """Synthetic labeled dataset for a single layer."""
    from models.train import FEATURE_COLS
    from ingestion.ticker_registry import tickers_in_layer
    rng = np.random.default_rng(42)
    tickers = tickers_in_layer(layer)
    rows = []
    for i in range(n_rows):
        ticker = tickers[i % len(tickers)]
        features = rng.normal(0, 1, len(FEATURE_COLS)).tolist()
        label = 0.3 * features[0] + rng.normal(0, 0.2)
        row = {"ticker": ticker, "date": _date(2020 + i // 365, 1, i % 12 + 1),
               "label_return": float(label)}
        for col, val in zip(FEATURE_COLS, features):
            row[col] = float(val)
        rows.append(row)
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_train_single_layer_creates_artifacts(tmp_path):
    from models.train import train_single_layer, FEATURE_COLS
    df = _make_layer_df("compute", n_rows=120)
    artifacts_dir = tmp_path / "layer_02_compute"
    train_single_layer(df, artifacts_dir)
    assert (artifacts_dir / "lgbm_q50.pkl").exists()
    assert (artifacts_dir / "rf_model.pkl").exists()
    assert (artifacts_dir / "feature_names.json").exists()
    import json
    names = json.loads((artifacts_dir / "feature_names.json").read_text())
    assert names == FEATURE_COLS


def test_train_all_layers_creates_15_dirs(tmp_path, monkeypatch):
    import models.train as train_module
    from ingestion.ticker_registry import layers
    # Monkeypatch build_training_dataset to return synthetic data per layer.
    # Must accept horizon_tag since train_all_layers now passes it.
    # Return 120 rows (≥100 coverage gate) with a 'label_return' column.
    def fake_build(ohlcv_dir, fundamentals_dir, layer=None, horizon_tag=None):
        if layer is None:
            return pl.DataFrame()
        from models.train import FEATURE_COLS
        rng = np.random.default_rng(42)
        from ingestion.ticker_registry import tickers_in_layer
        tickers = tickers_in_layer(layer)
        rows = []
        for i in range(120):
            ticker = tickers[i % len(tickers)]
            features = rng.normal(0, 1, len(FEATURE_COLS)).tolist()
            label = float(0.3 * features[0] + rng.normal(0, 0.2))
            row = {"ticker": ticker,
                   "date": _date(2020 + i // 365, 1, i % 12 + 1),
                   "label_return": label}
            for col, val in zip(FEATURE_COLS, features):
                row[col] = float(val)
            rows.append(row)
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

    monkeypatch.setattr(train_module, "build_training_dataset", fake_build)
    train_module.train_all_layers(
        ohlcv_dir=tmp_path / "ohlcv",
        fundamentals_dir=tmp_path / "fundamentals",
        artifacts_dir=tmp_path / "artifacts",
        horizon_tag="5d",
    )
    # 15 layer directories should exist (layers 1–15 including 3 robotics sub-layers
    # + 2 cyber layers), each containing a horizon_5d/ subdirectory.
    layer_dirs = list((tmp_path / "artifacts").glob("layer_*"))
    assert len(layer_dirs) == 15
    for layer_dir in layer_dirs:
        assert (layer_dir / "horizon_5d").exists(), f"horizon_5d/ missing under {layer_dir}"


def test_inference_merges_all_layers(tmp_path, monkeypatch):
    """run_inference returns one row per ticker across all 15 layers."""
    import models.train as train_module
    import models.inference as infer_module
    from models.train import train_single_layer, FEATURE_COLS
    from ingestion.ticker_registry import TICKERS, tickers_in_layer, layers as all_layers, LAYER_IDS

    # Train minimal artifacts for each layer using synthetic data
    artifacts_dir = tmp_path / "artifacts"
    for layer in all_layers():
        layer_id = LAYER_IDS[layer]
        layer_dir = artifacts_dir / f"layer_{layer_id:02d}_{layer}"
        df = _make_layer_df(layer, n_rows=80)
        train_single_layer(df, layer_dir)

    # Build a minimal price features DataFrame for all tickers
    from datetime import date as _date2
    import polars as pl
    import numpy as np
    rng = np.random.default_rng(0)
    feature_rows = []
    for t in TICKERS:
        row = {"ticker": t, "date": _date2(2024, 1, 15)}
        for col in FEATURE_COLS:
            row[col] = float(rng.normal(0, 1))
        feature_rows.append(row)
    feature_df = pl.DataFrame(feature_rows).with_columns(pl.col("date").cast(pl.Date))

    # Monkeypatch _build_feature_df to return our synthetic features
    monkeypatch.setattr(infer_module, "_build_feature_df", lambda *a, **kw: feature_df)

    result = infer_module.run_inference(
        date_str="2024-01-15",
        data_dir=tmp_path / "raw",
        artifacts_dir=artifacts_dir,
        output_dir=tmp_path / "predictions",
    )
    assert len(result) == len(TICKERS)
    assert result["rank"].min() == 1
    assert result["rank"].max() == len(TICKERS)
    assert "layer" in result.columns
