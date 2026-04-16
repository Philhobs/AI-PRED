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
               "label_return_1y": float(label)}
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


def test_train_all_layers_creates_10_dirs(tmp_path, monkeypatch):
    import models.train as train_module
    from ingestion.ticker_registry import layers
    # Monkeypatch build_training_dataset to return synthetic data per layer
    def fake_build(ohlcv_dir, fundamentals_dir, layer=None):
        if layer is None:
            return pl.DataFrame()
        return _make_layer_df(layer, n_rows=80)

    monkeypatch.setattr(train_module, "build_training_dataset", fake_build)
    train_module.train_all_layers(
        ohlcv_dir=tmp_path / "ohlcv",
        fundamentals_dir=tmp_path / "fundamentals",
        artifacts_dir=tmp_path / "artifacts",
    )
    # 10 layer directories should exist
    layer_dirs = list((tmp_path / "artifacts").glob("layer_*"))
    assert len(layer_dirs) == 10
