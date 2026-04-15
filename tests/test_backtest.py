"""Unit tests for walk-forward backtest. Uses synthetic data — no disk, no network."""
import pytest
import numpy as np
import polars as pl
from datetime import date as _date
from pathlib import Path


def _make_labeled_df(n_tickers: int = 5, n_months: int = 18) -> pl.DataFrame:
    """Synthetic labeled dataset with a planted signal: feature_0 predicts label_return_1y."""
    from models.train import FEATURE_COLS

    rng = np.random.default_rng(42)
    rows = []
    base_dates = []

    # Monthly dates spanning n_months
    from datetime import timedelta
    start = _date(2021, 1, 4)
    current = start
    for _ in range(n_months):
        base_dates.append(current)
        # advance ~30 days
        current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

    tickers = [f"T{i}" for i in range(n_tickers)]

    for d in base_dates:
        for t in tickers:
            features = rng.normal(0, 1, len(FEATURE_COLS)).tolist()
            # Plant signal: label = 0.5 * feature_0 + noise
            label = 0.5 * features[0] + rng.normal(0, 0.3)
            row = {"ticker": t, "date": d, "label_return_1y": float(label)}
            for col, val in zip(FEATURE_COLS, features):
                row[col] = float(val)
            rows.append(row)

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_backtest_runs_without_error(tmp_path, monkeypatch):
    """run_backtest returns a non-empty dict with required keys."""
    from models.backtest import run_backtest, TRAIN_CUTOFF, TEST_START, TEST_END
    import models.backtest as bt_module

    df = _make_labeled_df(n_tickers=6, n_months=24)

    # Monkeypatch build_training_dataset to return our synthetic data
    monkeypatch.setattr(
        "models.backtest.TRAIN_CUTOFF", "2022-06-30"
    )
    monkeypatch.setattr(
        "models.backtest.TEST_START", "2022-07-01"
    )
    monkeypatch.setattr(
        "models.backtest.TEST_END", "2022-12-31"
    )

    import models.train as train_module
    monkeypatch.setattr(train_module, "build_training_dataset", lambda *a, **kw: df)

    results = run_backtest(
        ohlcv_dir=tmp_path / "ohlcv",
        fundamentals_dir=tmp_path / "fundamentals",
        output_dir=tmp_path / "backtest",
    )

    assert isinstance(results, dict)
    assert "mean_ic" in results
    assert "ic_tstat" in results
    assert "hit_rate" in results
    assert "mean_spread" in results
    assert -1.0 <= results["mean_ic"] <= 1.0
    assert 0.0 <= results["hit_rate"] <= 1.0


def test_backtest_writes_json(tmp_path, monkeypatch):
    """Results JSON is written to output_dir."""
    import models.backtest as bt_module
    import models.train as train_module

    df = _make_labeled_df(n_tickers=6, n_months=24)

    monkeypatch.setattr(bt_module, "TRAIN_CUTOFF", "2022-06-30")
    monkeypatch.setattr(bt_module, "TEST_START", "2022-07-01")
    monkeypatch.setattr(bt_module, "TEST_END", "2022-12-31")
    monkeypatch.setattr(train_module, "build_training_dataset", lambda *a, **kw: df)

    from models.backtest import run_backtest
    run_backtest(
        ohlcv_dir=tmp_path / "ohlcv",
        fundamentals_dir=tmp_path / "fundamentals",
        output_dir=tmp_path / "backtest",
    )

    result_file = tmp_path / "backtest" / "walk_forward_results.json"
    assert result_file.exists()

    import json
    data = json.loads(result_file.read_text())
    assert data["feature_count"] > 0


def test_backtest_signal_detected(tmp_path, monkeypatch):
    """Planted feature_0 → label signal should produce positive mean IC."""
    import models.backtest as bt_module
    import models.train as train_module

    # Use more tickers and longer period for stronger detection
    df = _make_labeled_df(n_tickers=10, n_months=30)

    monkeypatch.setattr(bt_module, "TRAIN_CUTOFF", "2022-06-30")
    monkeypatch.setattr(bt_module, "TEST_START", "2022-07-01")
    monkeypatch.setattr(bt_module, "TEST_END", "2023-12-31")
    monkeypatch.setattr(train_module, "build_training_dataset", lambda *a, **kw: df)

    from models.backtest import run_backtest
    results = run_backtest(
        ohlcv_dir=tmp_path / "ohlcv",
        fundamentals_dir=tmp_path / "fundamentals",
        output_dir=tmp_path / "backtest",
    )

    # With a planted signal (r=0.5, noise=0.3), mean IC should be positive
    assert results["mean_ic"] > 0, f"Expected positive IC with planted signal, got {results['mean_ic']}"
