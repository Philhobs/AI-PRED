"""
Tests for walk_forward_results.json and per_ticker_accuracy.parquet.

These are integration-style tests — they load actual output files produced by train.py.
They are skipped if the files don't exist (run train.py first to generate them).
"""
from __future__ import annotations
import json
from pathlib import Path

import polars as pl
import pytest

_BACKTEST_DIR = Path("data/backtest")
_WF_JSON = _BACKTEST_DIR / "walk_forward_results.json"
_TICKER_PARQUET = _BACKTEST_DIR / "per_ticker_accuracy.parquet"


@pytest.fixture
def wf_results():
    if not _WF_JSON.exists():
        pytest.skip("walk_forward_results.json not found — run python models/train.py first")
    return json.loads(_WF_JSON.read_text())


@pytest.fixture
def ticker_df():
    if not _TICKER_PARQUET.exists():
        pytest.skip("per_ticker_accuracy.parquet not found — run python models/train.py first")
    return pl.read_parquet(_TICKER_PARQUET)


def test_walk_forward_json_top_level_keys(wf_results):
    assert "as_of" in wf_results
    assert "feature_count" in wf_results
    assert "folds" in wf_results
    assert "summary" in wf_results
    assert isinstance(wf_results["folds"], list)
    assert len(wf_results["folds"]) >= 1


def test_walk_forward_json_fold_keys(wf_results):
    for fold in wf_results["folds"]:
        assert "fold" in fold
        assert "train_end" in fold
        assert "test_start" in fold
        assert "ic" in fold
        assert "hit_rate" in fold
        assert "top_decile_return" in fold
        assert "per_layer" in fold
        assert 0.0 <= fold["hit_rate"] <= 1.0, "hit_rate must be in [0, 1]"


def test_walk_forward_json_per_layer_coverage(wf_results):
    for fold in wf_results["folds"]:
        layer_names = set(fold["per_layer"].keys())
        assert len(layer_names) >= 2, f"Expected ≥2 layers in per_layer, got {layer_names}"


def test_per_ticker_accuracy_schema(ticker_df):
    required_cols = {
        "ticker", "layer", "fold", "test_start", "test_end",
        "predicted_return", "actual_return",
        "predicted_direction", "actual_direction", "correct",
    }
    assert required_cols.issubset(set(ticker_df.columns)), \
        f"Missing columns: {required_cols - set(ticker_df.columns)}"


def test_per_ticker_accuracy_correct_is_bool(ticker_df):
    assert ticker_df["correct"].dtype == pl.Boolean, \
        f"'correct' column should be Boolean, got {ticker_df['correct'].dtype}"


def test_per_ticker_accuracy_directions_valid(ticker_df):
    valid = {-1, 0, 1}
    pred_dirs = set(ticker_df["predicted_direction"].unique().to_list())
    actual_dirs = set(ticker_df["actual_direction"].unique().to_list())
    assert pred_dirs.issubset(valid), f"Invalid predicted_direction values: {pred_dirs - valid}"
    assert actual_dirs.issubset(valid), f"Invalid actual_direction values: {actual_dirs - valid}"
