"""Tests for portfolio_metrics enrichment functions."""
from __future__ import annotations
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _make_predictions(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal predictions DataFrame for testing."""
    return pl.DataFrame({
        "rank": list(range(1, len(rows) + 1)),
        "ticker": [r["ticker"] for r in rows],
        "layer": [r.get("layer", "compute") for r in rows],
        "expected_annual_return": [r["ensemble"] for r in rows],
        "confidence_low": [0.0] * len(rows),
        "confidence_high": [1.0] * len(rows),
        "lgbm_return": [r["lgbm"] for r in rows],
        "rf_return": [r["rf"] for r in rows],
        "ridge_return": [r["ridge"] for r in rows],
        "as_of_date": [date(2026, 4, 15)] * len(rows),
    })


def test_model_agreement_all_agree():
    from processing.portfolio_metrics import _model_agreement
    df = _make_predictions([
        {"ticker": "NVDA", "ensemble": 0.5, "lgbm": 0.6, "rf": 0.4, "ridge": 0.3},
    ])
    result = _model_agreement(df)
    assert result[0] == pytest.approx(1.0), "All three sub-models positive → agreement=1.0"


def test_model_agreement_one_disagrees():
    from processing.portfolio_metrics import _model_agreement
    df = _make_predictions([
        {"ticker": "NVDA", "ensemble": 0.5, "lgbm": 0.6, "rf": 0.4, "ridge": -0.2},
    ])
    result = _model_agreement(df)
    assert result[0] == pytest.approx(2 / 3), "Two agree, one disagrees → agreement=0.667"


def test_model_agreement_none_agree():
    from processing.portfolio_metrics import _model_agreement
    df = _make_predictions([
        {"ticker": "NVDA", "ensemble": 0.1, "lgbm": -0.6, "rf": -0.4, "ridge": -0.2},
    ])
    result = _model_agreement(df)
    assert result[0] == pytest.approx(0.0), "All sub-models disagree → agreement=0.0"


def test_is_liquid_threshold():
    from processing.portfolio_metrics import _apply_liquidity
    df = _make_predictions([
        {"ticker": "APLD", "ensemble": 3.0, "lgbm": 2.0, "rf": 3.5, "ridge": 3.5},
        {"ticker": "TSM",  "ensemble": 2.0, "lgbm": 1.5, "rf": 2.5, "ridge": 2.0},
    ])
    caps = {"APLD": 0.3, "TSM": 650.0}
    result = _apply_liquidity(df, caps)
    assert result.filter(pl.col("ticker") == "APLD")["is_liquid"][0] == False
    assert result.filter(pl.col("ticker") == "TSM")["is_liquid"][0] == True


def test_is_liquid_none_market_cap():
    from processing.portfolio_metrics import _apply_liquidity
    df = _make_predictions([
        {"ticker": "NEWCO", "ensemble": 1.0, "lgbm": 0.8, "rf": 1.2, "ridge": 1.0},
    ])
    caps = {"NEWCO": None}
    result = _apply_liquidity(df, caps)
    assert result["is_liquid"][0] == False, "None market cap → not liquid"
    assert result["market_cap_b"][0] is None, "None market cap → null market_cap_b"


def test_enrich_writes_enriched_parquet(tmp_path, monkeypatch):
    from processing import portfolio_metrics

    date_str = "2026-04-15"
    pred_dir = tmp_path / f"date={date_str}"
    pred_dir.mkdir()
    df = _make_predictions([
        {"ticker": "NVDA", "ensemble": 0.9, "lgbm": 0.8, "rf": 0.9, "ridge": 1.0},
        {"ticker": "TSM",  "ensemble": 2.1, "lgbm": 1.5, "rf": 2.5, "ridge": 2.3},
    ])
    df.write_parquet(pred_dir / "predictions.parquet")

    monkeypatch.setattr(
        portfolio_metrics, "_get_market_caps",
        lambda tickers, as_of: {"NVDA": 2800.0, "TSM": 650.0},
    )
    monkeypatch.setattr(
        portfolio_metrics, "_peer_correlation",
        lambda df, ohlcv_dir, as_of, top_n=10: pl.Series("peer_correlation_90d", [0.3, 0.4]),
    )

    portfolio_metrics.enrich(date_str, predictions_dir=tmp_path)

    enriched_path = pred_dir / "predictions_enriched.parquet"
    assert enriched_path.exists(), "Enriched parquet must be written"
    enriched = pl.read_parquet(enriched_path)
    assert "market_cap_b" in enriched.columns
    assert "is_liquid" in enriched.columns
    assert "model_agreement" in enriched.columns
    assert "peer_correlation_90d" in enriched.columns
