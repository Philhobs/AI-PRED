"""Tests for tools/score_predictions.py — IC / hit-rate / top-decile scoring."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _write_ohlcv(ohlcv_dir: Path, ticker: str, rows: list[tuple[date, float]]) -> None:
    ticker_dir = ohlcv_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "ticker": [ticker] * len(rows),
            "date": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[1] for r in rows],
            "close_price": [r[1] for r in rows],
            "volume": [1_000_000] * len(rows),
        },
        schema={
            "ticker": pl.Utf8, "date": pl.Date,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close_price": pl.Float64, "volume": pl.Int64,
        },
    )
    df.write_parquet(ticker_dir / "2025.parquet", compression="snappy")


def _write_predictions(predictions_dir: Path, date_str: str, horizon: str,
                       rows: list[tuple[str, float]]) -> Path:
    out = predictions_dir / f"date={date_str}" / f"horizon={horizon}" / "predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "ticker": [r[0] for r in rows],
        "expected_annual_return": [r[1] for r in rows],
    })
    df.write_parquet(out, compression="snappy")
    return out


def test_score_one_returns_none_when_future_not_realized(tmp_path, monkeypatch):
    """If OHLCV doesn't have N+1 trading days after as_of, score returns None."""
    from tools import score_predictions

    ohlcv_dir = tmp_path / "ohlcv"
    pred_dir = tmp_path / "predictions"
    # Only 2 trading days starting at as_of — not enough for 5d horizon
    _write_ohlcv(ohlcv_dir, "NVDA", [(date(2025, 1, 1), 100.0), (date(2025, 1, 2), 105.0)])

    pred_path = _write_predictions(pred_dir, "2025-01-01", "5d", [("NVDA", 0.05)])
    monkeypatch.setattr(score_predictions, "_OHLCV_GLOB", str(ohlcv_dir / "*/*.parquet"))

    result = score_predictions._score_one(pred_path, date(2025, 1, 1), 5)
    assert result is None


def test_score_one_perfect_correlation_yields_ic_one(tmp_path, monkeypatch):
    """When predicted ranking matches realized ranking exactly, IC = 1.0 and hit_rate = 1.0."""
    from tools import score_predictions

    ohlcv_dir = tmp_path / "ohlcv"
    pred_dir = tmp_path / "predictions"

    # 10 tickers, predicted return = realized return (so perfect rank correlation)
    tickers = [f"T{i:02d}" for i in range(10)]
    as_of = date(2025, 1, 1)
    horizon = 5
    # Build OHLCV: each ticker starts at 100 on as_of, ends at 100*(1+0.01*i) on day 5
    for i, t in enumerate(tickers):
        rows = [(date(2025, 1, 1 + d), 100.0) for d in range(horizon)]
        rows.append((date(2025, 1, 1 + horizon), 100.0 * (1 + 0.01 * i)))
        _write_ohlcv(ohlcv_dir, t, rows)

    # Predicted = same ordering as realized
    preds = [(t, 0.01 * i) for i, t in enumerate(tickers)]
    pred_path = _write_predictions(pred_dir, as_of.isoformat(), "5d", preds)

    monkeypatch.setattr(score_predictions, "_OHLCV_GLOB", str(ohlcv_dir / "*/*.parquet"))

    result = score_predictions._score_one(pred_path, as_of, horizon)
    assert result is not None
    assert result["n_tickers"] == 10
    assert result["ic"] == pytest.approx(1.0, abs=1e-6)
    # hit_rate: all positive predicted (except T00 which is 0) match positive realized → 9/10
    # T00 has predicted 0 (sign 0) vs realized 0 (sign 0) — sign() == sign() is True for zeros
    assert result["hit_rate"] >= 0.9
    # Top decile (n=1) is T09 with realized 0.09
    assert result["top_decile_return"] == pytest.approx(0.09, abs=1e-6)


def test_score_one_inverse_correlation_yields_negative_ic(tmp_path, monkeypatch):
    """Inverted predictions → IC near -1."""
    from tools import score_predictions

    ohlcv_dir = tmp_path / "ohlcv"
    pred_dir = tmp_path / "predictions"

    tickers = [f"T{i:02d}" for i in range(10)]
    as_of = date(2025, 1, 1)
    horizon = 5
    for i, t in enumerate(tickers):
        rows = [(date(2025, 1, 1 + d), 100.0) for d in range(horizon)]
        rows.append((date(2025, 1, 1 + horizon), 100.0 * (1 + 0.01 * i)))
        _write_ohlcv(ohlcv_dir, t, rows)

    # Predicted = INVERSE of realized
    preds = [(t, -0.01 * i) for i, t in enumerate(tickers)]
    pred_path = _write_predictions(pred_dir, as_of.isoformat(), "5d", preds)
    monkeypatch.setattr(score_predictions, "_OHLCV_GLOB", str(ohlcv_dir / "*/*.parquet"))

    result = score_predictions._score_one(pred_path, as_of, horizon)
    assert result is not None
    assert result["ic"] == pytest.approx(-1.0, abs=1e-6)


def test_horizon_days_constant_matches_train_config():
    """_HORIZON_DAYS must enumerate the same horizons as train.HORIZON_CONFIGS."""
    from tools.score_predictions import _HORIZON_DAYS
    from models.train import HORIZON_CONFIGS

    # Score script handles all 8 horizons that train.py knows about
    assert set(_HORIZON_DAYS) == set(HORIZON_CONFIGS)
    for tag, days in _HORIZON_DAYS.items():
        assert days == HORIZON_CONFIGS[tag]["shift"]
