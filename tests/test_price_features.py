"""Tests for processing/price_features.py — windowed price feature computation."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from processing.price_features import build_price_features


def _write_ohlcv(ohlcv_dir: Path, ticker: str, rows: list[tuple[date, float, int]]) -> None:
    """Write a per-ticker OHLCV parquet under {ohlcv_dir}/{ticker}/2025.parquet.

    rows is a list of (date, close_price, volume) tuples.
    """
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
            "volume": [r[2] for r in rows],
        },
        schema={
            "ticker": pl.Utf8, "date": pl.Date,
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close_price": pl.Float64, "volume": pl.Int64,
        },
    )
    df.write_parquet(ticker_dir / "2025.parquet", compression="snappy")


def _trading_days(start: date, count: int) -> list[date]:
    """Return `count` consecutive calendar dates starting at `start` (no weekend skip — fine for tests)."""
    return [start.replace(day=start.day + i) for i in range(count)] if start.day + count <= 28 else (
        # spread across a longer span for >25 days
        [date(2025, 1, 1 + i) if 1 + i <= 31 else date(2025, 2, 1 + i - 31) for i in range(count)]
    )


def test_returns_columns_present(tmp_path: Path):
    """build_price_features outputs return_1d / 5d / 20d / sma / vol / volume_ratio columns."""
    rows = [(date(2025, 1, d), 100.0 + d, 1_000_000) for d in range(1, 26)]
    _write_ohlcv(tmp_path, "NVDA", rows)
    out = build_price_features(tmp_path)
    expected_cols = {"ticker", "date", "return_1d", "return_5d", "return_20d",
                     "sma_20_deviation", "volatility_20d", "volume_ratio"}
    assert expected_cols <= set(out.columns)


def test_return_1d_correct(tmp_path: Path):
    """return_1d = close_today / close_yesterday - 1."""
    rows = [(date(2025, 1, 1), 100.0, 1_000_000),
            (date(2025, 1, 2), 110.0, 1_000_000),
            (date(2025, 1, 3), 99.0, 1_000_000)]
    _write_ohlcv(tmp_path, "NVDA", rows)
    out = build_price_features(tmp_path).sort("date")
    # First day: no prior, NULL
    assert out["return_1d"][0] is None
    # Day 2: 110/100 - 1 = 0.10
    assert out["return_1d"][1] == pytest.approx(0.10, rel=1e-6)
    # Day 3: 99/110 - 1 = -0.10
    assert out["return_1d"][2] == pytest.approx(-0.10, rel=1e-6)


def test_return_5d_correct(tmp_path: Path):
    """return_5d uses close 5 rows back, NULL for first 5 rows."""
    rows = [(date(2025, 1, d), 100.0 + 10 * d, 1_000_000) for d in range(1, 8)]
    _write_ohlcv(tmp_path, "NVDA", rows)
    out = build_price_features(tmp_path).sort("date")
    # First 5 rows: not enough history for 5d lag
    for i in range(5):
        assert out["return_5d"][i] is None
    # Day 6: close=160, lag5=110 → 160/110-1 ≈ 0.4545
    assert out["return_5d"][5] == pytest.approx(160 / 110 - 1, rel=1e-6)


def test_zero_baseline_lag_returns_null(tmp_path: Path):
    """When the lag close is zero, NULLIF makes the division null instead of crashing."""
    rows = [(date(2025, 1, 1), 0.0, 1_000_000),
            (date(2025, 1, 2), 100.0, 1_000_000)]
    _write_ohlcv(tmp_path, "NVDA", rows)
    out = build_price_features(tmp_path).sort("date")
    # Day 2: lag=0 → NULL (no division by zero crash)
    assert out["return_1d"][1] is None


def test_filter_date_returns_only_one_date(tmp_path: Path):
    """filter_date="YYYY-MM-DD" trims the result to that single date."""
    rows = [(date(2025, 1, d), 100.0 + d, 1_000_000) for d in range(1, 26)]
    _write_ohlcv(tmp_path, "NVDA", rows)
    out = build_price_features(tmp_path, filter_date="2025-01-20")
    assert len(out) == 1
    assert out["date"][0] == date(2025, 1, 20)


def test_multi_ticker_isolation(tmp_path: Path):
    """Per-ticker windowed computations don't bleed across tickers."""
    nvda = [(date(2025, 1, d), 100.0, 1_000_000) for d in range(1, 6)]
    amd = [(date(2025, 1, d), 50.0, 500_000) for d in range(1, 6)]
    _write_ohlcv(tmp_path, "NVDA", nvda)
    _write_ohlcv(tmp_path, "AMD", amd)
    out = build_price_features(tmp_path).sort(["ticker", "date"])
    # All returns should be 0 within each ticker — flat prices
    nvda_rows = out.filter(pl.col("ticker") == "NVDA").sort("date")
    amd_rows = out.filter(pl.col("ticker") == "AMD").sort("date")
    # First rows null, rest 0
    for r in nvda_rows["return_1d"][1:]:
        assert r == pytest.approx(0.0, abs=1e-9)
    for r in amd_rows["return_1d"][1:]:
        assert r == pytest.approx(0.0, abs=1e-9)


def test_volume_ratio_one_for_flat_volume(tmp_path: Path):
    """volume / 20d_avg_volume = 1.0 when volume is constant (after the rolling window fills)."""
    rows = [(date(2025, 1, d), 100.0, 1_000_000) for d in range(1, 26)]
    _write_ohlcv(tmp_path, "NVDA", rows)
    out = build_price_features(tmp_path).sort("date")
    # Day 25: 20d avg = 1M, volume = 1M → ratio = 1.0
    assert out["volume_ratio"][24] == pytest.approx(1.0, rel=1e-9)


def test_date_dtype_is_pl_date(tmp_path: Path):
    """Returned 'date' column is pl.Date (per the cast at the end of build_price_features)."""
    rows = [(date(2025, 1, 1), 100.0, 1_000_000)]
    _write_ohlcv(tmp_path, "NVDA", rows)
    out = build_price_features(tmp_path)
    assert out["date"].dtype == pl.Date
