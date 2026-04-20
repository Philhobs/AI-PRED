"""Tests for FX-adjusted return features."""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


def _write_ohlcv(ticker_dir: Path, dates: list[date], prices: list[float]) -> None:
    """Write a minimal OHLCV parquet for a single ticker."""
    ticker_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "ticker":      [ticker_dir.name] * len(dates),
        "date":        dates,
        "close_price": prices,
    })
    year = dates[0].year
    df.write_parquet(ticker_dir / f"{year}.parquet")


def _write_fx(fx_dir: Path, pair: str, dates: list[date], rates: list[float]) -> None:
    """Write a minimal FX parquet."""
    fx_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": dates, "rate": rates}).write_parquet(fx_dir / f"{pair}.parquet")


def test_usd_ticker_passthrough(tmp_path):
    """USD ticker close prices are unchanged by build_usd_close_matrix."""
    from processing.fx_features import build_usd_close_matrix

    close = pl.DataFrame({
        "date":  [date(2025, 1, 1), date(2025, 1, 2)],
        "NVDA":  [100.0, 102.0],
    })
    fx_dir = tmp_path / "fx"
    fx_dir.mkdir()

    result = build_usd_close_matrix(close, fx_dir)
    assert result["NVDA"].to_list() == [100.0, 102.0]


def test_eur_ticker_converted_to_usd(tmp_path):
    """EUR ticker is multiplied by EURUSD rate on each date."""
    from processing.fx_features import build_usd_close_matrix

    fx_dir = tmp_path / "fx"
    _write_fx(fx_dir, "EURUSD", [date(2025, 1, 1), date(2025, 1, 2)], [1.10, 1.12])

    close = pl.DataFrame({
        "date":    [date(2025, 1, 1), date(2025, 1, 2)],
        "SAP.DE":  [100.0, 100.0],
    })

    result = build_usd_close_matrix(close, fx_dir)
    assert abs(result["SAP.DE"][0] - 110.0) < 0.001
    assert abs(result["SAP.DE"][1] - 112.0) < 0.001


def test_missing_fx_rate_produces_null(tmp_path):
    """Date with no FX rate → null for that ticker, no crash."""
    from processing.fx_features import build_usd_close_matrix

    fx_dir = tmp_path / "fx"
    # FX only covers Jan 1, not Jan 2
    _write_fx(fx_dir, "EURUSD", [date(2025, 1, 1)], [1.10])

    close = pl.DataFrame({
        "date":    [date(2025, 1, 1), date(2025, 1, 2)],
        "SAP.DE":  [100.0, 101.0],
    })

    result = build_usd_close_matrix(close, fx_dir)
    assert abs(result["SAP.DE"][0] - 110.0) < 0.001  # Jan 1: has rate
    assert result["SAP.DE"][1] is None               # Jan 2: no rate → null


def test_fx_adjusted_return_20d_correct(tmp_path):
    """EUR ticker with constant FX rate: 20d USD return == 20d local return."""
    from processing.fx_features import join_fx_features

    as_of = date(2025, 6, 1)
    n_days = 25
    dates = [as_of - timedelta(days=n_days - i) for i in range(n_days + 1)]

    # SAP.DE: EUR price grows 0.5%/day. EURUSD = 1.1 (constant).
    # 20d return in EUR == 20d return in USD (constant FX cancels out)
    eur_prices = [100.0 * (1.005 ** i) for i in range(n_days + 1)]
    ohlcv_dir = tmp_path / "ohlcv"
    _write_ohlcv(ohlcv_dir / "SAP.DE", dates, eur_prices)

    fx_dir = tmp_path / "fx"
    _write_fx(fx_dir, "EURUSD", dates, [1.1] * (n_days + 1))

    spine = pl.DataFrame({"ticker": ["SAP.DE"], "date": [as_of]})
    result = join_fx_features(spine, fx_dir=fx_dir, ohlcv_dir=ohlcv_dir)

    expected = (1.005 ** 20) - 1.0  # ≈ 0.1049
    actual = result["fx_adjusted_return_20d"][0]
    assert actual is not None
    assert abs(actual - expected) < 0.01, f"Expected ~{expected:.4f}, got {actual:.4f}"


def test_usd_ticker_gets_null_feature(tmp_path):
    """fx_adjusted_return_20d is always null for USD tickers."""
    from processing.fx_features import join_fx_features

    ohlcv_dir = tmp_path / "ohlcv"
    fx_dir = tmp_path / "fx"
    fx_dir.mkdir()

    spine = pl.DataFrame({
        "ticker": ["NVDA"],
        "date":   [date(2025, 6, 1)],
    })
    result = join_fx_features(spine, fx_dir=fx_dir, ohlcv_dir=ohlcv_dir)
    assert result["fx_adjusted_return_20d"][0] is None


def test_join_fx_features_adds_column(tmp_path):
    """join_fx_features adds exactly 1 Float64 column named fx_adjusted_return_20d."""
    from processing.fx_features import join_fx_features

    ohlcv_dir = tmp_path / "ohlcv"; ohlcv_dir.mkdir()
    fx_dir    = tmp_path / "fx";    fx_dir.mkdir()

    spine = pl.DataFrame({
        "ticker": ["NVDA", "SAP.DE"],
        "date":   [date(2025, 6, 1), date(2025, 6, 1)],
    })
    result = join_fx_features(spine, fx_dir=fx_dir, ohlcv_dir=ohlcv_dir)

    assert "fx_adjusted_return_20d" in result.columns
    assert result["fx_adjusted_return_20d"].dtype == pl.Float64
    assert len(result) == 2


def test_registry_coverage():
    """Every non-USD currency in TICKERS_INFO has a corresponding supported FX pair."""
    from ingestion.ticker_registry import TICKERS_INFO
    from ingestion.fx_ingestion import CURRENCY_TO_PAIR, SUPPORTED_CURRENCIES

    non_usd_currencies = {t.currency for t in TICKERS_INFO if t.currency != "USD"}
    uncovered = non_usd_currencies - SUPPORTED_CURRENCIES
    assert uncovered == set(), f"Currencies with no FX pair: {uncovered}"
