"""Tests that join_sentiment_features respects the 30-day tolerance window."""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from processing.sentiment_features import join_sentiment_features


def _make_sentiment_dir(tmp_path: Path, ticker: str, feature_date: date) -> Path:
    """Write a minimal sentiment features parquet for one ticker at one date."""
    features_dir = tmp_path / "sentiment_features"
    ticker_dir = features_dir / ticker
    ticker_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "ticker": [ticker],
        "date": [feature_date],
        "sentiment_mean_7d": [0.5],
        "sentiment_std_7d": [0.1],
        "article_count_7d": [3],
        "sentiment_momentum_14d": [0.2],
        "ticker_vs_market_7d": [0.1],
    })
    df.write_parquet(ticker_dir / "daily.parquet")
    return features_dir


def _make_spine(ticker: str, spine_date: date) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [ticker],
        "date": [spine_date],
    })


def test_sentiment_join_propagates_within_30d(tmp_path):
    """A spine row 29 days after sentiment data should get the sentiment value."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=29)  # near boundary, within 30-day window
    features_dir = _make_sentiment_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)
    result = join_sentiment_features(spine, features_dir)
    assert result["sentiment_mean_7d"][0] == pytest.approx(0.5)


def test_sentiment_join_null_beyond_30d(tmp_path):
    """A spine row 31 days after sentiment data should get null."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=31)
    features_dir = _make_sentiment_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)
    result = join_sentiment_features(spine, features_dir)
    assert result["sentiment_mean_7d"][0] is None


def test_sentiment_join_null_before_data_exists(tmp_path):
    """A spine row from 3 years before sentiment data should get null."""
    feature_date = date(2025, 1, 1)
    spine_date = date(2022, 1, 1)
    features_dir = _make_sentiment_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)
    result = join_sentiment_features(spine, features_dir)
    assert result["sentiment_mean_7d"][0] is None


def test_sentiment_join_at_exactly_30d_boundary(tmp_path):
    """A spine row exactly 30 days after sentiment data should still get the value (inclusive boundary)."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=30)  # exactly at boundary
    features_dir = _make_sentiment_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)
    result = join_sentiment_features(spine, features_dir)
    assert result["sentiment_mean_7d"][0] == pytest.approx(0.5), \
        "Exactly 30 days is within tolerance (inclusive boundary)"
