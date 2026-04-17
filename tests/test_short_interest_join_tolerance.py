"""Tests that join_short_interest_features respects the 7-day tolerance window."""
from __future__ import annotations
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from processing.short_interest_features import join_short_interest_features


def _make_short_interest_dir(tmp_path: Path, ticker: str, feature_date: date) -> Path:
    """Write a minimal short interest features parquet for one ticker at one date."""
    features_dir = tmp_path / "short_interest_features"
    ticker_dir = features_dir / ticker
    ticker_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "ticker": [ticker],
        "date": [feature_date],
        "short_vol_ratio_10d": [0.25],
        "short_vol_ratio_30d": [0.23],
        "short_ratio_momentum": [0.05],
    })
    df.write_parquet(ticker_dir / "si_daily.parquet")
    return features_dir


def _make_spine(ticker: str, spine_date: date) -> pl.DataFrame:
    return pl.DataFrame({
        "ticker": [ticker],
        "date": [spine_date],
    })


def test_short_interest_join_propagates_within_7d(tmp_path):
    """A spine row 6 days after short interest data should get the value."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=6)  # within 7-day window
    features_dir = _make_short_interest_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)
    result = join_short_interest_features(spine, features_dir)
    assert result["short_vol_ratio_10d"][0] == pytest.approx(0.25)


def test_short_interest_join_null_beyond_7d(tmp_path):
    """A spine row 8 days after short interest data should get null."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=8)  # beyond 7-day window
    features_dir = _make_short_interest_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)
    result = join_short_interest_features(spine, features_dir)
    assert result["short_vol_ratio_10d"][0] is None


def test_short_interest_join_null_no_data(tmp_path):
    """A spine row with no short interest data should get null."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=3)
    # Create empty features directory (no ticker data)
    features_dir = tmp_path / "short_interest_features"
    features_dir.mkdir(parents=True)
    spine = _make_spine("NVDA", spine_date)
    result = join_short_interest_features(spine, features_dir)
    assert result["short_vol_ratio_10d"][0] is None


def test_short_interest_join_at_exactly_7d_boundary(tmp_path):
    """A spine row exactly 7 days after short interest data should still get the value (inclusive boundary)."""
    feature_date = date(2025, 1, 1)
    spine_date = feature_date + timedelta(days=7)  # exactly at boundary
    features_dir = _make_short_interest_dir(tmp_path, "NVDA", feature_date)
    spine = _make_spine("NVDA", spine_date)
    result = join_short_interest_features(spine, features_dir)
    assert result["short_vol_ratio_10d"][0] == pytest.approx(0.25), \
        "Exactly 7 days is within tolerance (inclusive boundary)"
