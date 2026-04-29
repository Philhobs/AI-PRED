"""Tests for processing/ai_economics_features.py — Sequoia ratio + hyperscaler capex."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _write_quarterly(path: Path, rows: list[tuple[str, date, float, float]]) -> None:
    """Write hyperscalers_quarterly.parquet given (ticker, period_end, revenue, capex) rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ticker":     [r[0] for r in rows],
            "period_end": [r[1] for r in rows],
            "revenue":    [r[2] for r in rows],
            "capex":      [r[3] for r in rows],
        },
        schema={"ticker": pl.Utf8, "period_end": pl.Date,
                "revenue": pl.Float64, "capex": pl.Float64},
    ).write_parquet(path, compression="snappy")


def test_feature_cols_count_is_three():
    from processing.ai_economics_features import AI_ECONOMICS_FEATURE_COLS
    assert AI_ECONOMICS_FEATURE_COLS == [
        "ai_capex_coverage_ratio",
        "hyperscaler_capex_aggregate",
        "hyperscaler_capex_yoy",
    ]


def test_capex_coverage_ratio_perfect_data(tmp_path: Path):
    """Two tickers, 4 quarters each — ratio = total capex / total revenue."""
    from processing.ai_economics_features import join_ai_economics_features

    raw = tmp_path / "hyperscalers_quarterly.parquet"
    rows = []
    # MSFT: 4 quarters, $50B rev + $15B capex each = 200B+60B → 0.30
    for q in [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]:
        rows.append(("MSFT", q, 50e9, 15e9))
    # AMZN: 4 quarters, $100B rev + $25B capex each = 400B+100B → 0.25
    for q in [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]:
        rows.append(("AMZN", q, 100e9, 25e9))
    _write_quarterly(raw, rows)

    spine = pl.DataFrame({
        "ticker": ["NVDA", "AAPL"],
        "date":   [date(2026, 3, 1), date(2026, 3, 1)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_ai_economics_features(spine, raw_path=raw)
    # Aggregate ratio: (60B + 100B) / (200B + 400B) = 160B / 600B = 0.2667
    assert out["ai_capex_coverage_ratio"][0] == pytest.approx(160.0 / 600.0, rel=1e-6)
    # Aggregate capex in billions: 160
    assert out["hyperscaler_capex_aggregate"][0] == pytest.approx(160.0, rel=1e-6)


def test_yoy_computed_from_prior_year_aggregate(tmp_path: Path):
    """yoy = (current_TTM_capex - prior_TTM_capex) / prior_TTM_capex."""
    from processing.ai_economics_features import join_ai_economics_features

    raw = tmp_path / "hyperscalers_quarterly.parquet"
    rows = []
    # MSFT prior year (Q1-Q4 2024): $10B/quarter × 4 = $40B TTM
    for q in [date(2024, 3, 31), date(2024, 6, 30), date(2024, 9, 30), date(2024, 12, 31)]:
        rows.append(("MSFT", q, 40e9, 10e9))
    # MSFT current year (Q1-Q4 2025): $15B/quarter × 4 = $60B TTM
    for q in [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]:
        rows.append(("MSFT", q, 50e9, 15e9))
    _write_quarterly(raw, rows)

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2026, 3, 1)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_ai_economics_features(spine, raw_path=raw)
    # yoy: (60 - 40) / 40 = 0.5
    assert out["hyperscaler_capex_yoy"][0] == pytest.approx(0.5, rel=1e-6)


def test_returns_null_when_too_few_quarters(tmp_path: Path):
    """If a ticker has fewer than 4 quarters in the TTM window, aggregate is null."""
    from processing.ai_economics_features import join_ai_economics_features

    raw = tmp_path / "hyperscalers_quarterly.parquet"
    # Only 2 quarters — not enough for TTM
    _write_quarterly(raw, [
        ("MSFT", date(2025, 9, 30), 50e9, 15e9),
        ("MSFT", date(2025, 12, 31), 50e9, 15e9),
    ])

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2026, 3, 1)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_ai_economics_features(spine, raw_path=raw)
    assert out["ai_capex_coverage_ratio"][0] is None
    assert out["hyperscaler_capex_aggregate"][0] is None


def test_returns_null_when_raw_file_missing(tmp_path: Path):
    """Missing raw parquet → all features null, no exception."""
    from processing.ai_economics_features import join_ai_economics_features

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2026, 3, 1)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    # raw_path points to a file that doesn't exist
    out = join_ai_economics_features(spine, raw_path=tmp_path / "absent.parquet")
    assert out["ai_capex_coverage_ratio"][0] is None
    assert out["hyperscaler_capex_aggregate"][0] is None
    assert out["hyperscaler_capex_yoy"][0] is None


def test_features_apply_to_all_tickers(tmp_path: Path):
    """Macro features attach to every spine ticker on the same date (broadcast join)."""
    from processing.ai_economics_features import join_ai_economics_features

    raw = tmp_path / "hyperscalers_quarterly.parquet"
    rows = []
    for q in [date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30), date(2025, 12, 31)]:
        rows.append(("MSFT", q, 50e9, 15e9))
    _write_quarterly(raw, rows)

    spine = pl.DataFrame({
        "ticker": ["NVDA", "TSLA", "ROK", "1683.HK"],
        "date":   [date(2026, 3, 1)] * 4,
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_ai_economics_features(spine, raw_path=raw)
    assert len(out) == 4
    # All four rows get the same aggregate capex
    assert all(v == out["hyperscaler_capex_aggregate"][0]
               for v in out["hyperscaler_capex_aggregate"])
