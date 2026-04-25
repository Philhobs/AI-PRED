"""Tests for physical AI feature engineering and join."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _write_fred_parquet(out_dir: Path, series_id: str, rows: list[tuple[date, float | None]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {"date": [r[0] for r in rows], "value": [r[1] for r in rows]},
        schema={"date": pl.Date, "value": pl.Float64},
    )
    df.write_parquet(out_dir / f"{series_id}.parquet", compression="snappy")


def _write_jolts_parquet(out_dir: Path, rows: list[tuple[date, str, int, str, float]]) -> None:
    snapshot_dir = out_dir / "date=2025-04-01"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "date":      [r[0] for r in rows],
            "series_id": [r[1] for r in rows],
            "year":      [r[2] for r in rows],
            "period":    [r[3] for r in rows],
            "value":     [r[4] for r in rows],
        },
        schema={"date": pl.Date, "series_id": pl.Utf8, "year": pl.Int32,
                "period": pl.Utf8, "value": pl.Float64},
    )
    df.write_parquet(snapshot_dir / "openings.parquet", compression="snappy")


def _write_patent_parquet(out_dir: Path, bucket: str, rows: list[tuple[date, int]]) -> None:
    bucket_dir = out_dir / f"cpc_class={bucket}"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "quarter_end": [r[0] for r in rows],
            "cpc_class":   [bucket] * len(rows),
            "filing_count": [r[1] for r in rows],
        },
        schema={"quarter_end": pl.Date, "cpc_class": pl.Utf8, "filing_count": pl.Int64},
    )
    df.write_parquet(bucket_dir / "filings.parquet", compression="snappy")


def test_physical_ai_feature_cols_count_is_21():
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    assert len(PHYSICAL_AI_FEATURE_COLS) == 21


def test_physical_ai_feature_cols_exact_names():
    from processing.physical_ai_features import PHYSICAL_AI_FEATURE_COLS
    assert set(PHYSICAL_AI_FEATURE_COLS) == {
        "phys_ai_capgoods_orders_level",
        "phys_ai_capgoods_orders_yoy",
        "phys_ai_pmi_level",
        "phys_ai_machinery_prod_level",
        "phys_ai_machinery_prod_yoy",
        "phys_ai_machinery_ppi_level",
        "phys_ai_machinery_ppi_yoy",
        "phys_ai_machinery_jobs_level",
        "phys_ai_machinery_jobs_yoy",
        "phys_ai_patents_manipulators_count",
        "phys_ai_patents_manipulators_yoy",
        "phys_ai_patents_aerial_count",
        "phys_ai_patents_aerial_yoy",
        "phys_ai_patents_avs_count",
        "phys_ai_patents_avs_yoy",
        "phys_ai_patents_motion_count",
        "phys_ai_patents_motion_yoy",
        "phys_ai_patents_progcontrol_count",
        "phys_ai_patents_progcontrol_yoy",
        "phys_ai_patents_vision_count",
        "phys_ai_patents_vision_yoy",
    }


def test_yoy_handles_zero_baseline():
    """yoy with prior period value = 0 returns null (no division by zero)."""
    from processing.physical_ai_features import _yoy
    assert _yoy(current=10.0, prior=0.0) is None
    assert _yoy(current=10.0, prior=None) is None
    assert _yoy(current=None, prior=10.0) is None
    assert _yoy(current=110.0, prior=100.0) == pytest.approx(0.10, rel=1e-6)


def test_join_macro_features_forward_fills_within_60d(tmp_path: Path):
    """FRED level value within 60 days of query date propagates."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    _write_fred_parquet(fred_dir, "NEWORDER",
        [(date(2025, 1, 1), 100.0), (date(2025, 2, 1), 105.0)])
    _write_fred_parquet(fred_dir, "NAPM",
        [(date(2025, 2, 1), 52.0)])
    _write_fred_parquet(fred_dir, "IPG3331S",
        [(date(2025, 2, 1), 110.0)])
    _write_fred_parquet(fred_dir, "WPU114",
        [(date(2025, 2, 1), 250.0)])

    jolts_dir = tmp_path / "bls_jolts"
    _write_jolts_parquet(jolts_dir, [
        (date(2025, 4, 1), "JTS333000000000000JOL", 2025, "M01", 75.0),
    ])

    patents_dir = tmp_path / "uspto" / "physical_ai"
    for bucket in ["B25J", "B64", "B60W", "G05D1", "G05B19", "G06V"]:
        _write_patent_parquet(patents_dir, bucket,
            [(date(2024, 12, 31), 100), (date(2025, 3, 31), 120)])

    spine = pl.DataFrame({
        "ticker": ["NVDA", "AAPL"],
        "date":   [date(2025, 3, 15), date(2025, 3, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)

    assert "phys_ai_capgoods_orders_level" in out.columns
    nvda = out.filter(pl.col("ticker") == "NVDA").row(0, named=True)
    assert nvda["phys_ai_capgoods_orders_level"] == 105.0
    assert nvda["phys_ai_pmi_level"] == 52.0


def test_join_macro_features_null_beyond_tolerance(tmp_path: Path):
    """FRED level beyond 60 days returns null."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    _write_fred_parquet(fred_dir, "NEWORDER",
        [(date(2024, 12, 1), 100.0)])  # >60d before query
    _write_fred_parquet(fred_dir, "NAPM", [(date(2024, 12, 1), 50.0)])
    _write_fred_parquet(fred_dir, "IPG3331S", [(date(2024, 12, 1), 100.0)])
    _write_fred_parquet(fred_dir, "WPU114", [(date(2024, 12, 1), 250.0)])

    jolts_dir = tmp_path / "bls_jolts"
    patents_dir = tmp_path / "uspto" / "physical_ai"

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2025, 3, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert out["phys_ai_capgoods_orders_level"][0] is None


def test_join_patent_features_quarterly_tolerance(tmp_path: Path):
    """Patent count from 120 days ago propagates; beyond returns null."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    jolts_dir = tmp_path / "bls_jolts"
    patents_dir = tmp_path / "uspto" / "physical_ai"

    # Q4-2024 ends Dec 31, 2024 — 105 days before April 15, 2025 (within 120d)
    for bucket in ["B25J", "B64", "B60W", "G05D1", "G05B19", "G06V"]:
        _write_patent_parquet(patents_dir, bucket, [(date(2024, 12, 31), 50)])

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2025, 4, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert out["phys_ai_patents_manipulators_count"][0] == 50.0


def test_phys_ai_features_apply_to_all_tickers(tmp_path: Path):
    """Every ticker in the spine gets a row with all 21 columns present."""
    from processing.physical_ai_features import (
        join_physical_ai_features, PHYSICAL_AI_FEATURE_COLS,
    )

    fred_dir = tmp_path / "robotics_signals"
    jolts_dir = tmp_path / "bls_jolts"
    patents_dir = tmp_path / "uspto" / "physical_ai"

    spine = pl.DataFrame({
        "ticker": ["NVDA", "TSLA", "ROK", "1683.HK"],
        "date":   [date(2025, 3, 15)] * 4,
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert len(out) == 4
    for col in PHYSICAL_AI_FEATURE_COLS:
        assert col in out.columns, f"{col} missing"


def test_jolts_filter_uses_naics_333_only(tmp_path: Path):
    """Phys-AI labor features must read NAICS 333 rows only, not NAICS 51."""
    from processing.physical_ai_features import join_physical_ai_features

    fred_dir = tmp_path / "robotics_signals"
    patents_dir = tmp_path / "uspto" / "physical_ai"
    jolts_dir = tmp_path / "bls_jolts"
    _write_jolts_parquet(jolts_dir, [
        (date(2025, 4, 1), "JTS510000000000000JOL", 2025, "M02", 999.0),  # NAICS 51 — must be ignored
        (date(2025, 4, 1), "JTS333000000000000JOL", 2025, "M02", 75.0),   # NAICS 333 — must be picked up
        (date(2025, 4, 1), "JTS333000000000000JOL", 2024, "M02", 60.0),   # 1y prior for yoy
    ])

    spine = pl.DataFrame({
        "ticker": ["NVDA"], "date": [date(2025, 3, 15)],
    }, schema={"ticker": pl.Utf8, "date": pl.Date})

    out = join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir)
    assert out["phys_ai_machinery_jobs_level"][0] == 75.0
    # yoy = (75 - 60) / 60 = 0.25
    assert out["phys_ai_machinery_jobs_yoy"][0] == pytest.approx(0.25, rel=1e-6)


def test_value_at_tolerance_boundary_inclusive():
    """tolerance_days exactly equals diff → value passes; +1 day → None."""
    from datetime import date, timedelta
    import polars as pl
    from processing.physical_ai_features import _value_at

    obs_date = date(2025, 1, 14)
    df = pl.DataFrame(
        {"date": [obs_date], "value": [100.0]},
        schema={"date": pl.Date, "value": pl.Float64},
    )

    # Exactly 60 days later — should pass (60 > 60 is False)
    query_60 = obs_date + timedelta(days=60)
    assert _value_at(df, query_60, "value", "date", 60) == 100.0

    # 61 days later — should be None (61 > 60 is True)
    query_61 = obs_date + timedelta(days=61)
    assert _value_at(df, query_61, "value", "date", 60) is None
