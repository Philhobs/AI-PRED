"""Tests for energy geo feature computation."""
from __future__ import annotations
from datetime import date
from pathlib import Path

import polars as pl
import pytest


def _make_pjm(backlog_gw: float, as_of: date) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [as_of],
        "zone": ["ALL_VIRGINIA"],
        "queue_backlog_gw": [backlog_gw],
        "project_count": [10],
    }).with_columns(pl.col("project_count").cast(pl.Int32))


def _make_eia(nuc_gw: float, gas_gw: float, as_of: date) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [as_of, as_of],
        "fuel_type": ["nuclear", "natural_gas"],
        "capacity_gw": [nuc_gw, gas_gw],
    })


def _make_owid(tmp_path: Path) -> Path:
    """Write a minimal OWID country energy parquet."""
    df = pl.DataFrame({
        "country": ["United States", "United States", "Norway", "Norway"],
        "year": [2023, 2022, 2023, 2022],
        "renewables_pct": [0.22, 0.21, 0.98, 0.97],
        "carbon_intensity_gco2_per_kwh": [386.0, 390.0, 18.0, 19.0],
    })
    out = tmp_path / "energy_geo" / "country_energy.parquet"
    out.parent.mkdir(parents=True)
    df.write_parquet(out)
    return tmp_path


def test_us_power_moat_score_range(tmp_path):
    """us_power_moat_score is in [0, 1]."""
    from processing.energy_geo_features import compute_us_power_moat_score

    pjm = _make_pjm(backlog_gw=200.0, as_of=date(2025, 1, 1))
    eia = _make_eia(nuc_gw=95.0, gas_gw=618.0, as_of=date(2025, 1, 1))
    score = compute_us_power_moat_score(pjm, eia, as_of=date(2025, 1, 1))
    assert 0.0 <= score <= 1.0, f"Expected [0, 1], got {score}"


def test_power_moat_zero_when_no_data():
    """us_power_moat_score is 0.0 when EIA or PJM data is missing."""
    from processing.energy_geo_features import compute_us_power_moat_score

    empty_pjm = pl.DataFrame(schema={"date": pl.Date, "zone": pl.Utf8,
                                      "queue_backlog_gw": pl.Float64, "project_count": pl.Int32})
    eia = _make_eia(nuc_gw=95.0, gas_gw=618.0, as_of=date(2025, 1, 1))
    score = compute_us_power_moat_score(empty_pjm, eia, as_of=date(2025, 1, 1))
    assert score == 0.0


def test_geo_tailwind_uses_exposure_weights(tmp_path):
    """geo_weighted_tailwind_score is a weighted average of regional tailwinds."""
    from processing.energy_geo_features import compute_geo_tailwind_score

    owid_path = _make_owid(tmp_path)
    exposure = {"north_america": 0.5, "nordics": 0.5}  # equal split

    score_equal = compute_geo_tailwind_score(exposure, owid_path / "energy_geo" / "country_energy.parquet", as_of=date(2025, 1, 1))

    # All nordics should yield higher score than all north_america
    score_nordic  = compute_geo_tailwind_score({"nordics": 1.0}, owid_path / "energy_geo" / "country_energy.parquet", as_of=date(2025, 1, 1))
    score_us_only = compute_geo_tailwind_score({"north_america": 1.0}, owid_path / "energy_geo" / "country_energy.parquet", as_of=date(2025, 1, 1))

    assert score_nordic > score_us_only, "Nordic (98% renewable) should score higher than US"
    assert score_us_only <= score_equal <= score_nordic, "Equal-split score should be between the two"


def test_missing_ticker_defaults_to_north_america(tmp_path):
    """Ticker not in CSV gets north_america = 1.0 exposure."""
    from processing.energy_geo_features import load_geo_exposure

    csv_path = tmp_path / "ticker_geo_exposure.csv"
    csv_path.write_text("ticker,region,weight\nEQIX,nordics,1.0\n")

    exposure = load_geo_exposure(csv_path, ticker="NVDA")
    assert exposure == {"north_america": 1.0}, f"Expected north_america default, got {exposure}"
