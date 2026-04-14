from unittest.mock import patch, MagicMock
import polars as pl
import pytest
from ingestion.energy_geo_ingestion import (
    OWID_COLS, AI_INFRA_COUNTRIES, _parse_owid_csv, fetch_energy_geo, save_energy_geo,
)

FIXTURE_CSV = """\
country,year,renewables_share_elec,solar_share_elec,wind_share_elec,nuclear_share_elec,electricity_demand,per_capita_electricity,carbon_intensity_elec
Norway,2022,98.5,0.1,3.2,0.0,125.3,24100.0,20.1
Norway,2023,97.8,0.2,3.5,0.0,127.1,24300.0,19.8
Japan,2022,22.0,9.8,1.0,5.2,900.1,7200.0,450.0
Japan,2023,23.5,10.1,1.1,8.1,892.0,7150.0,420.0
FakeCountry,2022,50.0,10.0,10.0,0.0,50.0,5000.0,200.0
"""


def test_parse_owid_csv_filters_ai_infra_countries():
    """FakeCountry must be excluded; Norway and Japan must be included."""
    df = _parse_owid_csv(FIXTURE_CSV)
    countries = df["country"].to_list()
    assert "FakeCountry" not in countries
    assert "Norway" in countries
    assert "Japan" in countries


def test_parse_owid_csv_schema():
    """country is Utf8, year is Int32, renewables_share_elec is Float64."""
    df = _parse_owid_csv(FIXTURE_CSV)
    assert df.schema["country"] == pl.Utf8
    assert df.schema["year"] == pl.Int32
    assert df.schema["renewables_share_elec"] == pl.Float64


def test_parse_owid_csv_drops_all_null_signal_rows():
    """A row where every signal column is null (empty values) must be dropped."""
    csv_with_null = FIXTURE_CSV + "Iceland,2022,,,,,,\n"
    df = _parse_owid_csv(csv_with_null)
    iceland = df.filter(pl.col("country") == "Iceland")
    assert len(iceland) == 0


def test_fetch_energy_geo_returns_dataframe():
    """fetch_energy_geo() calls requests.get and returns a non-empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.text = FIXTURE_CSV
    mock_resp.raise_for_status = MagicMock()
    with patch("ingestion.energy_geo_ingestion.requests.get", return_value=mock_resp):
        df = fetch_energy_geo()
    assert isinstance(df, pl.DataFrame)
    assert "country" in df.columns
    assert len(df) > 0


def test_save_energy_geo_writes_parquet(tmp_path):
    """save_energy_geo writes a readable Parquet to <output_dir>/energy_geo/country_energy.parquet."""
    df = _parse_owid_csv(FIXTURE_CSV)
    save_energy_geo(df, tmp_path)
    out = tmp_path / "energy_geo" / "country_energy.parquet"
    assert out.exists()
    loaded = pl.read_parquet(str(out))
    assert len(loaded) == len(df)
    assert "country" in loaded.columns


def test_ai_infra_countries_contains_key_markets():
    """Key geography signals must all be present."""
    required = {"Norway", "Japan", "United States", "Canada"}
    assert required.issubset(set(AI_INFRA_COUNTRIES))
