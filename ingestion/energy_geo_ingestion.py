"""
OWID Energy Geography Ingestion
Downloads Our World in Data energy CSV and filters to AI infrastructure country signals.
"""
import io
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import requests

OWID_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

OWID_COLS = [
    "country", "year",
    "renewables_share_elec", "solar_share_elec", "wind_share_elec",
    "nuclear_share_elec", "electricity_demand", "per_capita_electricity",
    "carbon_intensity_elec",
]

AI_INFRA_COUNTRIES = [
    "Norway", "Sweden", "Iceland",
    "Canada",
    "Japan",
    "United Arab Emirates", "Saudi Arabia",
    "Malaysia", "Indonesia", "Singapore",
    "United States",
    "Morocco",
    "Germany", "France", "United Kingdom",
]

_SIGNAL_COLS = [
    "renewables_share_elec", "solar_share_elec", "wind_share_elec",
    "nuclear_share_elec", "electricity_demand", "per_capita_electricity",
    "carbon_intensity_elec",
]

_SCHEMA = pa.schema([
    pa.field("country", pa.string()),
    pa.field("year", pa.int32()),
    pa.field("renewables_share_elec", pa.float64()),
    pa.field("solar_share_elec", pa.float64()),
    pa.field("wind_share_elec", pa.float64()),
    pa.field("nuclear_share_elec", pa.float64()),
    pa.field("electricity_demand", pa.float64()),
    pa.field("per_capita_electricity", pa.float64()),
    pa.field("carbon_intensity_elec", pa.float64()),
])


def _parse_owid_csv(csv_text: str) -> pl.DataFrame:
    """
    Parse OWID CSV text:
    1. Read only OWID_COLS columns, treat empty strings and "NA" as null
    2. Filter rows where country is in AI_INFRA_COUNTRIES
    3. Cast year to Int32, all signal cols to Float64
    4. Drop rows where ALL signal columns are null
    5. Sort by ["country", "year"]
    Returns polars DataFrame.
    """
    df = pl.read_csv(
        io.StringIO(csv_text),
        columns=OWID_COLS,
        null_values=["", "NA"],
        infer_schema_length=5000,
    )

    # Filter to AI infrastructure countries
    df = df.filter(pl.col("country").is_in(AI_INFRA_COUNTRIES))

    # Cast year to Int32 and all signal cols to Float64
    cast_exprs = [pl.col("year").cast(pl.Int32)]
    for col in _SIGNAL_COLS:
        cast_exprs.append(pl.col(col).cast(pl.Float64))
    df = df.with_columns(cast_exprs)

    # Drop rows where ALL signal columns are null (keep rows with at least one non-null)
    df = df.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in _SIGNAL_COLS])
    )

    # Sort by country and year
    df = df.sort(["country", "year"])

    return df


def fetch_energy_geo() -> pl.DataFrame:
    """
    Download OWID CSV via requests.get(OWID_URL, timeout=60).
    Call resp.raise_for_status() before parsing.
    Returns _parse_owid_csv(resp.text).
    """
    resp = requests.get(OWID_URL, timeout=60)
    resp.raise_for_status()
    return _parse_owid_csv(resp.text)


def save_energy_geo(df: pl.DataFrame, output_dir: Path) -> None:
    """
    Write to <output_dir>/energy_geo/country_energy.parquet (snappy compression).
    Creates parent directories. Does nothing if df.is_empty() (prints warning).
    Uses PyArrow table: pa.Table.from_pylist(df.to_dicts(), schema=_SCHEMA).
    Prints: "[EnergyGeo] {len(df)} rows ({df['country'].n_unique()} countries) → {path}"
    """
    if df.is_empty():
        print("[EnergyGeo] Warning: DataFrame is empty, skipping write.")
        return

    path = Path(output_dir) / "energy_geo" / "country_energy.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pylist(df.to_dicts(), schema=_SCHEMA)
    pq.write_table(table, str(path), compression="snappy")

    print(f"[EnergyGeo] {len(df)} rows ({df['country'].n_unique()} countries) → {path}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    output_dir = Path("data/raw")
    print("[EnergyGeo] Downloading OWID energy data...")
    df = fetch_energy_geo()
    save_energy_geo(df, output_dir)
    print(f"[EnergyGeo] Done. Years: {df['year'].min()}–{df['year'].max()}")
