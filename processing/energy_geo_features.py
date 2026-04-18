"""
Energy geography features — grid moat signal and international tailwind score.

Produces 2 features per ticker per date:
  us_power_moat_score      — PJM Virginia queue / baseload capacity (normalized 0-1)
  geo_weighted_tailwind_score — weighted average of regional energy tailwinds

Called by models/train.py via join_energy_geo_features().
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

_LOG = logging.getLogger(__name__)

_POWER_TICKERS = frozenset({
    "CEG", "VST", "NRG", "TLN", "NEE", "SO", "EXC", "ETR",
    "GEV", "BWX", "OKLO", "SMR", "FSLR",
})

_REGION_COUNTRIES: dict[str, list[str]] = {
    "north_america": ["United States", "Canada"],
    "emea":          ["Germany", "United Kingdom", "France", "Netherlands"],
    "nordics":       ["Norway", "Sweden", "Iceland"],
    "asia_pacific":  ["Japan", "South Korea", "Singapore", "Malaysia"],
}


def load_geo_exposure(csv_path: Path, ticker: str) -> dict[str, float]:
    """
    Load {region: weight} for a ticker from the manual exposure CSV.
    Defaults to {"north_america": 1.0} if ticker not present.
    """
    if not csv_path.exists():
        return {"north_america": 1.0}
    df = pl.read_csv(csv_path).filter(pl.col("ticker") == ticker)
    if df.is_empty():
        return {"north_america": 1.0}
    return dict(zip(df["region"].to_list(), df["weight"].to_list()))


def compute_us_power_moat_score(
    pjm: pl.DataFrame,
    eia: pl.DataFrame,
    as_of: date,
    lookback_days: int = 365 * 3,
) -> float:
    """
    Compute US power moat score for as_of date.

    Formula: PJM Virginia queue backlog / (nuclear + gas capacity), normalized to [0, 1]
    using a rolling 3-year window. Returns 0.0 when data is unavailable.
    """
    recent_pjm = (
        pjm.filter((pl.col("zone") == "ALL_VIRGINIA") & (pl.col("date") <= as_of))
        .sort("date", descending=True)
        .head(1)
    )
    if recent_pjm.is_empty():
        return 0.0

    queue_gw = float(recent_pjm["queue_backlog_gw"][0])

    recent_eia = (
        eia.filter(
            pl.col("fuel_type").is_in(["nuclear", "natural_gas"]) & (pl.col("date") <= as_of)
        )
        .group_by("fuel_type")
        .agg(pl.col("capacity_gw").sort_by("date", descending=True).first())
    )
    if recent_eia.is_empty():
        return 0.0
    baseload_gw = float(recent_eia["capacity_gw"].sum())
    if baseload_gw == 0:
        return 0.0

    raw = queue_gw / baseload_gw

    window_start = as_of - timedelta(days=lookback_days)
    history = pjm.filter(
        (pl.col("zone") == "ALL_VIRGINIA") &
        (pl.col("date") >= window_start) &
        (pl.col("date") <= as_of)
    )
    if len(history) < 2:
        return float(min(raw, 1.0))

    ratios = (history["queue_backlog_gw"] / baseload_gw).to_numpy()
    lo, hi = float(ratios.min()), float(ratios.max())
    if hi == lo:
        return 0.5
    return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))


def compute_geo_tailwind_score(
    exposure: dict[str, float],
    owid_path: Path,
    as_of: date,
) -> float:
    """
    Compute geo-weighted energy tailwind score for a ticker.

    For each region, tailwind = 0.6 * renewable_growth_yoy + 0.4 * (1 - carbon_norm).
    Returns weighted average across the ticker's regional exposure.
    Returns 0.0 if OWID data unavailable.
    """
    if not owid_path.exists():
        return 0.0

    owid = pl.read_parquet(owid_path)
    as_of_year = as_of.year

    score_total = 0.0
    weight_total = 0.0

    for region, weight in exposure.items():
        countries = _REGION_COUNTRIES.get(region, [])
        if not countries:
            continue

        region_data = owid.filter(pl.col("country").is_in(countries))
        if region_data.is_empty():
            continue

        curr = (
            region_data.filter(pl.col("year") <= as_of_year)
            .group_by("country")
            .agg(
                pl.col("renewables_pct").sort_by("year", descending=True).first(),
                pl.col("carbon_intensity_gco2_per_kwh").sort_by("year", descending=True).first(),
            )
        )
        prev = (
            region_data.filter(pl.col("year") <= as_of_year - 1)
            .group_by("country")
            .agg(
                pl.col("renewables_pct").sort_by("year", descending=True).first(),
                pl.col("carbon_intensity_gco2_per_kwh").sort_by("year", descending=True).first(),
            )
        )

        if curr.is_empty():
            continue

        curr_ren = float(curr["renewables_pct"].mean())
        prev_ren = float(prev["renewables_pct"].mean()) if not prev.is_empty() else curr_ren
        ren_growth = max(0.0, curr_ren - prev_ren)

        curr_carbon = float(curr["carbon_intensity_gco2_per_kwh"].mean())
        carbon_norm = float(np.clip(curr_carbon / 500.0, 0.0, 1.0))
        carbon_tailwind = 1.0 - carbon_norm

        region_tailwind = 0.6 * ren_growth * 10 + 0.4 * carbon_tailwind  # scale growth to ~[0,1]
        score_total += weight * region_tailwind
        weight_total += weight

    if weight_total == 0:
        return 0.0
    return float(np.clip(score_total / weight_total, 0.0, 1.0))


def join_energy_geo_features(
    df: pl.DataFrame,
    energy_dir: Path | None = None,
    geo_csv: Path | None = None,
    owid_path: Path | None = None,
) -> pl.DataFrame:
    """
    Add us_power_moat_score and geo_weighted_tailwind_score to the training spine.

    Args:
        df: Training spine with columns [ticker, date, ...].
        energy_dir: Path to data/raw/energy/.
        geo_csv: Path to data/manual/ticker_geo_exposure.csv.
        owid_path: Path to data/raw/energy/energy_geo/country_energy.parquet.

    Returns df with two new float64 columns.
    """
    _ROOT = Path(__file__).parent.parent

    if energy_dir is None:
        energy_dir = _ROOT / "data" / "raw" / "energy"
    if geo_csv is None:
        geo_csv = _ROOT / "data" / "manual" / "ticker_geo_exposure.csv"
    if owid_path is None:
        owid_path = energy_dir / "energy_geo" / "country_energy.parquet"

    eia_path = energy_dir / "eia_capacity.parquet"
    pjm_path = energy_dir / "pjm_queue.parquet"

    eia = pl.read_parquet(eia_path) if eia_path.exists() else pl.DataFrame(
        schema={"date": pl.Date, "fuel_type": pl.Utf8, "capacity_gw": pl.Float64}
    )
    pjm = pl.read_parquet(pjm_path) if pjm_path.exists() else pl.DataFrame(
        schema={"date": pl.Date, "zone": pl.Utf8, "queue_backlog_gw": pl.Float64, "project_count": pl.Int32}
    )

    unique_tickers = df["ticker"].unique().to_list()
    geo_exposure: dict[str, dict[str, float]] = {
        t: load_geo_exposure(geo_csv, t) for t in unique_tickers
    }

    moat_scores = []
    tailwind_scores = []

    for row in df.select(["ticker", "date"]).iter_rows(named=True):
        ticker = row["ticker"]
        as_of  = row["date"]

        if ticker in _POWER_TICKERS:
            moat = compute_us_power_moat_score(pjm, eia, as_of)
        else:
            moat = 0.0

        tailwind = compute_geo_tailwind_score(geo_exposure[ticker], owid_path, as_of)

        moat_scores.append(moat)
        tailwind_scores.append(tailwind)

    return df.with_columns([
        pl.Series("us_power_moat_score", moat_scores, dtype=pl.Float64),
        pl.Series("geo_weighted_tailwind_score", tailwind_scores, dtype=pl.Float64),
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _LOG.info("Energy geo features are computed on-demand in train.py — no standalone run needed.")
    _LOG.info("Run python ingestion/eia_ingestion.py first to fetch the source data.")
