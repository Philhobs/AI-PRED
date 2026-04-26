"""Physical-AI feature engineering — macro + labor + patents (21 features).

Features (PHYSICAL_AI_FEATURE_COLS):
  Macro (FRED, monthly, 60-day forward-fill tolerance):
    phys_ai_capgoods_orders_level / _yoy        (NEWORDER)
    phys_ai_cfnai_level                          (CFNAI — PMI substitute, see ingestion docstring)
    phys_ai_machinery_prod_level / _yoy          (IPG3331S)
    phys_ai_machinery_ppi_level / _yoy           (WPU114)
  Labor (BLS JOLTS NAICS 333, monthly, 60-day tolerance):
    phys_ai_machinery_jobs_level / _yoy
  Patents (USPTO physical-AI, quarterly, 120-day tolerance):
    phys_ai_patents_{manipulators|aerial|avs|motion|progcontrol|vision}_count / _yoy

All 21 features apply uniformly to every ticker (model decides per-ticker weight).
Public entry: join_physical_ai_features(spine, fred_dir, jolts_dir, patents_dir).
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import polars as pl

_LOG = logging.getLogger(__name__)

# Public name list — imported by models/train.py
PHYSICAL_AI_FEATURE_COLS: list[str] = [
    "phys_ai_capgoods_orders_level",
    "phys_ai_capgoods_orders_yoy",
    "phys_ai_cfnai_level",
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
]

# (FRED series id) → (level column name, yoy column name | None)
_FRED_COL_MAP: dict[str, tuple[str, Optional[str]]] = {
    "NEWORDER": ("phys_ai_capgoods_orders_level", "phys_ai_capgoods_orders_yoy"),
    "CFNAI":    ("phys_ai_cfnai_level",           None),
    "IPG3331S": ("phys_ai_machinery_prod_level",  "phys_ai_machinery_prod_yoy"),
    "WPU114":   ("phys_ai_machinery_ppi_level",   "phys_ai_machinery_ppi_yoy"),
}

# (CPC bucket) → (count column name, yoy column name)
_PATENT_COL_MAP: dict[str, tuple[str, str]] = {
    "B25J":   ("phys_ai_patents_manipulators_count",  "phys_ai_patents_manipulators_yoy"),
    "B64":    ("phys_ai_patents_aerial_count",        "phys_ai_patents_aerial_yoy"),
    "B60W":   ("phys_ai_patents_avs_count",           "phys_ai_patents_avs_yoy"),
    "G05D1":  ("phys_ai_patents_motion_count",        "phys_ai_patents_motion_yoy"),
    "G05B19": ("phys_ai_patents_progcontrol_count",   "phys_ai_patents_progcontrol_yoy"),
    "G06V":   ("phys_ai_patents_vision_count",        "phys_ai_patents_vision_yoy"),
}

_JOLTS_NAICS_333 = "JTS333000000000000JOL"
_FRED_TOLERANCE_DAYS = 60
_PATENT_TOLERANCE_DAYS = 120


def _yoy(current: float | None, prior: float | None) -> float | None:
    """Year-over-year ratio. None if either value is missing or prior is 0."""
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / prior


def _load_fred_series(fred_dir: Path, series_id: str) -> pl.DataFrame:
    path = fred_dir / f"{series_id}.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"date": pl.Date, "value": pl.Float64})
    return pl.read_parquet(path).sort("date")


def _load_jolts(jolts_dir: Path) -> pl.DataFrame:
    """Load all BLS JOLTS snapshots, filter to NAICS 333. Each row: period_date + value."""
    if not jolts_dir.exists():
        return pl.DataFrame(schema={"period_date": pl.Date, "value": pl.Float64})
    files = sorted(jolts_dir.glob("date=*/openings.parquet"))
    if not files:
        return pl.DataFrame(schema={"period_date": pl.Date, "value": pl.Float64})
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = df.filter(pl.col("series_id") == _JOLTS_NAICS_333)
    df = df.filter(pl.col("period").str.starts_with("M"))
    df = df.with_columns(
        pl.date(pl.col("year"), pl.col("period").str.slice(1, 2).cast(pl.Int32), 1).alias("period_date")
    )
    df = df.unique(subset=["period_date"], keep="last").sort("period_date")
    return df.select(["period_date", "value"])


def _load_patent_bucket(patents_dir: Path, bucket: str) -> pl.DataFrame:
    path = patents_dir / f"cpc_class={bucket}" / "filings.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"quarter_end": pl.Date, "filing_count": pl.Int64})
    df = pl.read_parquet(path).select(["quarter_end", "filing_count"]).sort("quarter_end")
    return df


def _value_at(df: pl.DataFrame, query_date: date, value_col: str, date_col: str,
              tolerance_days: int) -> float | None:
    """Most recent observation in df where date_col <= query_date and within tolerance.
    Returns None if no row satisfies the constraint."""
    if df.is_empty():
        return None
    eligible = df.filter(pl.col(date_col) <= query_date)
    if eligible.is_empty():
        return None
    row = eligible.tail(1).row(0, named=True)
    if (query_date - row[date_col]).days > tolerance_days:
        return None
    return row[value_col]


def _value_one_year_prior(df: pl.DataFrame, query_date: date, value_col: str, date_col: str,
                          tolerance_days: int) -> float | None:
    """Most recent observation where date <= (query_date - 365 days), within tolerance."""
    from datetime import timedelta
    target = query_date - timedelta(days=365)
    return _value_at(df, target, value_col, date_col, tolerance_days)


def join_physical_ai_features(
    spine: pl.DataFrame,
    fred_dir: Path,
    jolts_dir: Path,
    patents_dir: Path,
) -> pl.DataFrame:
    """Join the 21 physical-AI features onto spine. Spine must have 'ticker' (Utf8) and 'date' (Date)."""
    fred_data: dict[str, pl.DataFrame] = {
        sid: _load_fred_series(fred_dir, sid) for sid in _FRED_COL_MAP
    }
    jolts = _load_jolts(jolts_dir)
    patent_data: dict[str, pl.DataFrame] = {
        bucket: _load_patent_bucket(patents_dir, bucket) for bucket in _PATENT_COL_MAP
    }

    # For each unique date, compute the 21 column values once, then join back to spine.
    unique_dates = spine.select("date").unique().sort("date")["date"].to_list()
    rows: list[dict] = []
    for d in unique_dates:
        row: dict = {"date": d}
        # FRED
        for series_id, (level_col, yoy_col) in _FRED_COL_MAP.items():
            df = fred_data[series_id]
            level = _value_at(df, d, "value", "date", _FRED_TOLERANCE_DAYS)
            row[level_col] = level
            if yoy_col is not None:
                prior = _value_one_year_prior(df, d, "value", "date", _FRED_TOLERANCE_DAYS)
                row[yoy_col] = _yoy(level, prior)
        # JOLTS NAICS 333
        jolts_level = _value_at(jolts, d, "value", "period_date", _FRED_TOLERANCE_DAYS)
        jolts_prior = _value_one_year_prior(jolts, d, "value", "period_date", _FRED_TOLERANCE_DAYS)
        row["phys_ai_machinery_jobs_level"] = jolts_level
        row["phys_ai_machinery_jobs_yoy"] = _yoy(jolts_level, jolts_prior)
        # Patents
        for bucket, (count_col, yoy_col) in _PATENT_COL_MAP.items():
            df = patent_data[bucket]
            count = _value_at(df, d, "filing_count", "quarter_end", _PATENT_TOLERANCE_DAYS)
            prior = _value_one_year_prior(df, d, "filing_count", "quarter_end", _PATENT_TOLERANCE_DAYS)
            row[count_col] = float(count) if count is not None else None
            row[yoy_col] = _yoy(
                float(count) if count is not None else None,
                float(prior) if prior is not None else None,
            )
        rows.append(row)

    schema = {"date": pl.Date}
    for col in PHYSICAL_AI_FEATURE_COLS:
        schema[col] = pl.Float64
    feature_df = pl.DataFrame(rows, schema=schema)

    return spine.join(feature_df, on="date", how="left")
