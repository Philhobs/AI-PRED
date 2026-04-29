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

# Publication lag — how many days after the period date the data becomes
# publicly observable. Applied at join time to prevent lookahead. These are
# conservative defaults (typically the data is published around or before
# these dates).
#
#   FRED macro (NEWORDER/CFNAI/IPG3331S/WPU114): mostly released 30-45 days
#       after period end (varies by series); 45d is a safe default.
#   BLS JOLTS NAICS 333: released ~30 business days after the reference month.
#   USPTO patent applications: pre-grant publications appear ~18 months after
#       filing by statute (35 USC 122(b)). When patent ingestion is restored
#       it should switch to the publication_date column directly; until then
#       the buckets are empty so this lag is moot.
_FRED_PUBLICATION_LAG_DAYS = 45
_JOLTS_PUBLICATION_LAG_DAYS = 30
_PATENT_PUBLICATION_LAG_DAYS = 540


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
    """[unit-test helper] Most recent observation in df where date_col <= query_date
    and within tolerance. Used by tests for boundary verification; the production
    join_physical_ai_features path uses vectorized join_asof — see below."""
    if df.is_empty():
        return None
    eligible = df.filter(pl.col(date_col) <= query_date)
    if eligible.is_empty():
        return None
    row = eligible.tail(1).row(0, named=True)
    if (query_date - row[date_col]).days > tolerance_days:
        return None
    return row[value_col]


def _asof_join_level(query_df: pl.DataFrame,
                     query_date_col: str,
                     src_df: pl.DataFrame,
                     src_date_col: str,
                     src_value_col: str,
                     alias: str,
                     tolerance_days: int,
                     publication_lag_days: int = 0) -> pl.DataFrame:
    """join_asof backward with tolerance — returns query_df + one new {alias: Float64} column.

    publication_lag_days: shift the source date forward by this many days
    before the asof match. This is the point-in-time correction: the data
    keyed at period_date is not publicly observable until period_date +
    publication_lag_days, so the spine's join must wait that long.

    When src is empty, fills the new column with nulls instead of crashing.
    """
    if src_df.is_empty():
        return query_df.with_columns(pl.lit(None).cast(pl.Float64).alias(alias))
    src_sorted = (
        src_df.sort(src_date_col)
        .select([
            (pl.col(src_date_col) + pl.duration(days=publication_lag_days))
                .cast(pl.Date).alias("_join_date"),
            pl.col(src_value_col).cast(pl.Float64).alias(alias),
        ])
        .sort("_join_date")
    )
    return (
        query_df.sort(query_date_col)
        .join_asof(
            src_sorted,
            left_on=query_date_col,
            right_on="_join_date",
            strategy="backward",
            tolerance=f"{tolerance_days}d",
        )
        .drop("_join_date")
    )


def join_physical_ai_features(
    spine: pl.DataFrame,
    fred_dir: Path,
    jolts_dir: Path,
    patents_dir: Path,
) -> pl.DataFrame:
    """Join the 21 physical-AI features onto spine. Spine must have 'ticker' (Utf8) and 'date' (Date).

    Vectorized via Polars join_asof — was a per-date Python loop in the first
    implementation; that became the training-time bottleneck on bigger spines
    (~26k Polars filter ops × N layers). This rewrite is O(unique_dates ×
    n_sources) but each op is a single C-level merge sort + join.
    """
    fred_data: dict[str, pl.DataFrame] = {
        sid: _load_fred_series(fred_dir, sid) for sid in _FRED_COL_MAP
    }
    jolts = _load_jolts(jolts_dir)
    patent_data: dict[str, pl.DataFrame] = {
        bucket: _load_patent_bucket(patents_dir, bucket) for bucket in _PATENT_COL_MAP
    }

    # Build query: unique sorted dates from spine, plus a "date - 365 days" column for yoy lookups.
    query_dates = (
        spine.select("date").unique().sort("date")
        .with_columns(
            (pl.col("date") - pl.duration(days=365)).cast(pl.Date).alias("_prior_date")
        )
    )
    feature_df = query_dates.clone()

    # ── FRED level + (optional) yoy ─────────────────────────────────────────
    for series_id, (level_col, yoy_col) in _FRED_COL_MAP.items():
        src = fred_data[series_id]
        feature_df = _asof_join_level(
            feature_df, "date", src, "date", "value", level_col,
            _FRED_TOLERANCE_DAYS, publication_lag_days=_FRED_PUBLICATION_LAG_DAYS,
        )
        if yoy_col is not None:
            feature_df = _asof_join_level(
                feature_df, "_prior_date", src, "date", "value",
                f"_{yoy_col}_prior", _FRED_TOLERANCE_DAYS,
                publication_lag_days=_FRED_PUBLICATION_LAG_DAYS,
            )
            # yoy = (level - prior) / prior; null when prior is null or zero
            feature_df = feature_df.with_columns(
                pl.when((pl.col(f"_{yoy_col}_prior").is_null()) | (pl.col(f"_{yoy_col}_prior") == 0))
                .then(None)
                .otherwise(
                    (pl.col(level_col) - pl.col(f"_{yoy_col}_prior"))
                    / pl.col(f"_{yoy_col}_prior")
                )
                .alias(yoy_col)
            ).drop(f"_{yoy_col}_prior")

    # ── JOLTS NAICS 333 (level + yoy) ───────────────────────────────────────
    feature_df = _asof_join_level(
        feature_df, "date", jolts, "period_date", "value",
        "phys_ai_machinery_jobs_level", _FRED_TOLERANCE_DAYS,
        publication_lag_days=_JOLTS_PUBLICATION_LAG_DAYS,
    )
    feature_df = _asof_join_level(
        feature_df, "_prior_date", jolts, "period_date", "value",
        "_jobs_prior", _FRED_TOLERANCE_DAYS,
        publication_lag_days=_JOLTS_PUBLICATION_LAG_DAYS,
    )
    feature_df = feature_df.with_columns(
        pl.when((pl.col("_jobs_prior").is_null()) | (pl.col("_jobs_prior") == 0))
        .then(None)
        .otherwise(
            (pl.col("phys_ai_machinery_jobs_level") - pl.col("_jobs_prior"))
            / pl.col("_jobs_prior")
        )
        .alias("phys_ai_machinery_jobs_yoy")
    ).drop("_jobs_prior")

    # ── Patent buckets (count + yoy) ────────────────────────────────────────
    for bucket, (count_col, yoy_col) in _PATENT_COL_MAP.items():
        src = patent_data[bucket]
        feature_df = _asof_join_level(
            feature_df, "date", src, "quarter_end", "filing_count",
            count_col, _PATENT_TOLERANCE_DAYS,
            publication_lag_days=_PATENT_PUBLICATION_LAG_DAYS,
        )
        feature_df = _asof_join_level(
            feature_df, "_prior_date", src, "quarter_end", "filing_count",
            f"_{yoy_col}_prior", _PATENT_TOLERANCE_DAYS,
            publication_lag_days=_PATENT_PUBLICATION_LAG_DAYS,
        )
        feature_df = feature_df.with_columns(
            pl.when((pl.col(f"_{yoy_col}_prior").is_null()) | (pl.col(f"_{yoy_col}_prior") == 0))
            .then(None)
            .otherwise(
                (pl.col(count_col) - pl.col(f"_{yoy_col}_prior"))
                / pl.col(f"_{yoy_col}_prior")
            )
            .alias(yoy_col)
        ).drop(f"_{yoy_col}_prior")

    feature_df = feature_df.drop("_prior_date")
    return spine.join(feature_df, on="date", how="left")
