"""Labor market signal features.

Features (LABOR_FEATURE_COLS):
    gov_ai_hiring_30d        — count of federal AI/ML job postings in 30d rolling window
    gov_ai_hiring_momentum   — recent 30d postings minus prior 30d (government AI investment)
    tech_job_openings_index  — BLS JOLTS NAICS 334 job openings, most recent month (thousands)
    tech_job_openings_momentum — current month openings minus previous month (hiring acceleration)

All 4 features are market-wide (joined on date only).
All features zero-filled when data is absent.

Tier routing: medium + long only (monthly data too slow for 5d/20d horizons).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

from ingestion.usajobs_ingestion import _SCHEMA as _POSTING_SCHEMA
from ingestion.bls_jolts_ingestion import _SCHEMA as _JOLTS_SCHEMA

_LOG = logging.getLogger(__name__)

LABOR_FEATURE_COLS: list[str] = [
    "gov_ai_hiring_30d",
    "gov_ai_hiring_momentum",
    "tech_job_openings_index",
    "tech_job_openings_momentum",
]


def _load_postings(usajobs_dir: Path) -> pl.DataFrame:
    files = sorted(usajobs_dir.glob("date=*/postings.parquet")) if usajobs_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_POSTING_SCHEMA)
    # Dedup: a posting active across multiple weekly snapshots should count once
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .sort("date")
        .unique(subset=["posting_id"], keep="last")
    )


def _load_jolts(jolts_dir: Path) -> pl.DataFrame:
    files = sorted(jolts_dir.glob("date=*/openings.parquet")) if jolts_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_JOLTS_SCHEMA)
    # Dedup: keep the most recent snapshot's value for each (year, period)
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .sort("date")
        .unique(subset=["year", "period"], keep="last")
    )


def join_labor_features(
    df: pl.DataFrame,
    usajobs_dir: Path,
    jolts_dir: Path,
) -> pl.DataFrame:
    """Left-join labor market features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'date' (Date) column.
        usajobs_dir: Root of data/raw/usajobs/ Hive tree.
        jolts_dir: Root of data/raw/bls_jolts/ Hive tree.

    Returns:
        df with LABOR_FEATURE_COLS appended (Float64). Zero-filled.
    """
    postings = _load_postings(usajobs_dir)
    jolts = _load_jolts(jolts_dir)
    query_dates = df.select(["date"]).unique()

    with duckdb.connect() as con:
        con.register("query_dates", query_dates.to_arrow())

        # USAJOBS: rolling 30d count and prior-30d momentum
        if not postings.is_empty():
            con.register("postings", postings.to_arrow())
            usajobs_result = con.execute("""
                SELECT
                    q.date,
                    COALESCE(CAST(SUM(CASE
                        WHEN p.posted_date >= q.date - INTERVAL 30 DAY
                             AND p.posted_date <= q.date
                        THEN 1 ELSE 0 END) AS DOUBLE), 0.0) AS gov_ai_hiring_30d,
                    COALESCE(CAST(SUM(CASE
                        WHEN p.posted_date >= q.date - INTERVAL 30 DAY
                             AND p.posted_date <= q.date
                        THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    - COALESCE(CAST(SUM(CASE
                        WHEN p.posted_date >= q.date - INTERVAL 60 DAY
                             AND p.posted_date < q.date - INTERVAL 30 DAY
                        THEN 1 ELSE 0 END) AS DOUBLE), 0.0)
                    AS gov_ai_hiring_momentum
                FROM query_dates q
                LEFT JOIN postings p
                    ON p.posted_date <= q.date
                    AND p.posted_date >= q.date - INTERVAL 60 DAY
                GROUP BY q.date
            """).pl()
        else:
            usajobs_result = pl.DataFrame(schema={
                "date": pl.Date,
                "gov_ai_hiring_30d": pl.Float64,
                "gov_ai_hiring_momentum": pl.Float64,
            })

        # BLS JOLTS: most recent month <= query date (index) and prior month (momentum)
        if not jolts.is_empty():
            con.register("jolts", jolts.to_arrow())
            jolts_result = con.execute("""
                WITH jolts_dated AS (
                    SELECT
                        value,
                        MAKE_DATE(year, CAST(SUBSTR(period, 2) AS INTEGER), 1) AS period_date
                    FROM jolts
                ),
                ranked AS (
                    SELECT
                        q.date,
                        j.value,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY j.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN jolts_dated j
                    WHERE j.period_date <= q.date
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN value END), 0.0)
                        AS tech_job_openings_index,
                    CASE
                        WHEN MAX(CASE WHEN rn = 2 THEN value END) IS NULL THEN 0.0
                        ELSE MAX(CASE WHEN rn = 1 THEN value END)
                           - MAX(CASE WHEN rn = 2 THEN value END)
                    END AS tech_job_openings_momentum
                FROM ranked
                GROUP BY date
            """).pl()
        else:
            jolts_result = pl.DataFrame(schema={
                "date": pl.Date,
                "tech_job_openings_index": pl.Float64,
                "tech_job_openings_momentum": pl.Float64,
            })

    # Join both feature sets directly to original df
    df = df.join(usajobs_result, on="date", how="left")
    df = df.join(jolts_result, on="date", how="left")

    # Zero-fill backstop: catches nulls from partial joins and empty windows
    for col in LABOR_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
