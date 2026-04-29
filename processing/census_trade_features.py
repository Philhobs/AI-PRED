"""Census international trade signal features.

Features (CENSUS_TRADE_FEATURE_COLS):
    semicon_import_value         — US semiconductor import value, most recent month ≤ query date (USD millions)
    semicon_import_momentum      — MoM change in semiconductor import value (USD millions)
    dc_equipment_import_value    — Data center equipment import value, most recent month (USD millions)
    dc_equipment_import_momentum — MoM change in DC equipment import value (USD millions)
    china_semicon_export_share   — US semiconductor exports to China / total (0–1 ratio)
    taiwan_semicon_import_share  — Taiwan's share of US semiconductor imports (0–1 ratio)

All 6 features are market-wide (joined on date only).
All features zero-filled when data is absent.

Tier routing: medium + long only (monthly data too slow for 5d/20d horizons).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

from ingestion.census_trade_ingestion import _SCHEMA as _TRADE_SCHEMA

_LOG = logging.getLogger(__name__)

CENSUS_TRADE_FEATURE_COLS: list[str] = [
    "semicon_import_value",
    "semicon_import_momentum",
    "dc_equipment_import_value",
    "dc_equipment_import_momentum",
    "china_semicon_export_share",
    "taiwan_semicon_import_share",
]


def _load_trade(census_trade_dir: Path) -> pl.DataFrame:
    files = sorted(census_trade_dir.glob("date=*/trade.parquet")) if census_trade_dir.exists() else []
    if not files:
        return pl.DataFrame(schema=_TRADE_SCHEMA)
    # Dedup: keep most recent snapshot's value for each (direction, hs_code, partner_code, year, month)
    return (
        pl.concat([pl.read_parquet(f) for f in files])
        .sort("date")
        .unique(subset=["direction", "hs_code", "partner_code", "year", "month"], keep="last")
    )


def join_census_trade_features(
    df: pl.DataFrame,
    census_trade_dir: Path,
) -> pl.DataFrame:
    """Left-join Census trade features to df. Missing rows zero-filled.

    Args:
        df: Input DataFrame with 'date' (Date) column.
        census_trade_dir: Root of data/raw/census_trade/ Hive tree.

    Returns:
        df with CENSUS_TRADE_FEATURE_COLS appended (Float64). Zero-filled.
    """
    trade = _load_trade(census_trade_dir)
    query_dates = df.select(["date"]).unique()

    with duckdb.connect() as con:
        con.register("query_dates", query_dates.to_arrow())

        if not trade.is_empty():
            con.register("trade", trade.to_arrow())

            # Semiconductor import value + momentum (HS 8541+8542, all partners)
            semicon_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month, SUM(value_usd) / 1e6 AS value_m
                    FROM trade
                    WHERE direction = 'import'
                      AND hs_code IN ('8541', '8542')
                      AND partner_code = 'ALL'
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.value_m,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date + INTERVAL '50' DAY <= q.date  -- Census FT900 publication lag ~50d (point-in-time)
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN value_m END), 0.0)
                        AS semicon_import_value,
                    CASE
                        WHEN MAX(CASE WHEN rn = 2 THEN value_m END) IS NULL THEN 0.0
                        ELSE MAX(CASE WHEN rn = 1 THEN value_m END)
                           - MAX(CASE WHEN rn = 2 THEN value_m END)
                    END AS semicon_import_momentum
                FROM ranked
                GROUP BY date
            """).pl()

            # DC equipment import value + momentum (HS 8471+8473, all partners)
            dc_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month, SUM(value_usd) / 1e6 AS value_m
                    FROM trade
                    WHERE direction = 'import'
                      AND hs_code IN ('8471', '8473')
                      AND partner_code = 'ALL'
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.value_m,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date + INTERVAL '50' DAY <= q.date  -- Census FT900 publication lag ~50d (point-in-time)
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN value_m END), 0.0)
                        AS dc_equipment_import_value,
                    CASE
                        WHEN MAX(CASE WHEN rn = 2 THEN value_m END) IS NULL THEN 0.0
                        ELSE MAX(CASE WHEN rn = 1 THEN value_m END)
                           - MAX(CASE WHEN rn = 2 THEN value_m END)
                    END AS dc_equipment_import_momentum
                FROM ranked
                GROUP BY date
            """).pl()

            # China semiconductor export share
            china_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month,
                        SUM(CASE WHEN partner_code = '5700' THEN value_usd ELSE 0.0 END) AS china_val,
                        SUM(CASE WHEN partner_code = 'ALL' THEN value_usd ELSE 0.0 END) AS total_val
                    FROM trade
                    WHERE direction = 'export'
                      AND hs_code IN ('8541', '8542')
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.china_val, d.total_val,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date + INTERVAL '50' DAY <= q.date  -- Census FT900 publication lag ~50d (point-in-time)
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN china_val END), 0.0)
                    / GREATEST(COALESCE(MAX(CASE WHEN rn = 1 THEN total_val END), 1.0), 1.0)
                        AS china_semicon_export_share
                FROM ranked
                GROUP BY date
            """).pl()

            # Taiwan semiconductor import share
            taiwan_result = con.execute("""
                WITH monthly AS (
                    SELECT year, month,
                        SUM(CASE WHEN partner_code = '5830' THEN value_usd ELSE 0.0 END) AS taiwan_val,
                        SUM(CASE WHEN partner_code = 'ALL' THEN value_usd ELSE 0.0 END) AS total_val
                    FROM trade
                    WHERE direction = 'import'
                      AND hs_code IN ('8541', '8542')
                    GROUP BY year, month
                ),
                dated AS (
                    SELECT *, MAKE_DATE(year, month, 1) AS period_date FROM monthly
                ),
                ranked AS (
                    SELECT q.date, d.taiwan_val, d.total_val,
                        ROW_NUMBER() OVER (
                            PARTITION BY q.date ORDER BY d.period_date DESC
                        ) AS rn
                    FROM query_dates q
                    CROSS JOIN dated d
                    WHERE d.period_date + INTERVAL '50' DAY <= q.date  -- Census FT900 publication lag ~50d (point-in-time)
                )
                SELECT
                    date,
                    COALESCE(MAX(CASE WHEN rn = 1 THEN taiwan_val END), 0.0)
                    / GREATEST(COALESCE(MAX(CASE WHEN rn = 1 THEN total_val END), 1.0), 1.0)
                        AS taiwan_semicon_import_share
                FROM ranked
                GROUP BY date
            """).pl()

        else:
            semicon_result = pl.DataFrame(schema={
                "date": pl.Date,
                "semicon_import_value": pl.Float64,
                "semicon_import_momentum": pl.Float64,
            })
            dc_result = pl.DataFrame(schema={
                "date": pl.Date,
                "dc_equipment_import_value": pl.Float64,
                "dc_equipment_import_momentum": pl.Float64,
            })
            china_result = pl.DataFrame(schema={
                "date": pl.Date,
                "china_semicon_export_share": pl.Float64,
            })
            taiwan_result = pl.DataFrame(schema={
                "date": pl.Date,
                "taiwan_semicon_import_share": pl.Float64,
            })

    # Join all feature sets directly to original df
    df = df.join(semicon_result, on="date", how="left")
    df = df.join(dc_result, on="date", how="left")
    df = df.join(china_result, on="date", how="left")
    df = df.join(taiwan_result, on="date", how="left")

    # Zero-fill backstop
    for col in CENSUS_TRADE_FEATURE_COLS:
        df = df.with_columns(pl.col(col).fill_null(0.0))

    return df
